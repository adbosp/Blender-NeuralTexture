"""Pure-numpy image helpers (no bpy).

Conventions used across the whole add-on:
  * Images are float32 numpy arrays shaped (H, W, C) or (H, W).
  * Row 0 is the BOTTOM row, matching Blender's ``Image.pixels`` layout and
    OpenGL/UV conventions (V grows with the row index).
  * Masks are float32 in [0, 1] or bool.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Basic conversions
# ----------------------------------------------------------------------------

def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of an (H, W, >=3) array."""
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def ensure_channels(img: np.ndarray, c: int) -> np.ndarray:
    """Return img with exactly ``c`` channels (repeat gray / drop / pad alpha)."""
    if img.ndim == 2:
        img = img[..., None]
    have = img.shape[2]
    if have == c:
        return img
    if have == 1:
        out = np.repeat(img, min(c, 3), axis=2)
        if c == 4:
            out = np.concatenate([out, np.ones_like(img)], axis=2)
        return out
    if have > c:
        return img[..., :c]
    # have in (2, 3), c == 4
    pad = np.ones(img.shape[:2] + (c - have,), dtype=img.dtype)
    return np.concatenate([img, pad], axis=2)


def resize_bilinear(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize with pixel-centre alignment (numpy only)."""
    src_h, src_w = img.shape[:2]
    if (src_h, src_w) == (h, w):
        return img.astype(np.float32, copy=True)
    ys = (np.arange(h, dtype=np.float32) + 0.5) * (src_h / h) - 0.5
    xs = (np.arange(w, dtype=np.float32) + 0.5) * (src_w / w) - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return sample_bilinear(img, xx, yy)


def sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample ``img`` at float pixel coordinates (x = column, y = row).

    Coordinates are clamped to the image. Output shape is x.shape (+ C).
    """
    h, w = img.shape[:2]
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    a = img[y0, x0]
    b = img[y0, x1]
    c = img[y1, x0]
    d = img[y1, x1]
    top = a * (1.0 - fx) + b * fx
    bot = c * (1.0 - fx) + d * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def sample_nearest(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    xi = np.clip(np.rint(x), 0, w - 1).astype(np.int64)
    yi = np.clip(np.rint(y), 0, h - 1).astype(np.int64)
    return img[yi, xi]


# ----------------------------------------------------------------------------
# Blurs
# ----------------------------------------------------------------------------

def _box_blur_axis(img: np.ndarray, r: int, axis: int) -> np.ndarray:
    if r <= 0:
        return img
    n = img.shape[axis]
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r + 1, r)
    p = np.pad(img, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def box_blur(img: np.ndarray, r: int) -> np.ndarray:
    return _box_blur_axis(_box_blur_axis(img, r, 0), r, 1)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Approximate Gaussian via three box passes (O(1) per pixel)."""
    if sigma <= 0.3:
        return img.astype(np.float32, copy=True)
    # Box radius giving the requested sigma after 3 passes.
    r = max(1, int(round((np.sqrt(4.0 * sigma * sigma + 1.0) - 1.0) / 2.0)))
    out = img.astype(np.float32)
    for _ in range(3):
        out = box_blur(out, r)
    return out


def masked_gaussian_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: blur only valid texels, ignore empty ones.

    Prevents UV-island borders from bleeding the background into the result.
    """
    m = mask.astype(np.float32)
    if img.ndim == 3:
        mb = m[..., None]
    else:
        mb = m
    num = gaussian_blur(img * mb, sigma)
    den = gaussian_blur(m, sigma)
    if img.ndim == 3:
        den = den[..., None]
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), img).astype(np.float32)


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary erosion with a square kernel of radius r."""
    m = mask.astype(bool)
    for _ in range(max(0, r)):
        p = np.pad(m, 1, mode="constant", constant_values=False)
        m = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    return m


# ----------------------------------------------------------------------------
# Hole filling (UV padding / unseen texels)
# ----------------------------------------------------------------------------

def pull_push_fill(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill texels where mask==0 using a pull-push pyramid.

    Valid texels are left untouched. Holes receive a smooth extrapolation
    of nearby valid data, which is ideal for UV-seam padding.
    """
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    w0 = mask.astype(np.float32)
    if w0.max() <= 0:
        return img[..., 0] if squeeze else img

    levels = []
    col = img.astype(np.float32) * w0[..., None]
    wt = w0.copy()
    # Pull: downsample premultiplied colour + weight.
    while True:
        levels.append((col, wt))
        h, w = wt.shape
        if h <= 1 and w <= 1:
            break
        ph, pw = h + (h & 1), w + (w & 1)
        c = np.pad(col, ((0, ph - h), (0, pw - w), (0, 0)))
        ww = np.pad(wt, ((0, ph - h), (0, pw - w)))
        c = c.reshape(ph // 2, 2, pw // 2, 2, -1).sum(axis=(1, 3))
        ww = ww.reshape(ph // 2, 2, pw // 2, 2).sum(axis=(1, 3))
        # Clamp weight so fully-valid coarse texels have weight 1.
        scale = np.minimum(ww, 1.0) / np.maximum(ww, 1e-8)
        col = c * scale[..., None]
        wt = np.minimum(ww, 1.0)

    # Push: upsample coarse estimates into holes.
    col, wt = levels[-1]
    est = col / np.maximum(wt, 1e-8)[..., None]
    for c, w in reversed(levels[:-1]):
        h, wd = w.shape
        up = np.repeat(np.repeat(est, 2, axis=0), 2, axis=1)[:h, :wd]
        # Soften blockiness from nearest upsampling.
        up = box_blur(up, 1)
        est = c + up * (1.0 - w)[..., None]
        est = est / np.maximum(w + (1.0 - w), 1e-8)[..., None]
    out = np.where(mask.astype(bool)[..., None], img, est).astype(np.float32)
    return out[..., 0] if squeeze else out


# ----------------------------------------------------------------------------
# Quantisation helpers for control images
# ----------------------------------------------------------------------------

def normalize_range(x: np.ndarray, mask: np.ndarray | None = None,
                    lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    vals = x[mask.astype(bool)] if mask is not None else x.reshape(-1)
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def flip_rows(img: np.ndarray) -> np.ndarray:
    """Convert between bottom-up (Blender) and top-down (PNG/PIL) row order."""
    return np.ascontiguousarray(img[::-1])
