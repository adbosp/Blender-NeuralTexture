"""Multi-view consistency helpers (V2), all in view space.

Sequential generation loop for view k (k > 0):
    reproj, conf = accumulator.reproject(view_k)       # what we already have
    init  = composite_init(reproj, conf, albedo_k)     # img2img start image
    mask  = inpaint_mask(conf, alpha)                   # 1 = free to generate
    gen   = model(init, mask, controls)                 # latent noise mask
    gen   = harmonize(gen, reproj, conf)                # remove colour drift
    accumulator.add_view(view_k, gen, lock=...)         # confident texels resist
"""

from __future__ import annotations

import numpy as np

from .imageops import erode, gaussian_blur


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    return ~erode(~mask.astype(bool), r)


def inpaint_mask(conf: np.ndarray, alpha: np.ndarray, threshold: float = 0.55,
                 grow_px: int = 6, feather_px: float = 5.0, background: float = 1.0) -> np.ndarray:
    """Soft mask for the image model: 1 = generate, 0 = keep.

    Low-confidence object pixels are regenerated (grown a little so seams
    are repainted rather than stitched); the background is left free so the
    silhouette is not constrained by the grey init colour.
    """
    obj = alpha > 0.5
    unknown = obj & (conf < threshold)
    unknown = dilate(unknown, grow_px) & obj
    m = unknown.astype(np.float32)
    if feather_px > 0:
        m = np.maximum(m, gaussian_blur(m, feather_px) * obj)
    return np.where(obj, np.clip(m, 0.0, 1.0), background).astype(np.float32)


def composite_init(reproj: np.ndarray, conf: np.ndarray, fallback: np.ndarray,
                   alpha: np.ndarray, bg: float = 0.5) -> np.ndarray:
    """Start image: known texture where confident, existing albedo elsewhere."""
    c = np.clip(conf, 0.0, 1.0)[..., None]
    img = reproj[..., :3] * c + fallback[..., :3] * (1.0 - c)
    return np.where((alpha > 0.5)[..., None], img, bg).astype(np.float32)


def harmonize(gen: np.ndarray, ref: np.ndarray, conf: np.ndarray, alpha: np.ndarray,
              min_conf: float = 0.6, strength: float = 1.0, max_gain: float = 1.5,
              min_pixels: int = 200) -> tuple:
    """Per-channel mean/std transfer so the new view matches the texture it
    overlaps. Statistics come only from reliable overlap pixels, the
    correction is applied to the whole view. Returns (image, info).
    """
    m = (alpha > 0.5) & (conf >= min_conf)
    info = {"overlap_pixels": int(m.sum()), "applied": False}
    if m.sum() < min_pixels:
        return gen, info
    out = gen[..., :3].astype(np.float32).copy()
    gains, offsets = [], []
    for c in range(3):
        g = out[..., c][m]
        r = ref[..., c][m]
        gs, rs = float(g.std()), float(r.std())
        gain = np.clip(rs / gs if gs > 1e-4 else 1.0, 1.0 / max_gain, max_gain)
        off = float(r.mean()) - gain * float(g.mean())
        gain = 1.0 + (gain - 1.0) * strength
        off *= strength
        out[..., c] = out[..., c] * gain + off
        gains.append(float(gain))
        offsets.append(off)
    info.update(applied=True, gains=gains, offsets=offsets)
    return np.clip(out, 0.0, 1.0), info


def view_color_error(gen: np.ndarray, ref: np.ndarray, conf: np.ndarray, alpha: np.ndarray,
                     min_conf: float = 0.6) -> float:
    """Mean absolute colour difference on reliable overlap (a consistency metric)."""
    m = (alpha > 0.5) & (conf >= min_conf)
    if not m.any():
        return 0.0
    return float(np.abs(gen[..., :3] - ref[..., :3])[m].mean())
