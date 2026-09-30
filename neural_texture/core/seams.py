"""UV seam refinement (V3).

Texels on either side of a UV seam are far apart in the texture but share the
same 3D position. Small colour differences there show up as visible seams.
We find border texels that coincide in 3D but not in UV, average their colours,
and diffuse the correction a few texels into each island (a cheap
Poisson-style blend). Fully vectorised; no island labelling needed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .imageops import gaussian_blur


def border_mask(coverage: np.ndarray) -> np.ndarray:
    c = coverage.astype(bool)
    p = np.pad(c, 1, constant_values=False)
    interior = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return c & ~interior


def _texel_spacing(pos: np.ndarray, coverage: np.ndarray) -> float:
    c = coverage.astype(bool)
    both = c[:, 1:] & c[:, :-1]
    if not both.any():
        return 1e-3
    d = np.linalg.norm(pos[:, 1:] - pos[:, :-1], axis=-1)[both]
    return float(np.median(d)) if d.size else 1e-3


def seam_corrections(color: np.ndarray, pos: np.ndarray, coverage: np.ndarray,
                     band: int = 2, min_uv_px: float = 4.0, cell_scale: float = 1.5
                     ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return (delta (H,W,C), weight (H,W), stats) at seam texels."""
    H, W = coverage.shape
    b = border_mask(coverage)
    for _ in range(max(0, band - 1)):  # include a thin band inside the border
        p = np.pad(b, 1)
        b = (b | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & coverage.astype(bool)
    ys, xs = np.nonzero(b)
    C = color.shape[2]
    delta = np.zeros((H, W, C), np.float32)
    wmap = np.zeros((H, W), np.float32)
    if ys.size == 0:
        return delta, wmap, {"seam_texels": 0}
    cell = _texel_spacing(pos, coverage) * cell_scale * max(1, band)
    P = pos[ys, xs].astype(np.float64)
    col = color[ys, xs].astype(np.float64)
    uv = np.stack([xs, ys], 1).astype(np.float64)

    acc_d = np.zeros((ys.size, C))
    acc_w = np.zeros(ys.size)
    for shift in (0.0, 0.5):  # two staggered grids catch pairs split by a cell boundary
        key3 = np.floor(P / cell + shift).astype(np.int64)
        _, inv = np.unique(key3, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        g = inv.max() + 1
        cnt = np.bincount(inv, minlength=g).astype(np.float64)
        mean_c = np.stack([np.bincount(inv, col[:, k], g) for k in range(C)], 1) / cnt[:, None]
        mean_uv = np.stack([np.bincount(inv, uv[:, k], g) for k in range(2)], 1) / cnt[:, None]
        # UV spread of the group: a real seam group has members far apart in UV.
        spread = np.sqrt(np.bincount(inv, np.sum((uv - mean_uv[inv]) ** 2, 1), g) / cnt)
        is_seam = (spread[inv] >= min_uv_px / 2.0) & (cnt[inv] >= 2)
        acc_d[is_seam] += mean_c[inv][is_seam] - col[is_seam]
        acc_w[is_seam] += 1.0
    ok = acc_w > 0
    delta[ys[ok], xs[ok]] = (acc_d[ok] / acc_w[ok, None]).astype(np.float32)
    wmap[ys[ok], xs[ok]] = 1.0
    return delta, wmap, {"seam_texels": int(ok.sum()), "cell": cell}


def refine_seams(color: np.ndarray, pos: np.ndarray, coverage: np.ndarray,
                 radius_px: float = 6.0, strength: float = 1.0, **kw) -> Tuple[np.ndarray, Dict]:
    delta, w, stats = seam_corrections(color, pos, coverage, **kw)
    if stats["seam_texels"] == 0 or strength <= 0:
        return color, stats
    num = gaussian_blur(delta * w[..., None], radius_px)
    den = gaussian_blur(w, radius_px)
    spread = num / np.maximum(den, 1e-6)[..., None]
    falloff = np.clip(den / max(float(den.max()), 1e-6) * 2.0, 0.0, 1.0)
    corr = np.where(w[..., None] > 0, delta, spread * falloff[..., None])
    out = color + strength * corr * coverage[..., None]
    stats["mean_abs_delta"] = float(np.abs(delta[w > 0]).mean())
    return np.clip(out, 0.0, 1.0).astype(np.float32), stats


def seam_error(color: np.ndarray, pos: np.ndarray, coverage: np.ndarray, **kw) -> float:
    """Mean colour mismatch across seams (quality metric)."""
    delta, w, st = seam_corrections(color, pos, coverage, **kw)
    return float(np.abs(delta[w > 0]).mean()) if st["seam_texels"] else 0.0
