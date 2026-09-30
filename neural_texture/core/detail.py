"""Detail normal / height abstraction (V2 + V3).

A DetailSource turns synthesis results into a tangent-space DETAIL normal
(only the bands the intent allows). The final normal is always
    RNM(existing base normal, detail, strength)
so macro structure from the asset is preserved (spec section 4).

Sources:
  luminance  - band-passed albedo luminance -> height -> normal (heuristic, V1)
  model      - per-view normals from a neural estimator, accumulated in tangent
               space, then macro removed by "unblending" a low-passed copy
  height     - user/agent supplied height map
  hybrid     - model where its confidence is high, luminance elsewhere
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import normals as nm
from .imageops import luminance, masked_gaussian_blur


# ----------------------------------------------------------------------------
# Rotations / band splitting of normal fields
# ----------------------------------------------------------------------------

def rotate_a_to_b(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply, per element, the minimal rotation taking unit a onto unit b to v."""
    a = nm.normalize(a)
    b = nm.normalize(b)
    k = np.cross(a, b)
    s = np.linalg.norm(k, axis=-1, keepdims=True)
    c = np.sum(a * b, axis=-1, keepdims=True)
    k = k / np.maximum(s, 1e-12)
    kv = np.cross(k, v)
    kdv = np.sum(k * v, axis=-1, keepdims=True)
    out = v * c + kv * s + k * kdv * (1.0 - c)
    # a == b: identity; a == -b: 180deg about X (irrelevant for z>0 hemispheres)
    return np.where(s < 1e-9, np.where(c > 0, v, v * np.array([1.0, -1.0, -1.0])), out).astype(np.float32)


def unblend(full: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Inverse of RNM: the detail d such that RNM(base, d) ~= full."""
    z = np.zeros_like(base)
    z[..., 2] = 1.0
    return nm.normalize(rotate_a_to_b(full, base, z))


def lowpass_normals(n: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.3:
        return n
    return nm.normalize(masked_gaussian_blur(n, mask, sigma))


def normal_band(n: np.ndarray, mask: np.ndarray, sigma_fine: float, sigma_coarse: float) -> np.ndarray:
    """Detail of ``n`` between two scales, as a tangent normal around +Z."""
    fine = lowpass_normals(n, mask, sigma_fine)
    coarse = lowpass_normals(n, mask, sigma_coarse)
    d = unblend(fine, coarse)
    flat = nm.flat_like(n.shape[:2])
    return np.where(mask[..., None] > 0.5, d, flat).astype(np.float32)


# ----------------------------------------------------------------------------
# Height <-> normal
# ----------------------------------------------------------------------------

def normals_to_height(n: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Frankot-Chellappa integration of a tangent normal field (bottom-up rows).

    Returns a zero-mean height (arbitrary units, consistent with
    ``normals.height_to_normal`` at strength 1).
    """
    nz = np.maximum(n[..., 2], 0.05)
    p = -n[..., 0] / nz   # dH/du
    q = -n[..., 1] / nz   # dH/dv (rows)
    if mask is not None:
        p = p * mask
        q = q * mask
    H, W = p.shape
    wx = np.fft.fftfreq(W) * 2 * np.pi
    wy = np.fft.fftfreq(H) * 2 * np.pi
    WX, WY = np.meshgrid(wx, wy)
    # np.gradient is a central difference: spectral response i*sin(w).
    sx, sy = np.sin(WX), np.sin(WY)
    denom = sx ** 2 + sy ** 2
    denom[0, 0] = 1.0
    Z = (-1j * sx * np.fft.fft2(p) - 1j * sy * np.fft.fft2(q)) / denom
    Z[0, 0] = 0.0
    Z[denom < 1e-6] = 0.0
    h = np.real(np.fft.ifft2(Z)).astype(np.float32)
    if mask is not None and mask.any():
        h -= h[mask > 0.5].mean()
    return h


# ----------------------------------------------------------------------------
# Detail sources
# ----------------------------------------------------------------------------

@dataclass
class DetailBands:
    micro: bool = True
    meso: bool = False
    meso_strength: float = 0.35
    sigma_micro: float = 0.75    # texels (finest kept scale)
    sigma_meso: float = 6.0
    sigma_macro: float = 24.0


@dataclass
class DetailResult:
    detail_normal: np.ndarray    # (H, W, 3) unit tangent normals
    height: np.ndarray           # (H, W) zero-centred, ~[-1, 1]
    source: str
    confidence: Optional[np.ndarray] = None


def _combine_bands(micro: Optional[np.ndarray], meso: Optional[np.ndarray], meso_strength: float,
                   shape) -> np.ndarray:
    out = nm.flat_like(shape)
    if meso is not None:
        out = nm.scale_strength(meso, meso_strength)
    if micro is not None:
        out = nm.rnm_combine(out, micro)
    return out


def luminance_detail(color: np.ndarray, mask: np.ndarray, bands: DetailBands,
                     gain: float = 0.35) -> DetailResult:
    lum = luminance(color)
    m = mask.astype(np.float32)
    height = np.zeros(lum.shape, np.float32)
    if bands.micro:
        height += nm.band_pass_height(lum, m, bands.sigma_micro * 0.5, bands.sigma_meso * 0.5)
    if bands.meso:
        height += bands.meso_strength * nm.band_pass_height(lum, m, bands.sigma_meso * 0.5, bands.sigma_macro)
    height *= m
    n = nm.height_to_normal(height, strength=1.0, texel_scale=gain)
    return DetailResult(detail_normal=n, height=np.clip(height, -1, 1), source="luminance")


def model_detail(tangent_normals: np.ndarray, weight: np.ndarray, mask: np.ndarray,
                 bands: DetailBands, weight_half: float = 0.2) -> DetailResult:
    m = mask.astype(np.float32) * (weight > 1e-8)
    micro = normal_band(tangent_normals, m, bands.sigma_micro, bands.sigma_meso) if bands.micro else None
    meso = normal_band(tangent_normals, m, bands.sigma_meso, bands.sigma_macro) if bands.meso else None
    n = _combine_bands(micro, meso, bands.meso_strength, m.shape)
    n = np.where(m[..., None] > 0.5, n, nm.flat_like(m.shape))
    h = normals_to_height(n, m)
    s = float(np.abs(h[m > 0.5]).std()) if (m > 0.5).any() else 1.0
    h = np.clip(h / max(3 * s, 1e-6), -1, 1) * m
    conf = 1.0 - np.exp(-weight / weight_half)
    return DetailResult(detail_normal=n.astype(np.float32), height=h, source="model", confidence=conf)


def height_detail(height: np.ndarray, mask: np.ndarray, bands: DetailBands, gain: float = 0.35) -> DetailResult:
    m = mask.astype(np.float32)
    h = nm.band_pass_height(height, m, bands.sigma_micro * 0.5 if bands.micro else bands.sigma_meso * 0.5,
                            bands.sigma_macro if bands.meso else bands.sigma_meso * 0.5)
    return DetailResult(detail_normal=nm.height_to_normal(h, 1.0, gain), height=h, source="height")


def hybrid_detail(model: DetailResult, lum: DetailResult) -> DetailResult:
    c = model.confidence if model.confidence is not None else np.ones(model.height.shape, np.float32)
    c = np.clip(c, 0, 1)[..., None]
    n = nm.normalize(model.detail_normal * c + lum.detail_normal * (1 - c))
    h = model.height * c[..., 0] + lum.height * (1 - c[..., 0])
    return DetailResult(detail_normal=n, height=h, source="hybrid", confidence=c[..., 0])


def select_detail(source: str, color: np.ndarray, mask: np.ndarray, bands: DetailBands,
                  tangent_normals: Optional[np.ndarray] = None, normal_weight: Optional[np.ndarray] = None,
                  height: Optional[np.ndarray] = None) -> DetailResult:
    has_model = tangent_normals is not None and normal_weight is not None and (normal_weight > 0).any()
    if source == "height" and height is not None:
        return height_detail(height, mask, bands)
    if source in ("model", "hybrid", "auto") and has_model:
        md = model_detail(tangent_normals, normal_weight, mask, bands)
        if source == "model":
            return md
        return hybrid_detail(md, luminance_detail(color, mask, bands))
    return luminance_detail(color, mask, bands)
