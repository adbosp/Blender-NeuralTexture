"""Normal-map math (pure numpy).

Spec rule (section 4): never regenerate the full normal map blindly.
    FinalNormal = Combine(ExistingBaseNormal, GeneratedDetailNormal, detail_strength)

All tangent-space normals here use the OpenGL / Blender convention
(+X right, +Y up, +Z out of the surface). Packed maps are RGB in [0, 1].
"""

from __future__ import annotations

import numpy as np

from .imageops import gaussian_blur, masked_gaussian_blur

FLAT = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.maximum(n, eps)).astype(np.float32)


def unpack(rgb: np.ndarray, flip_green: bool = False) -> np.ndarray:
    """[0,1] RGB -> unit vectors in [-1,1]. ``flip_green`` converts DirectX maps."""
    v = rgb[..., :3].astype(np.float32) * 2.0 - 1.0
    if flip_green:
        v[..., 1] *= -1.0
    return normalize(v)


def pack(n: np.ndarray, flip_green: bool = False) -> np.ndarray:
    v = n.astype(np.float32).copy()
    if flip_green:
        v[..., 1] *= -1.0
    return np.clip(v * 0.5 + 0.5, 0.0, 1.0)


def flat_like(shape_hw) -> np.ndarray:
    out = np.zeros(tuple(shape_hw) + (3,), dtype=np.float32)
    out[..., 2] = 1.0
    return out


def scale_strength(n: np.ndarray, strength: float) -> np.ndarray:
    """Scale the XY tilt of a tangent normal (0 = flat, 1 = unchanged)."""
    v = n.astype(np.float32).copy()
    v[..., :2] *= strength
    v[..., 2] = np.maximum(v[..., 2], 1e-4)
    return normalize(v)


def rnm_combine(base: np.ndarray, detail: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Reoriented Normal Mapping (Barré-Brisebois & Hill 2012).

    Rotates ``detail`` so that its "up" follows ``base``. Unlike linear/UDN
    blending, the macro normal is preserved exactly when detail is flat,
    and the detail is exactly reproduced when base is flat.
    Inputs are unpacked unit vectors.
    """
    d = scale_strength(detail, strength) if strength != 1.0 else detail
    t = base.astype(np.float32).copy()
    t[..., 2] += 1.0
    u = d.astype(np.float32).copy()
    u[..., 0] *= -1.0
    u[..., 1] *= -1.0
    dot = np.sum(t * u, axis=-1, keepdims=True)
    r = t * dot / np.maximum(t[..., 2:3], 1e-6) - u
    return normalize(r)


def height_to_normal(height: np.ndarray, strength: float = 1.0,
                     texel_scale: float = 1.0) -> np.ndarray:
    """Tangent-space normal from a height field.

    ``height`` rows are bottom-up (V increases with row), so the gradient
    along axis 0 is dH/dV and matches OpenGL +Y.
    """
    h = height.astype(np.float32) * (strength * texel_scale)
    dv, du = np.gradient(h)
    n = np.stack([-du, -dv, np.ones_like(h)], axis=-1)
    return normalize(n)


def band_pass_height(signal: np.ndarray, mask: np.ndarray,
                     sigma_fine: float, sigma_coarse: float) -> np.ndarray:
    """Keep only spatial frequencies between two blur scales.

    sigma_fine = 0 keeps everything down to single texels (micro detail).
    The coarse scale removes large-scale variation so the generated detail
    cannot fight the macro structure carried by the base normal.
    """
    lo = masked_gaussian_blur(signal, mask, sigma_coarse)
    hi = masked_gaussian_blur(signal, mask, sigma_fine) if sigma_fine > 0.3 else signal
    band = (hi - lo) * mask
    valid = mask > 0.5
    if valid.any():
        std = float(band[valid].std())
        if std > 1e-8:
            band = band / (3.0 * std)  # ~99% of values in [-1, 1]
    return np.clip(band, -1.0, 1.0).astype(np.float32)


def macro_component(n: np.ndarray, sigma: float) -> np.ndarray:
    """Low-pass a normal map (useful when a generated full normal must be split)."""
    return normalize(gaussian_blur(n, sigma))


def angular_difference_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.clip(np.sum(normalize(a) * normalize(b), axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))
