"""PBR texture-set synthesis (V3).

Inputs: the accumulated generated appearance (+ confidence), UV bakes of the
asset (position, material IDs, AO, curvature), optional model normals and
the existing maps. Output: BaseColor, DetailHeight, DetailNormal,
Normal = RNM(existing, detail), Roughness, Metallic, AO, ORM, Confidence.

Principles (spec 4, 6, 9):
  * macro structure comes from the asset (base normal is never replaced),
  * lighting is removed from BaseColor as far as possible,
  * existing data is used wherever generation is unreliable,
  * roughness / metallic follow physically plausible, material-aware priors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import normals as nm
from . import materials as mats
from .detail import DetailBands, select_detail
from .imageops import gaussian_blur, luminance, masked_gaussian_blur, pull_push_fill, resize_bilinear
from .intent import Intent
from .seams import refine_seams, seam_error


@dataclass
class ExistingMaps:
    """Existing asset maps already resampled to output resolution (may be None)."""
    basecolor: Optional[np.ndarray] = None   # (H, W, 3) display-encoded
    normal: Optional[np.ndarray] = None      # (H, W, 3) packed tangent normal
    roughness: Optional[np.ndarray] = None   # (H, W)
    metallic: Optional[np.ndarray] = None    # (H, W)
    ao: Optional[np.ndarray] = None          # (H, W)
    normal_flip_green: bool = False          # True for DirectX-style source maps
    default_roughness: float = 0.5
    default_metallic: float = 0.0


@dataclass
class SurfaceData:
    """UV-space data about the asset + synthesis result."""
    color: np.ndarray                         # (H, W, 3) accumulated appearance
    seen: np.ndarray                          # (H, W) bool
    coverage: np.ndarray                      # (H, W) bool, inside UV islands
    confidence: Optional[np.ndarray] = None   # (H, W) 0..1
    position: Optional[np.ndarray] = None     # (H, W, 3) world position (seams)
    material_id: Optional[np.ndarray] = None  # (H, W) slot index, -1 outside
    slot_classes: List[str] = field(default_factory=lambda: ["generic"])
    ao: Optional[np.ndarray] = None           # baked AO (H, W)
    curvature: Optional[np.ndarray] = None    # baked curvature, 0.5 = flat
    tangent_normals: Optional[np.ndarray] = None  # model-estimated (H, W, 3)
    normal_weight: Optional[np.ndarray] = None
    height: Optional[np.ndarray] = None       # user/agent supplied height


@dataclass
class PBRParams:
    delight_amount: float = 0.6        # remove low-frequency shading baked into views
    ao_delight: float = 0.35           # undo model-painted occlusion using baked AO
    highlight_suppress: float = 0.6    # compress specular highlights in albedo
    preserve_existing_color: float = 0.0  # 0 = new colour, 1 = keep existing macro tone
    confidence_blend: bool = True      # soft blend to existing albedo where unsure
    seam_strength: float = 1.0
    seam_radius_px: float = 6.0
    detail_source: str = "auto"        # auto | luminance | model | hybrid | height
    detail_gain: float = 1.0
    roughness_mode: str = "auto"       # auto | material | existing | blend
    roughness_blend: float = 0.5
    metallic_mode: str = "auto"        # auto | material | existing
    ao_strength: float = 1.0
    albedo_clamp: bool = True
    export_orm: bool = True
    macro_sigma_px: float = 48.0
    reference_resolution: int = 2048   # pixel sizes are defined at this resolution


@dataclass
class PBRResult:
    maps: Dict[str, np.ndarray] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# BaseColor
# ----------------------------------------------------------------------------

def delight(color: np.ndarray, mask: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """Divide out large-scale luminance variation (baked light gradients)."""
    if amount <= 0:
        return color
    lum = luminance(color)
    low = masked_gaussian_blur(lum, mask, sigma)
    valid = mask > 0.5
    target = float(np.median(low[valid])) if valid.any() else 0.5
    ratio = np.clip(target / np.maximum(low, 1e-3), 0.4, 2.5) ** amount
    return np.clip(color * ratio[..., None], 0.0, 1.0).astype(np.float32)


def ao_delight(color: np.ndarray, ao: Optional[np.ndarray], amount: float) -> np.ndarray:
    if ao is None or amount <= 0:
        return color
    shade = np.clip(1.0 - amount * (1.0 - np.clip(ao, 0, 1)), 0.35, 1.0)
    return np.clip(color / shade[..., None], 0.0, 1.0).astype(np.float32)


def suppress_highlights(color: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Soft-compress bright, desaturated spots (specular reflections)."""
    if amount <= 0:
        return color
    lum = luminance(color)
    valid = mask > 0.5
    if not valid.any():
        return color
    knee = float(np.percentile(lum[valid], 97))
    mx, mn = color.max(axis=2), color.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    spec = np.clip((lum - knee) / max(1.0 - knee, 1e-3), 0, 1) * np.clip(1.0 - sat * 2.0, 0, 1)
    local = masked_gaussian_blur(color, mask.astype(np.float32), 4.0)
    w = (amount * spec)[..., None]
    return (color * (1 - w) + np.minimum(color, local) * w).astype(np.float32)


def clamp_albedo(color: np.ndarray, class_map: np.ndarray, classes: List[str]) -> np.ndarray:
    lum = luminance(color)
    out = color.copy()
    for i, cname in enumerate(classes):
        m = class_map == i
        if not m.any():
            continue
        lo, hi = mats.get(cname).albedo_range
        target = np.clip(lum[m], lo, hi)
        scale = target / np.maximum(lum[m], 1e-4)
        out[m] = np.clip(color[m] * scale[:, None], 0, 1)
    return out


def transfer_macro_tone(color, existing, mask, sigma, amount):
    if amount <= 0 or existing is None:
        return color
    g_lo = masked_gaussian_blur(color, mask, sigma)
    e_lo = masked_gaussian_blur(existing, mask, sigma)
    return np.clip(color + amount * (e_lo - g_lo), 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _class_map(s: SurfaceData) -> np.ndarray:
    H, W = s.coverage.shape
    if s.material_id is None:
        return np.where(s.coverage, 0, -1).astype(np.int32)
    mid = s.material_id.astype(np.int32)
    return np.where(mid >= len(s.slot_classes), 0, mid)


def _per_texel(class_map: np.ndarray, classes: List[str], getter) -> np.ndarray:
    lut = np.array([getter(mats.get(c)) for c in classes] + [getter(mats.get("generic"))], np.float32)
    return lut[np.where(class_map < 0, len(classes), class_map)]


def build_texture_set(s: SurfaceData, existing: ExistingMaps, intent: Intent,
                      params: Optional[PBRParams] = None) -> PBRResult:
    p = params or PBRParams()
    H, W = s.coverage.shape
    scale = max(H, W) / float(p.reference_resolution)
    cov = s.coverage.astype(np.float32)
    seen_f = s.seen.astype(np.float32)
    cmap = _class_map(s)
    classes = s.slot_classes or ["generic"]
    res = PBRResult()
    conf = s.confidence if s.confidence is not None else seen_f

    # ---- BaseColor --------------------------------------------------------
    color = s.color[..., :3].astype(np.float32)
    color = ao_delight(color, s.ao, p.ao_delight)
    color = delight(color, seen_f, p.macro_sigma_px * scale * 2.0, p.delight_amount)
    color = suppress_highlights(color, seen_f, p.highlight_suppress)
    if p.albedo_clamp:
        color = clamp_albedo(color, np.where(s.seen, cmap, -1), classes)
    fallback = existing.basecolor[..., :3] if existing.basecolor is not None else None
    if fallback is not None:
        color = transfer_macro_tone(color, fallback, seen_f, p.macro_sigma_px * scale, p.preserve_existing_color)
        if p.confidence_blend:
            c = np.clip(conf * 2.0, 0.0, 1.0)[..., None] * s.seen[..., None]
            color = color * c + fallback * (1 - c)
        else:
            color = np.where(s.seen[..., None], color, fallback)
        known = s.seen | s.coverage
    else:
        known = s.seen
    color = pull_push_fill(color, known.astype(np.float32))
    if s.position is not None and p.seam_strength > 0:
        res.stats["seam_error_before"] = seam_error(color, s.position, s.coverage)
        color, st = refine_seams(color, s.position, s.coverage, p.seam_radius_px * max(scale, 0.25),
                                 p.seam_strength)
        res.stats["seam_error_after"] = seam_error(color, s.position, s.coverage)
        res.stats["seam_texels"] = st.get("seam_texels", 0)
    color = pull_push_fill(color, cov)  # UV padding
    res.maps["basecolor"] = color

    # ---- Detail -------------------------------------------------------------
    micro_sigma = float(np.mean([mats.get(c).micro_sigma for c in classes]))
    bands = DetailBands(micro=intent.generate_micro, meso=intent.generate_meso,
                        meso_strength=intent.meso_strength,
                        sigma_micro=micro_sigma * scale, sigma_meso=6.0 * scale * 2,
                        sigma_macro=24.0 * scale * 2)
    det = select_detail(p.detail_source, color, cov * seen_f, bands,
                        s.tangent_normals, s.normal_weight, s.height)
    gain = _per_texel(cmap, classes, lambda c: c.detail_gain) * p.detail_gain
    detail_n = nm.normalize(np.concatenate([det.detail_normal[..., :2] * gain[..., None],
                                            det.detail_normal[..., 2:]], axis=2))
    detail_n = np.where(s.seen[..., None], detail_n, nm.flat_like((H, W)))  # no invented detail
    res.maps["detail_normal"] = nm.pack(detail_n)
    height = det.height * s.seen
    res.maps["detail_height"] = np.clip(height * 0.5 + 0.5, 0.0, 1.0)
    res.stats["detail_source"] = det.source

    # ---- Final normal ------------------------------------------------------
    if existing.normal is not None and intent.preserve_macro_normal:
        base_n = nm.unpack(existing.normal, flip_green=existing.normal_flip_green)
    else:
        base_n = nm.flat_like((H, W))
    final_n = nm.rnm_combine(base_n, detail_n, intent.detail_strength)
    res.maps["normal"] = nm.pack(final_n)
    if existing.normal is not None and s.coverage.any():
        diff = nm.angular_difference_deg(base_n, final_n)[s.coverage]
        res.stats["normal_mean_deviation_deg"] = float(diff.mean())
        res.stats["normal_p99_deviation_deg"] = float(np.percentile(diff, 99))

    # ---- AO -----------------------------------------------------------------
    ao = None
    if existing.ao is not None:
        ao = existing.ao
        if s.ao is not None:
            ao = np.minimum(ao, np.clip(s.ao, 0, 1) ** (0.5 * p.ao_strength))
    elif s.ao is not None:
        ao = np.clip(s.ao, 0, 1) ** p.ao_strength
    if ao is not None:
        res.maps["ao"] = pull_push_fill(ao.astype(np.float32), cov)

    # ---- Roughness ------------------------------------------------------------
    r_lo = _per_texel(cmap, classes, lambda c: c.roughness[0])
    r_hi = _per_texel(cmap, classes, lambda c: c.roughness[1])
    cav_k = _per_texel(cmap, classes, lambda c: c.cavity_rough)
    lum = luminance(color)
    lum_hp = lum - masked_gaussian_blur(lum, cov, 8.0 * max(scale, 0.25))
    t = 0.5 - 1.5 * lum_hp - 0.35 * height
    if ao is not None:
        t = t + (1.0 - ao) * 1.2
    if s.curvature is not None:
        t = t + (s.curvature - 0.5) * 0.6   # convex edges are more worn/rough
    r_mat = r_lo + (r_hi - r_lo) * np.clip(t, 0, 1) + cav_k * (1.0 - (ao if ao is not None else 1.0))
    mode = p.roughness_mode
    restyle = bool(getattr(intent, "material_class", "")) and existing.roughness is not None
    if mode == "auto":
        mode = "material" if (existing.roughness is None or restyle) else "blend"
    ex_r = existing.roughness if existing.roughness is not None else \
        np.full((H, W), existing.default_roughness, np.float32)
    if mode == "existing":
        rough = ex_r
    elif mode == "blend":
        rough = ex_r * (1 - p.roughness_blend) + r_mat * p.roughness_blend
    else:
        rough = r_mat
    rough = rough + intent.roughness_offset
    res.maps["roughness"] = pull_push_fill(np.clip(rough, 0.02, 1.0).astype(np.float32), cov)
    res.stats["roughness_mode"] = mode

    # ---- Metallic -------------------------------------------------------------
    metal_v = _per_texel(cmap, classes, lambda c: c.metallic)
    any_metal_class = any(mats.get(c).metallic > 0 for c in classes)
    mmode = p.metallic_mode
    if mmode == "auto":
        mmode = "material" if (any_metal_class or existing.metallic is None) else "existing"
    if mmode == "existing" and existing.metallic is not None:
        metal = existing.metallic
    else:
        metal = metal_v.copy()
        partial = (metal_v > 0) & (metal_v < 1)   # e.g. rusted metal: bare metal where grey & bright
        if partial.any():
            mx, mn = color.max(axis=2), color.min(axis=2)
            sat = (mx - mn) / np.maximum(mx, 1e-4)
            bare = np.clip((0.3 - sat) / 0.15, 0, 1) * np.clip((lum - 0.2) / 0.2, 0, 1)
            metal = np.where(partial, bare, metal)
    if existing.metallic is not None or any_metal_class or intent.metallic_hint:
        res.maps["metallic"] = pull_push_fill(np.clip(metal, 0, 1).astype(np.float32), cov)
    if intent.metallic_hint and not any_metal_class:
        res.notes.append("Prompt mentions metal but no slot was classified as metal; set "
                         "'material_classes' in the intent to control metallic explicitly.")

    # ---- ORM / confidence --------------------------------------------------------
    if p.export_orm:
        o = res.maps.get("ao", np.ones((H, W), np.float32))
        m = res.maps.get("metallic", np.zeros((H, W), np.float32))
        res.maps["orm"] = np.stack([o, res.maps["roughness"], m], axis=2)
    res.maps["confidence"] = np.clip(conf, 0, 1).astype(np.float32)

    res.stats["seen_fraction"] = float(s.seen[s.coverage].mean()) if s.coverage.any() else 0.0
    res.stats["mean_confidence"] = float(conf[s.coverage].mean()) if s.coverage.any() else 0.0
    res.stats["slot_classes"] = list(classes)
    return res


def resample_existing(maps: Dict[str, np.ndarray], res: int) -> Dict[str, np.ndarray]:
    return {k: resize_bilinear(v, res, res) for k, v in maps.items() if v is not None}
