"""View <-> UV-texture projection (pure numpy).

For every covered texel we know its world position P and normal N (UV bakes).
A texel receives colour from a view when
  1. it projects inside the frame,
  2. the view's rendered position at that pixel equals P (not occluded),
  3. its normal faces the camera.
The per-texel weight favours head-on, non-silhouette pixels. The same weights
drive the V2 view planner, the UV accumulator and the confidence map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .camera import CameraModel
from .imageops import erode, gaussian_blur, sample_bilinear, sample_nearest


@dataclass
class ViewSample:
    camera: CameraModel
    image: Optional[np.ndarray]  # (h, w, C) generated appearance, bottom-up rows (may be None)
    position: np.ndarray         # (h, w, 3) rendered world position
    alpha: np.ndarray            # (h, w) object coverage from the render
    weight_scale: float = 1.0    # per-view trust (e.g. lower for top/bottom)
    uv: Optional[np.ndarray] = None  # (h, w, 2) rendered UV coordinates (for reprojection)
    name: str = ""


@dataclass
class ProjectionParams:
    facing_power: float = 3.0         # higher = prefer head-on views more
    min_facing: float = 0.08          # ignore grazing angles below this cos
    position_tolerance: float = 0.004 # fraction of bbox diagonal
    silhouette_erode_px: int = 3      # distrust pixels this close to the silhouette
    silhouette_feather_px: float = 4.0
    sharpen: float = 1.0              # >1 approaches "best view wins"


@dataclass
class ProjectionResult:
    color: np.ndarray        # (H, W, C) blended colour (0 where unseen)
    weight: np.ndarray       # (H, W) total weight
    seen: np.ndarray         # (H, W) bool, texel received any view
    best_view: np.ndarray    # (H, W) int, index of dominant view (-1 unseen)


def silhouette_weight(alpha: np.ndarray, erode_px: int, feather: float) -> np.ndarray:
    inner = erode(alpha > 0.5, erode_px).astype(np.float32)
    if feather > 0:
        inner = gaussian_blur(inner, feather) * inner
    return inner


def view_texel_weights(P: np.ndarray, N: np.ndarray, v: ViewSample, bbox_diag: float,
                       params: ProjectionParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weight of view ``v`` for texels at world positions P (n,3) with normals N (n,3).

    Returns (w, px, py): weights (0 = not visible) and pixel coordinates.
    """
    cam = v.camera
    px, py, depth = cam.world_to_pixel(P)
    inside = (px >= -0.5) & (px <= cam.width - 0.5) & (py >= -0.5) & (py <= cam.height - 0.5) & (depth > 0)
    w = np.zeros(len(P), dtype=np.float64)
    if not inside.any():
        return w, px, py
    rendered = sample_nearest(v.position, px, py)
    dist = np.linalg.norm(rendered.astype(np.float64) - P, axis=1)
    tol = np.maximum(params.position_tolerance * bbox_diag, 2.0 * cam.pixel_world_size(depth))
    a = sample_nearest(v.alpha, px, py)
    visible = inside & (a > 0.5) & (dist <= tol)
    facing = np.sum(N * cam.view_dirs_to(P), axis=1)
    facing = np.where(facing > params.min_facing, facing, 0.0)
    sil = silhouette_weight(v.alpha, params.silhouette_erode_px, params.silhouette_feather_px)
    sil_w = sample_bilinear(sil, px, py)
    w = (np.maximum(facing, 0.0) ** params.facing_power) * sil_w * v.weight_scale
    w = np.where(visible, w, 0.0)
    if params.sharpen != 1.0:
        w = w ** params.sharpen
    return w.astype(np.float64), px, py


def texels_from_maps(texel_pos: np.ndarray, texel_nrm: np.ndarray, coverage: np.ndarray):
    idx = np.nonzero(coverage.reshape(-1))[0]
    P = texel_pos.reshape(-1, 3)[idx].astype(np.float64)
    N = texel_nrm.reshape(-1, 3)[idx].astype(np.float32)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-8)
    return idx, P, N


def project_views(texel_pos: np.ndarray, texel_nrm: np.ndarray, coverage: np.ndarray,
                  views: List[ViewSample], bbox_diag: float,
                  params: Optional[ProjectionParams] = None) -> ProjectionResult:
    """Single-shot weighted blend of all views (V1 path; V2 uses UVAccumulator)."""
    params = params or ProjectionParams()
    H, W = coverage.shape
    idx, P, N = texels_from_maps(texel_pos, texel_nrm, coverage)
    C = views[0].image.shape[2] if views and views[0].image.ndim == 3 else 1
    acc = np.zeros((idx.size, C), dtype=np.float64)
    wsum = np.zeros(idx.size, dtype=np.float64)
    best_w = np.zeros(idx.size, dtype=np.float64)
    best_i = np.full(idx.size, -1, dtype=np.int32)

    for vi, v in enumerate(views):
        w, px, py = view_texel_weights(P, N, v, bbox_diag, params)
        if not np.any(w > 0):
            continue
        img = v.image if v.image.ndim == 3 else v.image[..., None]
        col = sample_bilinear(img, px, py).reshape(idx.size, -1)
        acc += col[:, :C] * w[:, None]
        wsum += w
        better = w > best_w
        best_w = np.where(better, w, best_w)
        best_i = np.where(better, vi, best_i)

    color = np.zeros((H * W, C), dtype=np.float32)
    weight = np.zeros(H * W, dtype=np.float32)
    best = np.full(H * W, -1, dtype=np.int32)
    ok = wsum > 1e-8
    color[idx[ok]] = (acc[ok] / wsum[ok, None]).astype(np.float32)
    weight[idx] = wsum.astype(np.float32)
    best[idx] = best_i
    return ProjectionResult(color=color.reshape(H, W, C), weight=weight.reshape(H, W),
                            seen=(weight > 1e-8).reshape(H, W), best_view=best.reshape(H, W))
