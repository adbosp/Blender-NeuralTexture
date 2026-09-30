"""UV-space accumulation of generated views (V2).

Each generated view is splatted into texture space with per-texel weights
(visibility x facing x silhouette distance). The accumulator keeps
    colour sum, squared-colour sum (for cross-view disagreement),
    weight sum, contributing-view count, dominant view,
    tangent-space normal sum (when a normal estimator is used),
so it can answer at any time:
    current texture, confidence map, and "what does the texture look like
    from view k" (reprojection) for consistency-aware generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .imageops import luminance, pull_push_fill, sample_bilinear, sample_nearest
from .projection import ProjectionParams, ViewSample, texels_from_maps, view_texel_weights


@dataclass
class ConfidenceParams:
    weight_half: float = 0.25     # accumulated weight giving ~63% "coverage" confidence
    disagreement_sigma: float = 0.12  # luminance std across views that halves confidence
    min_views_for_agreement: int = 2


def orthonormal_tbn(T: np.ndarray, B: np.ndarray, N: np.ndarray):
    """Gram-Schmidt so tangent frames are orthonormal (keeps bitangent handedness)."""
    N = N / np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-8)
    T = T - N * np.sum(T * N, axis=-1, keepdims=True)
    T = T / np.maximum(np.linalg.norm(T, axis=-1, keepdims=True), 1e-8)
    Bo = np.cross(N, T)
    sign = np.sign(np.sum(Bo * B, axis=-1, keepdims=True))
    sign[sign == 0] = 1.0
    return T.astype(np.float32), (Bo * sign).astype(np.float32), N.astype(np.float32)


class UVAccumulator:
    def __init__(self, texel_pos: np.ndarray, texel_nrm: np.ndarray, coverage: np.ndarray,
                 bbox_diag: float, params: Optional[ProjectionParams] = None,
                 tangent: Optional[np.ndarray] = None, bitangent: Optional[np.ndarray] = None,
                 conf_params: Optional[ConfidenceParams] = None):
        self.H, self.W = coverage.shape
        self.coverage = coverage.astype(bool)
        self.idx, self.P, self.N = texels_from_maps(texel_pos, texel_nrm, coverage)
        self.bbox_diag = float(bbox_diag)
        self.params = params or ProjectionParams()
        self.conf_params = conf_params or ConfidenceParams()
        n = self.idx.size
        self.color_sum = np.zeros((n, 3), np.float64)
        self.sq_sum = np.zeros((n, 3), np.float64)
        self.w_sum = np.zeros(n, np.float64)
        self.count = np.zeros(n, np.int16)
        self.best_w = np.zeros(n, np.float32)
        self.best_view = np.full(n, -1, np.int16)
        self.nrm_sum = np.zeros((n, 3), np.float64)
        self.nrm_w = np.zeros(n, np.float64)
        self.view_names: List[str] = []
        self.tbn = None
        if tangent is not None and bitangent is not None:
            T = tangent.reshape(-1, 3)[self.idx]
            B = bitangent.reshape(-1, 3)[self.idx]
            self.tbn = orthonormal_tbn(T, B, self.N)

    # ------------------------------------------------------------------ core
    @property
    def n(self) -> int:
        return int(self.idx.size)

    def weights(self, view: ViewSample) -> tuple:
        return view_texel_weights(self.P, self.N, view, self.bbox_diag, self.params)

    def confidence_flat(self) -> np.ndarray:
        cp = self.conf_params
        cov = 1.0 - np.exp(-self.w_sum / max(cp.weight_half, 1e-6))
        mean = self.color_sum / np.maximum(self.w_sum, 1e-12)[:, None]
        var = self.sq_sum / np.maximum(self.w_sum, 1e-12)[:, None] - mean ** 2
        std = np.sqrt(np.maximum(luminance(np.maximum(var, 0.0)), 0.0))
        agree = np.where(self.count >= cp.min_views_for_agreement,
                         np.exp(-(std / cp.disagreement_sigma) ** 2), 1.0)
        return (cov * agree).astype(np.float32)

    def add_view(self, view: ViewSample, image: np.ndarray, weight_mult: float = 1.0,
                 texel_mask: Optional[np.ndarray] = None, lock: float = 0.0,
                 normals_cam: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Splat one generated view.

        lock in [0,1]: how strongly already-confident texels resist change
        (consistency: earlier views win where they were reliable).
        normals_cam: optional (h, w, 3) camera-space unit normals predicted from
        the generated image (+X right, +Y up, +Z toward camera).
        """
        w, px, py = self.weights(view)
        w = w * weight_mult
        if texel_mask is not None:
            w = w * texel_mask
        if lock > 0 and self.view_names:
            w = w * (1.0 - lock * self.confidence_flat())
        vis = w > 1e-8
        vi = len(self.view_names)
        self.view_names.append(view.name or f"view{vi}")
        if not vis.any():
            return {"texels": 0, "new_texels": 0}
        img = image[..., :3] if image.ndim == 3 else np.repeat(image[..., None], 3, axis=2)
        col = sample_bilinear(img, px[vis], py[vis]).astype(np.float64)
        wv = w[vis]
        new = int(((self.w_sum[vis] <= 1e-8)).sum())
        self.color_sum[vis] += col * wv[:, None]
        self.sq_sum[vis] += col * col * wv[:, None]
        self.w_sum[vis] += wv
        self.count[vis] += 1
        better = wv > self.best_w[vis]
        sel = np.nonzero(vis)[0][better]
        self.best_w[sel] = wv[better]
        self.best_view[sel] = vi
        if normals_cam is not None and self.tbn is not None:
            nc = sample_bilinear(normals_cam, px[vis], py[vis]).astype(np.float64)
            R = view.camera.mat[:3, :3]
            nw = nc @ R.T
            T, B, N = (a[vis] for a in self.tbn)
            nts = np.stack([np.sum(nw * T, 1), np.sum(nw * B, 1), np.sum(nw * N, 1)], axis=1)
            nts[:, 2] = np.maximum(nts[:, 2], 0.05)
            nts /= np.linalg.norm(nts, axis=1, keepdims=True)
            self.nrm_sum[vis] += nts * wv[:, None]
            self.nrm_w[vis] += wv
        return {"texels": int(vis.sum()), "new_texels": new}

    def add_uv(self, image: np.ndarray, weight: np.ndarray, name: str = "uv_inpaint") -> Dict[str, float]:
        """Splat a UV-space (atlas) image directly, e.g. texture-space inpainting."""
        w = weight.reshape(-1)[self.idx].astype(np.float64)
        vis = w > 1e-8
        vi = len(self.view_names)
        self.view_names.append(name)
        col = image[..., :3].reshape(-1, 3)[self.idx][vis].astype(np.float64)
        new = int((self.w_sum[vis] <= 1e-8).sum())
        self.color_sum[vis] += col * w[vis, None]
        self.sq_sum[vis] += col * col * w[vis, None]
        self.w_sum[vis] += w[vis]
        self.count[vis] += 1
        better = w[vis] > self.best_w[vis]
        sel = np.nonzero(vis)[0][better]
        self.best_w[sel] = w[vis][better]
        self.best_view[sel] = vi
        return {"texels": int(vis.sum()), "new_texels": new}

    def reset(self, texel_mask: np.ndarray):
        """Forget accumulated data for masked texels (region regeneration)."""
        m = texel_mask.astype(bool)
        for a in (self.color_sum, self.sq_sum, self.nrm_sum):
            a[m] = 0
        for a in (self.w_sum, self.nrm_w):
            a[m] = 0
        self.count[m] = 0
        self.best_w[m] = 0
        self.best_view[m] = -1

    def flat_mask_from_map(self, uv_mask: np.ndarray) -> np.ndarray:
        return uv_mask.reshape(-1)[self.idx].astype(np.float32)

    # ------------------------------------------------------------------ maps
    def _to_map(self, flat: np.ndarray, fill=0.0) -> np.ndarray:
        shape = (self.H * self.W,) + flat.shape[1:]
        out = np.full(shape, fill, dtype=np.float32)
        out[self.idx] = flat
        return out.reshape((self.H, self.W) + flat.shape[1:])

    def color_map(self) -> np.ndarray:
        mean = self.color_sum / np.maximum(self.w_sum, 1e-12)[:, None]
        return self._to_map(np.where(self.w_sum[:, None] > 1e-8, mean, 0.0).astype(np.float32))

    def seen_map(self) -> np.ndarray:
        return self._to_map((self.w_sum > 1e-8).astype(np.float32)) > 0.5

    def weight_map(self) -> np.ndarray:
        return self._to_map(self.w_sum.astype(np.float32))

    def confidence_map(self) -> np.ndarray:
        return self._to_map(self.confidence_flat())

    def best_view_map(self) -> np.ndarray:
        return self._to_map(self.best_view.astype(np.float32), fill=-1).astype(np.int32)

    def tangent_normal_map(self) -> Optional[np.ndarray]:
        if self.tbn is None or not (self.nrm_w > 0).any():
            return None
        n = self.nrm_sum / np.maximum(self.nrm_w, 1e-12)[:, None]
        n = np.where(self.nrm_w[:, None] > 1e-8, n, np.array([0.0, 0.0, 1.0]))
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
        m = self._to_map(n.astype(np.float32))
        m[~self.coverage] = (0.0, 0.0, 1.0)
        return m

    def normal_weight_map(self) -> np.ndarray:
        return self._to_map(self.nrm_w.astype(np.float32))

    # -------------------------------------------------------- reprojection
    def reproject(self, view: ViewSample, filled_color: Optional[np.ndarray] = None):
        """Render the current texture into ``view`` via its UV pass.

        Returns (image (h,w,3), confidence (h,w)). Requires view.uv.
        """
        if view.uv is None:
            raise ValueError(f"view '{view.name}' has no UV pass; cannot reproject")
        color = filled_color if filled_color is not None else \
            pull_push_fill(self.color_map(), self.seen_map().astype(np.float32))
        conf = self.confidence_map()
        u = view.uv[..., 0] * self.W - 0.5
        v = view.uv[..., 1] * self.H - 0.5
        img = sample_bilinear(color, u, v)
        c = sample_nearest(conf, u, v)
        m = view.alpha > 0.5
        img = np.where(m[..., None], img, 0.0).astype(np.float32)
        c = np.where(m, c, 0.0).astype(np.float32)
        return img, c

    # ------------------------------------------------------------- persist
    def save(self, path: str):
        """Uncompressed (compression cost 3x the whole generation step) and atomic."""
        import os
        tmp = path[:-4] + ".tmp.npz" if path.endswith(".npz") else path + ".tmp.npz"
        np.savez(tmp, color_sum=self.color_sum, sq_sum=self.sq_sum, w_sum=self.w_sum,
                 count=self.count, best_w=self.best_w, best_view=self.best_view,
                 nrm_sum=self.nrm_sum, nrm_w=self.nrm_w,
                 view_names=np.array(self.view_names, dtype=object), idx=self.idx)
        os.replace(tmp, path)

    def load(self, path: str):
        d = np.load(path, allow_pickle=True)
        if d["idx"].shape != self.idx.shape or not np.array_equal(d["idx"], self.idx):
            raise ValueError("accumulator state does not match the current UV layout")
        for k in ("color_sum", "sq_sum", "w_sum", "count", "best_w", "best_view", "nrm_sum", "nrm_w"):
            setattr(self, k, d[k].copy())
        self.view_names = list(d["view_names"])
        return self
