"""View planning (V2): choose and order views for sequential generation.

Goal: cover as much of the UV surface as possible with few views, while
ordering them so each new view OVERLAPS what is already generated — the
overlap is what the consistency step (reprojection + inpainting) anchors to.
Texels no candidate view can see are reported as "unreachable" and handled
by UV-space inpainting / fallback fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .projection import ProjectionParams, ViewSample, view_texel_weights

PRIORITY = ["front", "front_left", "front_right", "left", "right", "back", "top", "bottom"]


@dataclass
class ViewPlan:
    order: List[str]
    coverage_after: List[float]          # fraction of (sampled) texels covered after each view
    gains: List[float]
    final_coverage: float
    unreachable_fraction: float          # no candidate sees these texels
    per_view_visible: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__.copy()


def sample_texels(P: np.ndarray, N: np.ndarray, max_samples: int = 60000, seed: int = 0):
    if len(P) <= max_samples:
        return P, N
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(P), max_samples, replace=False)
    return P[sel], N[sel]


def plan_views(P: np.ndarray, N: np.ndarray, candidates: Sequence, bbox_diag: float,
               params: Optional[ProjectionParams] = None, first: Optional[str] = "front",
               max_views: int = 10, target_coverage: float = 0.985, min_gain: float = 0.004,
               weight_half: float = 0.25, overlap_bias: float = 0.6,
               required: Sequence[str] = (), loader: Optional[Callable[[str], ViewSample]] = None
               ) -> ViewPlan:
    """``candidates``: ViewSamples, or view names together with ``loader(name)``
    (views are then loaded one at a time to bound memory)."""
    params = params or ProjectionParams()
    names = [c if isinstance(c, str) else c.name for c in candidates]
    rows = []
    for c in candidates:
        v = loader(c) if isinstance(c, str) else c
        rows.append(view_texel_weights(P, N, v, bbox_diag, params)[0].astype(np.float32))
    Wm = np.stack(rows)
    n = Wm.shape[1]
    reach = 1.0 - np.exp(-Wm.sum(axis=0) / weight_half)
    reachable = reach > 0.05
    per_view = {nm: float((Wm[i] > 0).mean()) for i, nm in enumerate(names)}

    acc = np.zeros(n, np.float32)
    covered = np.zeros(n, np.float32)
    order: List[str] = []
    cov_hist: List[float] = []
    gains: List[float] = []
    remaining = set(range(len(names)))

    def pick(i):
        nonlocal acc, covered
        acc = acc + Wm[i]
        newcov = 1.0 - np.exp(-acc / weight_half)
        gains.append(float((newcov - covered).mean()))
        covered = newcov
        order.append(names[i])
        cov_hist.append(float(covered.mean()))
        remaining.discard(i)

    forced = [f for f in ([first] if first else []) + list(required) if f in names]
    for f in dict.fromkeys(forced):
        pick(names.index(f))

    while remaining and len(order) < max_views:
        if covered.mean() >= target_coverage * max(reachable.mean(), 1e-6):
            break
        best, best_score, best_gain = None, -1.0, 0.0
        for i in remaining:
            cand = 1.0 - np.exp(-(acc + Wm[i]) / weight_half)
            gain = float((cand - covered).mean())
            vis = Wm[i] > 0
            overlap = float(covered[vis].mean()) if vis.any() and order else 0.0
            prio = 1.0 + 0.05 * (len(PRIORITY) - PRIORITY.index(names[i])) if names[i] in PRIORITY else 1.0
            score = gain * ((1.0 - overlap_bias) + overlap_bias * min(1.0, overlap * 2.0)) * prio
            if score > best_score:
                best, best_score, best_gain = i, score, gain
        if best is None or best_gain < min_gain:
            break
        pick(best)

    return ViewPlan(order=order, coverage_after=cov_hist, gains=gains,
                    final_coverage=float(covered.mean()),
                    unreachable_fraction=float(1.0 - reachable.mean()),
                    per_view_visible=per_view)
