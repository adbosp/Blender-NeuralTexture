"""Consistent multi-view synthesis, region regeneration and map building (V2/V3).

Pure Python + numpy. Runs from the Blender add-on (worker thread), the CLI,
or the MCP server. Inputs are a guidance package (see package.py) and an
image generator (see generators.py).

    run_synthesis()      plan views -> sequential consistent generation -> UV inpaint -> finalize
    regenerate_region()  redo only masked texels (painted mask or low confidence)
    finalize()           rebuild PBR maps from the accumulator (fast; for slider tweaks)

State lives in <out>/synth/ (accumulator + state.json) so runs can resume after
a crash/cancel, and every finalize() writes a new version under <out>/history/.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import materials as mats
from . import normals as nm
from . import pngio
from .accumulate import UVAccumulator
from .config import JobConfig
from .consistency import composite_init, dilate, harmonize, inpaint_mask, view_color_error
from .generators import BaseGenerator, FolderGenerator, GeneratorCancelled, ViewJob
from .imageops import gaussian_blur, luminance, pull_push_fill, resize_bilinear, sample_nearest
from .intent import Intent, resolve_intent
from .package import Package
from .pbr import PBRParams, SurfaceData, build_texture_set
from .planner import plan_views, sample_texels
from .projection import ProjectionParams
from .prompts import build_prompts

Progress = Optional[Callable[[float, str], None]]


Cancelled = GeneratorCancelled  # one exception type for user cancel + generator interrupt


class CancelToken:
    def __init__(self):
        self._ev = threading.Event()

    def cancel(self):
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def check(self):
        if self._ev.is_set():
            raise Cancelled("cancelled by user")


@dataclass
class SynthParams:
    max_views: int = 10
    target_coverage: float = 0.985
    min_gain: float = 0.004
    first_view: str = "front"
    consistency: bool = True
    lock: float = 0.8                 # confident texels resist later views
    harmonize_strength: float = 0.85
    inpaint_threshold: float = 0.55
    inpaint_grow_px: int = 6
    inpaint_feather_px: float = 5.0
    uv_inpaint: bool = True           # texture-space inpaint of never-seen texels
    uv_inpaint_weight: float = 0.3
    planner_samples: int = 60000
    calibrate_normals: bool = True
    region_max_views: int = 4


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _params(cls, d: Optional[Dict]):
    return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def _synth_dir(out: str) -> str:
    d = os.path.join(out, "synth")
    os.makedirs(d, exist_ok=True)
    return d


def load_state(out: str) -> Dict:
    p = os.path.join(_synth_dir(out), "state.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(out: str, state: Dict):
    p = os.path.join(_synth_dir(out), "state.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, p)  # atomic: a crash never leaves a half-written state


def make_accumulator(pkg: Package, cfg: JobConfig) -> UVAccumulator:
    return UVAccumulator(pkg.uv("position"), pkg.uv("normal"), pkg.uv("coverage"), pkg.bbox_diag,
                         _params(ProjectionParams, cfg.projection),
                         tangent=pkg.uv("tangent"), bitangent=pkg.uv("bitangent"))


def _acc_path(out: str) -> str:
    return os.path.join(_synth_dir(out), "accumulator.npz")


def load_accumulator(out: str, pkg: Package, cfg: JobConfig) -> UVAccumulator:
    acc = make_accumulator(pkg, cfg)
    if not os.path.exists(_acc_path(out)):
        raise FileNotFoundError("No synthesis state yet; run synthesis first.")
    return acc.load(_acc_path(out))


def _fit(img: np.ndarray, h: int, w: int) -> np.ndarray:
    img = img[..., :3].astype(np.float32)
    return img if img.shape[:2] == (h, w) else resize_bilinear(img, h, w)


def calibrate_normal_axes(pred: np.ndarray, gt: np.ndarray, alpha: np.ndarray) -> Dict:
    """Find X/Y sign flips so estimator output matches our convention.

    ``gt`` are exact camera-space geometry normals from the guidance render.
    Compared after blurring (the estimator also sees fine detail).
    """
    m = alpha > 0.5
    best = {"flip_x": False, "flip_y": False, "agreement": -1.0}
    if m.sum() < 100:
        return best
    g = nm.normalize(gaussian_blur(gt, 3.0))[m]
    for fx in (False, True):
        for fy in (False, True):
            p = pred.copy()
            if fx:
                p[..., 0] *= -1
            if fy:
                p[..., 1] *= -1
            p = nm.normalize(gaussian_blur(p, 3.0))[m]
            agree = float(np.mean(np.sum(p * g, axis=1)))
            if agree > best["agreement"]:
                best = {"flip_x": fx, "flip_y": fy, "agreement": agree}
    return best


# ----------------------------------------------------------------------------
# Synthesis
# ----------------------------------------------------------------------------

def plan(pkg: Package, cfg: JobConfig, acc: UVAccumulator, sp: SynthParams) -> Dict:
    if not cfg.use_planner:
        order = [v for v in cfg.views if v in pkg.views]
        return {"order": order, "planner": False}
    P, N = sample_texels(acc.P, acc.N, sp.planner_samples)
    first = sp.first_view if sp.first_view in pkg.views else None
    vp = plan_views(P, N, pkg.views, pkg.bbox_diag, acc.params, first=first, max_views=sp.max_views,
                    target_coverage=sp.target_coverage, min_gain=sp.min_gain,
                    loader=lambda n: pkg.view_sample(n, with_uv=False))
    d = vp.to_dict()
    d["planner"] = True
    return d


def _generate_one(pkg: Package, cfg: JobConfig, intent: Intent, gen: BaseGenerator, acc: UVAccumulator,
                  name: str, index: int, sp: SynthParams, state: Dict,
                  texel_mask: Optional[np.ndarray] = None, region_view_mask: Optional[np.ndarray] = None,
                  lock: Optional[float] = None) -> Dict:
    v = pkg.view_sample(name)
    h, w = v.alpha.shape
    vd = pkg.view_dir(name)
    albedo = pkg.control(name, "albedo")
    if albedo is None:
        albedo = np.full((h, w, 3), 0.5, np.float32)
    files = {k: pkg.control_path(name, k) for k in ("depth", "normal", "albedo")}
    first = not (acc.w_sum > 0).any()
    info: Dict = {"view": name, "first": first}
    reproj = conf = None
    if not first and sp.consistency:
        filled = pull_push_fill(acc.color_map(), acc.seen_map().astype(np.float32))
        reproj, conf = acc.reproject(v, filled)
        init = composite_init(reproj, conf, albedo, v.alpha)
        mask = inpaint_mask(conf, v.alpha, sp.inpaint_threshold, sp.inpaint_grow_px, sp.inpaint_feather_px)
        if region_view_mask is not None:
            mask = np.where(v.alpha > 0.5, np.maximum(mask * region_view_mask, region_view_mask), mask)
        files["init"] = os.path.join(vd, "init.png")
        files["inpaint_mask"] = os.path.join(vd, "inpaint_mask.png")
        pngio.write_png(files["init"], init)
        pngio.write_png(files["inpaint_mask"], np.repeat(mask[..., None], 3, axis=2))
        denoise = cfg.model.consistency_denoise
        info["known_fraction"] = float(((conf >= sp.inpaint_threshold) & (v.alpha > 0.5)).sum()
                                       / max(1, (v.alpha > 0.5).sum()))
    else:
        denoise = cfg.model.denoise
    job = ViewJob(name=name, index=index, width=w, height=h, files={k: p for k, p in files.items() if p},
                  prompts=build_prompts(intent, name), first=first, denoise=denoise, seed=cfg.model.seed,
                  reference_images=list(cfg.reference_images), reference_role=intent.reference_role)
    t0 = time.time()
    img = _fit(gen.generate_view(job), h, w)
    info["generate_s"] = round(time.time() - t0, 2)
    if not isinstance(gen, FolderGenerator):
        pngio.write_png(os.path.join(vd, "generated.png"), img)
    if reproj is not None and sp.harmonize_strength > 0:
        info["overlap_error_before"] = view_color_error(img, reproj, conf, v.alpha)
        img, hinfo = harmonize(img, reproj, conf, v.alpha, strength=sp.harmonize_strength)
        info["overlap_error_after"] = view_color_error(img, reproj, conf, v.alpha)
        info["harmonize"] = hinfo
    final_path = os.path.join(vd, "final.png")
    pngio.write_png(final_path, img)

    normals = None
    if gen.supports_normals and cfg.model.normal_estimator != "off":
        normals = gen.estimate_normals(job, final_path)
        if normals is not None:
            normals = nm.normalize(_fit(normals, h, w) if normals.shape[:2] != (h, w) else normals)
            calib = state.get("normal_calibration")
            if calib is None and sp.calibrate_normals:
                gt = pkg.control(name, "normal")
                if gt is not None:
                    calib = calibrate_normal_axes(normals, gt * 2.0 - 1.0, v.alpha)
                    state["normal_calibration"] = calib
            if calib:
                if calib.get("flip_x"):
                    normals[..., 0] *= -1
                if calib.get("flip_y"):
                    normals[..., 1] *= -1
                if calib.get("agreement", 1.0) < 0.25:
                    normals = None  # estimator disagrees with geometry: don't trust it
                    info["normals"] = "rejected (low agreement with geometry)"
    lk = (0.0 if first else sp.lock) if lock is None else lock
    info.update(acc.add_view(v, img, lock=lk, texel_mask=texel_mask, normals_cam=normals))
    return info


def run_synthesis(out: str, cfg: JobConfig, gen: BaseGenerator, intent: Optional[Intent] = None,
                  progress: Progress = None, cancel: Optional[CancelToken] = None,
                  resume: bool = True, finalize_maps: bool = True) -> Dict:
    t_start = time.time()
    progress = progress or (lambda f, m: None)
    cancel = cancel or CancelToken()
    pkg = Package(out)
    intent = intent or resolve_intent(cfg)
    if intent.requires_geometry_mode:
        raise ValueError("Request needs geometry editing: " +
                         "; ".join(w.clause for w in intent.geometry_warnings))
    sp = _params(SynthParams, cfg.synth)
    acc = make_accumulator(pkg, cfg)
    state = load_state(out) if resume else {}
    if state.get("status") in ("running", "synthesized", "cancelled", "error") and resume \
            and os.path.exists(_acc_path(out)) and state.get("resolution") == pkg.resolution:
        acc.load(_acc_path(out))
    else:
        state = {}
    if not state.get("plan"):
        progress(0.01, "planning views")
        state = {"status": "running", "resolution": pkg.resolution, "done": [], "steps": [],
                 "plan": plan(pkg, cfg, acc, sp), "started": time.time()}
        save_state(out, state)
    order = state["plan"]["order"]
    state["status"] = "running"
    gen.begin({"package": pkg, "cfg": cfg, "intent": intent, "out": out})
    try:
        for i, name in enumerate(order):
            if name in state["done"]:
                continue
            cancel.check()
            progress(0.05 + 0.8 * i / max(1, len(order)), f"generating view {i + 1}/{len(order)}: {name}")
            info = _generate_one(pkg, cfg, intent, gen, acc, name, i, sp, state)
            state["done"].append(name)
            state["steps"].append(info)
            acc.save(_acc_path(out))
            save_state(out, state)

        cancel.check()
        seen = acc.seen_map()
        cov = pkg.uv("coverage").astype(bool)
        holes = cov & ~seen
        state["unseen_fraction_before_inpaint"] = float(holes.sum() / max(1, cov.sum()))
        if sp.uv_inpaint and holes.any() and gen.supports_atlas_inpaint and not state.get("uv_inpaint_done"):
            progress(0.87, "inpainting unseen texels in UV space")
            _uv_inpaint(pkg, cfg, intent, gen, acc, holes, sp, out)
            state["uv_inpaint_done"] = True
            acc.save(_acc_path(out))
        state["status"] = "synthesized"
        state["synth_seconds"] = round(time.time() - t_start, 1)
        save_state(out, state)
    except Cancelled:
        gen.interrupt()
        state["status"] = "cancelled"
        save_state(out, state)
        raise
    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        state["traceback"] = traceback.format_exc()
        save_state(out, state)
        raise
    finally:
        gen.end()

    if not finalize_maps:
        return {"state": state}
    progress(0.9, "building PBR maps")
    res = finalize(out, cfg, intent, acc=acc, pkg=pkg, label="synthesis")
    res["synthesis"] = {"plan": state["plan"], "steps": state["steps"],
                        "unseen_before_inpaint": state.get("unseen_fraction_before_inpaint")}
    _write_json(os.path.join(res["history_dir"], "result.json"), res)
    progress(1.0, "done")
    return res


def _uv_inpaint(pkg, cfg, intent, gen, acc, holes, sp, out):
    res = pkg.resolution
    color = acc.color_map()
    known = acc.seen_map()
    ex = pkg.existing().basecolor
    if ex is not None:
        color = np.where(known[..., None], color, ex[..., :3])
        known = known | pkg.uv("coverage").astype(bool)
    atlas = pull_push_fill(color, known.astype(np.float32))
    mask = dilate(holes, 3).astype(np.float32)
    d = _synth_dir(out)
    ap, mp = os.path.join(d, "atlas_init.png"), os.path.join(d, "atlas_mask.png")
    pngio.write_png(ap, atlas)
    pngio.write_png(mp, np.repeat(mask[..., None], 3, axis=2))
    prompts = build_prompts(intent, "texture atlas")
    img = gen.inpaint_atlas(ap, mp, prompts, res)
    if img is None:
        return
    img = _fit(img, res, res)
    pngio.write_png(os.path.join(d, "atlas_inpainted.png"), img)
    acc.add_uv(img, holes.astype(np.float32) * sp.uv_inpaint_weight)


# ----------------------------------------------------------------------------
# Region regeneration (V4 "fix this area")
# ----------------------------------------------------------------------------

def low_confidence_mask(out: str, cfg: JobConfig, threshold: float = 0.35) -> np.ndarray:
    pkg = Package(out)
    acc = load_accumulator(out, pkg, cfg)
    return (acc.confidence_map() < threshold) & pkg.uv("coverage").astype(bool)


def regenerate_region(out: str, cfg: JobConfig, gen: BaseGenerator, uv_mask: np.ndarray,
                      intent: Optional[Intent] = None, progress: Progress = None,
                      cancel: Optional[CancelToken] = None, seed_offset: int = 1) -> Dict:
    progress = progress or (lambda f, m: None)
    cancel = cancel or CancelToken()
    pkg = Package(out)
    intent = intent or resolve_intent(cfg)
    sp = _params(SynthParams, cfg.synth)
    acc = load_accumulator(out, pkg, cfg)
    region = (uv_mask > 0.5) & pkg.uv("coverage").astype(bool)
    if region.shape != (pkg.resolution, pkg.resolution):
        region = resize_bilinear(region.astype(np.float32), pkg.resolution, pkg.resolution) > 0.5
    m_flat = acc.flat_mask_from_map(region)
    if m_flat.sum() == 0:
        raise ValueError("Mask does not cover any texel of the mesh.")
    P, N = acc.P[m_flat > 0], acc.N[m_flat > 0]
    P, N = sample_texels(P, N, 20000)
    vp = plan_views(P, N, pkg.views, pkg.bbox_diag, acc.params, first=None, max_views=sp.region_max_views,
                    target_coverage=0.98, min_gain=0.01, overlap_bias=0.0,
                    loader=lambda n: pkg.view_sample(n, with_uv=False))
    acc.reset(m_flat)
    cfg.model.seed = int(cfg.model.seed) + seed_offset
    state = load_state(out)
    steps = []
    gen.begin({"package": pkg, "cfg": cfg, "intent": intent, "out": out})
    try:
        for i, name in enumerate(vp.order):
            cancel.check()
            progress(0.1 + 0.7 * i / max(1, len(vp.order)), f"regenerating region from {name}")
            v = pkg.view_sample(name)
            rv = sample_nearest(region.astype(np.float32), v.uv[..., 0] * pkg.resolution - 0.5,
                                v.uv[..., 1] * pkg.resolution - 0.5) * (v.alpha > 0.5)
            rv = gaussian_blur(dilate(rv > 0.5, sp.inpaint_grow_px).astype(np.float32), 3.0)
            steps.append(_generate_one(pkg, cfg, intent, gen, acc, name, i, sp, state,
                                       texel_mask=m_flat, region_view_mask=rv, lock=0.0))
    except Cancelled:
        gen.interrupt()
        raise
    finally:
        gen.end()
    acc.save(_acc_path(out))
    progress(0.9, "rebuilding maps")
    res = finalize(out, cfg, intent, acc=acc, pkg=pkg, label="region")
    res["region"] = {"texels": int(m_flat.sum()), "views": vp.order, "steps": steps}
    _write_json(os.path.join(res["history_dir"], "result.json"), res)
    progress(1.0, "done")
    return res


# ----------------------------------------------------------------------------
# Finalize: accumulator -> PBR set -> history version
# ----------------------------------------------------------------------------

def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def slot_classes(pkg: Package, intent: Intent, object_name: str = "") -> List[str]:
    slots = pkg.man.get("slots") or [{"index": 0, "material": None}]
    names = [s.get("material") for s in slots]
    if intent.material_class and not intent.material_classes:
        return [intent.material_class] * len(names)
    prompt = "" if intent.material_classes else intent.appearance_prompt
    return mats.classify_slots(prompt, names, object_name, intent.material_classes)


BITS = {"normal": 16, "detail_normal": 16, "detail_height": 16}


def finalize(out: str, cfg: JobConfig, intent: Optional[Intent] = None, acc: Optional[UVAccumulator] = None,
             pkg: Optional[Package] = None, label: str = "rebuild") -> Dict:
    t0 = time.time()
    pkg = pkg or Package(out)
    intent = intent or resolve_intent(cfg)
    acc = acc or load_accumulator(out, pkg, cfg)
    cov = pkg.uv("coverage").astype(bool)
    classes = slot_classes(pkg, intent, pkg.man.get("object", ""))
    s = SurfaceData(color=acc.color_map(), seen=acc.seen_map(), coverage=cov,
                    confidence=acc.confidence_map(), position=pkg.uv("position"),
                    material_id=pkg.uv("material_id"), slot_classes=classes, ao=pkg.uv("ao"),
                    curvature=pkg.uv("curvature"), tangent_normals=acc.tangent_normal_map(),
                    normal_weight=acc.normal_weight_map())
    existing = pkg.existing()
    pbr = build_texture_set(s, existing, intent, _params(PBRParams, cfg.pbr))

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int((time.time() % 1) * 1000):03d}"
    hdir = os.path.join(out, "history", run_id)
    tdir = os.path.join(hdir, "textures")
    os.makedirs(tdir, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in pkg.man.get("object", "asset"))
    files = {}
    for key, arr in pbr.maps.items():
        p = os.path.join(tdir, f"{stem}_{key}.png")
        pngio.write_png(p, arr, BITS.get(key, 8))
        files[key] = os.path.relpath(p, out)
    _write_preview(os.path.join(hdir, "preview.png"), existing.basecolor, pbr.maps["basecolor"])

    validation = validate(pbr.stats, pkg, acc)
    result = {"run_id": run_id, "label": label, "history_dir": hdir, "textures": files,
              "stats": pbr.stats, "notes": pbr.notes + intent.notes, "validation": validation,
              "intent": intent.to_dict(), "slot_classes": classes,
              "finalize_seconds": round(time.time() - t0, 2)}
    _write_json(os.path.join(hdir, "result.json"), result)
    _write_json(os.path.join(hdir, "job.json"), cfg.to_dict())
    _record_history(out, run_id, label, validation)
    return result


def _write_preview(path, before, after, size: int = 512):
    a = resize_bilinear(after[..., :3], size, size)
    b = resize_bilinear(before[..., :3], size, size) if before is not None else np.full_like(a, 0.5)
    sep = np.ones((size, 4, 3), np.float32)
    pngio.write_png(path, np.concatenate([b, sep, a], axis=1))


def validate(stats: Dict, pkg: Package, acc: UVAccumulator) -> Dict:
    issues = []
    seen = stats.get("seen_fraction", 0.0)
    if seen < 0.9:
        issues.append(f"Only {seen:.0%} of UV texels were seen by any generated view; the rest use "
                      "UV inpainting / existing albedo. Consider regenerating low-confidence regions.")
    conf = stats.get("mean_confidence", 1.0)
    if conf < 0.5:
        issues.append(f"Mean confidence is {conf:.2f}; views disagree or coverage is thin.")
    dev = stats.get("normal_mean_deviation_deg")
    if dev is not None and dev > 12.0:
        issues.append(f"Final normal deviates {dev:.1f} deg on average from the base normal; lower detail strength.")
    se = stats.get("seam_error_after")
    if se is not None and se > 0.06:
        issues.append(f"Visible UV seams remain (mean mismatch {se:.3f}).")
    return {"ok": not issues, "issues": issues}


def _record_history(out: str, run_id: str, label: str, validation: Dict):
    idx_path = os.path.join(out, "history", "index.json")
    idx = []
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            idx = json.load(f)
    idx.append({"run_id": run_id, "label": label, "time": time.time(), "ok": validation["ok"]})
    _write_json(idx_path, idx)
    _write_json(os.path.join(out, "latest.json"), {"run_id": run_id})


def history(out: str) -> List[Dict]:
    p = os.path.join(out, "history", "index.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def set_latest(out: str, run_id: str) -> Dict:
    hdir = os.path.join(out, "history", run_id)
    if not os.path.isdir(hdir):
        raise FileNotFoundError(run_id)
    _write_json(os.path.join(out, "latest.json"), {"run_id": run_id})
    with open(os.path.join(hdir, "result.json"), encoding="utf-8") as f:
        return json.load(f)


def latest_result(out: str) -> Optional[Dict]:
    p = os.path.join(out, "latest.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        rid = json.load(f)["run_id"]
    with open(os.path.join(out, "history", rid, "result.json"), encoding="utf-8") as f:
        return json.load(f)
