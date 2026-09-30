"""Stage orchestration inside Blender (V2+).

Blender does only what needs Blender:
  analyze    read-only inspection            -> <out>/analysis.json
  guidance   renders + UV bakes + maps       -> <out>/package.json (+ views/, uv/, existing/)
  material   build/assign NT_<object>        -> from <out>/latest.json
Everything else is pure Python (core.synth) and runs in a worker thread:
  synthesize consistent multi-view generation -> accumulator + history/<run>/textures
  regenerate masked / low-confidence regions
  rebuild    maps from the accumulator (slider changes, no regeneration)

Geometry, UVs, rig and the user's materials are never modified.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

import bpy
import numpy as np

from .core import synth
from .core.config import JobConfig
from .core.generators import BaseGenerator, FolderGenerator, IdentityGenerator
from .core.intent import Intent, resolve_intent
from .blender import guidance, inspect_asset, material_build
from .blender.runner import JobRunner

Progress = Optional[Callable[[float, str], None]]


class StructuralWarning(RuntimeError):
    """Raised when a request needs geometry editing (spec section 5)."""

    def __init__(self, warnings):
        self.warnings = warnings
        super().__init__("; ".join(f"{w.kind}: '{w.clause}'" for w in warnings))


# ----------------------------------------------------------------------------
def output_dir(cfg: JobConfig, obj: bpy.types.Object) -> str:
    base = bpy.path.abspath(cfg.output_dir or "//neural_texture_out")
    if base.startswith("//") or not os.path.isabs(base):
        base = os.path.join(bpy.app.tempdir or "/tmp", "neural_texture_out")  # unsaved .blend
    d = os.path.join(base, bpy.path.clean_name(obj.name))
    os.makedirs(d, exist_ok=True)
    return d


def _write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def resolve_paths(cfg: JobConfig) -> JobConfig:
    cfg.reference_images = [bpy.path.abspath(p) for p in cfg.reference_images]
    missing = [p for p in cfg.reference_images if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Reference image(s) not found: {missing}")
    if cfg.model.custom_workflow:
        cfg.model.custom_workflow = bpy.path.abspath(cfg.model.custom_workflow)
    return cfg


# ----------------------------------------------------------------------------
# Blender stages
# ----------------------------------------------------------------------------

def stage_analyze(obj, cfg: JobConfig, out: str) -> Dict:
    dg = bpy.context.evaluated_depsgraph_get()
    mesh = inspect_asset.analyze_mesh(obj, dg, cfg.uv_map)
    maps = inspect_asset.extract_material_maps(obj)
    intent = resolve_intent(cfg)
    slot_tris = [s["triangles"] for s in mesh["material_slots"]]
    prim = inspect_asset.primary_maps(maps, slot_tris)
    warnings = list(mesh["uv"].get("warnings", []))
    for m in maps:
        warnings += [f"[{m.material}] {w}" for w in m.warnings]
    report = {"mesh": mesh, "materials": [m.to_dict() for m in maps],
              "primary_material_slot": prim.slot if prim else None,
              "intent": intent.to_dict(), "requires_geometry_mode": intent.requires_geometry_mode,
              "warnings": warnings, "job": cfg.to_dict(), "blender": bpy.app.version_string}
    _write(os.path.join(out, "analysis.json"), report)
    cfg.save(os.path.join(out, "job.json"))
    return report


def guidance_steps(obj, cfg: JobConfig, out: str):
    """Iterator of (fraction, message); renders + bakes the guidance package."""
    maps = inspect_asset.extract_material_maps(obj)
    slot_tris = [0] * len(maps)
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        slot_tris = [s["triangles"] for s in inspect_asset.analyze_mesh(obj, dg, cfg.uv_map)["material_slots"]]
    except Exception:
        pass
    return guidance.export_package_steps(obj, cfg, out, maps, slot_tris)


def package_is_current(out: str, cfg: JobConfig) -> bool:
    p = os.path.join(out, "package.json")
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8") as f:
        man = json.load(f)
    want = set(cfg.views) | (set(cfg.candidate_views) if cfg.use_planner else set())
    return (man.get("version") == 2 and man["uv"]["resolution"] == cfg.texture_resolution
            and want <= set(man["views"])
            and all(v["camera"]["width"] == cfg.view_resolution for v in man["views"].values()))


def apply_material(obj, out: str, cfg: JobConfig, assign: Optional[bool] = None) -> Optional[str]:
    res = synth.latest_result(out)
    if not res:
        return None
    files = {k: os.path.join(out, v) for k, v in res["textures"].items()}
    with open(os.path.join(out, "package.json"), encoding="utf-8") as f:
        man = json.load(f)
    mat = material_build.build_material(f"NT_{obj.name}", files, man["uv"].get("uv_map", ""),
                                        float(man.get("existing", {}).get("normal_strength", 1.0)))
    mat["nt_run_id"] = res["run_id"]
    if cfg.assign_material if assign is None else assign:
        material_build.assign_material(obj, mat)
    return mat.name


def read_mask_image(obj) -> Optional[np.ndarray]:
    img = bpy.data.images.get(mask_image_name(obj))
    if img is None:
        return None
    from .blender.imageio import image_to_array
    return image_to_array(img)[..., 0]


def mask_image_name(obj) -> str:
    return f"NT_mask_{obj.name}"


# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------

def make_generator(cfg: JobConfig, kind: str = "comfyui", out: str = "", log=print) -> BaseGenerator:
    if kind == "identity":
        return IdentityGenerator()
    if kind == "folder":
        return FolderGenerator(out)
    from .bridge.comfy_generator import ComfyGenerator
    from .bridge.generate import is_local_url
    if not is_local_url(cfg.model.server_url) and not getattr(bpy.app, "online_access", True):
        raise RuntimeError("Remote ComfyUI URL blocked: enable 'Allow Online Access' in Preferences > System.")
    return ComfyGenerator(cfg.model, log=log, reference_images=cfg.reference_images)


# ----------------------------------------------------------------------------
# Jobs (non-blocking)
# ----------------------------------------------------------------------------

def _check_intent(cfg: JobConfig) -> Intent:
    intent = resolve_intent(cfg)
    if intent.requires_geometry_mode:
        raise StructuralWarning(intent.geometry_warnings)
    return intent


def build_generate_job(obj, cfg: JobConfig, generator_kind: str = "comfyui", force_guidance: bool = False,
                       resume: bool = True, on_done=None) -> JobRunner:
    cfg = resolve_paths(cfg)
    out = output_dir(cfg, obj)
    intent = _check_intent(cfg)
    stage_analyze(obj, cfg, out)
    gen = make_generator(cfg, generator_kind, out)
    name = obj.name
    steps = []
    if force_guidance or not package_is_current(out, cfg):
        steps.append(("main", "guidance", lambda: guidance_steps(obj, cfg, out)))
        resume = False
    steps.append(("thread", "synthesis",
                  lambda p, c: synth.run_synthesis(out, cfg, gen, intent, p, c, resume=resume)))
    steps.append(("main", "material", lambda: apply_material(bpy.data.objects[name], out, cfg)))
    return JobRunner(f"generate {name}", steps, on_done)


def build_regenerate_job(obj, cfg: JobConfig, uv_mask: np.ndarray, generator_kind: str = "comfyui",
                         on_done=None) -> JobRunner:
    cfg = resolve_paths(cfg)
    out = output_dir(cfg, obj)
    intent = _check_intent(cfg)
    gen = make_generator(cfg, generator_kind, out)
    name = obj.name
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    np.save(os.path.join(out, "masks", f"mask_{int(time.time())}.npy"), uv_mask.astype(np.float32))
    steps = [("thread", "regenerate",
              lambda p, c: synth.regenerate_region(out, cfg, gen, uv_mask, intent, p, c)),
             ("main", "material", lambda: apply_material(bpy.data.objects[name], out, cfg))]
    return JobRunner(f"regenerate {name}", steps, on_done)


def build_rebuild_job(obj, cfg: JobConfig, on_done=None) -> JobRunner:
    out = output_dir(cfg, obj)
    intent = resolve_intent(cfg)
    name = obj.name
    steps = [("thread", "rebuild maps", lambda p, c: synth.finalize(out, cfg, intent, label="rebuild")),
             ("main", "material", lambda: apply_material(bpy.data.objects[name], out, cfg))]
    return JobRunner(f"rebuild {name}", steps, on_done)


# ----------------------------------------------------------------------------
# Blocking entry points (CLI / background)
# ----------------------------------------------------------------------------

def run_all(obj, cfg: JobConfig, generator_kind: str = "comfyui", progress: Progress = None) -> Dict:
    job = build_generate_job(obj, cfg, generator_kind)
    if progress:
        orig = job._set

        def _set(frac, msg):
            orig(frac, msg)
            progress(job.progress, job.message)
        job._set = _set
    job.run_blocking()
    return synth.latest_result(output_dir(cfg, obj)) or {}


def stage_guidance(obj, cfg: JobConfig, out: str, progress: Progress = None) -> Dict:
    for frac, msg in guidance_steps(obj, cfg, out):
        if progress:
            progress(frac, msg)
    with open(os.path.join(out, "package.json"), encoding="utf-8") as f:
        return json.load(f)
