"""ComfyUI compatibility layer (V6): model auto-detection, graph completion
and validation against the server's /object_info.

ComfyUI node signatures drift between versions and custom-node packs. Rather
than hard-coding them, graphs are completed with the server's own defaults
for any missing required input and checked (class exists, enum values valid)
before queueing, so failures are reported clearly instead of as opaque 400s.
"""

from __future__ import annotations

import copy
import re
from typing import Dict, List, Optional, Tuple

from ..core.config import ModelConfig

NORMAL_ESTIMATORS = ["DSINE-NormalMapPreprocessor", "Metric3D-NormalMapPreprocessor",
                     "BAE-NormalMapPreprocessor", "MiDaS-NormalMapPreprocessor"]
IPADAPTER_NODES = ("IPAdapterUnifiedLoader", "IPAdapterAdvanced")


class GraphError(ValueError):
    pass


# ----------------------------------------------------------------------------
# object_info helpers
# ----------------------------------------------------------------------------

def _inputs(info: Dict, cls: str) -> Tuple[Dict, Dict]:
    spec = info.get(cls, {}).get("input", {})
    return spec.get("required", {}) or {}, spec.get("optional", {}) or {}


def choices(info: Dict, cls: str, name: str) -> Optional[List]:
    req, opt = _inputs(info, cls)
    s = req.get(name) or opt.get(name)
    if not s:
        return None
    first = s[0]
    if isinstance(first, list):
        return first
    if first == "COMBO" and len(s) > 1 and isinstance(s[1], dict):
        return s[1].get("options")  # newer ComfyUI combo format
    return None


def _is_upload(info: Dict, cls: str, name: str) -> bool:
    req, opt = _inputs(info, cls)
    s = req.get(name) or opt.get(name) or []
    opts = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
    return bool(opts.get("image_upload")) or (cls.startswith("LoadImage") and name == "image")


def _default(spec) -> Tuple[bool, object]:
    if not spec:
        return False, None
    first = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(first, list):
        return (True, first[0]) if first else (False, None)
    if first == "COMBO":
        o = opts.get("options") or []
        return (True, o[0]) if o else (False, None)
    if "default" in opts:
        return True, opts["default"]
    if first in ("INT", "FLOAT"):
        return True, opts.get("min", 0)
    if first == "BOOLEAN":
        return True, False
    if first == "STRING":
        return True, ""
    return False, None  # a link type (IMAGE, MODEL...) we cannot invent


def complete_and_validate(graph: Dict, info: Dict) -> Tuple[Dict, List[str]]:
    """Return (completed graph, notes). Raises GraphError on unfixable problems."""
    g = copy.deepcopy(graph)
    notes: List[str] = []
    missing_cls = sorted({n["class_type"] for n in g.values() if n["class_type"] not in info})
    if missing_cls:
        raise GraphError("ComfyUI is missing node types: " + ", ".join(missing_cls))
    problems = []
    for nid, node in g.items():
        cls = node["class_type"]
        req, opt = _inputs(info, cls)
        inp = node.setdefault("inputs", {})
        for name, spec in req.items():
            if name not in inp:
                ok, val = _default(spec)
                if ok:
                    inp[name] = val
                    notes.append(f"{cls}.{name} defaulted to {val!r}")
                else:
                    problems.append(f"node {nid} ({cls}) missing required input '{name}'")
        for name, val in list(inp.items()):
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                if val[0] not in g:
                    problems.append(f"node {nid} ({cls}).{name} links to missing node {val[0]}")
                continue
            ch = choices(info, cls, name)
            if _is_upload(info, cls, name):
                ch = None  # uploaded files are not in the server's cached list
            if ch is not None and val not in ch:
                fixed = closest_choice(val, ch)
                if fixed is None:
                    problems.append(f"{cls}.{name} = {val!r} not available; options: {ch[:12]}")
                else:
                    inp[name] = fixed
                    notes.append(f"{cls}.{name}: {val!r} -> {fixed!r}")
            if name not in req and name not in opt and req:
                del inp[name]  # input not supported by this server version
                notes.append(f"{cls}.{name} dropped (unsupported by server)")
    if problems:
        raise GraphError("Workflow incompatible with this ComfyUI:\n- " + "\n- ".join(problems))
    return g, notes


def closest_choice(val, options: List) -> Optional[str]:
    if not isinstance(val, str) or not options:
        return None
    low = val.lower()
    stem = re.sub(r"\.(safetensors|ckpt|pt|pth|bin)$", "", low)
    for o in options:
        if isinstance(o, str) and (o.lower() == low or re.sub(r"\.(safetensors|ckpt|pt|pth|bin)$", "", o.lower()) == stem):
            return o
    for o in options:
        if isinstance(o, str) and stem and stem in o.lower():
            return o
    return None


# ----------------------------------------------------------------------------
# Auto-detection
# ----------------------------------------------------------------------------

def _pick(options: List[str], prefer: List[str], avoid: List[str] = ()) -> Optional[str]:
    opts = [o for o in options if isinstance(o, str)]
    for pat in prefer:
        for o in opts:
            lo = o.lower()
            if re.search(pat, lo) and not any(re.search(a, lo) for a in avoid):
                return o
    return None


def autodetect(model: ModelConfig, info: Dict, vram_gb: float = 0.0) -> Dict:
    """Adjust ``model`` in place to what the server actually has. Returns a report."""
    rep: Dict = {"changes": [], "warnings": []}

    ckpts = choices(info, "CheckpointLoaderSimple", "ckpt_name") or []
    if ckpts and model.checkpoint not in ckpts:
        pick = closest_choice(model.checkpoint, ckpts) or _pick(
            ckpts, [r"juggernaut.*xl", r"realvis.*xl", r"sd_xl_base", r"sdxl", r"\bxl\b", r"xl"],
            avoid=[r"refiner", r"inpaint", r"turbo", r"lightning"]) or ckpts[0]
        rep["changes"].append(f"checkpoint: {model.checkpoint} -> {pick}")
        model.checkpoint = pick
    elif not ckpts:
        rep["warnings"].append("No checkpoints found in ComfyUI/models/checkpoints.")

    cns = choices(info, "ControlNetLoader", "control_net_name") or []
    if cns and model.controlnet not in cns:
        union = _pick(cns, [r"union.*(xl|sdxl)", r"(xl|sdxl).*union", r"union"])
        if union:
            model.controlnet, model.controlnet_is_union = union, True
        else:
            depth = _pick(cns, [r"depth.*(xl|sdxl)", r"(xl|sdxl).*depth", r"depth"])
            normal = _pick(cns, [r"normal.*(xl|sdxl)", r"(xl|sdxl).*normal", r"normal"])
            model.controlnet_is_union = False
            model.controlnet = depth or cns[0]
            model.controlnet_normal = normal or ""
            if not normal:
                model.normal_strength = 0.0
                rep["warnings"].append("No normal ControlNet found; using depth only.")
        rep["changes"].append(f"controlnet: {model.controlnet} (union={model.controlnet_is_union})")
    elif not cns:
        rep["warnings"].append("No ControlNet models found; generation will not be geometry-guided!")
        model.depth_strength = model.normal_strength = 0.0

    if model.controlnet_is_union and "SetUnionControlNetType" not in info:
        rep["warnings"].append("SetUnionControlNetType missing (old ComfyUI); union type left on auto.")

    if model.use_ipadapter and not all(n in info for n in IPADAPTER_NODES):
        model.use_ipadapter = False
        rep["warnings"].append("ComfyUI_IPAdapter_plus not installed; reference images will only "
                               "guide colour through the init image.")

    est = model.normal_estimator
    if est == "auto":
        found = next((n for n in NORMAL_ESTIMATORS if n in info), None)
        model.normal_estimator = found or "off"
        rep["changes"].append(f"normal estimator: {model.normal_estimator}")
    elif est not in ("off",) and est not in info:
        rep["warnings"].append(f"Normal estimator node '{est}' not installed; detail normals use luminance.")
        model.normal_estimator = "off"

    if vram_gb:
        rep["vram_gb"] = round(vram_gb, 1)
        if vram_gb < 7.5:
            rep["warnings"].append(f"Only {vram_gb:.1f} GB VRAM: SDXL + ControlNet may run out of memory; "
                                   "view resolution will be reduced automatically on OOM.")
    return rep
