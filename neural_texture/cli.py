"""Headless entry point inside Blender (for agents / CI).

    blender -b scene.blend --python neural_texture/cli.py -- \
        --job job.json --stage all [--object Head] [--generator comfyui] [--preset ancient_stone] [--save]

Stages:
  analyze     read-only report                           (Blender)
  guidance    render/bake the guidance package           (Blender)
  synth       consistent multi-view generation + maps    (pure Python; also: python -m neural_texture.headless)
  rebuild     rebuild maps from the accumulator          (pure Python)
  regenerate  redo --mask file.npy or low-confidence     (pure Python)
  material    build/assign NT_<object> from latest       (Blender)
  all         analyze + guidance (if stale) + synth + material
The last line printed is  NT_RESULT_JSON: {...}
Exit codes: 0 ok, 2 bad arguments, 3 geometry editing required, 4 failure, 5 cancelled.
"""

import argparse
import json
import os
import sys
import traceback

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from neural_texture import pipeline  # noqa: E402
from neural_texture.core import synth  # noqa: E402
from neural_texture.core.config import JobConfig  # noqa: E402

STAGES = ["analyze", "guidance", "synth", "rebuild", "regenerate", "material", "all"]


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="neural_texture.cli")
    p.add_argument("--job", required=True, help="JobConfig JSON")
    p.add_argument("--stage", default="all", choices=STAGES)
    p.add_argument("--object", default=None, help="overrides job.object_name")
    p.add_argument("--generator", default="comfyui", choices=["comfyui", "identity", "folder"])
    p.add_argument("--preset", default=None)
    p.add_argument("--mask", default=None, help="UV mask .npy for --stage regenerate")
    p.add_argument("--threshold", type=float, default=0.35, help="low-confidence threshold")
    p.add_argument("--fresh", action="store_true", help="ignore saved synthesis progress")
    p.add_argument("--save", action="store_true", help="save the .blend after material/all")
    return p.parse_args(argv)


def _emit(payload, code):
    print("NT_RESULT_JSON: " + json.dumps(payload, default=str))
    sys.stdout.flush()
    if bpy.app.background:
        os._exit(code)  # Blender ignores the script's exit status otherwise
    return code


def _prog(f, s):
    print(f"[NeuralTexture] {f:6.1%} {s}", flush=True)


def main():
    try:
        a = _args()
    except SystemExit:
        return _emit({"ok": False, "error": "bad arguments"}, 2)
    cfg = JobConfig.load(a.job)
    if a.preset:
        from neural_texture.core.presets import apply_preset
        apply_preset(cfg, a.preset)
    name = a.object or cfg.object_name
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "MESH":
        return _emit({"ok": False, "error": f"mesh object '{name}' not found"}, 2)
    out = pipeline.output_dir(cfg, obj)
    try:
        if a.stage == "analyze":
            res = pipeline.stage_analyze(obj, cfg, out)
        elif a.stage == "guidance":
            pipeline.stage_analyze(obj, cfg, out)
            res = {"views": list(pipeline.stage_guidance(obj, cfg, out, _prog)["views"])}
        elif a.stage == "synth":
            cfg = pipeline.resolve_paths(cfg)
            res = synth.run_synthesis(out, cfg, pipeline.make_generator(cfg, a.generator, out), None, _prog,
                                      resume=not a.fresh)
        elif a.stage == "rebuild":
            res = synth.finalize(out, cfg)
        elif a.stage == "regenerate":
            cfg = pipeline.resolve_paths(cfg)
            mask = np.load(a.mask) if a.mask else synth.low_confidence_mask(out, cfg, a.threshold)
            res = synth.regenerate_region(out, cfg, pipeline.make_generator(cfg, a.generator, out),
                                          mask.astype(np.float32), None, _prog)
        elif a.stage == "material":
            res = {"material": pipeline.apply_material(obj, out, cfg)}
        else:
            res = pipeline.run_all(obj, cfg, a.generator, _prog)
        if a.save and a.stage in ("material", "all") and bpy.data.filepath:
            bpy.ops.wm.save_mainfile()
        return _emit({"ok": True, "stage": a.stage, "output_dir": out, "result": res}, 0)
    except pipeline.StructuralWarning as e:
        return _emit({"ok": False, "requires_geometry_mode": True,
                      "geometry_warnings": [w.to_dict() for w in e.warnings]}, 3)
    except synth.Cancelled:
        return _emit({"ok": False, "cancelled": True, "output_dir": out}, 5)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        with open(os.path.join(out, "error.json"), "w", encoding="utf-8") as f:
            json.dump({"stage": a.stage, "error": str(e), "traceback": traceback.format_exc()}, f, indent=2)
        return _emit({"ok": False, "error": str(e), "stage": a.stage, "output_dir": out}, 4)


if __name__ == "__main__":
    main()
