"""Pure-Python command line (no Blender needed) for agents, CI and the MCP server.

    python -m neural_texture.headless intent "làm model này thành đá cổ nhưng giữ normal gốc"
    python -m neural_texture.headless synth   <out_dir> --job job.json [--generator comfyui|identity|folder]
    python -m neural_texture.headless rebuild <out_dir> --job job.json
    python -m neural_texture.headless regenerate <out_dir> --job job.json [--mask m.npy | --threshold 0.35]
    python -m neural_texture.headless status  <out_dir>
    python -m neural_texture.headless comfy   --url http://127.0.0.1:8188      (auto-detect report)
    python -m neural_texture.headless presets

<out_dir> is the per-object folder containing package.json (written by the
Blender 'guidance' stage). Output: JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .core import synth
from .core.config import JobConfig
from .core.generators import FolderGenerator, IdentityGenerator
from .core.intent import parse_prompt
from .core.presets import PRESETS, apply_preset


def _gen(kind: str, cfg: JobConfig, out: str):
    if kind == "identity":
        return IdentityGenerator()
    if kind == "folder":
        return FolderGenerator(out)
    from .bridge.comfy_generator import ComfyGenerator
    return ComfyGenerator(cfg.model, log=lambda m: print(m, file=sys.stderr),
                          reference_images=cfg.reference_images)


def _cfg(a) -> JobConfig:
    cfg = JobConfig.load(a.job) if a.job else JobConfig.load(os.path.join(a.out, "job.json"))
    if getattr(a, "preset", None):
        apply_preset(cfg, a.preset)
    return cfg


def status(out: str) -> dict:
    st = synth.load_state(out)
    return {"package": os.path.exists(os.path.join(out, "package.json")),
            "synthesis": {k: st.get(k) for k in ("status", "done", "error", "synth_seconds")},
            "plan": (st.get("plan") or {}).get("order"),
            "latest": synth.latest_result(out), "history": synth.history(out)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="neural_texture.headless", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("intent")
    s.add_argument("text")
    s.add_argument("--reference", action="store_true")
    for name in ("synth", "rebuild", "regenerate"):
        s = sub.add_parser(name)
        s.add_argument("out")
        s.add_argument("--job", default=None)
        s.add_argument("--preset", default=None)
        s.add_argument("--generator", default="comfyui", choices=["comfyui", "identity", "folder"])
        s.add_argument("--fresh", action="store_true")
        s.add_argument("--mask", default=None)
        s.add_argument("--threshold", type=float, default=0.35)
    s = sub.add_parser("status")
    s.add_argument("out")
    s = sub.add_parser("comfy")
    s.add_argument("--url", default="http://127.0.0.1:8188")
    sub.add_parser("presets")
    a = p.parse_args(argv)

    prog = lambda f, m: print(f"[NeuralTexture] {f:6.1%} {m}", file=sys.stderr, flush=True)  # noqa: E731
    try:
        if a.cmd == "intent":
            res = parse_prompt(a.text, has_reference=a.reference).to_dict()
        elif a.cmd == "synth":
            cfg = _cfg(a)
            res = synth.run_synthesis(a.out, cfg, _gen(a.generator, cfg, a.out), None, prog, resume=not a.fresh)
        elif a.cmd == "rebuild":
            res = synth.finalize(a.out, _cfg(a))
        elif a.cmd == "regenerate":
            cfg = _cfg(a)
            mask = np.load(a.mask) if a.mask else synth.low_confidence_mask(a.out, cfg, a.threshold)
            res = synth.regenerate_region(a.out, cfg, _gen(a.generator, cfg, a.out), mask.astype(np.float32),
                                          None, prog)
        elif a.cmd == "status":
            res = status(a.out)
        elif a.cmd == "comfy":
            from .bridge import compat
            from .bridge.comfyui import ComfyClient
            from .core.config import ModelConfig
            cli = ComfyClient(a.url, timeout=10)
            m = ModelConfig(server_url=a.url)
            vram = cli.vram()
            rep = compat.autodetect(m, cli.object_info(), vram.get("total_gb", 0))
            res = {"vram": vram, "report": rep, "model": m.__dict__}
        else:
            res = {k: {"label": v["label"], "description": v["description"]} for k, v in PRESETS.items()}
        print(json.dumps({"ok": True, "result": res}, indent=2, default=str))
        return 0
    except synth.Cancelled:
        print(json.dumps({"ok": False, "cancelled": True}))
        return 5
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(e)}))
        return 4


if __name__ == "__main__":
    sys.exit(main())
