#!/usr/bin/env python3
"""Benchmark the pure-Python pipeline (V6) on synthetic assets — no Blender/GPU.

    python tools/benchmark.py [--res 256 512 1024] [--shapes sphere cube] [--out bench.json]

Measures, per shape x texture resolution: package build, planning, synthesis
(fake model), finalize (PBR + seams), peak RSS, plus quality metrics
(seen fraction, confidence, seam error, albedo error). Use it to catch
performance regressions and to size texture resolution for your machine.
The ComfyUI generation time is excluded (the fake model is instant); add
~5-20 s per view for SDXL on a modern GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from synthetic import Cube, FakeModel, Sphere, build_package, true_albedo  # noqa: E402
from neural_texture.core import synth  # noqa: E402
from neural_texture.core.config import JobConfig  # noqa: E402
from neural_texture.core.generators import read_image  # noqa: E402
from neural_texture.core.package import Package  # noqa: E402


def peak_rss_mb() -> float:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024 if platform.system() != "Darwin" else r / 1024 / 1024
    except ImportError:
        return float("nan")


def run(shape_name: str, res: int, view_res: int) -> dict:
    shape = {"sphere": Sphere, "cube": Cube}[shape_name]()
    with tempfile.TemporaryDirectory() as td:
        t0 = time.time()
        build_package(td, shape, tex_res=res, view_res=view_res)
        t_pkg = time.time() - t0
        cfg = JobConfig(prompt="ancient stone")
        cfg.synth = {"max_views": 10}
        steps = {}
        t1 = time.time()
        res_ = synth.run_synthesis(td, cfg, FakeModel(td, drift=0.1, light=0.3),
                                   progress=lambda f, m: steps.setdefault(m.split(":")[0], time.time()))
        t_syn = time.time() - t1
        t2 = time.time()
        synth.finalize(td, cfg)
        t_fin = time.time() - t2
        pkg = Package(td)
        bc = read_image(os.path.join(td, res_["textures"]["basecolor"]))[..., :3]
        cov = pkg.uv("coverage")
        err = float(np.abs(bc - true_albedo(pkg.uv("position")))[cov].mean())
        st = res_["stats"]
        return {"shape": shape_name, "texture_res": res, "view_res": view_res,
                "views": len(res_["synthesis"]["plan"]["order"]),
                "seconds": {"package": round(t_pkg, 2), "synthesis_total": round(t_syn, 2),
                            "finalize": round(t_fin, 2)},
                "quality": {"seen": round(st["seen_fraction"], 4), "confidence": round(st["mean_confidence"], 3),
                            "seam_before": round(st.get("seam_error_before", 0), 4),
                            "seam_after": round(st.get("seam_error_after", 0), 4), "albedo_abs_err": round(err, 4)},
                "peak_rss_mb": round(peak_rss_mb(), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--view-res", type=int, default=256)
    ap.add_argument("--shapes", nargs="+", default=["sphere", "cube"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for shape in a.shapes:
        for r in a.res:
            row = run(shape, r, a.view_res)
            rows.append(row)
            print(json.dumps(row))
    info = {"python": sys.version.split()[0], "numpy": np.__version__, "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "results": rows}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
