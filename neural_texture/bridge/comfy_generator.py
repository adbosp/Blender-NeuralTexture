"""ComfyUI implementation of the synthesis generator interface (V2 + V6).

Adds on top of the raw client:
  * model / custom-node auto-detection and graph completion (compat.py)
  * consistency inpainting (init image + latent noise mask)
  * optional neural normal estimation per view
  * texture-space (atlas) inpainting of unseen texels
  * OOM recovery: /free + retry at reduced resolution
  * cancellation: interrupts the running prompt
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from ..core import pngio
from ..core.config import ModelConfig
from ..core.generators import BaseGenerator, GeneratorCancelled, ViewJob, read_image
from ..core.imageops import resize_bilinear
from . import compat
from .comfyui import ComfyCancelled, ComfyClient, ComfyError, is_oom
from .workflow import (build_atlas_inpaint_graph, build_default_graph, build_normal_estimator_graph,
                       fill_template, load_template)


class ComfyGenerator(BaseGenerator):
    name = "comfyui"
    supports_atlas_inpaint = True

    def __init__(self, model: ModelConfig, log: Callable[[str], None] = print,
                 client: Optional[ComfyClient] = None, reference_images: Optional[List[str]] = None):
        self.m = model
        self.log = log
        self.client = client or ComfyClient(model.server_url)
        self._stop = threading.Event()
        self._refs_local = list(reference_images or [])
        self.refs: List[str] = []
        self.scale = 1.0
        self.report: Dict = {}
        self.tag = f"nt{int(time.time())}"
        self.template = None
        self._tmp = tempfile.mkdtemp(prefix="nt_upload_")

    # ------------------------------------------------------------ lifecycle
    def begin(self, context: Dict) -> None:
        cfg = context.get("cfg")
        if cfg is not None and not self._refs_local:
            self._refs_local = list(cfg.reference_images)
        self.client.ping()
        info = self.client.object_info()
        vram = self.client.vram()
        if self.m.auto_detect_models:
            self.report = compat.autodetect(self.m, info, vram.get("total_gb", 0.0))
            for c in self.report.get("changes", []):
                self.log(f"[NeuralTexture] auto-detect: {c}")
            for w in self.report.get("warnings", []):
                self.log(f"[NeuralTexture] warning: {w}")
        self.supports_normals = self.m.normal_estimator not in ("off", "auto", "") and \
            self.m.normal_estimator in info
        self.template = load_template(self.m.custom_workflow) if self.m.custom_workflow else None
        self.refs = [self.client.upload_image(p, f"{self.tag}_ref{i}{os.path.splitext(p)[1] or '.png'}")
                     for i, p in enumerate(self._refs_local)]

    def interrupt(self) -> None:
        self._stop.set()
        self.client.interrupt()

    # ------------------------------------------------------------ helpers
    def _upload(self, path: str, name: str, size=None) -> str:
        """Upload, resizing to ``size`` (w, h) when it differs (VRAM fallback / big atlases)."""
        if size is not None and path.lower().endswith(".png"):
            img = read_image(path)
            if img.shape[1] != size[0] or img.shape[0] != size[1]:
                small = os.path.join(self._tmp, name)
                pngio.write_png(small, resize_bilinear(img, size[1], size[0]))
                path = small
        return self.client.upload_image(path, name)

    def _run(self, make_graph: Callable[[], Dict], label: str) -> np.ndarray:
        """Queue a graph with validation, OOM retries and cancellation."""
        attempts = 0
        while True:
            if self._stop.is_set():
                raise GeneratorCancelled("cancelled")
            graph = make_graph()
            try:
                graph, notes = compat.complete_and_validate(graph, self.client.object_info())
                for n in notes:
                    self.log(f"[NeuralTexture] compat: {n}")
                pid = self.client.queue(graph)
                entry = self.client.wait(pid, timeout=self.m.timeout_s, should_stop=self._stop.is_set)
                imgs = self.client.output_images(entry)
                if not imgs:
                    raise ComfyError(f"No output image for {label}")
                dest = os.path.join(self._tmp, f"{label}_{pid}.png")
                self.client.download(imgs[0], dest)
                return read_image(dest)[..., :3]
            except ComfyCancelled as e:
                raise GeneratorCancelled(str(e)) from e
            except ComfyError as e:
                attempts += 1
                if is_oom(e) and attempts <= self.m.max_retries:
                    self.client.free()
                    new_scale = self.scale * 0.75
                    self.log(f"[NeuralTexture] out of VRAM on {label}; retrying at {new_scale:.2f}x resolution")
                    self.scale = new_scale
                    continue
                if attempts <= self.m.max_retries and "Cannot reach" in str(e):
                    time.sleep(2.0 * attempts)  # transient connection loss
                    continue
                raise

    def _size(self, w: int, h: int, cap: int = 0):
        s = max(self.scale, min(1.0, self.m.min_resolution / max(w, h, 1)))
        if cap:
            s = min(s, cap / max(w, h, 1))
        return max(64, int(w * s) // 8 * 8), max(64, int(h * s) // 8 * 8)

    # ------------------------------------------------------------ interface
    def generate_view(self, job: ViewJob) -> np.ndarray:
        def make():
            w, h = self._size(job.width, job.height)
            up = {k: self._upload(p, f"{self.tag}_{job.name}_{k}.png", (w, h))
                  for k, p in job.files.items() if k in ("depth", "normal", "albedo", "init", "inpaint_mask") and p}
            if "inpaint_mask" in up:
                up["mask"] = up.pop("inpaint_mask")
            if self.template:
                return fill_template(self.template, {
                    **job.prompts, "depth_image": up.get("depth", ""), "normal_image": up.get("normal", ""),
                    "albedo_image": up.get("albedo", ""), "init_image": up.get("init", up.get("albedo", "")),
                    "mask_image": up.get("mask", ""), "reference_image": self.refs[0] if self.refs else "",
                    "seed": job.seed, "steps": self.m.steps, "cfg": self.m.cfg, "denoise": job.denoise,
                    "width": w, "height": h, "view": job.name, "filename_prefix": f"{self.tag}_{job.name}"})
            return build_default_graph(self.m, job.prompts, {**up, "references": self.refs}, w, h,
                                       job.reference_role, f"{self.tag}_{job.name}", denoise=job.denoise)
        self.log(f"[NeuralTexture] ComfyUI: view '{job.name}'" + ("" if job.first else " (consistency inpaint)"))
        return self._run(make, job.name)

    def estimate_normals(self, job: ViewJob, image_path: str) -> Optional[np.ndarray]:
        if not self.supports_normals:
            return None
        try:
            img = self._run(lambda: build_normal_estimator_graph(
                self.m.normal_estimator, self._upload(image_path, f"{self.tag}_{job.name}_final.png"),
                self.m.normal_estimator_resolution, f"{self.tag}_{job.name}_nrm"), f"{job.name}_normals")
        except (ComfyError, compat.GraphError) as e:
            self.log(f"[NeuralTexture] normal estimation failed ({e}); falling back to luminance detail")
            self.supports_normals = False
            return None
        return img * 2.0 - 1.0

    def inpaint_atlas(self, atlas_path: str, mask_path: str, prompts: Dict[str, str], size: int):
        def make():
            sz = self._size(size, size, cap=1536)
            a = self._upload(atlas_path, f"{self.tag}_atlas.png", sz)
            m = self._upload(mask_path, f"{self.tag}_atlas_mask.png", sz)
            return build_atlas_inpaint_graph(self.m, prompts, a, m, self.refs, f"{self.tag}_atlas")
        try:
            return self._run(make, "atlas")
        except (ComfyError, compat.GraphError) as e:
            self.log(f"[NeuralTexture] UV inpainting skipped: {e}")
            return None
