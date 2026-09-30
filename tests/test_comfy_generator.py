"""ComfyGenerator + full synthesis against a richer mock ComfyUI (object_info specs,
OOM on first try, normal estimator, atlas inpaint, interrupt/free)."""
import json
import os
import re
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synthetic import Sphere, build_package  # noqa: E402
from neural_texture.bridge import compat  # noqa: E402
from neural_texture.bridge.comfy_generator import ComfyGenerator  # noqa: E402
from neural_texture.core import synth  # noqa: E402
from neural_texture.core.config import JobConfig, ModelConfig  # noqa: E402

I = lambda **k: ("INT", k)  # noqa: E731
F = lambda **k: ("FLOAT", k)  # noqa: E731
OBJECT_INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["v1-5.safetensors", "juggernautXL_v9.safetensors", "sdxl_refiner.safetensors"]]}}},
    "CLIPTextEncode": {"input": {"required": {"text": ("STRING", {}), "clip": ("CLIP",)}}},
    "LoadImage": {"input": {"required": {"image": [["x.png"], {"image_upload": True}]}}},
    "ControlNetLoader": {"input": {"required": {"control_net_name": [["diffusers_xl_depth.safetensors", "xinsir-controlnet-union-sdxl-promax.safetensors"]]}}},
    "SetUnionControlNetType": {"input": {"required": {"control_net": ("CONTROL_NET",), "type": [["auto", "depth", "normal", "canny"]]}}},
    "ControlNetApplyAdvanced": {"input": {"required": {"positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
                                                       "control_net": ("CONTROL_NET",), "image": ("IMAGE",),
                                                       "strength": F(default=1.0), "start_percent": F(default=0.0),
                                                       "end_percent": F(default=1.0)},
                                          "optional": {"vae": ("VAE",)}}},
    "VAEEncode": {"input": {"required": {"pixels": ("IMAGE",), "vae": ("VAE",)}}},
    "EmptyLatentImage": {"input": {"required": {"width": I(default=1024), "height": I(default=1024), "batch_size": I(default=1)}}},
    "ImageToMask": {"input": {"required": {"image": ("IMAGE",), "channel": [["red", "green", "blue", "alpha"]]}}},
    "SetLatentNoiseMask": {"input": {"required": {"samples": ("LATENT",), "mask": ("MASK",)}}},
    "KSampler": {"input": {"required": {k: ("X",) for k in ["model", "positive", "negative", "latent_image"]} | {
        "seed": I(default=0), "steps": I(default=20), "cfg": F(default=8.0),
        "sampler_name": [["euler", "dpmpp_2m"]], "scheduler": [["normal", "karras"]], "denoise": F(default=1.0)}}},
    "VAEDecode": {"input": {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}},
    "SaveImage": {"input": {"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "ComfyUI"})}}},
    "BAE-NormalMapPreprocessor": {"input": {"required": {"image": ("IMAGE",)}, "optional": {"resolution": I(default=512)}}},
    # no IPAdapter nodes installed
}


class RichMock:
    def __init__(self, oom_first=True):
        self.uploads, self.prompts, self.graphs = {}, {}, []
        self.oom_pending = oom_first
        self.freed = 0
        self.interrupted = 0

    def serve(self):
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/system_stats":
                    return self._send(200, {"devices": [{"name": "mock", "vram_total": 6 * 1024 ** 3, "vram_free": 5 * 1024 ** 3}]})
                if self.path == "/object_info":
                    return self._send(200, OBJECT_INFO)
                m = re.match(r"/history/(\w+)", self.path)
                if m:
                    pid = m.group(1)
                    g = mock.prompts[pid]
                    if mock.oom_pending and any(n["class_type"] == "KSampler" for n in g.values()):
                        mock.oom_pending = False
                        return self._send(200, {pid: {"status": {"status_str": "error", "completed": False,
                                          "messages": [["execution_error", {"exception_message": "CUDA out of memory. Tried to allocate"}]]},
                                          "outputs": {}}})
                    return self._send(200, {pid: {"status": {"status_str": "success", "completed": True},
                                                  "outputs": {"9": {"images": [{"filename": f"{pid}.png", "subfolder": "", "type": "output"}]}}}})
                if self.path.startswith("/view"):
                    pid = re.search(r"filename=(\w+)\.png", self.path).group(1)
                    names = [n["inputs"]["image"] for n in mock.prompts[pid].values() if n["class_type"] == "LoadImage"]
                    for key in ("_init", "_albedo", "_final", "_atlas.png"):
                        hit = [n for n in names if key in n]
                        if hit:
                            return self._send(200, mock.uploads[hit[0].split("/")[-1]], "image/png")
                    return self._send(404, {})
                self._send(404, {"error": self.path})

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path == "/upload/image":
                    name = re.search(rb'filename="([^"]+)"', body).group(1).decode()
                    start = body.index(b"\r\n\r\n", body.index(b'name="image"')) + 4
                    mock.uploads[name] = body[start:body.rindex(b"\r\n--")]
                    return self._send(200, {"name": name, "subfolder": "nt", "type": "input"})
                if self.path == "/prompt":
                    g = json.loads(body)["prompt"]
                    pid = f"p{len(mock.prompts)}"
                    mock.prompts[pid] = g
                    mock.graphs.append(g)
                    return self._send(200, {"prompt_id": pid, "number": 0, "node_errors": {}})
                if self.path == "/free":
                    mock.freed += 1
                    return self._send(200, {})
                if self.path == "/interrupt":
                    mock.interrupted += 1
                    return self._send(200, {})
                self._send(404, {})

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


def test_autodetect_picks_models():
    m = ModelConfig(checkpoint="missing.safetensors", controlnet="nope.safetensors")
    rep = compat.autodetect(m, OBJECT_INFO, vram_gb=6.0)
    assert m.checkpoint == "juggernautXL_v9.safetensors"
    assert "union" in m.controlnet and m.controlnet_is_union
    assert not m.use_ipadapter
    assert m.normal_estimator == "BAE-NormalMapPreprocessor"
    assert any("VRAM" in w for w in rep["warnings"])


def test_complete_and_validate_fills_and_fixes():
    g = {"1": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["2", 0], "negative": ["2", 0],
                                                     "latent_image": ["2", 0], "sampler_name": "dpmpp_2m",
                                                     "scheduler": "karras", "bogus": 3}},
         "2": {"class_type": "EmptyLatentImage", "inputs": {}}}
    out, notes = compat.complete_and_validate(g, OBJECT_INFO)
    assert out["1"]["inputs"]["steps"] == 20 and "bogus" not in out["1"]["inputs"]
    assert out["2"]["inputs"]["width"] == 1024
    try:
        compat.complete_and_validate({"1": {"class_type": "KSampler", "inputs": {"sampler_name": "nope"}}}, OBJECT_INFO)
        assert False
    except compat.GraphError as e:
        assert "sampler_name" in str(e)
    try:
        compat.complete_and_validate({"1": {"class_type": "IPAdapterAdvanced", "inputs": {}}}, OBJECT_INFO)
        assert False
    except compat.GraphError as e:
        assert "IPAdapterAdvanced" in str(e)


def test_full_synthesis_through_mock_comfyui_with_oom_recovery():
    mock = RichMock(oom_first=True)
    srv = mock.serve()
    try:
        with tempfile.TemporaryDirectory() as td:
            build_package(td, Sphere(), tex_res=64, view_res=96)
            ref = os.path.join(td, "ref.png")
            from neural_texture.core import pngio
            pngio.write_png(ref, np.full((16, 16, 3), 0.6, np.float32))
            cfg = JobConfig(prompt="ancient stone", reference_images=[ref])
            cfg.model.server_url = f"http://127.0.0.1:{srv.server_port}"
            cfg.model.timeout_s = 10
            cfg.model.min_resolution = 64
            cfg.synth = {"max_views": 6}
            logs = []
            gen = ComfyGenerator(cfg.model, log=logs.append)
            res = synth.run_synthesis(td, cfg, gen)
            assert mock.freed == 1 and any("out of VRAM" in l for l in logs)
            assert res["stats"]["seen_fraction"] > 0.9
            kinds = [{n["class_type"] for n in g.values()} for g in mock.graphs]
            assert any("SetLatentNoiseMask" in k for k in kinds)          # consistency inpaint
            assert any("BAE-NormalMapPreprocessor" in k for k in kinds)   # normal estimation
            assert not any("IPAdapterAdvanced" in k for k in kinds)        # auto-disabled
            ks = [n for g in mock.graphs for n in g.values() if n["class_type"] == "KSampler"]
            assert all("steps" in k["inputs"] for k in ks)
            # The mock "estimator" echoes the colour image: calibration must detect that
            # it disagrees with the geometry and fall back to luminance detail.
            cal = synth.load_state(td)["normal_calibration"]
            assert cal["agreement"] < 0.25 and res["stats"]["detail_source"] == "luminance", (cal, res["stats"])
    finally:
        srv.shutdown()


def test_cancel_interrupts_comfyui():
    mock = RichMock(oom_first=False)
    srv = mock.serve()
    try:
        with tempfile.TemporaryDirectory() as td:
            build_package(td, Sphere(), tex_res=32, view_res=64, views=["front", "back", "left"])
            cfg = JobConfig(prompt="stone")
            cfg.model.server_url = f"http://127.0.0.1:{srv.server_port}"
            cfg.model.min_resolution = 64
            tok = synth.CancelToken()
            gen = ComfyGenerator(cfg.model, log=lambda m: None)
            calls = []

            def prog(f, m):
                calls.append(m)
                if len([c for c in calls if c.startswith("generating")]) == 2:
                    tok.cancel()
            try:
                synth.run_synthesis(td, cfg, gen, progress=prog, cancel=tok)
                assert False
            except synth.Cancelled:
                pass
            assert synth.load_state(td)["status"] == "cancelled"
            assert mock.interrupted >= 1
    finally:
        srv.shutdown()
