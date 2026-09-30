"""ComfyUI bridge against an in-process mock server (no GPU, no network)."""
import json
import os
import re
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural_texture.core import pngio
from neural_texture.core.camera import make_views
from neural_texture.core.config import JobConfig
from neural_texture.core.intent import parse_prompt
from neural_texture.bridge.comfyui import ComfyClient, ComfyError
from neural_texture.bridge.generate import generate_views
from neural_texture.bridge.workflow import build_default_graph, fill_template, build_prompts

KNOWN_NODES = {"CheckpointLoaderSimple", "CLIPTextEncode", "LoadImage", "ControlNetLoader",
               "SetUnionControlNetType", "ControlNetApplyAdvanced", "VAEEncode", "EmptyLatentImage",
               "KSampler", "VAEDecode", "SaveImage", "IPAdapterUnifiedLoader", "IPAdapterAdvanced",
               "ImageBatch"}


class Mock:
    def __init__(self, known=KNOWN_NODES):
        self.known = known
        self.uploads = {}
        self.prompts = {}

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
                    return self._send(200, {"devices": [{"name": "mock"}]})
                if self.path == "/object_info":
                    return self._send(200, {k: {} for k in mock.known})
                m = re.match(r"/history/(\w+)", self.path)
                if m:
                    pid = m.group(1)
                    return self._send(200, {pid: {"status": {"status_str": "success", "completed": True},
                                                  "outputs": {"99": {"images": [
                                                      {"filename": f"{pid}.png", "subfolder": "", "type": "output"}]}}}})
                if self.path.startswith("/view"):
                    # Echo the albedo control image back as the "generated" result.
                    pid = re.search(r"filename=(\w+)\.png", self.path).group(1)
                    g = mock.prompts[pid]
                    alb = [n["inputs"]["image"] for n in g.values()
                           if n["class_type"] == "LoadImage" and "albedo" in n["inputs"]["image"]]
                    return self._send(200, mock.uploads[alb[0].split("/")[-1]], "image/png")
                self._send(404, {"error": self.path})

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/upload/image":
                    name = re.search(rb'filename="([^"]+)"', body).group(1).decode()
                    start = body.index(b"\r\n\r\n", body.index(b'name="image"')) + 4
                    end = body.rindex(b"\r\n--")
                    mock.uploads[name] = body[start:end]
                    return self._send(200, {"name": name, "subfolder": "neural_texture", "type": "input"})
                if self.path == "/prompt":
                    g = json.loads(body)["prompt"]
                    pid = f"p{len(mock.prompts)}"
                    mock.prompts[pid] = g
                    return self._send(200, {"prompt_id": pid, "number": 0, "node_errors": {}})
                self._send(404, {})

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


def _package(td, views=("front", "right")):
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    cams = make_views(corners, np.eye(3), list(views), resolution=64)
    man = {"views": {}}
    rng = np.random.default_rng(0)
    for c in cams:
        vd = os.path.join(td, "views", c.name)
        os.makedirs(vd)
        files = {}
        for k in ("depth", "normal", "albedo"):
            files[k] = f"views/{c.name}/{k}.png"
            pngio.write_png(os.path.join(td, files[k]), rng.random((64, 64, 3)).astype(np.float32))
        man["views"][c.name] = {"camera": c.to_dict(), "files": files}
    return man


def test_generate_views_roundtrip():
    mock = Mock()
    srv = mock.serve()
    try:
        with tempfile.TemporaryDirectory() as td:
            man = _package(td)
            ref = os.path.join(td, "face.png")
            pngio.write_png(ref, np.full((32, 32, 3), 0.7, np.float32))
            cfg = JobConfig(prompt="Make the character resemble this face. Add pores.",
                            reference_images=[ref])
            cfg.model.server_url = f"http://127.0.0.1:{srv.server_port}"
            cfg.model.timeout_s = 5
            rec = generate_views(td, cfg, man, log=lambda m: None)
            assert set(rec["views"]) == {"front", "right"}
            for v in ("front", "right"):
                gen = pngio.read_png(os.path.join(td, "views", v, "generated.png"))
                alb = pngio.read_png(os.path.join(td, "views", v, "albedo.png"))
                assert np.array_equal(gen, alb)
            g = mock.prompts["p0"]
            types = [n["class_type"] for n in g.values()]
            assert types.count("ControlNetApplyAdvanced") == 2
            assert "IPAdapterAdvanced" in types and "VAEEncode" in types
            loader = next(n for n in g.values() if n["class_type"] == "IPAdapterUnifiedLoader")
            assert "FACE" in loader["inputs"]["preset"]  # identity reference -> face preset
            ks = next(n for n in g.values() if n["class_type"] == "KSampler")
            assert ks["inputs"]["denoise"] == cfg.model.denoise
            # all links point to existing nodes
            for n in g.values():
                for v in n["inputs"].values():
                    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                        assert v[0] in g
            assert os.path.exists(os.path.join(td, "generation.json"))
    finally:
        srv.shutdown()


def test_missing_custom_nodes_reported():
    mock = Mock(known=KNOWN_NODES - {"IPAdapterAdvanced", "IPAdapterUnifiedLoader"})
    srv = mock.serve()
    try:
        with tempfile.TemporaryDirectory() as td:
            man = _package(td, ("front",))
            ref = os.path.join(td, "r.png")
            pngio.write_png(ref, np.zeros((8, 8, 3), np.float32))
            cfg = JobConfig(prompt="stone", reference_images=[ref])
            cfg.model.server_url = f"http://127.0.0.1:{srv.server_port}"
            try:
                generate_views(td, cfg, man, log=lambda m: None)
                assert False, "expected ComfyError"
            except ComfyError as e:
                assert "IPAdapter" in str(e)
    finally:
        srv.shutdown()


def test_geometry_request_blocks_generation():
    with tempfile.TemporaryDirectory() as td:
        cfg = JobConfig(prompt="Make the nose twice as large")
        try:
            generate_views(td, cfg, {"views": {}}, client=ComfyClient("http://127.0.0.1:9"))
            assert False
        except ComfyError as e:
            assert "geometry" in str(e)


def test_unreachable_server_error():
    try:
        ComfyClient("http://127.0.0.1:9", timeout=1).ping()
        assert False
    except ComfyError as e:
        assert "Cannot reach" in str(e)


def test_txt2img_graph_without_ipadapter():
    cfg = JobConfig()
    cfg.model.use_ipadapter = False
    cfg.model.denoise = 1.0
    cfg.model.controlnet_is_union = False
    p = build_prompts(parse_prompt("weathered bronze"), "top")
    g = build_default_graph(cfg.model, p, {"depth": "d.png", "normal": "n.png", "albedo": "a.png"}, 1024, 1024)
    types = [n["class_type"] for n in g.values()]
    assert "EmptyLatentImage" in types and "VAEEncode" not in types
    assert "SetUnionControlNetType" not in types and "IPAdapterAdvanced" not in types
    assert "view from directly above" in p["positive"]


def test_fill_template_types():
    t = {"3": {"class_type": "KSampler", "inputs": {"seed": "{{seed}}", "text": "a {{view}} b"}}}
    out = fill_template(t, {"seed": 5, "view": "front"})
    assert out["3"]["inputs"]["seed"] == 5 and out["3"]["inputs"]["text"] == "a front b"
    try:
        fill_template({"x": "{{nope}}"}, {})
        assert False
    except ValueError:
        pass


def test_example_template_fills_completely():
    from neural_texture.bridge.workflow import load_template, PLACEHOLDERS
    t = load_template(os.path.join(os.path.dirname(__file__), "..", "examples", "workflow_template_sdxl.json"))
    vals = {k: (1 if k in ("seed", "steps", "width", "height") else 0.5 if k in ("cfg", "denoise") else "x")
            for k in PLACEHOLDERS}
    g = fill_template(t, vals)
    ks = next(n for n in g.values() if n["class_type"] == "KSampler")
    assert ks["inputs"]["seed"] == 1 and ks["inputs"]["denoise"] == 0.5


def test_example_job_loads():
    cfg = JobConfig.load(os.path.join(os.path.dirname(__file__), "..", "examples", "job_face_example.json"))
    from neural_texture.bridge.generate import resolve_intent
    it = resolve_intent(cfg)
    assert it.reference_role == "identity" and not it.requires_geometry_mode
