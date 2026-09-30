"""Agent bridge (V5): lets the MCP server / Claude / Codex drive a *running* Blender.

Protocol: newline-delimited JSON over TCP on 127.0.0.1 only.
    request  {"id": 1, "token": "...", "cmd": "status", "args": {...}}
    response {"id": 1, "ok": true, "result": {...}}  |  {"id": 1, "ok": false, "error": "..."}
Commands are a fixed whitelist of pipeline operations (no arbitrary Python).
The access token and port are written to ~/.neural_texture/bridge.json so only
local processes of the same user can connect.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import socket
import threading
import traceback
from typing import Callable, Dict

import bpy
import numpy as np

DEFAULT_PORT = 9877
INFO_PATH = os.path.join(os.path.expanduser("~"), ".neural_texture", "bridge.json")

_server = None


def _job_cfg(args: Dict):
    from ..core.config import JobConfig
    cfg = JobConfig.from_dict(args.get("job") or {})
    if args.get("object"):
        cfg.object_name = args["object"]
    if args.get("preset"):
        from ..core.presets import apply_preset
        apply_preset(cfg, args["preset"])
    return cfg


def _mesh(name: str):
    obj = bpy.data.objects.get(name or "")
    if obj is None or obj.type != "MESH":
        raise ValueError(f"mesh object '{name}' not found")
    return obj


# ----------------------------------------------------------------------------- commands
def cmd_ping(args):
    from .. import bl_info
    return {"addon": ".".join(map(str, bl_info["version"])), "blender": bpy.app.version_string,
            "file": bpy.data.filepath}


def cmd_scene_info(args):
    objs = []
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        objs.append({"name": o.name, "selected": o.select_get() if o.name in bpy.context.view_layer.objects else False,
                     "materials": [s.material.name if s.material else None for s in o.material_slots],
                     "uv_maps": [l.name for l in o.data.uv_layers], "vertices": len(o.data.vertices)})
    act = bpy.context.view_layer.objects.active if bpy.context.view_layer else None
    return {"meshes": objs, "active": act.name if act else None}


def cmd_analyze(args):
    from .. import pipeline
    obj = _mesh(args.get("object"))
    cfg = _job_cfg(args)
    return pipeline.stage_analyze(obj, cfg, pipeline.output_dir(cfg, obj))


def cmd_start_generate(args):
    from .. import pipeline
    obj = _mesh(args.get("object"))
    cfg = _job_cfg(args)
    job = pipeline.build_generate_job(obj, cfg, args.get("generator", "comfyui"),
                                      bool(args.get("force_guidance")), resume=not args.get("fresh"))
    job.start()
    return {"started": job.name, "output_dir": pipeline.output_dir(cfg, obj)}


def cmd_start_regenerate(args):
    from .. import pipeline
    from ..core import synth
    obj = _mesh(args.get("object"))
    cfg = _job_cfg(args)
    out = pipeline.output_dir(cfg, obj)
    if args.get("mask_path"):
        mask = np.load(args["mask_path"])
    elif args.get("use_painted_mask"):
        mask = pipeline.read_mask_image(obj)
        if mask is None:
            raise ValueError("no painted mask image")
    else:
        mask = synth.low_confidence_mask(out, cfg, float(args.get("threshold", 0.35))).astype(np.float32)
    job = pipeline.build_regenerate_job(obj, cfg, mask, args.get("generator", "comfyui"))
    job.start()
    return {"started": job.name, "texels": int((mask > 0.5).sum())}


def cmd_start_rebuild(args):
    from .. import pipeline
    obj = _mesh(args.get("object"))
    job = pipeline.build_rebuild_job(obj, _job_cfg(args))
    job.start()
    return {"started": job.name}


def cmd_status(args):
    from . import runner
    j = runner.current()
    if j is None:
        return {"status": "idle"}
    return {"name": j.name, "status": j.status, "progress": round(j.progress, 3), "message": j.message,
            "error": j.error}


def cmd_cancel(args):
    from . import runner
    j = runner.current()
    if j is not None and j.running:
        j.cancel()
        return {"cancelling": j.name}
    return {"cancelling": None}


def cmd_result(args):
    from .. import pipeline
    from ..core import synth
    obj = _mesh(args.get("object"))
    out = pipeline.output_dir(_job_cfg(args), obj)
    return {"output_dir": out, "latest": synth.latest_result(out), "history": synth.history(out)}


def cmd_apply_material(args):
    from .. import pipeline
    from ..core import synth
    obj = _mesh(args.get("object"))
    cfg = _job_cfg(args)
    out = pipeline.output_dir(cfg, obj)
    if args.get("run_id"):
        synth.set_latest(out, args["run_id"])
    return {"material": pipeline.apply_material(obj, out, cfg, assign=args.get("assign", True))}


def cmd_before_after(args):
    from . import material_build
    obj = _mesh(args.get("object"))
    if args.get("show") == "before":
        return {"restored": material_build.restore_materials(obj)}
    mat = bpy.data.materials.get(f"NT_{obj.name}")
    if mat is None:
        raise ValueError("no generated material yet")
    material_build.assign_material(obj, mat)
    return {"assigned": mat.name}


COMMANDS: Dict[str, Callable] = {k[4:]: v for k, v in globals().items() if k.startswith("cmd_")}


# ----------------------------------------------------------------------------- server
class _Server:
    def __init__(self, port: int):
        self.token = secrets.token_hex(16)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.jobs: "queue.Queue" = queue.Queue()
        self.alive = True
        threading.Thread(target=self._accept, daemon=True, name="NT-bridge").start()
        bpy.app.timers.register(self._pump, first_interval=0.1, persistent=True)
        os.makedirs(os.path.dirname(INFO_PATH), exist_ok=True)
        with open(INFO_PATH, "w", encoding="utf-8") as f:
            json.dump({"port": self.port, "token": self.token, "pid": os.getpid()}, f)
        try:
            os.chmod(INFO_PATH, 0o600)
        except OSError:
            pass

    def _accept(self):
        while self.alive:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                break
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn):
        with conn:
            f = conn.makefile("rwb")
            for line in f:
                try:
                    req = json.loads(line)
                except ValueError:
                    continue
                if req.get("token") != self.token:
                    resp = {"id": req.get("id"), "ok": False, "error": "bad token"}
                elif req.get("cmd") not in COMMANDS:
                    resp = {"id": req.get("id"), "ok": False, "error": f"unknown cmd; known: {sorted(COMMANDS)}"}
                else:
                    done = threading.Event()
                    box = {}
                    self.jobs.put((req, box, done))
                    done.wait(timeout=600)
                    resp = box.get("resp") or {"id": req.get("id"), "ok": False, "error": "timeout"}
                f.write((json.dumps(resp, default=str) + "\n").encode())
                f.flush()

    def _pump(self):
        """Main-thread executor (bpy is not thread-safe)."""
        if not self.alive:
            return None
        while not self.jobs.empty():
            req, box, done = self.jobs.get()
            try:
                res = COMMANDS[req["cmd"]](req.get("args") or {})
                box["resp"] = {"id": req.get("id"), "ok": True, "result": res}
            except Exception as e:  # noqa: BLE001
                box["resp"] = {"id": req.get("id"), "ok": False, "error": str(e),
                               "traceback": traceback.format_exc()[-2000:]}
            done.set()
        return 0.1

    def close(self):
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass
        try:
            os.remove(INFO_PATH)
        except OSError:
            pass


def start(port: int = DEFAULT_PORT) -> int:
    global _server
    if _server is None:
        try:
            _server = _Server(port)
        except OSError:
            _server = _Server(0)  # port busy: pick a free one (written to bridge.json)
    return _server.port


def stop():
    global _server
    if _server is not None:
        _server.close()
        _server = None


def is_running() -> bool:
    return _server is not None
