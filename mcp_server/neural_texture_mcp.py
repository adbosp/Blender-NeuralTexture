#!/usr/bin/env python3
"""Neural Texture MCP server (V5) — stdio, dependency-free (Python 3.9+ + numpy).

Lets Claude Code / Codex / any MCP client drive the pipeline with natural
language ("làm model này thành đá cổ nhưng giữ normal gốc"):

  * Live mode   — talks to a running Blender through the add-on's Agent Bridge
                  (N panel > Neural Texture > ComfyUI > Agent Bridge).
  * Headless    — runs `blender -b file.blend --python cli.py` (set NT_BLENDER).
  * Pure Python — synthesis / rebuild / regenerate on an existing guidance
                  package, no Blender needed.

Register (Claude Code):
    claude mcp add neural-texture -- python /path/to/mcp_server/neural_texture_mcp.py
Env: NT_BLENDER=/path/to/blender   NT_ADDON_ROOT=/path/containing/neural_texture
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_ROOT = os.environ.get("NT_ADDON_ROOT") or os.path.dirname(HERE)
if ADDON_ROOT not in sys.path:
    sys.path.insert(0, ADDON_ROOT)

BRIDGE_INFO = os.path.join(os.path.expanduser("~"), ".neural_texture", "bridge.json")
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "neural-texture", "version": "0.2.0"}


# ----------------------------------------------------------------------------- live bridge client
class BridgeError(RuntimeError):
    pass


def bridge_call(cmd: str, args: Optional[Dict] = None, timeout: float = 120.0) -> Any:
    if not os.path.exists(BRIDGE_INFO):
        raise BridgeError("Blender Agent Bridge is not running. In Blender: N panel > Neural Texture > "
                          "ComfyUI > 'Agent Bridge', or use headless mode (blend_file + NT_BLENDER).")
    with open(BRIDGE_INFO, encoding="utf-8") as f:
        info = json.load(f)
    try:
        with socket.create_connection(("127.0.0.1", int(info["port"])), timeout=timeout) as s:
            s.sendall((json.dumps({"id": 1, "token": info["token"], "cmd": cmd, "args": args or {}}) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
    except OSError as e:
        raise BridgeError(f"Cannot reach Blender Agent Bridge on port {info.get('port')}: {e}") from e
    resp = json.loads(buf or b"{}")
    if not resp.get("ok"):
        raise BridgeError(resp.get("error", "bridge error"))
    return resp.get("result")


def live_available() -> bool:
    try:
        bridge_call("ping", timeout=3)
        return True
    except BridgeError:
        return False


# ----------------------------------------------------------------------------- headless blender
def run_blender_cli(blend_file: str, job: Dict, stage: str, extra: List[str] = (), timeout: float = 7200) -> Dict:
    blender = os.environ.get("NT_BLENDER", "blender")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(job, f)
        job_path = f.name
    cmd = [blender, "-b", blend_file, "--python", os.path.join(ADDON_ROOT, "neural_texture", "cli.py"), "--",
           "--job", job_path, "--stage", stage, *extra]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise RuntimeError(f"Blender executable not found ('{blender}'); set NT_BLENDER.") from e
    finally:
        os.unlink(job_path)
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("NT_RESULT_JSON: "):
            return json.loads(line[len("NT_RESULT_JSON: "):])
    return {"ok": False, "error": "no result from Blender", "exit_code": p.returncode,
            "stderr_tail": p.stderr[-3000:], "stdout_tail": p.stdout[-3000:]}


# ----------------------------------------------------------------------------- helpers
def build_job(a: Dict) -> Dict:
    from neural_texture.core.config import JobConfig
    from neural_texture.core.presets import apply_preset
    cfg = JobConfig.from_dict(a.get("job") or {})
    if a.get("object"):
        cfg.object_name = a["object"]
    if a.get("prompt"):
        cfg.prompt = a["prompt"]
    if a.get("reference_images"):
        cfg.reference_images = list(a["reference_images"])
    if a.get("intent"):
        cfg.intent = a["intent"]
    if a.get("comfy_url"):
        cfg.model.server_url = a["comfy_url"]
    if a.get("output_dir"):
        cfg.output_dir = a["output_dir"]
    for k in ("texture_resolution", "view_resolution"):
        if a.get(k):
            setattr(cfg, k, int(a[k]))
    if a.get("preset"):
        apply_preset(cfg, a["preset"])
    return cfg.to_dict()


def _mode(a: Dict) -> str:
    if a.get("blend_file"):
        return "headless"
    return "live"


# ----------------------------------------------------------------------------- tools
TOOLS: Dict[str, Dict] = {}


def tool(name: str, description: str, props: Dict, required: List[str] = ()):
    def deco(fn: Callable):
        TOOLS[name] = {"fn": fn, "schema": {"name": name, "description": description,
                                            "inputSchema": {"type": "object", "properties": props,
                                                            "required": list(required)}}}
        return fn
    return deco


JOB_PROPS = {
    "object": {"type": "string", "description": "Mesh object name in the .blend"},
    "prompt": {"type": "string", "description": "Appearance request (English or Vietnamese)"},
    "intent": {"type": "object", "description": "Structured intent (preferred; see agent/intent.schema.json)"},
    "preset": {"type": "string", "description": "Preset name (see nt_list_presets)"},
    "reference_images": {"type": "array", "items": {"type": "string"}},
    "comfy_url": {"type": "string"},
    "output_dir": {"type": "string"},
    "texture_resolution": {"type": "integer"},
    "view_resolution": {"type": "integer"},
    "job": {"type": "object", "description": "Full JobConfig overrides"},
    "blend_file": {"type": "string", "description": "If set, run headless Blender on this file instead of the live bridge"},
}


@tool("nt_parse_intent", "Parse a natural-language texture request (EN/VI) into the structured intent the "
      "pipeline uses. Shows material class, preserved structure, detail bands and GEOMETRY WARNINGS "
      "(requests textures cannot honestly implement). Call this first and review it.",
      {"text": {"type": "string"}, "has_reference": {"type": "boolean"}}, ["text"])
def t_parse_intent(a):
    from neural_texture.core.intent import parse_prompt
    return parse_prompt(a["text"], bool(a.get("has_reference"))).to_dict()


@tool("nt_list_presets", "List named presets (enhance, restyle, skin_detail, ancient_stone, ...).", {})
def t_presets(a):
    from neural_texture.core.presets import PRESETS
    return {k: {"label": v["label"], "description": v["description"], "values": v["values"]}
            for k, v in PRESETS.items()}


@tool("nt_blender_status", "Is a live Blender reachable? Returns add-on/Blender version and mesh objects.", {})
def t_status(a):
    try:
        return {"live": True, "ping": bridge_call("ping", timeout=5), "scene": bridge_call("scene_info", timeout=10)}
    except BridgeError as e:
        return {"live": False, "reason": str(e), "headless_blender": os.environ.get("NT_BLENDER", "blender")}


@tool("nt_check_comfyui", "Check the ComfyUI server: VRAM, which checkpoints/ControlNets/custom nodes are "
      "installed and what the pipeline will auto-select.", {"url": {"type": "string"}})
def t_comfy(a):
    from neural_texture.bridge import compat
    from neural_texture.bridge.comfyui import ComfyClient
    from neural_texture.core.config import ModelConfig
    url = a.get("url") or "http://127.0.0.1:8188"
    cli = ComfyClient(url, timeout=10)
    m = ModelConfig(server_url=url)
    vram = cli.vram()
    return {"vram": vram, "report": compat.autodetect(m, cli.object_info(), vram.get("total_gb", 0)),
            "selected": {k: getattr(m, k) for k in ("checkpoint", "controlnet", "controlnet_is_union",
                                                     "use_ipadapter", "normal_estimator")}}


@tool("nt_analyze", "Read-only analysis of a mesh: topology, rig, UV health, existing maps, parsed intent.",
      JOB_PROPS, ["object"])
def t_analyze(a):
    job = build_job(a)
    if _mode(a) == "headless":
        return run_blender_cli(a["blend_file"], job, "analyze")
    return bridge_call("analyze", {"object": a["object"], "job": job}, timeout=300)


@tool("nt_generate", "Generate a new PBR texture set for an existing mesh (structure and base normal preserved). "
      "Live mode starts a background job (poll nt_job_status) unless wait=true. Refuses geometry edits.",
      dict(JOB_PROPS, generator={"type": "string", "enum": ["comfyui", "identity", "folder"]},
           wait={"type": "boolean"}, fresh={"type": "boolean"},
           timeout_s={"type": "number", "description": "max wait in live mode (default 3600)"}), ["object"])
def t_generate(a):
    from neural_texture.core.intent import Intent, parse_prompt
    job = build_job(a)
    intent = Intent.from_dict(job["intent"]) if job.get("intent") else parse_prompt(job.get("prompt", ""))
    if intent.requires_geometry_mode:
        return {"ok": False, "requires_geometry_mode": True,
                "geometry_warnings": [w.to_dict() for w in intent.geometry_warnings]}
    gen = a.get("generator", "comfyui")
    if _mode(a) == "headless":
        extra = ["--generator", gen] + (["--fresh"] if a.get("fresh") else [])
        return run_blender_cli(a["blend_file"], job, "all", extra)
    started = bridge_call("start_generate", {"object": a["object"], "job": job, "generator": gen,
                                             "fresh": bool(a.get("fresh"))})
    if not a.get("wait"):
        return {"started": started, "hint": "poll nt_job_status; then nt_result"}
    return _wait_live(a["object"], job, float(a.get("timeout_s", 3600)))


def _wait_live(obj: str, job: Dict, timeout: float) -> Dict:
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        last = bridge_call("status", timeout=10)
        if last.get("status") in ("done", "error", "cancelled", "idle"):
            break
        time.sleep(3)
    res = bridge_call("result", {"object": obj, "job": job}, timeout=30)
    return {"job": last, "latest": res.get("latest")}


@tool("nt_job_status", "Progress of the running live job (status, progress 0-1, message, error).", {})
def t_job_status(a):
    return bridge_call("status", timeout=10)


@tool("nt_cancel", "Cancel the running live job (finished views are kept; the next generate resumes).", {})
def t_cancel(a):
    return bridge_call("cancel", timeout=10)


@tool("nt_result", "Latest result for an object: texture paths, validation (ok/issues), stats, history.",
      {"object": {"type": "string"}, "output_dir": {"type": "string"},
       "package_dir": {"type": "string", "description": "per-object folder (pure-Python mode)"}})
def t_result(a):
    if a.get("package_dir"):
        from neural_texture.headless import status
        return status(a["package_dir"])
    return bridge_call("result", {"object": a.get("object"), "job": build_job(a)}, timeout=30)


@tool("nt_regenerate", "Regenerate only part of the texture: a UV mask (.npy), the mask painted in Blender, "
      "or all texels below a confidence threshold. Everything else stays identical.",
      dict(JOB_PROPS, mask_path={"type": "string"}, use_painted_mask={"type": "boolean"},
           threshold={"type": "number"}, generator={"type": "string"}), ["object"])
def t_regenerate(a):
    job = build_job(a)
    if _mode(a) == "headless":
        extra = ["--generator", a.get("generator", "comfyui")]
        extra += ["--mask", a["mask_path"]] if a.get("mask_path") else ["--threshold", str(a.get("threshold", 0.35))]
        return run_blender_cli(a["blend_file"], job, "regenerate", extra)
    return bridge_call("start_regenerate", {"object": a["object"], "job": job, "mask_path": a.get("mask_path"),
                                            "use_painted_mask": a.get("use_painted_mask"),
                                            "threshold": a.get("threshold", 0.35),
                                            "generator": a.get("generator", "comfyui")})


@tool("nt_rebuild_maps", "Rebuild PBR maps from existing generation with new settings (detail strength, "
      "keep colours, roughness...) — seconds, no image generation.", JOB_PROPS, ["object"])
def t_rebuild(a):
    job = build_job(a)
    if _mode(a) == "headless":
        return run_blender_cli(a["blend_file"], job, "rebuild")
    return bridge_call("start_rebuild", {"object": a["object"], "job": job})


@tool("nt_before_after", "Show original ('before') or generated ('after') material on the object.",
      {"object": {"type": "string"}, "show": {"type": "string", "enum": ["before", "after"]}}, ["object", "show"])
def t_before_after(a):
    return bridge_call("before_after", {"object": a["object"], "show": a["show"]})


@tool("nt_synthesize_package", "Pure-Python synthesis on an existing guidance package folder (no Blender): "
      "consistent multi-view generation, UV accumulation, inpainting, PBR maps.",
      {"package_dir": {"type": "string"}, "job": {"type": "object"}, "prompt": {"type": "string"},
       "preset": {"type": "string"}, "generator": {"type": "string", "enum": ["comfyui", "identity", "folder"]},
       "fresh": {"type": "boolean"}}, ["package_dir"])
def t_synth_pkg(a):
    from neural_texture.core import synth
    from neural_texture.core.config import JobConfig
    from neural_texture.headless import _gen
    cfg = JobConfig.from_dict(build_job(a))
    out = a["package_dir"]
    res = synth.run_synthesis(out, cfg, _gen(a.get("generator", "comfyui"), cfg, out), None,
                              lambda f, m: None, resume=not a.get("fresh"))
    return {k: res[k] for k in ("run_id", "textures", "validation", "stats", "slot_classes") if k in res}


# ----------------------------------------------------------------------------- JSON-RPC loop
def _result(rid, result):
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _error(rid, code, msg):
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


def handle(req: Dict) -> Optional[Dict]:
    method, rid = req.get("method"), req.get("id")
    if method == "initialize":
        want = (req.get("params") or {}).get("protocolVersion")
        return _result(rid, {"protocolVersion": want if want in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                             "capabilities": {"tools": {"listChanged": False}}, "serverInfo": SERVER_INFO,
                             "instructions": "Texture/appearance generation for existing Blender meshes. "
                                             "Always nt_parse_intent first; never fake geometry changes."})
    if method in ("notifications/initialized", "notifications/cancelled") or rid is None:
        return None
    if method == "ping":
        return _result(rid, {})
    if method == "tools/list":
        return _result(rid, {"tools": [t["schema"] for t in TOOLS.values()]})
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        if name not in TOOLS:
            return _error(rid, -32602, f"unknown tool {name}")
        try:
            out = TOOLS[name]["fn"](params.get("arguments") or {})
            is_err = isinstance(out, dict) and out.get("ok") is False
            return _result(rid, {"content": [{"type": "text", "text": json.dumps(out, indent=2, default=str)}],
                                 "isError": is_err})
        except Exception as e:  # noqa: BLE001 - reported to the model as a tool error
            return _result(rid, {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}\n"
                                              + traceback.format_exc()[-1500:]}], "isError": True})
    return _error(rid, -32601, f"method not found: {method}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            sys.stdout.write(json.dumps(_error(None, -32700, "parse error")) + "\n")
            sys.stdout.flush()
            continue
        resp = handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, default=str) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
