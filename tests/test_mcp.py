"""MCP server over stdio: protocol, tools, pure-Python synthesis, live-bridge client."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(__file__))
from synthetic import Sphere, build_package  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "mcp_server", "neural_texture_mcp.py")


def _session(requests, home):
    env = dict(os.environ, HOME=home, USERPROFILE=home)
    p = subprocess.run([sys.executable, SERVER], input="\n".join(json.dumps(r) for r in requests) + "\n",
                       capture_output=True, text=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr
    return [json.loads(l) for l in p.stdout.splitlines() if l.strip()]


def _call(i, name, args):
    return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}


def _payload(resp):
    return json.loads(resp["result"]["content"][0]["text"])


def test_protocol_and_tools():
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as pkg:
        build_package(pkg, Sphere(), tex_res=48, view_res=64, views=["front", "back", "left", "right", "top", "bottom"])
        out = _session([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            _call(3, "nt_parse_intent", {"text": "làm model này thành đá cổ nhưng giữ normal gốc"}),
            _call(4, "nt_parse_intent", {"text": "make the nose twice as large"}),
            _call(5, "nt_blender_status", {}),
            _call(6, "nt_synthesize_package", {"package_dir": pkg, "prompt": "ancient stone", "generator": "identity",
                                               "job": {"synth": {"uv_inpaint": False}}}),
            _call(7, "nt_result", {"package_dir": pkg}),
            _call(8, "nt_generate", {"object": "Head", "prompt": "làm mũi to gấp đôi"}),
            {"jsonrpc": "2.0", "id": 9, "method": "nope"},
        ], home)
        by_id = {r.get("id"): r for r in out}
        assert by_id[1]["result"]["protocolVersion"] == "2025-06-18"
        names = {t["name"] for t in by_id[2]["result"]["tools"]}
        assert {"nt_generate", "nt_parse_intent", "nt_regenerate", "nt_synthesize_package"} <= names
        it = _payload(by_id[3])
        assert it["material_class"] == "stone" and it["preserve_macro_normal"] and not it["geometry_warnings"]
        assert _payload(by_id[4])["geometry_warnings"]
        assert _payload(by_id[5])["live"] is False
        syn = _payload(by_id[6])
        assert "basecolor" in syn["textures"] and syn["slot_classes"] == ["stone"]
        assert _payload(by_id[7])["latest"]["run_id"] == syn["run_id"]
        g = by_id[8]["result"]
        assert g["isError"] and _payload(by_id[8])["requires_geometry_mode"]
        assert by_id[9]["error"]["code"] == -32601


def test_live_bridge_client_roundtrip():
    """Fake Blender bridge speaking the add-on protocol; MCP must pass the token and relay results."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    port = srv.getsockname()[1]
    seen = []

    def serve():
        for _ in range(2):
            c, _ = srv.accept()
            with c:
                f = c.makefile("rwb")
                req = json.loads(f.readline())
                seen.append(req)
                ok = req.get("token") == "secret"
                res = {"status": "running", "progress": 0.5, "message": "view 3/8"} if req["cmd"] == "status" else {"x": 1}
                f.write((json.dumps({"id": req["id"], "ok": ok, "result": res, "error": "bad token"}) + "\n").encode())
                f.flush()
    threading.Thread(target=serve, daemon=True).start()
    with tempfile.TemporaryDirectory() as home:
        os.makedirs(os.path.join(home, ".neural_texture"))
        with open(os.path.join(home, ".neural_texture", "bridge.json"), "w") as f:
            json.dump({"port": port, "token": "secret"}, f)
        out = _session([_call(1, "nt_job_status", {}), _call(2, "nt_cancel", {})], home)
    st = _payload(out[0])
    assert st["progress"] == 0.5 and seen[0]["cmd"] == "status" and seen[1]["cmd"] == "cancel"
    srv.close()
