"""Minimal ComfyUI HTTP client (stdlib only, no websockets needed).

Endpoints used:  POST /upload/image   POST /prompt   GET /history/{id}
                 GET /view            GET /object_info (optional validation)
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Dict, List, Optional


class ComfyError(RuntimeError):
    pass


class ComfyCancelled(ComfyError):
    pass


def is_oom(err: Exception) -> bool:
    import re
    t = str(err).lower()
    return bool(re.search(r"out of memory|outofmemory|\boom\b|failed to allocate|not enough memory|"
                          r"allocation on device", t))


class ComfyClient:
    def __init__(self, server_url: str = "http://127.0.0.1:8188", timeout: float = 30.0):
        self.base = server_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    # ------------------------------------------------------------------ http
    def _request(self, method: str, path: str, data: Optional[bytes] = None,
                 headers: Optional[Dict[str, str]] = None) -> bytes:
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise ComfyError(f"ComfyUI {method} {path} -> HTTP {e.code}: {body[:2000]}") from e
        except urllib.error.URLError as e:
            raise ComfyError(f"Cannot reach ComfyUI at {self.base}: {e.reason}") from e

    def _json(self, method: str, path: str, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        hdr = {"Content-Type": "application/json"} if data else {}
        return json.loads(self._request(method, path, data, hdr) or b"{}")

    # --------------------------------------------------------------- api
    def ping(self) -> Dict:
        return self._json("GET", "/system_stats")

    def object_info(self, refresh: bool = False) -> Dict:
        if refresh or not getattr(self, "_object_info", None):
            self._object_info = self._json("GET", "/object_info")
        return self._object_info

    def interrupt(self) -> None:
        try:
            self._request("POST", "/interrupt", b"{}", {"Content-Type": "application/json"})
        except ComfyError:
            pass

    def free(self, unload_models: bool = True) -> None:
        """Ask ComfyUI to drop cached models / free VRAM (used after OOM)."""
        try:
            self._json("POST", "/free", {"unload_models": unload_models, "free_memory": True})
        except ComfyError:
            pass

    def vram(self) -> Dict[str, float]:
        try:
            dev = (self.ping().get("devices") or [{}])[0]
        except ComfyError:
            return {}
        gb = 1024 ** 3
        return {"name": dev.get("name", ""), "total_gb": dev.get("vram_total", 0) / gb,
                "free_gb": dev.get("vram_free", 0) / gb}

    def missing_nodes(self, graph: Dict) -> List[str]:
        info = self.object_info()
        return sorted({n["class_type"] for n in graph.values() if n["class_type"] not in info})

    def upload_image(self, path: str, name: Optional[str] = None, subfolder: str = "neural_texture",
                     overwrite: bool = True) -> str:
        """Upload a local image; returns the name to use in LoadImage."""
        name = name or os.path.basename(path)
        boundary = uuid.uuid4().hex
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            content = f.read()
        parts = []
        for k, v in (("overwrite", "true" if overwrite else "false"), ("subfolder", subfolder), ("type", "input")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                      f"Content-Type: {ctype}\r\n\r\n").encode() + content + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        res = json.loads(self._request("POST", "/upload/image", b"".join(parts),
                                       {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
        sub = res.get("subfolder") or ""
        return f"{sub}/{res['name']}" if sub else res["name"]

    def queue(self, graph: Dict) -> str:
        res = self._json("POST", "/prompt", {"prompt": graph, "client_id": self.client_id})
        if res.get("node_errors"):
            raise ComfyError(f"Workflow rejected: {json.dumps(res['node_errors'])[:2000]}")
        if "prompt_id" not in res:
            raise ComfyError(f"Unexpected /prompt response: {res}")
        return res["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = 600.0, poll: float = 1.0,
             on_tick=None, should_stop=None) -> Dict:
        t0 = time.time()
        while True:
            if should_stop is not None and should_stop():
                self.interrupt()
                raise ComfyCancelled("cancelled while waiting for ComfyUI")
            hist = self._json("GET", f"/history/{prompt_id}")
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    raise ComfyError(f"ComfyUI execution error: {json.dumps(msgs)[:2000]}")
                if status.get("completed", True) and entry.get("outputs") is not None:
                    return entry
            if time.time() - t0 > timeout:
                raise ComfyError(f"Timed out after {timeout:.0f}s waiting for prompt {prompt_id}")
            if on_tick:
                on_tick(time.time() - t0)
            time.sleep(poll)

    def download(self, image_info: Dict, dest: str) -> str:
        q = urllib.parse.urlencode({"filename": image_info["filename"],
                                    "subfolder": image_info.get("subfolder", ""),
                                    "type": image_info.get("type", "output")})
        data = self._request("GET", f"/view?{q}")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return dest

    @staticmethod
    def output_images(entry: Dict, node_id: Optional[str] = None) -> List[Dict]:
        outs = entry.get("outputs", {})
        ids = [node_id] if node_id and node_id in outs else list(outs)
        imgs: List[Dict] = []
        for nid in ids:
            imgs.extend(i for i in outs[nid].get("images", []) if i.get("type", "output") == "output")
        if not imgs:  # e.g. PreviewImage only
            for nid in ids:
                imgs.extend(outs[nid].get("images", []))
        return imgs
