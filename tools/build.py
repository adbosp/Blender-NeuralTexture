#!/usr/bin/env python3
"""Package the add-on (V6).

    python tools/build.py                 # dist/neural_texture-<ver>.zip (+ manifest checks)
    python tools/build.py --blender /path/to/blender --install
        # validates with `blender --command extension validate` and installs it

Produces:
  dist/neural_texture-<ver>.zip            Blender 4.2+ extension / legacy add-on zip
  dist/neural_texture_agent-<ver>.zip      MCP server + SKILL.md + CLAUDE.md for agents
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "neural_texture")
EXCLUDE = re.compile(r"(__pycache__|\.pyc$|\.DS_Store|/tests?/)")


def check() -> str:
    with open(os.path.join(PKG, "blender_manifest.toml"), "rb") as f:
        man = tomllib.load(f)
    init = open(os.path.join(PKG, "__init__.py"), encoding="utf-8").read()
    ver = tuple(int(x) for x in re.search(r'"version": \((\d+), (\d+), (\d+)\)', init).groups())
    mver = tuple(int(x) for x in man["version"].split("."))
    if ver != mver:
        sys.exit(f"version mismatch: bl_info {ver} vs manifest {mver}")
    for key in ("id", "name", "blender_version_min", "license", "type"):
        if key not in man:
            sys.exit(f"manifest missing '{key}'")
    for py in _files(PKG):
        if py.endswith(".py"):
            src = open(py, encoding="utf-8").read()
            compile(src, py, "exec")
            if "/core/" in py.replace(os.sep, "/") and re.search(r"^\s*import bpy|^\s*from bpy", src, re.M):
                sys.exit(f"core must stay bpy-free: {py}")
    return man["version"]


def _files(base):
    for d, _, fs in os.walk(base):
        for f in fs:
            p = os.path.join(d, f)
            if not EXCLUDE.search(p.replace(os.sep, "/")):
                yield p


def build(ver: str) -> list:
    dist = os.path.join(ROOT, "dist")
    os.makedirs(dist, exist_ok=True)
    addon = os.path.join(dist, f"neural_texture-{ver}.zip")
    with zipfile.ZipFile(addon, "w", zipfile.ZIP_DEFLATED) as z:
        for p in _files(PKG):
            z.write(p, os.path.relpath(p, ROOT))
    agent = os.path.join(dist, f"neural_texture_agent-{ver}.zip")
    with zipfile.ZipFile(agent, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("mcp_server", "agent"):
            for p in _files(os.path.join(ROOT, sub)):
                z.write(p, os.path.relpath(p, ROOT))
        for f in ("CLAUDE.md", "AGENTS.md", ".mcp.json", "README.md"):
            if os.path.exists(os.path.join(ROOT, f)):
                z.write(os.path.join(ROOT, f), f)
        for p in _files(PKG):  # MCP server imports the pure-Python parts
            z.write(p, os.path.relpath(p, ROOT))
    return [addon, agent]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=None)
    ap.add_argument("--install", action="store_true")
    a = ap.parse_args()
    ver = check()
    outs = build(ver)
    for o in outs:
        print("built", o, f"({os.path.getsize(o) // 1024} KB)")
    if a.blender:
        subprocess.run([a.blender, "--command", "extension", "validate", outs[0]], check=True)
        if a.install:
            subprocess.run([a.blender, "--command", "extension", "install-file", "-r", "user_default",
                            "--enable", outs[0]], check=True)
            print("installed into Blender user_default repository")


if __name__ == "__main__":
    main()
