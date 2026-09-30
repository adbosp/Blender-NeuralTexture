"""LEGACY (V1) independent per-view generation. Superseded by core/synth.py +
bridge/comfy_generator.py (consistent sequential generation). Kept for
`is_local_url` and backwards compatibility with V1 guidance folders.

Generate one appearance image per guidance view through ComfyUI.

Pure Python: can run inside Blender or standalone, e.g. by the agent:
    python -m neural_texture.bridge.generate <output_dir> [job.json]
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
from typing import Callable, Dict, List, Optional

from ..core.config import JobConfig
from ..core.intent import Intent, parse_prompt
from .comfyui import ComfyClient, ComfyError
from .workflow import build_default_graph, build_prompts, fill_template, load_template


def is_local_url(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("192.168.") or host.startswith("10.")


from ..core.intent import resolve_intent  # noqa: E402,F401  (re-export)


def generate_views(out_dir: str, cfg: JobConfig, manifest: Dict, intent: Optional[Intent] = None,
                   views: Optional[List[str]] = None, log: Callable[[str], None] = print,
                   client: Optional[ComfyClient] = None) -> Dict:
    intent = intent or resolve_intent(cfg)
    if intent.requires_geometry_mode:
        raise ComfyError("Refusing to generate: the request needs geometry editing:\n" +
                         "\n".join(f"- {w.clause} ({w.kind})" for w in intent.geometry_warnings))
    m = cfg.model
    client = client or ComfyClient(m.server_url)
    client.ping()

    job = f"nt{int(time.time())}"
    refs = [client.upload_image(p, f"{job}_ref{i}{os.path.splitext(p)[1] or '.png'}")
            for i, p in enumerate(cfg.reference_images)]
    template = load_template(m.custom_workflow) if m.custom_workflow else None

    record = {"job": job, "intent": intent.to_dict(), "model": m.__dict__, "views": {}}
    names = views or list(manifest["views"])
    checked = False
    for name in names:
        v = manifest["views"][name]
        files = {k: os.path.join(out_dir, p) for k, p in v["files"].items()}
        up = {k: client.upload_image(files[k], f"{job}_{name}_{k}.png")
              for k in ("depth", "normal", "albedo") if k in files}
        prompts = build_prompts(intent, name)
        w = h = int(v["camera"]["width"])
        if template:
            graph = fill_template(template, {
                **prompts, "depth_image": up.get("depth", ""), "normal_image": up.get("normal", ""),
                "albedo_image": up.get("albedo", ""), "reference_image": refs[0] if refs else "",
                "seed": m.seed, "steps": m.steps, "cfg": m.cfg, "denoise": m.denoise,
                "width": w, "height": h, "view": name, "filename_prefix": f"{job}_{name}"})
        else:
            graph = build_default_graph(m, prompts, {**up, "references": refs}, w, h,
                                        intent.reference_role, filename_prefix=f"{job}_{name}")
        if not checked:
            try:
                missing = client.missing_nodes(graph)
            except ComfyError:
                missing = []  # older servers: let /prompt report problems
            if missing:
                raise ComfyError("ComfyUI is missing node types: " + ", ".join(missing) +
                                 ". Install the custom nodes (e.g. ComfyUI_IPAdapter_plus) or "
                                 "disable IPAdapter / use a custom workflow.")
            checked = True
        log(f"[NeuralTexture] generating view '{name}'")
        pid = client.queue(graph)
        entry = client.wait(pid, timeout=m.timeout_s)
        imgs = client.output_images(entry)
        if not imgs:
            raise ComfyError(f"No output image for view '{name}'")
        dest = os.path.join(out_dir, "views", name, "generated.png")
        client.download(imgs[0], dest)
        record["views"][name] = {"prompt_id": pid, "prompts": prompts, "output": os.path.relpath(dest, out_dir)}
        with open(os.path.join(out_dir, "views", name, "graph.json"), "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=1)
    with open(os.path.join(out_dir, "generation.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return record


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    out_dir = argv[0]
    cfg = JobConfig.load(argv[1]) if len(argv) > 1 else JobConfig.load(os.path.join(out_dir, "job.json"))
    with open(os.path.join(out_dir, "views", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    generate_views(out_dir, cfg, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
