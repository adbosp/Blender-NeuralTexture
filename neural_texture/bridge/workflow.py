"""ComfyUI API-format graph construction and prompt building.

Default graph (SDXL):
    Checkpoint -> [IPAdapter(reference)] -> KSampler
    positive/negative -> ControlNet(depth) -> ControlNet(normal) -> KSampler
    latent: VAEEncode(existing albedo render) when denoise < 1, else empty
Custom graphs: any API-format JSON using {{placeholders}} (see PLACEHOLDERS).
"""

from __future__ import annotations

import copy
import json
import re
from typing import Dict, List, Optional

from ..core.config import ModelConfig
from ..core.intent import Intent

PLACEHOLDERS = ("positive", "negative", "depth_image", "normal_image", "albedo_image",
                "init_image", "mask_image", "reference_image", "seed", "steps", "cfg", "denoise",
                "width", "height", "view", "filename_prefix")

from ..core.prompts import ALBEDO_STYLE, VIEW_PHRASES, build_prompts  # noqa: F401


class GraphBuilder:
    def __init__(self):
        self.g: Dict[str, Dict] = {}
        self._n = 0

    def add(self, class_type: str, **inputs) -> str:
        self._n += 1
        nid = str(self._n)
        self.g[nid] = {"class_type": class_type, "inputs": inputs}
        return nid


def _model_chain(b: GraphBuilder, m: ModelConfig, refs: List[str], reference_role: str):
    ckpt = b.add("CheckpointLoaderSimple", ckpt_name=m.checkpoint)
    model, clip, vae = [ckpt, 0], [ckpt, 1], [ckpt, 2]
    if m.use_ipadapter and refs:
        ref_nodes = [b.add("LoadImage", image=r) for r in refs]
        img = [ref_nodes[0], 0]
        for rn in ref_nodes[1:]:
            img = [b.add("ImageBatch", image1=img, image2=[rn, 0]), 0]
        preset = m.ipadapter_face_preset if reference_role == "identity" else m.ipadapter_preset
        loader = b.add("IPAdapterUnifiedLoader", model=model, preset=preset)
        ipa = b.add("IPAdapterAdvanced", model=[loader, 0], ipadapter=[loader, 1], image=img,
                     weight=m.ipadapter_weight, weight_type="linear", combine_embeds="concat",
                     start_at=0.0, end_at=1.0, embeds_scaling="V only")
        model = [ipa, 0]
    return model, clip, vae


def _latent(b: GraphBuilder, images: Dict[str, str], vae, width, height, denoise: float):
    """img2img from init/albedo (+ latent noise mask for consistency inpainting)."""
    src = images.get("init") or (images.get("albedo") if denoise < 0.999 else None)
    if src:
        pix = b.add("LoadImage", image=src)
        latent = [b.add("VAEEncode", pixels=[pix, 0], vae=vae), 0]
        if images.get("mask"):
            mimg = b.add("LoadImage", image=images["mask"])
            mask = b.add("ImageToMask", image=[mimg, 0], channel="red")
            latent = [b.add("SetLatentNoiseMask", samples=latent, mask=[mask, 0]), 0]
        return latent, denoise
    return [b.add("EmptyLatentImage", width=width, height=height, batch_size=1), 0], 1.0


def build_default_graph(m: ModelConfig, prompts: Dict[str, str], images: Dict[str, str],
                        width: int, height: int, reference_role: str = "appearance",
                        filename_prefix: str = "neural_texture", denoise: Optional[float] = None) -> Dict:
    """``images``: depth/normal/albedo/init/mask/references -> uploaded ComfyUI names."""
    b = GraphBuilder()
    model, clip, vae = _model_chain(b, m, images.get("references") or [], reference_role)
    pos = [b.add("CLIPTextEncode", text=prompts["positive"], clip=clip), 0]
    neg = [b.add("CLIPTextEncode", text=prompts["negative"], clip=clip), 0]

    loaders = {}
    for kind, strength in (("depth", m.depth_strength), ("normal", m.normal_strength)):
        if strength <= 0 or kind not in images:
            continue
        name = m.controlnet if (m.controlnet_is_union or kind == "depth") else (m.controlnet_normal or "")
        if not name:
            continue
        if name not in loaders:
            loaders[name] = b.add("ControlNetLoader", control_net_name=name)
        net = [loaders[name], 0]
        if m.controlnet_is_union:
            net = [b.add("SetUnionControlNetType", control_net=net, type=kind), 0]
        img = [b.add("LoadImage", image=images[kind]), 0]
        apply = b.add("ControlNetApplyAdvanced", positive=pos, negative=neg, control_net=net,
                      image=img, strength=strength, start_percent=0.0, end_percent=m.control_end)
        pos, neg = [apply, 0], [apply, 1]

    latent, dn = _latent(b, images, vae, width, height, m.denoise if denoise is None else denoise)
    ks = b.add("KSampler", model=model, seed=m.seed, steps=m.steps, cfg=m.cfg,
               sampler_name=m.sampler, scheduler=m.scheduler, positive=pos, negative=neg,
               latent_image=latent, denoise=dn)
    dec = b.add("VAEDecode", samples=[ks, 0], vae=vae)
    b.add("SaveImage", images=[dec, 0], filename_prefix=filename_prefix)
    return b.g


def build_atlas_inpaint_graph(m: ModelConfig, prompts: Dict[str, str], atlas: str, mask: str,
                              references: List[str], filename_prefix: str = "nt_atlas") -> Dict:
    """Texture-space inpainting of texels no view could see (no ControlNet: UV space)."""
    b = GraphBuilder()
    model, clip, vae = _model_chain(b, m, references, "appearance")
    pos = [b.add("CLIPTextEncode", text=prompts["positive"] + ", seamless texture", clip=clip), 0]
    neg = [b.add("CLIPTextEncode", text=prompts["negative"], clip=clip), 0]
    latent, dn = _latent(b, {"init": atlas, "mask": mask}, vae, 0, 0, 1.0)
    ks = b.add("KSampler", model=model, seed=m.seed + 7, steps=m.steps, cfg=m.cfg, sampler_name=m.sampler,
               scheduler=m.scheduler, positive=pos, negative=neg, latent_image=latent, denoise=dn)
    dec = b.add("VAEDecode", samples=[ks, 0], vae=vae)
    b.add("SaveImage", images=[dec, 0], filename_prefix=filename_prefix)
    return b.g


def build_normal_estimator_graph(node_class: str, image: str, resolution: int,
                                 filename_prefix: str = "nt_normal") -> Dict:
    b = GraphBuilder()
    img = b.add("LoadImage", image=image)
    est = b.add(node_class, image=[img, 0], resolution=resolution)
    b.add("SaveImage", images=[est, 0], filename_prefix=filename_prefix)
    return b.g


_FULL = re.compile(r"^\{\{(\w+)\}\}$")
_PART = re.compile(r"\{\{(\w+)\}\}")


def fill_template(template: Dict, values: Dict) -> Dict:
    """Substitute {{key}} placeholders in an API-format workflow.

    A string that is exactly "{{key}}" becomes the typed value (numbers stay
    numbers); placeholders inside longer strings are text-substituted.
    """
    def sub(v):
        if isinstance(v, str):
            m = _FULL.match(v)
            if m and m.group(1) in values:
                return values[m.group(1)]
            return _PART.sub(lambda mm: str(values.get(mm.group(1), mm.group(0))), v)
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sub(x) for x in v]
        return v
    out = sub(copy.deepcopy(template))
    left = set(_PART.findall(json.dumps(out)))
    if left:
        raise ValueError(f"Unfilled workflow placeholders: {sorted(left)}")
    return out


def load_template(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        wf = json.load(f)
    if "nodes" in wf and "links" in wf:
        raise ValueError("This is a UI-format workflow. In ComfyUI use 'Save (API Format)'.")
    return wf
