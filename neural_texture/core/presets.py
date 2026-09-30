"""Named presets (V4). Each preset is a flat dict of JobConfig paths -> values.

Paths: "model.denoise", "synth.max_views", "pbr.preserve_existing_color",
"intent.material_class", "view_resolution", ... Used by the UI, CLI and MCP.
"""

from __future__ import annotations

from typing import Dict

from .config import JobConfig

PRESETS: Dict[str, Dict] = {
    "enhance": {
        "label": "Enhance (keep look)",
        "description": "Keep the existing colours, add believable detail and fix blur.",
        "values": {"model.denoise": 0.45, "model.consistency_denoise": 0.7, "model.ipadapter_weight": 0.5,
                   "pbr.preserve_existing_color": 0.6, "pbr.roughness_mode": "blend", "detail_strength": 0.6},
    },
    "restyle": {
        "label": "Restyle (new look)",
        "description": "New appearance from prompt/reference; structure and base normal kept.",
        "values": {"model.denoise": 1.0, "model.consistency_denoise": 0.95, "pbr.preserve_existing_color": 0.0,
                   "pbr.roughness_mode": "material", "detail_strength": 0.7},
    },
    "skin_detail": {
        "label": "Skin detail",
        "description": "Pores and fine wrinkles on an existing head; colours mostly kept.",
        "values": {"intent.material_class": "skin", "model.denoise": 0.35, "model.consistency_denoise": 0.6,
                   "pbr.preserve_existing_color": 0.85, "detail_strength": 0.45, "intent.generate_meso": True,
                   "intent.meso_strength": 0.2},
    },
    "ancient_stone": {
        "label": "Ancient stone",
        "description": "Weathered stone statue look, macro normal preserved.",
        "values": {"intent.material_class": "stone", "model.denoise": 1.0, "pbr.preserve_existing_color": 0.0,
                   "detail_strength": 0.85, "intent.generate_meso": True, "intent.meso_strength": 0.4,
                   "prompt_suffix": "ancient weathered stone, moss in crevices, chipped edges"},
    },
    "worn_metal": {
        "label": "Worn metal",
        "description": "Bare metal with wear; metallic map generated per material.",
        "values": {"intent.material_class": "metal", "model.denoise": 0.95, "detail_strength": 0.6,
                   "prompt_suffix": "worn bare metal, edge wear, fine scratches"},
    },
    "rusty_metal": {
        "label": "Rusty metal",
        "description": "Corroded metal; bare metal where the rust is flaked off.",
        "values": {"intent.material_class": "rusted_metal", "model.denoise": 1.0, "detail_strength": 0.8,
                   "prompt_suffix": "heavily rusted iron, flaking paint and oxide"},
    },
    "fabric": {
        "label": "Fabric",
        "description": "Cloth weave and fibres, high roughness.",
        "values": {"intent.material_class": "fabric", "model.denoise": 0.9, "detail_strength": 0.5},
    },
    "wood": {
        "label": "Wood",
        "description": "Natural wood grain.",
        "values": {"intent.material_class": "wood", "model.denoise": 0.95, "detail_strength": 0.6},
    },
    "fast_preview": {
        "label": "Fast preview",
        "description": "Low resolution, few views: check the look quickly.",
        "values": {"view_resolution": 768, "texture_resolution": 1024, "synth.max_views": 6,
                   "model.steps": 20, "ao_samples": 8},
    },
    "production": {
        "label": "Production",
        "description": "High resolution, more views, more steps.",
        "values": {"view_resolution": 1024, "texture_resolution": 4096, "synth.max_views": 12,
                   "model.steps": 32, "ao_samples": 64},
    },
}


def apply_preset(cfg: JobConfig, name: str) -> JobConfig:
    """Apply preset values to a JobConfig. Intent keys go into cfg.intent_overrides."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}'. Known: {sorted(PRESETS)}")
    for path, val in PRESETS[name]["values"].items():
        if path == "prompt_suffix":
            if val not in (cfg.prompt or ""):
                cfg.prompt = f"{cfg.prompt}. {val}".strip(". ") if cfg.prompt else val
        elif path == "detail_strength":
            cfg.intent_overrides["detail_strength"] = val
        elif path.startswith("intent."):
            cfg.intent_overrides[path.split(".", 1)[1]] = val
        elif path.startswith("model."):
            setattr(cfg.model, path.split(".", 1)[1], val)
        elif path.startswith(("synth.", "pbr.", "projection.")):
            group, key = path.split(".", 1)
            getattr(cfg, group)[key] = val
        else:
            setattr(cfg, path, val)
    cfg.preset = name
    return cfg
