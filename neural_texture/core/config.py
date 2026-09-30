"""Job configuration shared by the UI operators and the headless CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from .camera import CANDIDATE_VIEWS, DEFAULT_VIEWS


@dataclass
class ModelConfig:
    """ComfyUI model settings. Names must match files in the ComfyUI models dirs."""
    server_url: str = "http://127.0.0.1:8188"
    checkpoint: str = "sd_xl_base_1.0.safetensors"
    controlnet: str = "controlnet-union-sdxl-1.0.safetensors"  # union: depth + normal
    controlnet_is_union: bool = True
    controlnet_normal: str = ""          # separate normal ControlNet when not using a union model
    depth_strength: float = 0.75
    normal_strength: float = 0.55
    control_end: float = 0.85
    use_ipadapter: bool = True           # needs ComfyUI_IPAdapter_plus custom nodes
    ipadapter_preset: str = "PLUS (high strength)"
    ipadapter_face_preset: str = "PLUS FACE (portraits)"
    ipadapter_weight: float = 0.7
    steps: int = 28
    cfg: float = 5.5
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    denoise: float = 0.75                # <1 = img2img from existing albedo render
    seed: int = 12345
    timeout_s: float = 600.0
    custom_workflow: str = ""            # optional API-format JSON with {{placeholders}}
    # V2 consistency (inpainting of already-textured regions)
    consistency_denoise: float = 0.9     # denoise inside the "unknown" mask for views after the first
    # V2 detail normals from a neural estimator (comfyui_controlnet_aux)
    normal_estimator: str = "auto"       # auto | off | <ComfyUI node class name>
    normal_estimator_resolution: int = 1024
    # V6 robustness
    auto_detect_models: bool = True
    max_retries: int = 2
    min_resolution: int = 640


@dataclass
class JobConfig:
    object_name: str = ""
    output_dir: str = "//neural_texture_out"
    prompt: str = ""
    intent: Optional[dict] = None        # structured intent from the agent (preferred)
    reference_images: List[str] = field(default_factory=list)
    views: List[str] = field(default_factory=lambda: list(DEFAULT_VIEWS))
    # V2: extra views rendered as candidates; the planner picks from views + candidates.
    candidate_views: List[str] = field(default_factory=lambda: [v for v in CANDIDATE_VIEWS if v not in DEFAULT_VIEWS])
    use_planner: bool = True
    synth: dict = field(default_factory=dict)       # SynthParams overrides
    preset: str = ""
    intent_overrides: dict = field(default_factory=dict)  # applied on top of parsed/agent intent
    view_resolution: int = 1024
    texture_resolution: int = 2048
    orthographic: bool = True
    uv_map: str = ""                     # empty = active render UV map
    ao_samples: int = 32
    use_gpu: bool = False
    keep_temp_scene: bool = False        # debugging only
    projection: dict = field(default_factory=dict)  # ProjectionParams overrides
    pbr: dict = field(default_factory=dict)         # PBRParams overrides
    model: ModelConfig = field(default_factory=ModelConfig)
    create_material: bool = True
    assign_material: bool = False        # never replace the user's material unless asked

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JobConfig":
        d = dict(d)
        model = ModelConfig(**{k: v for k, v in d.pop("model", {}).items()
                               if k in ModelConfig.__dataclass_fields__})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.model = model
        return cfg

    @classmethod
    def load(cls, path: str) -> "JobConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
