"""Image-generator abstraction used by the synthesis loop.

Implementations:
  IdentityGenerator  - returns the init/albedo image (offline tests, "no AI" runs)
  FolderGenerator    - uses views/<name>/generated.png supplied by the user/agent
  ComfyGenerator     - neural_texture.bridge.comfy_generator (ComfyUI)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import pngio


def read_image(path: str) -> np.ndarray:
    """(H, W, 3|4) float32 bottom-up. Uses Pillow when present (fast), else pngio."""
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
            a = np.asarray(im, dtype=np.float32) / 255.0
        return np.ascontiguousarray(a[::-1])
    except ImportError:
        return pngio.read_png(path)


@dataclass
class ViewJob:
    name: str
    index: int
    width: int
    height: int
    files: Dict[str, str]                 # depth, normal, albedo, init, mask (absolute paths)
    prompts: Dict[str, str]
    first: bool = True
    denoise: float = 0.75
    seed: int = 0
    reference_images: List[str] = field(default_factory=list)
    reference_role: str = "appearance"
    extra: Dict = field(default_factory=dict)


class GeneratorCancelled(RuntimeError):
    pass


class BaseGenerator:
    name = "base"
    supports_normals = False
    supports_atlas_inpaint = False

    def begin(self, context: Dict) -> None:
        """Called once before the loop (upload references, check server...)."""

    def generate_view(self, job: ViewJob) -> np.ndarray:
        raise NotImplementedError

    def estimate_normals(self, job: ViewJob, image_path: str) -> Optional[np.ndarray]:
        """Camera-space unit normals (+X right, +Y up, +Z to camera) or None."""
        return None

    def inpaint_atlas(self, atlas_path: str, mask_path: str, prompts: Dict[str, str],
                      size: int) -> Optional[np.ndarray]:
        return None

    def interrupt(self) -> None:
        """Best-effort cancellation of in-flight work."""

    def end(self) -> None:
        pass


class IdentityGenerator(BaseGenerator):
    """Returns the init image (or existing albedo). Proves the geometry path
    end-to-end without any model and is the 'enhance only' baseline."""
    name = "identity"

    def generate_view(self, job: ViewJob) -> np.ndarray:
        src = job.files.get("init") or job.files["albedo"]
        return read_image(src)[..., :3]


class FolderGenerator(BaseGenerator):
    """Images produced elsewhere: <views>/<name>/generated.png must exist."""
    name = "folder"

    def __init__(self, root: str, filename: str = "generated.png"):
        self.root = root
        self.filename = filename

    def generate_view(self, job: ViewJob) -> np.ndarray:
        p = os.path.join(self.root, "views", job.name, self.filename)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return read_image(p)[..., :3]
