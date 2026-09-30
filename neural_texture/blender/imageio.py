"""bpy.types.Image <-> numpy, plus file loading that prefers Blender's loaders."""

from __future__ import annotations

import os

import bpy
import numpy as np

from ..core import pngio


def image_to_array(img: bpy.types.Image) -> np.ndarray:
    """(H, W, 4) float32, bottom-up rows.

    Byte images return their stored (display-encoded) values; float images
    return scene-linear values. Callers decide on colour conversions.
    """
    n = len(img.pixels)  # forces the image buffer to load
    w, h = img.size
    if n == 0 or w == 0 or h == 0:
        raise RuntimeError(f"Image '{img.name}' has no pixel data (missing file?)")
    arr = np.empty(n, dtype=np.float32)
    img.pixels.foreach_get(arr)
    return arr.reshape(h, w, img.channels)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def image_to_display_rgb(img: bpy.types.Image) -> np.ndarray:
    """Colour texture as sRGB-encoded RGB (matching 8-bit texture files)."""
    arr = image_to_array(img)
    rgb = arr[..., :3] if arr.shape[2] >= 3 else np.repeat(arr[..., :1], 3, axis=2)
    if img.is_float and img.colorspace_settings.name.lower().startswith("srgb"):
        rgb = linear_to_srgb(rgb)
    return np.clip(rgb, 0.0, 1.0)


def load_file(path: str, as_data: bool = True) -> np.ndarray:
    """Load an image file to (H, W, 4) float32 bottom-up.

    Uses Blender's loader (fast, many formats); PNG falls back to pngio.
    """
    path = bpy.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        img = bpy.data.images.load(path, check_existing=False)
        try:
            if as_data:
                img.colorspace_settings.name = "Non-Color"
            arr = image_to_array(img)
        finally:
            bpy.data.images.remove(img)
    except RuntimeError:
        if not path.lower().endswith(".png"):
            raise
        arr = pngio.read_png(path)
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones(arr.shape[:2] + (1,), np.float32)], axis=2)
    elif arr.shape[2] == 1:
        arr = np.concatenate([np.repeat(arr, 3, axis=2), np.ones(arr.shape[:2] + (1,), np.float32)], axis=2)
    return arr


def new_float_image(name: str, w: int, h: int, scope=None) -> bpy.types.Image:
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    img.generated_color = (0.0, 0.0, 0.0, 0.0)
    if scope is not None:
        scope.own(img)
    return img
