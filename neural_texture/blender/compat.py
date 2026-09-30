"""Blender version compatibility helpers (V6). Supported: 4.2 LTS .. 5.x.

Everything version-sensitive goes through here so the rest of the add-on
can stay version-agnostic.
"""

from __future__ import annotations

import bpy

MIN_VERSION = (4, 2, 0)
NON_COLOR_NAMES = ("Non-Color", "Non-Colour Data", "Raw", "Linear Rec.709", "Generic Data")
SRGB_NAMES = ("sRGB", "sRGB 2.1", "sRGB IEC61966-2.1")


def version_ok() -> bool:
    return tuple(bpy.app.version[:3]) >= MIN_VERSION


def version_str() -> str:
    return ".".join(map(str, bpy.app.version[:3]))


def _set_colorspace(img, names) -> bool:
    for n in names:
        try:
            img.colorspace_settings.name = n
            return True
        except TypeError:
            continue
    return False


def set_non_color(img) -> bool:
    return _set_colorspace(img, NON_COLOR_NAMES)


def set_srgb(img) -> bool:
    return _set_colorspace(img, SRGB_NAMES)


def ensure_nodes(mat) -> None:
    """Material.use_nodes is deprecated (always on) in Blender 5.0+."""
    if bpy.app.version < (5, 0, 0) or not getattr(mat, "node_tree", None):
        try:
            mat.use_nodes = True
        except AttributeError:
            pass


def has_progress_widget() -> bool:
    return hasattr(bpy.types.UILayout, "progress")


def temp_override(**kw):
    return bpy.context.temp_override(**kw)


def is_background() -> bool:
    return bool(bpy.app.background)


def online_access_allowed() -> bool:
    return bool(getattr(bpy.app, "online_access", True))


def cycles_available() -> bool:
    return "cycles" in bpy.context.preferences.addons or hasattr(bpy.types.Scene, "cycles")


def set_gpu(scene, enable: bool) -> str:
    """Enable GPU rendering when a compute device is configured; returns device used."""
    if not enable:
        scene.cycles.device = "CPU"
        return "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        if prefs.compute_device_type in ("", "NONE"):
            scene.cycles.device = "CPU"
            return "CPU"
        scene.cycles.device = "GPU"
        return f"GPU ({prefs.compute_device_type})"
    except (KeyError, AttributeError):
        scene.cycles.device = "CPU"
        return "CPU"


def brush_color_white(context) -> None:
    """Set the texture-paint colour to white across brush API changes (4.3 brush assets)."""
    ts = context.tool_settings
    for target in (getattr(ts, "unified_paint_settings", None),
                   getattr(getattr(ts, "image_paint", None), "brush", None)):
        if target is not None and hasattr(target, "color"):
            try:
                target.color = (1.0, 1.0, 1.0)
            except (AttributeError, TypeError):
                pass
