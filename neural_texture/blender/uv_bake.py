"""UV-space bakes of the proxy mesh (world position/normal/tangent frame,
material ID, AO, curvature). They tell, for every texel, where it lives on
the surface: the bridge between view-space images and the texture.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import bpy
import numpy as np

from . import shaders
from .imageio import image_to_array, new_float_image


def _select_only(scene, obj):
    vl = scene.view_layers[0]
    for o in scene.objects:
        o.select_set(False, view_layer=vl)
    obj.select_set(True, view_layer=vl)
    vl.objects.active = obj
    return vl


def _run_bake(scene, obj, uv_name: str, margin: int = 0):
    vl = _select_only(scene, obj)
    kwargs = dict(type="EMIT", margin=margin, use_clear=True, target="IMAGE_TEXTURES",
                  save_mode="INTERNAL", uv_layer=uv_name or "")
    try:
        with bpy.context.temp_override(scene=scene, view_layer=vl, active_object=obj, object=obj,
                                       selected_objects=[obj], selected_editable_objects=[obj]):
            bpy.ops.object.bake(**kwargs)
        return
    except RuntimeError as e:
        err = e
    # Fallback: temporarily switch the window's scene (restored immediately).
    win = bpy.context.window
    if win is None:
        raise err
    prev = win.scene
    try:
        win.scene = scene
        with bpy.context.temp_override(window=win, active_object=obj, object=obj,
                                       selected_objects=[obj], selected_editable_objects=[obj]):
            bpy.ops.object.bake(**kwargs)
    finally:
        win.scene = prev


def _bake_pass(scope, proxy, mats: List[bpy.types.Material], res: int, name: str,
               samples: int = 1) -> np.ndarray:
    img = new_float_image(f"NT_bake_{name}", res, res, scope)
    for m in {m.name: m for m in mats}.values():
        shaders.add_bake_target(m, img)
    proxy.set_materials(mats)
    scope.scene.cycles.samples = samples
    _run_bake(scope.scene, proxy.obj, proxy.uv_map, 0)
    return image_to_array(img)


def _unit(enc: np.ndarray) -> np.ndarray:
    n = enc[..., :3] * 2.0 - 1.0
    return (n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-8)).astype(np.float32)


def bake_uv_steps(scope, proxy, res: int, ao_samples: int = 32) -> Iterator[Tuple[float, str, Dict]]:
    from .guidance import decode_position

    rb = scope.scene.render.bake
    rb.margin = 0
    rb.use_clear = True
    steps = 7

    nrm = _bake_pass(scope, proxy, proxy.pass_materials("normal"), res, "normal")
    cov = nrm[..., :3].sum(axis=2) > 1e-4
    yield 1 / steps, "bake UV normal", {"coverage": cov, "normal": _unit(nrm)}

    pos = _bake_pass(scope, proxy, proxy.pass_materials("position"), res, "position")
    yield 2 / steps, "bake UV position", {"position": decode_position(pos, proxy.bbox_min, proxy.bbox_size)}

    t = _bake_pass(scope, proxy, proxy.pass_materials("tangent"), res, "tangent")
    b = _bake_pass(scope, proxy, proxy.pass_materials("bitangent"), res, "bitangent")
    yield 4 / steps, "bake tangent frame", {"tangent": _unit(t), "bitangent": _unit(b)}

    ids = _bake_pass(scope, proxy, proxy.pass_materials("mask"), res, "mask")
    palette = np.array([shaders.id_color(i) for i in range(proxy.n_slots)], np.float32)
    d = np.linalg.norm(ids[..., None, :3] - palette[None, None], axis=3)
    yield 5 / steps, "bake material IDs", {"material_id": np.where(cov, d.argmin(axis=2), -1).astype(np.int16)}

    ao = _bake_pass(scope, proxy, proxy.pass_materials("ao", ao_samples=ao_samples), res, "ao", ao_samples)
    yield 6 / steps, "bake AO", {"ao": np.clip(ao[..., 0], 0.0, 1.0)}

    cu = _bake_pass(scope, proxy, proxy.pass_materials("curvature"), res, "curvature")
    yield 1.0, "bake curvature", {"curvature": np.clip(cu[..., 0], 0.0, 1.0)}
