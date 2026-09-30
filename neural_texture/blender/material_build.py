"""Create the output PBR material from baked texture files.

Never edits the user's existing material. A new material is created; it is
assigned only when explicitly requested, and the original assignment is
recorded on the object so it can be restored.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import bpy

from . import compat

ORIGINAL_KEY = "nt_original_materials"


def _load(path: str, non_color: bool) -> bpy.types.Image:
    img = bpy.data.images.load(path, check_existing=True)
    img.reload()
    (compat.set_non_color if non_color else compat.set_srgb)(img)
    return img


def build_material(name: str, files: Dict[str, str], uv_map: str = "",
                   normal_strength: float = 1.0) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    compat.ensure_nodes(mat)
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_map
    uv.location = (-900, 0)

    def tex(key, y, non_color):
        if key not in files or not os.path.exists(files[key]):
            return None
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = _load(files[key], non_color)
        n.label = key
        n.location = (-550, y)
        nt.links.new(uv.outputs["UV"], n.inputs["Vector"])
        return n

    bc = tex("basecolor", 300, False)
    if bc:
        nt.links.new(bc.outputs["Color"], bsdf.inputs["Base Color"])
    ro = tex("roughness", 0, True)
    if ro:
        nt.links.new(ro.outputs["Color"], bsdf.inputs["Roughness"])
    me = tex("metallic", -300, True)
    if me:
        nt.links.new(me.outputs["Color"], bsdf.inputs["Metallic"])
    no = tex("normal", -600, True)
    if no:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.space = "TANGENT"
        nm.uv_map = uv_map
        nm.inputs["Strength"].default_value = normal_strength
        nm.location = (-150, -600)
        nt.links.new(no.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    # AO / detail maps are loaded but left unconnected (physically, AO is not albedo).
    y = -900
    for key in ("ao", "detail_normal", "detail_height", "confidence", "orm"):
        n = tex(key, y, True)
        y -= 300
    mat["nt_generated"] = True
    mat.use_fake_user = True
    return mat


def assign_material(obj: bpy.types.Object, mat: bpy.types.Material, slot: Optional[int] = None):
    """Assign ``mat`` (all slots, or one) and remember the originals for restore."""
    if ORIGINAL_KEY not in obj:
        obj[ORIGINAL_KEY] = json.dumps([s.material.name if s.material else "" for s in obj.material_slots])
    if not obj.material_slots:
        obj.data.materials.append(mat)
        return
    for i, s in enumerate(obj.material_slots):
        if slot is None or i == slot:
            s.material = mat


def restore_materials(obj: bpy.types.Object) -> bool:
    if ORIGINAL_KEY not in obj:
        return False
    names = json.loads(obj[ORIGINAL_KEY])
    for s, n in zip(obj.material_slots, names):
        s.material = bpy.data.materials.get(n) if n else None
    del obj[ORIGINAL_KEY]
    return True
