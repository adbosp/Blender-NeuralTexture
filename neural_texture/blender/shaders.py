"""Emission-only pass materials for guidance renders and UV bakes (Cycles).

Every pass writes its data through an Emission shader so the rendered/baked
value is exactly the encoded quantity, independent of scene lighting
(spec section 9: lighting never leaks into data passes).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import bpy

from . import compat

# Encodings shared with the decoders in guidance.py / uv_bake.py
POS_LO, POS_HI = 0.05, 0.95   # world position mapped into this range within the bbox


def _new_material(name: str, scope) -> tuple:
    mat = bpy.data.materials.new(name)
    scope.own(mat)
    compat.ensure_nodes(mat)
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.target = "ALL"
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat, nt, emit


def _vec_math(nt, op, a, b=None, c=None):
    n = nt.nodes.new("ShaderNodeVectorMath")
    n.operation = op
    for i, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, bpy.types.NodeSocket):
            nt.links.new(v, n.inputs[i])
        else:
            n.inputs[i].default_value = v
    return n.outputs[0]


def _uv_socket(nt, uv_map: str):
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_map or ""
    return uv.outputs["UV"]


def uv_material(scope, uv_map: str = "") -> bpy.types.Material:
    """Encode the UV coordinate of every rendered pixel (for view<->texture reprojection)."""
    mat, nt, emit = _new_material("NT_pass_uv", scope)
    nt.links.new(_uv_socket(nt, uv_map), emit.inputs["Color"])
    return mat


def tangent_frame_material(scope, axis: str, uv_map: str = "") -> bpy.types.Material:
    """World-space tangent ('T') or bitangent ('B') exactly as Blender's Normal Map
    node defines them: a constant tangent-space normal of +X / +Y decoded by
    the node yields T / B (including the bitangent sign)."""
    mat, nt, emit = _new_material(f"NT_pass_{axis}", scope)
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.space = "TANGENT"
    nmap.uv_map = uv_map or ""
    nmap.inputs["Strength"].default_value = 1.0
    nmap.inputs["Color"].default_value = (1.0, 0.5, 0.5, 1.0) if axis == "T" else (0.5, 1.0, 0.5, 1.0)
    enc = _vec_math(nt, "MULTIPLY_ADD", nmap.outputs["Normal"], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    nt.links.new(enc, emit.inputs["Color"])
    return mat


def world_normal_material(scope) -> bpy.types.Material:
    mat, nt, emit = _new_material("NT_pass_normal", scope)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    enc = _vec_math(nt, "MULTIPLY_ADD", geo.outputs["Normal"], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    nt.links.new(enc, emit.inputs["Color"])
    return mat


def world_position_material(scope, bbox_min: Sequence[float], bbox_size: Sequence[float]) -> bpy.types.Material:
    """Encode world position: POS_LO + (P - min) / size * (POS_HI - POS_LO)."""
    mat, nt, emit = _new_material("NT_pass_position", scope)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    rel = _vec_math(nt, "SUBTRACT", geo.outputs["Position"], tuple(bbox_min))
    span = POS_HI - POS_LO
    scale = tuple(span / max(s, 1e-9) for s in bbox_size)
    enc = _vec_math(nt, "MULTIPLY_ADD", rel, scale, (POS_LO, POS_LO, POS_LO))
    nt.links.new(enc, emit.inputs["Color"])
    return mat


def depth_material(scope) -> bpy.types.Material:
    mat, nt, emit = _new_material("NT_pass_depth", scope)
    cam = nt.nodes.new("ShaderNodeCameraData")
    nt.links.new(cam.outputs["View Z Depth"], emit.inputs["Color"])
    return mat


def ao_material(scope, distance: float, samples: int = 16) -> bpy.types.Material:
    mat, nt, emit = _new_material("NT_pass_ao", scope)
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = max(1, int(samples))
    ao.only_local = True
    ao.inputs["Distance"].default_value = distance
    nt.links.new(ao.outputs["AO"], emit.inputs["Color"])
    return mat


def curvature_material(scope, contrast: float = 6.0) -> bpy.types.Material:
    """Curvature approximation from Cycles Pointiness (0.5 = flat)."""
    mat, nt, emit = _new_material("NT_pass_curvature", scope)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    m = nt.nodes.new("ShaderNodeMath")
    m.operation = "MULTIPLY_ADD"
    m.use_clamp = True
    nt.links.new(geo.outputs["Pointiness"], m.inputs[0])
    m.inputs[1].default_value = contrast
    m.inputs[2].default_value = 0.5 - 0.5 * contrast
    nt.links.new(m.outputs[0], emit.inputs["Color"])
    return mat


def flat_color_material(scope, name: str, color: Sequence[float]) -> bpy.types.Material:
    mat, nt, emit = _new_material(name, scope)
    emit.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
    return mat


def albedo_material(scope, name: str, image: Optional[bpy.types.Image], color: Sequence[float],
                    uv_map: str = "") -> bpy.types.Material:
    """Unlit existing BaseColor (texture if available, else the constant)."""
    mat, nt, emit = _new_material(name, scope)
    if image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.interpolation = "Linear"
        nt.links.new(_uv_socket(nt, uv_map), tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    else:
        emit.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
    return mat


def add_bake_target(mat: bpy.types.Material, image: bpy.types.Image) -> None:
    """Add the active Image Texture node Cycles bakes into."""
    nt = mat.node_tree
    for n in nt.nodes:
        n.select = False
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.name = "NT_bake_target"
    tex.image = image
    tex.select = True
    nt.nodes.active = tex


def id_color(i: int) -> tuple:
    """Distinct, deterministic colour for material/object IDs."""
    golden = 0.61803398875
    h = (i * golden) % 1.0
    import colorsys
    return colorsys.hsv_to_rgb(h, 0.75, 0.95)
