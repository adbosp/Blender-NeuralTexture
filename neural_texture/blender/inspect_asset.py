"""Mesh / UV / material inspection (spec sections 6 and 7). Read-only.

Produces a JSON-serialisable report the orchestrating agent can reason about,
and locates existing texture maps so they are enhanced, never discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import bpy
import numpy as np

# ----------------------------------------------------------------------------
# Mesh analysis
# ----------------------------------------------------------------------------


def evaluated_mesh_arrays(obj: bpy.types.Object, depsgraph, uv_map: str = ""):
    """World-space triangles + UVs of the evaluated mesh (modifiers applied)."""
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        nv, nt = len(me.vertices), len(me.loop_triangles)
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mw = np.array(obj.matrix_world, dtype=np.float64)
        co_w = (co @ mw[:3, :3].T + mw[:3, 3]).astype(np.float32)
        tri_v = np.empty(nt * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tri_v)
        tri_l = np.empty(nt * 3, np.int32)
        me.loop_triangles.foreach_get("loops", tri_l)
        mat_idx = np.empty(nt, np.int32)
        me.loop_triangles.foreach_get("material_index", mat_idx)
        uv = None
        layer = me.uv_layers.get(uv_map) if uv_map else _render_uv(me)
        if layer is not None:
            uv = np.empty(len(me.loops) * 2, np.float32)
            layer.data.foreach_get("uv", uv)
            uv = uv.reshape(-1, 2)
        return {
            "co": co_w, "tri_v": tri_v.reshape(-1, 3), "tri_l": tri_l.reshape(-1, 3),
            "mat_idx": mat_idx, "uv": uv, "uv_name": layer.name if layer else None,
            "n_edges": len(me.edges), "n_faces": len(me.polygons),
        }
    finally:
        ev.to_mesh_clear()


def _render_uv(me):
    for layer in me.uv_layers:
        if layer.active_render:
            return layer
    return me.uv_layers.active


def analyze_mesh(obj: bpy.types.Object, depsgraph, uv_map: str = "") -> Dict:
    if obj.type != "MESH":
        raise TypeError(f"'{obj.name}' is a {obj.type}, expected MESH")
    arr = evaluated_mesh_arrays(obj, depsgraph, uv_map)
    co = arr["co"]
    mn, mx = co.min(axis=0), co.max(axis=0)
    me = obj.data
    armatures = [m.object.name for m in obj.modifiers if m.type == "ARMATURE" and m.object]
    slot_faces = np.bincount(arr["mat_idx"], minlength=len(obj.material_slots))
    return {
        "object": obj.name,
        "mesh": me.name,
        "vertices": int(len(co)),
        "faces": arr["n_faces"],
        "triangles": int(len(arr["tri_v"])),
        "bbox_min": mn.tolist(),
        "bbox_max": mx.tolist(),
        "bbox_diagonal": float(np.linalg.norm(mx - mn)),
        "modifiers": [{"name": m.name, "type": m.type} for m in obj.modifiers],
        "armatures": armatures,
        "parent": obj.parent.name if obj.parent else None,
        "shape_keys": len(me.shape_keys.key_blocks) if me.shape_keys else 0,
        "vertex_groups": len(obj.vertex_groups),
        "material_slots": [
            {"index": i, "material": s.material.name if s.material else None,
             "link": s.link, "triangles": int(slot_faces[i]) if i < len(slot_faces) else 0}
            for i, s in enumerate(obj.material_slots)
        ],
        "uv_layers": [{"name": l.name, "active": l.active, "active_render": l.active_render}
                      for l in me.uv_layers],
        "color_attributes": [a.name for a in getattr(me, "color_attributes", [])],
        "uv": analyze_uv(arr),
    }


def analyze_uv(arr: Dict) -> Dict:
    uv = arr["uv"]
    if uv is None:
        return {"present": False, "warnings": ["Mesh has no UV map; texture generation needs UVs."]}
    tri_uv = uv[arr["tri_l"]]                     # (T, 3, 2)
    tri_co = arr["co"][arr["tri_v"]]              # (T, 3, 3)
    e1, e2 = tri_uv[:, 1] - tri_uv[:, 0], tri_uv[:, 2] - tri_uv[:, 0]
    signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    uv_area = np.abs(signed)
    area3d = 0.5 * np.linalg.norm(np.cross(tri_co[:, 1] - tri_co[:, 0], tri_co[:, 2] - tri_co[:, 0]), axis=1)
    tiles = np.floor(tri_uv.mean(axis=1)).astype(int)
    uniq = np.unique(tiles, axis=0) if len(tiles) else np.zeros((0, 2), int)
    tile_ids = sorted(1001 + int(u) + 10 * int(v) for u, v in uniq if 0 <= u < 10 and v >= 0)
    total = float(uv_area.sum())
    valid = (uv_area > 1e-12) & (area3d > 1e-12)
    density = np.sqrt(uv_area[valid] / area3d[valid]) if valid.any() else np.array([0.0])
    warnings: List[str] = []
    per_tile = total / max(1, len(tile_ids))
    if per_tile > 1.02:
        warnings.append(f"UV area per tile is {per_tile:.2f} (>1): overlapping or stacked UVs. "
                        "Overlapping texels receive blended appearance from all overlapping surfaces.")
    flipped = float((signed < 0).mean()) if len(signed) else 0.0
    if 0.02 < flipped < 0.98:
        warnings.append(f"{flipped:.0%} of triangles have mirrored UVs (common for symmetric models); "
                        "if mirrored halves share texels, the result will be symmetric.")
    if len(tile_ids) > 1:
        warnings.append(f"UDIM layout detected ({len(tile_ids)} tiles). V1 generates tile 1001 only.")
    degenerate = float((uv_area <= 1e-12).mean())
    if degenerate > 0.01:
        warnings.append(f"{degenerate:.1%} of triangles have zero UV area.")
    cv = float(density.std() / max(density.mean(), 1e-9))
    return {
        "present": True,
        "name": arr["uv_name"],
        "total_uv_area": total,
        "udim_tiles": tile_ids,
        "mirrored_fraction": flipped,
        "degenerate_fraction": degenerate,
        "texel_density_cv": cv,
        "warnings": warnings,
    }


# ----------------------------------------------------------------------------
# Existing material maps
# ----------------------------------------------------------------------------

@dataclass
class MapRef:
    image: Optional[str] = None       # bpy image name
    filepath: Optional[str] = None
    channel: Optional[int] = None     # packed-channel index (ORM etc.)
    uv_map: Optional[str] = None
    modified: bool = False            # passes through nodes other than simple routing
    value: Optional[object] = None    # default value when unlinked
    via: List[str] = field(default_factory=list)


@dataclass
class MaterialMaps:
    material: Optional[str]
    slot: int
    basecolor: MapRef = field(default_factory=MapRef)
    roughness: MapRef = field(default_factory=MapRef)
    metallic: MapRef = field(default_factory=MapRef)
    normal: MapRef = field(default_factory=MapRef)
    normal_strength: float = 1.0
    bump_height: MapRef = field(default_factory=MapRef)
    displacement: MapRef = field(default_factory=MapRef)
    ao: MapRef = field(default_factory=MapRef)
    alpha: MapRef = field(default_factory=MapRef)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


_PASSTHROUGH = {"REROUTE", "SEPARATE_COLOR", "SEPRGB", "SEPARATE_RGB", "NORMAL_MAP", "BUMP",
                "DISPLACEMENT", "MAPPING", "UVMAP", "TEX_COORD"}
_CHANNEL_NAMES = {"Red": 0, "R": 0, "Green": 1, "G": 1, "Blue": 2, "B": 2}


def _trace_image(socket, depth: int = 0, ref: Optional[MapRef] = None) -> MapRef:
    ref = ref or MapRef()
    if not socket.is_linked:
        if depth == 0 and hasattr(socket, "default_value"):
            dv = socket.default_value
            try:
                ref.value = list(dv)
            except TypeError:
                ref.value = float(dv)
        return ref
    link = socket.links[0]
    node = link.from_node
    ref.via.append(node.bl_idname)
    if node.type in ("SEPARATE_COLOR", "SEPRGB", "SEPARATE_RGB"):
        ref.channel = _CHANNEL_NAMES.get(link.from_socket.name)
    if node.type == "TEX_IMAGE":
        if node.image is not None:
            ref.image = node.image.name
            ref.filepath = bpy.path.abspath(node.image.filepath) if node.image.filepath else None
        vec = node.inputs["Vector"]
        if vec.is_linked and vec.links[0].from_node.type == "UVMAP":
            ref.uv_map = vec.links[0].from_node.uv_map or None
        if link.from_socket.name == "Alpha":
            ref.channel = 3
        return ref
    if node.type not in _PASSTHROUGH:
        ref.modified = True
    if depth > 8:
        return ref
    # Prefer the most "colour-like" input first.
    preferred = [s for s in node.inputs if s.is_linked and s.name in ("Color", "Height", "Image", "A", "Color1")]
    others = [s for s in node.inputs if s.is_linked and s not in preferred]
    for s in preferred + others:
        r = _trace_image(s, depth + 1, ref)
        if r.image:
            return r
    return ref


def _find_principled(mat: bpy.types.Material):
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None, None
    nt = mat.node_tree
    out = nt.get_output_node("CYCLES") or nt.get_output_node("ALL")
    if out is None:
        return None, None
    stack = [out.inputs["Surface"]]
    seen = set()
    while stack:
        s = stack.pop()
        for l in s.links:
            n = l.from_node
            if n.name in seen:
                continue
            seen.add(n.name)
            if n.type == "BSDF_PRINCIPLED":
                return n, out
            stack.extend(i for i in n.inputs if i.is_linked)
    return None, out


def extract_material_maps(obj: bpy.types.Object) -> List[MaterialMaps]:
    result = []
    for i, slot in enumerate(obj.material_slots):
        mat = slot.material
        mm = MaterialMaps(material=mat.name if mat else None, slot=i)
        bsdf, out = _find_principled(mat)
        if bsdf is None:
            mm.warnings.append("No Principled BSDF found; existing maps cannot be reused.")
            result.append(mm)
            continue
        mm.basecolor = _trace_image(bsdf.inputs["Base Color"])
        mm.roughness = _trace_image(bsdf.inputs["Roughness"])
        mm.metallic = _trace_image(bsdf.inputs["Metallic"])
        mm.alpha = _trace_image(bsdf.inputs["Alpha"])
        nsock = bsdf.inputs["Normal"]
        if nsock.is_linked:
            n = nsock.links[0].from_node
            if n.type == "NORMAL_MAP":
                if n.space != "TANGENT":
                    mm.warnings.append(f"Normal map space is {n.space}; V1 combines tangent-space only.")
                mm.normal = _trace_image(n.inputs["Color"])
                mm.normal.uv_map = mm.normal.uv_map or (n.uv_map or None)
                mm.normal_strength = float(n.inputs["Strength"].default_value)
            elif n.type == "BUMP":
                mm.bump_height = _trace_image(n.inputs["Height"])
                nn = n.inputs["Normal"]
                if nn.is_linked and nn.links[0].from_node.type == "NORMAL_MAP":
                    mm.normal = _trace_image(nn.links[0].from_node.inputs["Color"])
            else:
                mm.normal = _trace_image(nsock)
        if out is not None and out.inputs["Displacement"].is_linked:
            mm.displacement = _trace_image(out.inputs["Displacement"])
        # glTF-style occlusion: a group node with an "Occlusion" input.
        for n in mat.node_tree.nodes:
            if n.type == "GROUP" and "Occlusion" in n.inputs and n.inputs["Occlusion"].is_linked:
                mm.ao = _trace_image(n.inputs["Occlusion"])
        if mm.basecolor.modified:
            mm.warnings.append("Base Color passes through colour-modifying nodes; the raw image "
                               "is used as guidance, the node effect is not reproduced.")
        result.append(mm)
    return result


def primary_maps(maps: List[MaterialMaps], slot_tris: List[int]) -> Optional[MaterialMaps]:
    """V1 produces one texture set: pick the slot covering most triangles that has maps."""
    order = sorted(range(len(maps)), key=lambda i: -(slot_tris[i] if i < len(slot_tris) else 0))
    for i in order:
        if maps[i].basecolor.image or maps[i].normal.image:
            return maps[i]
    return maps[order[0]] if maps else None


def export_existing_maps(mm: Optional[MaterialMaps], res: int, normal_flip_green: bool = False):
    """Existing maps of the primary slot resampled to ``res`` -> (arrays, meta).

    Colour maps are returned display-encoded (sRGB) like 8-bit texture files;
    data maps use the stored channel (packed ORM supported).
    """
    from ..core.imageops import resize_bilinear
    from .imageio import image_to_array, image_to_display_rgb

    out: Dict[str, np.ndarray] = {}
    meta: Dict = {"normal_flip_green": normal_flip_green, "default_roughness": 0.5,
                  "default_metallic": 0.0, "normal_strength": 1.0, "material": None}
    if mm is None:
        return out, meta
    meta["material"] = mm.material
    meta["normal_strength"] = mm.normal_strength

    def img(ref: MapRef):
        return bpy.data.images.get(ref.image) if ref.image else None

    def scalar(ref: MapRef, arr):
        ch = ref.channel
        return arr[..., ch if ch is not None and ch < arr.shape[2] else 0]

    try:
        if img(mm.basecolor):
            out["basecolor"] = resize_bilinear(image_to_display_rgb(img(mm.basecolor)), res, res)
        if img(mm.normal):
            out["normal"] = resize_bilinear(image_to_array(img(mm.normal))[..., :3], res, res)
        for key in ("roughness", "metallic", "ao"):
            ref = getattr(mm, key)
            if img(ref):
                out[key] = resize_bilinear(scalar(ref, image_to_array(img(ref))), res, res)
            elif isinstance(ref.value, (int, float)) and key != "ao":
                meta[f"default_{key}"] = float(ref.value)
    except RuntimeError as e:  # missing image files: keep what we could read
        meta["warning"] = f"could not read an existing map: {e}"
    return out, meta
