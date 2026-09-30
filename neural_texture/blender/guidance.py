"""Guidance package export (V2): multi-view passes + UV bakes + existing maps.

Renders a static proxy of the evaluated mesh from temporary cameras in an
isolated scene with emission-only pass materials, bakes UV-space data, and
writes a *package* (see core/package.py) that the pure-Python synthesis reads.

    per view:  position, uv (projection/reprojection), depth + camera-space
               normal (ControlNet), albedo (existing BaseColor, img2img init),
               material mask
    UV space:  position, normal, tangent, bitangent, material id, AO, curvature
    existing:  basecolor / normal / roughness / metallic / ao resampled to output res

Nothing here touches the user's scene; everything lives in a TempScope.
Implemented as a generator of small steps so the UI stays responsive and the
job can be cancelled between renders.
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Tuple

import bpy
import numpy as np
from mathutils import Matrix

from ..core.camera import CameraModel, make_views
from ..core.config import JobConfig
from ..core.imageops import normalize_range
from ..core.package import PackageWriter
from . import compat, shaders
from .imageio import linear_to_srgb, load_file
from .inspect_asset import MaterialMaps, export_existing_maps, primary_maps
from .scope import TempScope

VIEW_PASSES = ("position", "uv", "normal", "depth", "mask", "albedo")
VIEW_WEIGHT = {"top": 0.6, "bottom": 0.4, "bottom_front": 0.6, "bottom_back": 0.6,
               "bottom_left": 0.6, "bottom_right": 0.6}

Step = Tuple[float, str]


# ----------------------------------------------------------------------------
# Scene / proxy setup
# ----------------------------------------------------------------------------

def configure_cycles(scene: bpy.types.Scene, res_x: int, res_y: int, use_gpu: bool = False):
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.use_file_extension = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    cy = scene.cycles
    cy.samples = 1
    cy.use_denoising = False
    cy.use_adaptive_sampling = False
    cy.filter_width = 0.01          # one exact sample per pixel centre
    cy.max_bounces = 0
    compat.set_gpu(scene, use_gpu)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    s = scene.render.image_settings
    s.file_format = "OPEN_EXR"
    s.color_mode = "RGBA"
    s.color_depth = "32"
    s.exr_codec = "ZIP"


class Proxy:
    """Static copy of the evaluated mesh, linked only into the temp scene."""

    def __init__(self, scope: TempScope, obj: bpy.types.Object, depsgraph, uv_map: str = ""):
        self.scope = scope
        ev = obj.evaluated_get(depsgraph)
        self.mesh = scope.own(bpy.data.meshes.new_from_object(
            ev, preserve_all_data_layers=True, depsgraph=depsgraph))
        self.mesh.name = "NT_proxy_mesh"   # reserved prefix: crash cleanup can find it
        self.obj = bpy.data.objects.new("NT_proxy", self.mesh)
        self.obj.matrix_world = obj.matrix_world.copy()
        scope.link(self.obj)
        if len(self.mesh.materials) == 0:
            self.mesh.materials.append(None)
        self.n_slots = len(self.mesh.materials)
        self.uv_map = uv_map or next((l.name for l in self.mesh.uv_layers if l.active_render), "")
        co = np.empty(len(self.mesh.vertices) * 3, np.float32)
        self.mesh.vertices.foreach_get("co", co)
        mw = np.array(self.obj.matrix_world, dtype=np.float64)
        w = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        self.bbox_min = w.min(axis=0)
        self.bbox_max = w.max(axis=0)
        self.bbox_size = np.maximum(self.bbox_max - self.bbox_min, 1e-6)
        self.rotation = np.array(obj.matrix_world.to_3x3(), dtype=np.float64)
        self.corners = np.array([[x, y, z] for x in (self.bbox_min[0], self.bbox_max[0])
                                 for y in (self.bbox_min[1], self.bbox_max[1])
                                 for z in (self.bbox_min[2], self.bbox_max[2])])
        self._cache: Dict[str, List[bpy.types.Material]] = {}

    @property
    def diag(self) -> float:
        return float(np.linalg.norm(self.bbox_size))

    def set_materials(self, mats: List[bpy.types.Material]):
        for i in range(self.n_slots):
            self.mesh.materials[i] = mats[i % len(mats)]

    def pass_materials(self, name: str, maps: Optional[List[MaterialMaps]] = None,
                       ao_samples: int = 16) -> List[bpy.types.Material]:
        if name in self._cache:
            return self._cache[name]
        sc = self.scope
        if name == "position":
            mats = [shaders.world_position_material(sc, self.bbox_min, self.bbox_size)]
        elif name == "normal":
            mats = [shaders.world_normal_material(sc)]
        elif name == "uv":
            mats = [shaders.uv_material(sc, self.uv_map)]
        elif name in ("tangent", "bitangent"):
            mats = [shaders.tangent_frame_material(sc, "T" if name == "tangent" else "B", self.uv_map)]
        elif name == "depth":
            mats = [shaders.depth_material(sc)]
        elif name == "ao":
            mats = [shaders.ao_material(sc, self.diag * 0.15, ao_samples)]
        elif name == "curvature":
            mats = [shaders.curvature_material(sc)]
        elif name == "mask":
            mats = [shaders.flat_color_material(sc, f"NT_id_{i}", shaders.id_color(i))
                    for i in range(self.n_slots)]
        elif name == "albedo":
            mats = []
            for i in range(self.n_slots):
                mm = maps[i] if maps and i < len(maps) else None
                img = bpy.data.images.get(mm.basecolor.image) if mm and mm.basecolor.image else None
                val = mm.basecolor.value if mm and isinstance(mm.basecolor.value, list) else [0.5, 0.5, 0.5]
                uv = (mm.basecolor.uv_map if mm else None) or self.uv_map
                mats.append(shaders.albedo_material(sc, f"NT_albedo_{i}", img, val, uv))
        else:
            raise ValueError(name)
        self._cache[name] = mats
        return mats


def add_camera(scope: TempScope, cm: CameraModel) -> bpy.types.Object:
    data = scope.own(bpy.data.cameras.new(f"NT_cam_{cm.name}"))
    data.type = "ORTHO" if cm.ortho else "PERSP"
    data.ortho_scale = cm.ortho_scale
    data.lens = cm.lens
    data.sensor_width = cm.sensor_width
    data.sensor_fit = "AUTO"
    data.clip_start = cm.clip_start
    data.clip_end = cm.clip_end
    ob = bpy.data.objects.new(f"NT_cam_{cm.name}", data)
    ob.matrix_world = Matrix(cm.matrix_world)
    scope.link(ob)
    return ob


def render_still(scene: bpy.types.Scene, path: str, samples: int = 1):
    scene.cycles.samples = samples
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True, scene=scene.name)
    if not os.path.exists(path):
        raise RuntimeError(f"Render did not produce {path}")


# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------

def decode_position(rgba: np.ndarray, bbox_min, bbox_size) -> np.ndarray:
    span = shaders.POS_HI - shaders.POS_LO
    return ((rgba[..., :3] - shaders.POS_LO) / span) * np.asarray(bbox_size, np.float32) \
        + np.asarray(bbox_min, np.float32)


def camera_space_normals(world_n: np.ndarray, cam: CameraModel) -> np.ndarray:
    r = cam.mat[:3, :3]  # columns: right, up, back
    return (world_n @ r).astype(np.float32)


def controls_from_raw(cam: CameraModel, raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Control images for the image model (PNG-ready, sRGB where colour)."""
    m = raw["position"][..., 3] > 0.5
    d = raw["depth"][..., 0]
    near_white = 1.0 - normalize_range(d, m, 0.5, 99.5)
    depth_img = np.where(m, 0.1 + 0.9 * near_white, 0.0)
    cn = camera_space_normals(raw["normal"][..., :3] * 2.0 - 1.0, cam)
    nimg = np.where(m[..., None], cn * 0.5 + 0.5, np.array([0.5, 0.5, 1.0], np.float32))
    alb = np.where(m[..., None], linear_to_srgb(raw["albedo"][..., :3]), 0.5)
    return {"depth": np.repeat(depth_img[..., None], 3, 2), "normal": nimg, "albedo": alb,
            "mask": np.where(m[..., None], raw["mask"][..., :3], 0.0)}


# ----------------------------------------------------------------------------
# Export (generator of steps)
# ----------------------------------------------------------------------------

def export_package_steps(obj: bpy.types.Object, cfg: JobConfig, out_dir: str,
                         maps: List[MaterialMaps], slot_tris: List[int]) -> Iterator[Step]:
    from .uv_bake import bake_uv_steps  # local import avoids a cycle

    depsgraph = bpy.context.evaluated_depsgraph_get()
    names = list(dict.fromkeys(list(cfg.views) + (list(cfg.candidate_views) if cfg.use_planner else [])))
    tmp_dir = os.path.join(out_dir, "tmp_exr")
    os.makedirs(tmp_dir, exist_ok=True)
    with TempScope("NT_Guidance", keep=cfg.keep_temp_scene) as scope:
        scene = scope.scene
        configure_cycles(scene, cfg.view_resolution, cfg.view_resolution, cfg.use_gpu)
        proxy = Proxy(scope, obj, depsgraph, cfg.uv_map)
        if not proxy.mesh.uv_layers:
            raise RuntimeError(f"'{obj.name}' has no UV map; unwrap it first (texture generation needs UVs).")
        writer = PackageWriter(out_dir, obj.name, proxy.bbox_min, proxy.bbox_size, proxy.uv_map, list(cfg.views))
        cams = make_views(proxy.corners, proxy.rotation, names, cfg.view_resolution, ortho=cfg.orthographic)
        total = len(cams) * len(VIEW_PASSES) + 8
        step = 0
        for cm in cams:
            scene.camera = add_camera(scope, cm)
            raw = {}
            for p in VIEW_PASSES:
                proxy.set_materials(proxy.pass_materials(p, maps))
                path = os.path.join(tmp_dir, f"{cm.name}_{p}.exr")
                render_still(scene, path, 1)
                raw[p] = load_file(path)
                if not cfg.keep_temp_scene:
                    os.remove(path)
                step += 1
                yield step / total * 0.85, f"render {cm.name}: {p}"
            pos = decode_position(raw["position"], proxy.bbox_min, proxy.bbox_size)
            alpha = raw["position"][..., 3]
            writer.add_view(cm.name, cm, pos, alpha, raw["uv"][..., :2], controls_from_raw(cm, raw),
                            VIEW_WEIGHT.get(cm.name, 1.0))
            del raw

        # UV-space bakes (same proxy, same bbox encoding)
        uv = {}
        for frac, msg, data in bake_uv_steps(scope, proxy, cfg.texture_resolution, cfg.ao_samples):
            uv.update(data)
            yield 0.85 + 0.12 * frac, msg
        writer.set_uv(uv)

        prim = primary_maps(maps, slot_tris)
        ex, meta = export_existing_maps(prim, cfg.texture_resolution,
                                        bool(cfg.pbr.get("normal_flip_green", False)))
        writer.set_existing(ex, meta)
        writer.set_slots([{"index": i, "material": m.material, "triangles": slot_tris[i] if i < len(slot_tris) else 0}
                          for i, m in enumerate(maps)])
        writer.finish()
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    yield 1.0, "guidance package written"


def export_package(obj, cfg: JobConfig, out_dir: str, maps, slot_tris, progress=None) -> None:
    for frac, msg in export_package_steps(obj, cfg, out_dir, maps, slot_tris):
        if progress:
            progress(frac, msg)
