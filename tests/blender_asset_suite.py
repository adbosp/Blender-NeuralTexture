"""Multi-asset robustness suite in real Blender (V6). No ComfyUI needed.

    blender -b --factory-startup --python tests/blender_asset_suite.py [-- --out report.json --generator identity]

Covers asset types that break naive texture pipelines:
  uv_sphere (poles + seam), cube (hard edges, many islands), suzanne (concave, occlusion),
  torus (self-occlusion), multi_material (2 slots), mirrored_uv (overlapping mirrored islands),
  rigged (armature modifier, posed), subsurf (modifier stack), rotated_scaled (object transform),
  packed_orm (roughness/metallic from one ORM texture), no_texture (constant colours only).
For each: runs the full pipeline, checks structural preservation and cleanup, and
records timings + quality stats into a JSON report (benchmark across asset types).
"""
import json
import math
import os
import sys
import tempfile
import time
import traceback

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neural_texture import pipeline  # noqa: E402
from neural_texture.core.config import JobConfig  # noqa: E402


def _mat(name, color=(0.6, 0.5, 0.4, 1), grid=True):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = color
    if grid:
        img = bpy.data.images.new(name + "_tex", 256, 256)
        img.generated_type = "COLOR_GRID"
        t = m.node_tree.nodes.new("ShaderNodeTexImage")
        t.image = img
        m.node_tree.links.new(t.outputs["Color"], b.inputs["Base Color"])
    return m


def _finish(obj, name, mats=None):
    obj.name = name
    for m in mats or [_mat(name + "_mat")]:
        obj.data.materials.append(m)
    return obj


def a_uv_sphere():
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
    return _finish(bpy.context.active_object, "uv_sphere")


def a_cube():
    bpy.ops.mesh.primitive_cube_add()
    return _finish(bpy.context.active_object, "cube")


def a_suzanne():
    bpy.ops.mesh.primitive_monkey_add()
    o = bpy.context.active_object
    if not o.data.uv_layers:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode="OBJECT")
    return _finish(o, "suzanne")


def a_torus():
    bpy.ops.mesh.primitive_torus_add()
    return _finish(bpy.context.active_object, "torus")


def a_multi_material():
    bpy.ops.mesh.primitive_cylinder_add(vertices=32)
    o = _finish(bpy.context.active_object, "multi_material",
                [_mat("Body_Skin", (0.8, 0.6, 0.5, 1)), _mat("Armor_Steel", (0.7, 0.7, 0.75, 1), grid=False)])
    for p in o.data.polygons:
        p.material_index = 1 if abs(p.normal.z) > 0.5 else 0
    return o


def a_mirrored_uv():
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
    o = bpy.context.active_object
    uv = o.data.uv_layers.active.data
    for l in uv:  # fold U onto itself: left/right halves share texels
        u, v = l.uv
        l.uv = (0.5 - abs(u - 0.5), v)
    return _finish(o, "mirrored_uv")


def a_rigged():
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, depth=2)
    o = bpy.context.active_object
    bpy.ops.object.armature_add(location=(0, 0, -1))
    arm = bpy.context.active_object
    bpy.ops.object.mode_set(mode="EDIT")
    arm.data.edit_bones[0].tail = (0, 0, 1)
    bpy.ops.object.mode_set(mode="POSE")
    arm.pose.bones[0].rotation_mode = "XYZ"
    arm.pose.bones[0].rotation_euler = (0.4, 0, 0)
    bpy.ops.object.mode_set(mode="OBJECT")
    mod = o.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    vg = o.vertex_groups.new(name=arm.data.bones[0].name)
    vg.add([v.index for v in o.data.vertices], 1.0, "REPLACE")
    bpy.context.view_layer.objects.active = o
    return _finish(o, "rigged")


def a_subsurf():
    bpy.ops.mesh.primitive_cube_add()
    o = bpy.context.active_object
    o.modifiers.new("Sub", "SUBSURF").levels = 2
    return _finish(o, "subsurf")


def a_rotated_scaled():
    bpy.ops.mesh.primitive_monkey_add(rotation=(0.3, 0.2, 1.2))
    o = bpy.context.active_object
    o.scale = (2.0, 0.7, 1.3)
    return _finish(o, "rotated_scaled")


def a_packed_orm():
    bpy.ops.mesh.primitive_uv_sphere_add()
    o = bpy.context.active_object
    m = _mat("orm_mat")
    orm = bpy.data.images.new("orm_tex", 128, 128)
    px = np.zeros((128, 128, 4), np.float32)
    px[..., 0], px[..., 1], px[..., 2], px[..., 3] = 1.0, 0.8, 0.0, 1.0
    orm.pixels.foreach_set(px.ravel())
    nt = m.node_tree
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = orm
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(t.outputs["Color"], sep.inputs["Color"])
    b = nt.nodes["Principled BSDF"]
    nt.links.new(sep.outputs["Green"], b.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], b.inputs["Metallic"])
    return _finish(o, "packed_orm", [m])


def a_no_texture():
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3)
    o = bpy.context.active_object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")
    return _finish(o, "no_texture", [_mat("flat", grid=False)])


ASSETS = [a_uv_sphere, a_cube, a_suzanne, a_torus, a_multi_material, a_mirrored_uv, a_rigged, a_subsurf,
          a_rotated_scaled, a_packed_orm, a_no_texture]


def snapshot(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    me = obj.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    return {"co": co.tobytes(), "slots": [s.material.name if s.material else None for s in obj.material_slots],
            "mods": [(m.name, m.type) for m in obj.modifiers], "matrix": [list(r) for r in obj.matrix_world],
            "uv": [l.name for l in me.uv_layers]}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(tempfile.gettempdir(), "nt_asset_report.json")
    gen = argv[argv.index("--generator") + 1] if "--generator" in argv else "identity"
    root = tempfile.mkdtemp(prefix="nt_suite_")
    report, failures = [], []
    for make in ASSETS:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        rec = {"asset": make.__name__[2:]}
        t0 = time.time()
        try:
            obj = make()
            bpy.context.view_layer.objects.active = obj
            before = snapshot(obj)
            cfg = JobConfig(object_name=obj.name, output_dir=root, prompt="weathered stone", view_resolution=256,
                            texture_resolution=512, ao_samples=4, assign_material=False)
            cfg.synth = {"max_views": 8}
            res = pipeline.run_all(obj, cfg, gen)
            rec.update(seconds=round(time.time() - t0, 1), stats=res.get("stats"), validation=res.get("validation"))
            after = snapshot(obj)
            changed = [k for k in before if before[k] != after[k]]
            assert not changed, f"user data changed: {changed}"
            assert not [s for s in bpy.data.scenes if s.name.startswith("NT_")], "temp scene left behind"
            assert res["stats"]["seen_fraction"] > 0.6, res["stats"]
            rec["ok"] = True
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=str(e), traceback=traceback.format_exc()[-2000:])
            failures.append(rec["asset"])
        report.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("asset", "ok", "seconds", "error")}))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"blender": bpy.app.version_string, "generator": gen, "assets": report}, f, indent=2, default=str)
    print(f"report: {out_path}")
    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("ASSET SUITE PASSED")


if __name__ == "__main__":
    main()
