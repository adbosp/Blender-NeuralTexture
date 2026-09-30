"""Real-Blender smoke test (no ComfyUI needed).

    blender -b --factory-startup --python tests/blender_smoke.py

1. Builds a UV sphere with a colour-grid BaseColor texture.
2. Runs the full pipeline (analyze -> guidance package -> synthesis -> material)
   with the 'identity' generator (re-projects the existing albedo: no AI).
3. Checks: BaseColor round-trips, the user's scene/mesh/materials are
   untouched, temp data is gone, and history/package files exist.
"""
import os
import sys
import tempfile

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neural_texture import pipeline  # noqa: E402
from neural_texture.core import pngio, synth  # noqa: E402
from neural_texture.core.config import JobConfig  # noqa: E402
from neural_texture.core.imageops import gaussian_blur, resize_bilinear  # noqa: E402
from neural_texture.blender.imageio import image_to_display_rgb  # noqa: E402


def snapshot(obj):
    me = obj.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    return {"scenes": sorted(s.name for s in bpy.data.scenes),
            "objects": sorted(o.name for o in bpy.data.objects),
            "cameras": len(bpy.data.cameras), "meshes": len(bpy.data.meshes),
            "co": co.tobytes(), "uv": uv.tobytes(),
            "slots": [s.material.name if s.material else None for s in obj.material_slots],
            "matrix": [list(r) for r in obj.matrix_world]}


def make_sphere():
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
    obj = bpy.context.active_object
    obj.name = "SmokeSphere"
    img = bpy.data.images.new("SmokeGrid", 512, 512)
    img.generated_type = "COLOR_GRID"
    mat = bpy.data.materials.new("SmokeMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = img
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.4
    obj.data.materials.append(mat)
    return obj, img


def main():
    obj, img = make_sphere()
    before = snapshot(obj)
    cfg = JobConfig(object_name=obj.name, output_dir=tempfile.mkdtemp(prefix="nt_smoke_"),
                    prompt="Add fine pores. Keep the silhouette.", view_resolution=384,
                    texture_resolution=512, ao_samples=4, assign_material=False,
                    pbr={"delight_amount": 0.0, "ao_delight": 0.0, "highlight_suppress": 0.0,
                         "albedo_clamp": False, "confidence_blend": False, "seam_strength": 0.0})
    cfg.synth = {"max_views": 8, "uv_inpaint": False}
    res = pipeline.run_all(obj, cfg, "identity", lambda f, m: print(f"{f:5.1%} {m}"))
    out = pipeline.output_dir(cfg, obj)
    print("validation:", res["validation"], "stats:", res["stats"])

    for f in ("package.json", "analysis.json", "latest.json", "synth/state.json"):
        assert os.path.exists(os.path.join(out, f)), f
    bc = pngio.read_png(os.path.join(out, res["textures"]["basecolor"]))[..., :3]
    src = resize_bilinear(image_to_display_rgb(img), 512, 512)
    cov = np.load(os.path.join(out, "uv", "coverage.npy"))
    err = np.abs(gaussian_blur(bc, 1.5) - gaussian_blur(src, 1.5))[cov].mean()
    print(f"seen {res['stats']['seen_fraction']:.3f}, basecolor round-trip error {err:.4f}")
    assert res["stats"]["seen_fraction"] > 0.9
    assert err < 0.05, err

    # tangent frame sanity: T and B orthogonal to N and each other
    T = np.load(os.path.join(out, "uv", "tangent.npy"))[cov]
    N = np.load(os.path.join(out, "uv", "normal.npy"))[cov]
    assert np.abs((T * N).sum(1)).mean() < 0.05

    after = snapshot(obj)
    for k in before:
        assert before[k] == after[k], f"user data changed: {k}"
    assert bpy.data.materials.get("NT_SmokeSphere") is not None
    assert not [s for s in bpy.data.scenes if s.name.startswith("NT_")]
    assert not [o for o in bpy.data.objects if o.name.startswith("NT_")]
    print("BLENDER SMOKE TEST PASSED ->", out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
