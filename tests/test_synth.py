"""End-to-end V2/V3 synthesis on synthetic packages (no Blender, no GPU)."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synthetic import Cube, FakeModel, Sphere, build_package, true_albedo  # noqa: E402
from neural_texture.core import normals as nm  # noqa: E402
from neural_texture.core import synth  # noqa: E402
from neural_texture.core.config import JobConfig  # noqa: E402
from neural_texture.core.consistency import harmonize, inpaint_mask  # noqa: E402
from neural_texture.core.detail import normals_to_height, unblend  # noqa: E402
from neural_texture.core.generators import IdentityGenerator, read_image  # noqa: E402
from neural_texture.core.package import Package  # noqa: E402
from neural_texture.core.seams import refine_seams, seam_error  # noqa: E402
from neural_texture.core import materials as mats  # noqa: E402
from neural_texture.core.intent import parse_prompt  # noqa: E402

PBR_RAW = {"delight_amount": 0.0, "ao_delight": 0.0, "highlight_suppress": 0.0, "albedo_clamp": False,
           "confidence_blend": False}


def _cfg(**kw):
    cfg = JobConfig(prompt="ancient weathered stone", **kw)
    cfg.synth = dict({"max_views": 10}, **cfg.synth)
    return cfg


def _gain_fit_error(img, ref, mask):
    a, b = img[mask], ref[mask]
    g = (a * b).sum(0) / np.maximum((a * a).sum(0), 1e-9)
    return float(np.abs(a * g - b).mean())


def test_sphere_consistency_beats_independent_views():
    errs = {}
    for consistent in (True, False):
        with tempfile.TemporaryDirectory() as td:
            build_package(td, Sphere(), tex_res=96, view_res=128)
            cfg = _cfg()
            cfg.pbr = dict(PBR_RAW)
            cfg.synth.update(consistency=consistent, lock=0.8 if consistent else 0.0,
                             harmonize_strength=0.85 if consistent else 0.0)
            res = synth.run_synthesis(td, cfg, FakeModel(td, drift=0.2, light=0.0, seed=3))
            pkg = Package(td)
            bc = read_image(os.path.join(td, res["textures"]["basecolor"]))[..., :3]
            cov = pkg.uv("coverage")
            errs[consistent] = _gain_fit_error(bc, true_albedo(pkg.uv("position")), cov)
            assert res["stats"]["seen_fraction"] > 0.97
    assert errs[True] < errs[False] * 0.8, errs


def test_cube_planner_coverage_seams_and_materials():
    with tempfile.TemporaryDirectory() as td:
        build_package(td, Cube(), tex_res=128, view_res=128)
        cfg = _cfg()
        res = synth.run_synthesis(td, cfg, FakeModel(td, drift=0.1, light=0.3))
        st = res["stats"]
        assert st["seen_fraction"] > 0.97, st
        assert st["seam_error_after"] < st["seam_error_before"], st
        assert res["slot_classes"] == ["stone", "stone"]
        plan = res["synthesis"]["plan"]
        assert plan["order"][0] == "front" and len(plan["order"]) <= 10
        for k in ("basecolor", "normal", "detail_normal", "detail_height", "roughness", "ao", "orm", "confidence"):
            assert os.path.exists(os.path.join(td, res["textures"][k])), k
        rough = read_image(os.path.join(td, res["textures"]["roughness"]))[..., 0]
        cov = Package(td).uv("coverage")
        assert 0.55 < rough[cov].mean() < 0.95  # stone prior, not the existing 0.5
        assert os.path.exists(os.path.join(res["history_dir"], "preview.png"))


def test_cancel_then_resume_only_generates_remaining_views():
    with tempfile.TemporaryDirectory() as td:
        build_package(td, Sphere(), tex_res=64, view_res=96)
        cfg = _cfg()
        m1 = FakeModel(td)
        m1.cancel_after = 2
        try:
            synth.run_synthesis(td, cfg, m1)
            assert False, "expected cancel"
        except synth.Cancelled:
            pass
        st = synth.load_state(td)
        assert st["status"] == "cancelled" and len(st["done"]) == 2
        total = len(st["plan"]["order"])
        m2 = FakeModel(td)
        res = synth.run_synthesis(td, cfg, m2)
        assert len(m2.calls) == total - 2
        assert res["stats"]["seen_fraction"] > 0.95


def test_region_regeneration_only_touches_mask():
    with tempfile.TemporaryDirectory() as td:
        build_package(td, Sphere(), tex_res=64, view_res=96)
        cfg = _cfg()
        synth.run_synthesis(td, cfg, FakeModel(td, drift=0.0, light=0.0))
        pkg = Package(td)
        before = synth.load_accumulator(td, pkg, cfg).color_map()
        mask = np.zeros((64, 64), np.float32)
        mask[24:40, 20:36] = 1

        class Red(FakeModel):
            def generate_view(self, job):
                img = super().generate_view(job)
                if "inpaint_mask" in job.files:
                    m = read_image(job.files["inpaint_mask"])[..., :1]
                    img = img * (1 - m) + np.array([0.9, 0.1, 0.1]) * m
                return img

        res = synth.regenerate_region(td, cfg, Red(td, drift=0.0, light=0.0), mask)
        after = synth.load_accumulator(td, pkg, cfg).color_map()
        inside = mask > 0.5
        outside = ~inside
        assert np.allclose(before[outside], after[outside], atol=1e-5)
        assert after[inside][:, 0].mean() > before[inside][:, 0].mean() + 0.2
        assert res["label"] == "region" and len(synth.history(td)) == 2


def test_model_normals_calibrated_and_used():
    with tempfile.TemporaryDirectory() as td:
        build_package(td, Sphere(), tex_res=64, view_res=96)
        cfg = _cfg()
        res = synth.run_synthesis(td, cfg, FakeModel(td, normal_flip_x=True))
        cal = synth.load_state(td)["normal_calibration"]
        assert cal["flip_x"] and not cal["flip_y"] and cal["agreement"] > 0.9, cal
        assert res["stats"]["detail_source"] in ("hybrid", "model")


def test_identity_generator_and_history_restore():
    with tempfile.TemporaryDirectory() as td:
        build_package(td, Sphere(), tex_res=48, view_res=64, views=["front", "back", "left", "right", "top", "bottom"])
        cfg = _cfg()
        cfg.synth["uv_inpaint"] = False
        r1 = synth.run_synthesis(td, cfg, IdentityGenerator())
        cfg.pbr = {"detail_gain": 0.2}
        r2 = synth.finalize(td, cfg)
        assert r1["run_id"] != r2["run_id"]
        assert synth.latest_result(td)["run_id"] == r2["run_id"]
        synth.set_latest(td, r1["run_id"])
        assert synth.latest_result(td)["run_id"] == r1["run_id"]


def test_geometry_intent_refuses():
    with tempfile.TemporaryDirectory() as td:
        build_package(td, Sphere(), tex_res=32, view_res=48, views=["front"])
        cfg = JobConfig(prompt="làm mũi to gấp đôi")
        try:
            synth.run_synthesis(td, cfg, IdentityGenerator())
            assert False
        except ValueError as e:
            assert "geometry" in str(e)


# ------------------------------------------------------------------ unit pieces
def test_frankot_chellappa_roundtrip():
    y, x = np.mgrid[0:64, 0:64].astype(np.float32)
    h = np.sin(x / 64 * 2 * np.pi * 3) * np.cos(y / 64 * 2 * np.pi * 2)
    n = nm.height_to_normal(h, strength=2.0)
    h2 = normals_to_height(n)
    assert np.corrcoef(h.ravel(), h2.ravel())[0, 1] > 0.98


def test_unblend_inverts_rnm():
    rng = np.random.default_rng(0)
    def rn():
        v = rng.normal(size=(32, 3)); v[:, :2] *= 0.4; v[:, 2] = np.abs(v[:, 2]) + 0.5
        return nm.normalize(v)
    base, det = rn(), rn()
    full = nm.rnm_combine(base, det)
    assert np.allclose(unblend(full, base), det, atol=1e-4)


def test_seam_refinement_on_cube():
    c = Cube()
    P, N, T, B, cov, mid = c.uv_maps(96)
    col = np.where(cov[..., None], true_albedo(P), 0).astype(np.float32)
    # simulate a per-island colour offset (what separate views cause)
    x0, y0, w, h = c._cell(0)
    v, u = np.mgrid[0:96, 0:96]
    isl = (u / 96 >= x0) & (u / 96 < x0 + w) & (v / 96 >= y0) & (v / 96 < y0 + h)
    col[isl] += 0.15
    e0 = seam_error(col, P, cov)
    out, st = refine_seams(col, P, cov, radius_px=4)
    assert st["seam_texels"] > 0 and seam_error(out, P, cov) < e0 * 0.5


def test_harmonize_removes_gain_drift():
    rng = np.random.default_rng(1)
    ref = rng.random((64, 64, 3)).astype(np.float32) * 0.6 + 0.2
    gen = np.clip(ref * np.array([1.2, 0.9, 1.1]) + 0.03, 0, 1)
    alpha = np.ones((64, 64), np.float32)
    conf = np.ones((64, 64), np.float32)
    out, info = harmonize(gen, ref, conf, alpha, strength=1.0)
    assert info["applied"] and np.abs(out - ref).mean() < np.abs(gen - ref).mean() * 0.2


def test_inpaint_mask_regions():
    alpha = np.zeros((32, 32), np.float32)
    alpha[4:28, 4:28] = 1
    conf = np.zeros((32, 32), np.float32)
    conf[4:28, 4:16] = 1
    m = inpaint_mask(conf, alpha, grow_px=2, feather_px=0)
    assert m[10, 25] == 1 and m[10, 6] == 0 and m[0, 0] == 1


def test_material_classification_en_vi():
    assert mats.classify("làm model này thành đá cổ") == "stone"
    assert mats.classify("thêm lỗ chân lông cho da mặt") == "skin"
    assert mats.classify("rusty iron barrel") == "rusted_metal"
    assert mats.classify("gỗ sồi cũ") == "wood"
    assert mats.classify_slots("", ["Body_Skin", "Armor_Steel", None], "Hero") == ["skin", "metal", "generic"]


def test_vietnamese_prompt_intent():
    it = parse_prompt("làm model này thành đá cổ nhưng giữ normal gốc")
    assert it.material_class == "stone" and it.mode == "restyle" and it.language == "vi"
    assert it.preserve_macro_normal and not it.geometry_warnings
    assert "stone" in it.appearance_prompt
    it2 = parse_prompt("thêm nếp nhăn nhẹ và lỗ chân lông, không thay đổi hình dạng")
    assert it2.generate_meso and it2.generate_micro and not it2.geometry_warnings
    for t in ("làm mũi to gấp đôi", "thêm sừng trên đầu", "mở miệng nhân vật"):
        assert parse_prompt(t).requires_geometry_mode, t
