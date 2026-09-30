import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural_texture.core import imageops as io_
from neural_texture.core import normals as nm
from neural_texture.core.camera import CameraModel, make_views, look_at
from neural_texture.core.intent import parse_prompt, Intent
from neural_texture.core.config import JobConfig
from neural_texture.core.pbr import build_texture_set, ExistingMaps, SurfaceData, PBRParams


# ---------------------------------------------------------------- normals
def rand_normals(shape, rng, tilt=0.6):
    v = rng.normal(size=shape + (3,)).astype(np.float32)
    v[..., :2] *= tilt
    v[..., 2] = np.abs(v[..., 2]) + 0.3
    return nm.normalize(v)


def test_rnm_identity_properties():
    rng = np.random.default_rng(0)
    base = rand_normals((16, 16), rng)
    flat = nm.flat_like((16, 16))
    assert np.allclose(nm.rnm_combine(base, flat), base, atol=1e-5)
    assert np.allclose(nm.rnm_combine(flat, base), base, atol=1e-5)
    # zero strength preserves the macro normal exactly
    assert np.allclose(nm.rnm_combine(base, rand_normals((16, 16), rng), 0.0), base, atol=1e-5)


def test_pack_roundtrip_and_green_flip():
    rng = np.random.default_rng(1)
    n = rand_normals((8, 8), rng)
    assert np.allclose(nm.unpack(nm.pack(n)), n, atol=1e-5)
    dx = nm.pack(n, flip_green=True)
    assert np.allclose(nm.unpack(dx, flip_green=True), n, atol=1e-5)


def test_height_to_normal_orientation():
    # Height increasing with +U (columns) should tilt normal towards -X.
    h = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (32, 1))
    n = nm.height_to_normal(h, strength=10)
    assert (n[..., 0] < 0).all()
    # Height increasing with +V (rows, bottom-up) should tilt towards -Y.
    n2 = nm.height_to_normal(h.T.copy(), strength=10)
    assert (n2[..., 1] < 0).all()


def test_band_pass_removes_macro():
    y, x = np.mgrid[0:128, 0:128].astype(np.float32)
    macro = x / 128.0
    rng = np.random.default_rng(2)
    micro = rng.normal(scale=0.05, size=(128, 128)).astype(np.float32)
    mask = np.ones((128, 128), np.float32)
    band = nm.band_pass_height(macro + micro, mask, 0.0, 6.0)
    # correlates with micro noise, not with the ramp
    inner = (slice(16, -16), slice(16, -16))
    assert np.corrcoef(band[inner].ravel(), micro[inner].ravel())[0, 1] > 0.8
    assert abs(np.corrcoef(band[inner].ravel(), macro[inner].ravel())[0, 1]) < 0.1


# ---------------------------------------------------------------- imageops
def test_gaussian_blur_preserves_mean_and_smooths():
    rng = np.random.default_rng(3)
    img = rng.random((64, 64)).astype(np.float32)
    b = io_.gaussian_blur(img, 3.0)
    assert abs(b.mean() - img.mean()) < 0.01
    assert b.std() < img.std() * 0.5


def test_pull_push_fill_keeps_valid_and_fills_holes():
    img = np.zeros((40, 40, 3), np.float32)
    img[:, :20] = [1, 0, 0]
    mask = np.zeros((40, 40), np.float32)
    mask[:, :20] = 1
    out = io_.pull_push_fill(img, mask)
    assert np.allclose(out[:, :20], img[:, :20])
    assert np.all(out[:, 20:, 0] > 0.9)  # extrapolates red into the hole


def test_bilinear_resize_constant():
    img = np.full((10, 20, 3), 0.25, np.float32)
    assert np.allclose(io_.resize_bilinear(img, 33, 7), 0.25)


# ---------------------------------------------------------------- camera
def test_front_camera_projects_centre_and_axes():
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    cams = make_views(corners, np.eye(3), ["front", "right", "top"], resolution=100)
    front = cams[0]
    assert np.allclose(front.forward, [0, 1, 0], atol=1e-6)
    px, py, d = front.world_to_pixel(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], float))
    assert abs(px[0] - 49.5) < 1e-3 and abs(py[0] - 49.5) < 1e-3
    assert px[1] > px[0]            # +X is to the right in the front view
    assert py[2] > py[0]            # +Z is up (rows bottom-up)
    # everything fits in frame
    for c in cams:
        px, py, d = c.world_to_pixel(corners)
        assert (px >= 0).all() and (px <= 99).all() and (py >= 0).all() and (py <= 99).all()
        assert (d > 0).all()


def test_rotated_object_views_follow_object_axes():
    ang = np.radians(90)
    rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    front = make_views(corners, rot, ["front"])[0]
    # object's -Y rotated by 90deg about Z is +X, camera sits at +X looking -X
    assert np.allclose(front.forward, [-1, 0, 0], atol=1e-6)


def test_perspective_projection_centre():
    m = look_at(np.array([0, -5.0, 0]), np.zeros(3), np.array([0, 0, 1.0]))
    cam = CameraModel("p", m.tolist(), 64, 64, ortho=False, lens=50, sensor_width=36)
    px, py, d = cam.world_to_pixel(np.array([[0, 0, 0.0]]))
    assert abs(px[0] - 31.5) < 1e-3 and abs(d[0] - 5) < 1e-6
    assert CameraModel.from_dict(cam.to_dict()).to_dict() == cam.to_dict()


# ---------------------------------------------------------------- intent
SPEC_PROMPT = """Make the character resemble this face.
Keep the existing head proportions.
Keep the original large-scale facial normal.
Make the skin slightly smoother.
Add subtle lip and nose details.
Generate high-frequency skin detail only.
Do not change the silhouette."""


def test_spec_example_prompt():
    it = parse_prompt(SPEC_PROMPT, has_reference=True)
    assert not it.geometry_warnings, it.geometry_warnings
    assert it.preserve_macro_normal and it.preserve_silhouette
    assert it.reference_role == "identity"
    assert it.roughness_offset < 0
    assert it.generate_micro and it.generate_meso and it.meso_strength <= 0.2
    assert it.detail_strength < 0.6
    assert "silhouette" not in it.appearance_prompt.lower()


@pytest.mark.parametrize("text", [
    "Make the nose twice as large",
    "make the ears bigger",
    "Add horns to the head",
    "change the silhouette of the jaw",
    "Open the mouth",
])
def test_geometry_requests_warn(text):
    it = parse_prompt(text)
    assert it.requires_geometry_mode, text


@pytest.mark.parametrize("text", [
    "Make the skin rougher and more weathered",
    "Add pores and fine wrinkles",
    "Do not make the nose bigger",
])
def test_texture_requests_do_not_warn(text):
    assert not parse_prompt(text).requires_geometry_mode, text


def test_intent_roundtrip():
    it = parse_prompt("make the nose bigger. add scratches")
    it2 = Intent.from_dict(it.to_dict())
    assert it2.to_dict() == it.to_dict()


def test_job_config_roundtrip(tmp_path):
    cfg = JobConfig(object_name="Head", prompt="x", reference_images=["a.png"])
    cfg.model.seed = 7
    p = tmp_path / "job.json"
    cfg.save(str(p))
    cfg2 = JobConfig.load(str(p))
    assert cfg2.to_dict() == cfg.to_dict()


# ---------------------------------------------------------------- pbr
def test_build_texture_set_preserves_macro_normal():
    H = W = 128
    rng = np.random.default_rng(5)
    cov = np.zeros((H, W), bool)
    cov[8:-8, 8:-8] = True
    seen = cov.copy()
    seen[:, 100:] = False
    col = np.clip(0.5 + rng.normal(scale=0.05, size=(H, W, 3)), 0, 1).astype(np.float32)
    base_n = rand_normals((H, W), rng, tilt=0.3)
    base_n = nm.macro_component(base_n, 8)
    ex = ExistingMaps(basecolor=np.full((H, W, 3), 0.3, np.float32), normal=nm.pack(base_n),
                      roughness=np.full((H, W), 0.6, np.float32))
    it = parse_prompt("Make the skin slightly smoother. Generate high-frequency skin detail only.")
    out = build_texture_set(SurfaceData(color=col, seen=seen, coverage=cov), ex, it,
                            PBRParams(confidence_blend=False, delight_amount=0.6, highlight_suppress=0.0))
    for k in ("basecolor", "detail_normal", "normal", "roughness"):
        assert k in out.maps and np.isfinite(out.maps[k]).all()
    # macro structure survives: low-passed final normal ~ low-passed base
    lo_f = nm.macro_component(nm.unpack(out.maps["normal"]), 8)[cov]
    lo_b = nm.macro_component(base_n, 8)[cov]
    assert nm.angular_difference_deg(lo_f, lo_b).mean() < 3.0
    # unseen texels keep existing albedo
    assert np.allclose(out.maps["basecolor"][20:100, 110:118], 0.3, atol=1e-5)
    assert out.maps["roughness"][cov].mean() < 0.62
    assert out.maps["orm"].shape == (H, W, 3)
