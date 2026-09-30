"""Synthetic guidance packages (sphere, cube) and a fake 'neural model'.

Lets the whole V2/V3 synthesis run without Blender or a GPU:
  * geometry is ray-cast analytically (stands in for Blender's passes/bakes),
  * the fake model paints a known procedural albedo, adds per-view lighting
    and colour drift (what real diffusion models do), and honours init/mask
    like latent-noise-mask inpainting.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural_texture.core import normals as nm  # noqa: E402
from neural_texture.core.camera import CANDIDATE_VIEWS, DEFAULT_VIEWS, make_views  # noqa: E402
from neural_texture.core.generators import BaseGenerator, read_image  # noqa: E402
from neural_texture.core.package import write_package  # noqa: E402

CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)


# ----------------------------------------------------------------------------- albedo
def true_albedo(P: np.ndarray) -> np.ndarray:
    """Procedural 'stone' albedo defined on the surface (world position)."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    base = 0.45 + 0.12 * np.sin(3.1 * x + 1.3) * np.cos(2.7 * y) + 0.08 * np.sin(4.3 * z)
    fine = 0.05 * np.sin(23 * x + 7 * z) * np.sin(19 * y - 5 * x)
    lum = base + fine
    return np.stack([lum * 1.05, lum * 0.97, lum * 0.88], -1).clip(0, 1).astype(np.float32)


def existing_albedo(P):
    return np.full(P.shape[:-1] + (3,), 0.35, np.float32)


# ----------------------------------------------------------------------------- shapes
class Sphere:
    name = "sphere"

    def uv_maps(self, res):
        v, u = np.mgrid[0:res, 0:res].astype(np.float32)
        u = (u + 0.5) / res
        v = (v + 0.5) / res
        th = (u - 0.5) * 2 * np.pi
        ph = (v - 0.5) * np.pi * 0.98  # avoid exact poles
        P = np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], -1)
        T = np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], -1)
        N = P.copy()
        B = np.cross(N, T)
        return (P.astype(np.float32), N.astype(np.float32), T.astype(np.float32), B.astype(np.float32),
                np.ones((res, res), bool), np.zeros((res, res), np.int16))

    def uv_of(self, P):
        th = np.arctan2(P[..., 1], P[..., 0])
        ph = np.arcsin(np.clip(P[..., 2], -1, 1))
        return np.stack([th / (2 * np.pi) + 0.5, ph / (np.pi * 0.98) + 0.5], -1)

    def raycast(self, o, d):
        b = np.sum(o * d, -1)
        c = np.sum(o * o, -1) - 1.0
        disc = b * b - c
        hit = disc > 0
        t = -b - np.sqrt(np.maximum(disc, 0))
        P = o + d * t[..., None]
        return P, hit, P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-9), np.zeros(hit.shape, int)


class Cube:
    """Unit cube (half-size 0.8), each face its own UV island in a 3x2 atlas with gaps."""
    name = "cube"
    s = 0.8
    # face: (normal, u axis, v axis)
    FACES = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
             ((0, 1, 0), (-1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
             ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, -1, 0))]
    GAP = 0.02

    def _cell(self, f):
        cx, cy = f % 3, f // 3
        w = 1 / 3 - 2 * self.GAP
        h = 1 / 2 - 2 * self.GAP
        return cx / 3 + self.GAP, cy / 2 + self.GAP, w, h

    def uv_maps(self, res):
        v, u = np.mgrid[0:res, 0:res].astype(np.float32)
        u = (u + 0.5) / res
        v = (v + 0.5) / res
        P = np.zeros((res, res, 3), np.float32)
        N = np.zeros_like(P)
        T = np.zeros_like(P)
        B = np.zeros_like(P)
        cov = np.zeros((res, res), bool)
        mid = np.full((res, res), -1, np.int16)
        for f, (n, a, b) in enumerate(self.FACES):
            x0, y0, w, h = self._cell(f)
            m = (u >= x0) & (u < x0 + w) & (v >= y0) & (v < y0 + h)
            s = ((u - x0) / w * 2 - 1) * self.s
            t = ((v - y0) / h * 2 - 1) * self.s
            n, a, b = map(np.array, (n, a, b))
            P[m] = (n * self.s + a * s[..., None] + b * t[..., None])[m]
            N[m], T[m], B[m] = n, a, b
            cov |= m
            mid[m] = 0 if f < 4 else 1  # two material slots: sides / caps
        return P, N, T, B, cov, mid

    def uv_of_face(self, P, f):
        n, a, b = map(np.array, self.FACES[f])
        x0, y0, w, h = self._cell(f)
        s = np.sum(P * a, -1) / self.s
        t = np.sum(P * b, -1) / self.s
        return np.stack([x0 + (s + 1) / 2 * w, y0 + (t + 1) / 2 * h], -1)

    def raycast(self, o, d):
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        t1 = (-self.s - o) * inv
        t2 = (self.s - o) * inv
        tmin = np.max(np.minimum(t1, t2), -1)
        tmax = np.min(np.maximum(t1, t2), -1)
        hit = (tmax >= tmin) & (tmax > 0)
        P = o + d * tmin[..., None]
        ax = np.argmax(np.abs(P) / self.s, -1)
        sign = np.sign(np.take_along_axis(P, ax[..., None], -1)[..., 0])
        N = np.zeros_like(P)
        np.put_along_axis(N, ax[..., None], sign[..., None], -1)
        face = np.select([(ax == 0) & (sign > 0), (ax == 0) & (sign < 0), (ax == 1) & (sign > 0),
                          (ax == 1) & (sign < 0), (ax == 2) & (sign > 0)], [0, 1, 2, 3, 4], 5)
        return P, hit, N, face

    def uv_of(self, P, face):
        out = np.zeros(P.shape[:-1] + (2,), np.float32)
        for f in range(6):
            m = face == f
            if m.any():
                out[m] = self.uv_of_face(P[m], f)
        return out


# ----------------------------------------------------------------------------- package
def render_view(shape, cam):
    m = cam.mat
    right, up, fwd = m[:3, 0], m[:3, 1], -m[:3, 2]
    fw, fh = cam._frame_size()
    py, px = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    x = ((px + 0.5) / cam.width - 0.5) * fw
    y = ((py + 0.5) / cam.height - 0.5) * fh
    o = cam.location + right * x[..., None] + up * y[..., None]
    d = np.broadcast_to(fwd, o.shape)
    P, hit, N, face = shape.raycast(o, d)
    P[~hit] = 0
    N[~hit] = 0
    uv = shape.uv_of(P, face) if isinstance(shape, Cube) else shape.uv_of(P)
    uv[~hit] = 0
    depth = np.sum((P - cam.location) * fwd, -1)
    return P.astype(np.float32), hit, N.astype(np.float32), uv.astype(np.float32), depth


def build_package(out_dir, shape, tex_res=128, view_res=128, views=None, with_existing=True):
    names = views or list(CANDIDATE_VIEWS)
    cams = make_views(CORNERS * (0.8 if isinstance(shape, Cube) else 1.0), np.eye(3), names, view_res)
    P, N, T, B, cov, mid = shape.uv_maps(tex_res)
    vdata = {}
    for cam in cams:
        pos, hit, n, uv, depth = render_view(shape, cam)
        a = hit.astype(np.float32)
        ncam = n @ cam.mat[:3, :3]
        dn = np.where(hit, 1 - (depth - depth[hit].min()) / max(np.ptp(depth[hit]), 1e-6), 0)
        vdata[cam.name] = {
            "camera": cam, "position": pos, "alpha": a, "uv": uv,
            "weight": {"top": 0.6, "bottom": 0.4}.get(cam.name, 1.0),
            "controls": {"depth": np.repeat(dn[..., None], 3, 2),
                         "normal": np.where(hit[..., None], ncam * 0.5 + 0.5, [0.5, 0.5, 1.0]),
                         "albedo": np.where(hit[..., None], existing_albedo(pos), 0.5),
                         "mask": np.repeat(a[..., None], 3, 2)},
        }
    existing = None
    if with_existing:
        existing = {"basecolor": np.where(cov[..., None], existing_albedo(P), 0),
                    "normal": np.where(cov[..., None], nm.pack(nm.flat_like(cov.shape)), 0.5),
                    "roughness": np.full(cov.shape, 0.5, np.float32)}
    slots = [{"index": 0, "material": "Rock"}] + ([{"index": 1, "material": "RockCap"}] if isinstance(shape, Cube) else [])
    size = (1.6, 1.6, 1.6) if isinstance(shape, Cube) else (2, 2, 2)
    return write_package(out_dir, object_name=shape.name, bbox_min=(-size[0] / 2,) * 3, bbox_size=size,
                         uv={"position": P, "normal": N, "tangent": T, "bitangent": B, "coverage": cov,
                             "material_id": mid, "ao": np.ones(cov.shape, np.float32),
                             "curvature": np.full(cov.shape, 0.5, np.float32)},
                         views=vdata, existing=existing, slots=slots, default_views=list(DEFAULT_VIEWS))


# ----------------------------------------------------------------------------- fake model
class FakeModel(BaseGenerator):
    """Paints true_albedo with view-dependent lighting and colour drift; honours init/mask."""
    name = "fake"
    supports_normals = True
    supports_atlas_inpaint = True

    def __init__(self, pkg_dir, drift=0.12, light=0.35, seed=0, normal_flip_x=False):
        self.dir = pkg_dir
        self.rng = np.random.default_rng(seed)
        self.drift = drift
        self.light = light
        self.calls = []
        self.normal_flip_x = normal_flip_x
        self.cancel_after = None

    def generate_view(self, job):
        self.calls.append(job.name)
        if self.cancel_after is not None and len(self.calls) > self.cancel_after:
            from neural_texture.core.synth import Cancelled
            raise Cancelled("test cancel")
        pos = np.load(os.path.join(self.dir, "views", job.name, "position.npy"))
        alpha = np.load(os.path.join(self.dir, "views", job.name, "alpha.npy"))
        ncam = read_image(os.path.join(self.dir, "views", job.name, "normal.png"))[..., :3] * 2 - 1
        img = true_albedo(pos)
        shade = 1 - self.light + self.light * np.clip(ncam @ nm.normalize(np.array([0.4, 0.6, 0.7])), 0, 1)
        gain = 1 + self.rng.uniform(-self.drift, self.drift, 3)
        img = np.clip(img * shade[..., None] * gain, 0, 1)
        img = np.where(alpha[..., None], img, 0.5)
        if "inpaint_mask" in job.files and "init" in job.files and not job.first:
            init = read_image(job.files["init"])[..., :3]
            m = read_image(job.files["inpaint_mask"])[..., :1]
            img = init * (1 - m) + img * m
        return img.astype(np.float32)

    def estimate_normals(self, job, image_path):
        n = read_image(os.path.join(self.dir, "views", job.name, "normal.png"))[..., :3] * 2 - 1
        if self.normal_flip_x:
            n[..., 0] *= -1
        return n

    def inpaint_atlas(self, atlas_path, mask_path, prompts, size):
        a = read_image(atlas_path)[..., :3]
        return np.clip(a * 1.0, 0, 1)
