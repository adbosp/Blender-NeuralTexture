"""Synthetic end-to-end check of view -> UV projection with an analytic sphere.

We "render" a unit sphere from the standard views by ray casting (standing in
for Blender's position pass), paint each view with a known function of world
position (standing in for the neural model), project back to an
equirectangular UV layout, and compare with the ground truth.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural_texture.core.camera import make_views, DEFAULT_VIEWS
from neural_texture.core.projection import project_views, ViewSample, ProjectionParams


def sphere_uv_maps(res):
    v, u = np.mgrid[0:res, 0:res].astype(np.float32)
    u = (u + 0.5) / res
    v = (v + 0.5) / res  # row 0 = bottom = v near 0
    th = (u - 0.5) * 2 * np.pi
    ph = (v - 0.5) * np.pi
    pos = np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], -1)
    return pos.astype(np.float32), pos.astype(np.float32), np.ones((res, res), bool)


def render_sphere(cam):
    m = cam.mat
    right, up, fwd = m[:3, 0], m[:3, 1], -m[:3, 2]
    fw, fh = cam._frame_size()
    py, px = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    x = ((px + 0.5) / cam.width - 0.5) * fw
    y = ((py + 0.5) / cam.height - 0.5) * fh
    o = cam.location + right * x[..., None] + up * y[..., None]
    b = np.sum(o * fwd, -1)
    c = np.sum(o * o, -1) - 1.0
    disc = b * b - c
    hit = disc > 0
    t = -b - np.sqrt(np.maximum(disc, 0))
    p = o + fwd * t[..., None]
    p[~hit] = 0
    return p.astype(np.float32), hit.astype(np.float32)


def paint(pos):
    return (pos * 0.5 + 0.5).astype(np.float32)


def test_sphere_projection_matches_ground_truth():
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    cams = make_views(corners, np.eye(3), DEFAULT_VIEWS, resolution=256)
    views = []
    for cam in cams:
        pos, alpha = render_sphere(cam)
        views.append(ViewSample(cam, paint(pos) * alpha[..., None], pos, alpha))
    tpos, tnrm, cov = sphere_uv_maps(128)
    res = project_views(tpos, tnrm, cov, views, bbox_diag=2 * np.sqrt(3), params=ProjectionParams())
    assert res.seen.mean() > 0.99, res.seen.mean()
    err = np.abs(res.color - paint(tpos))[res.seen]
    assert err.mean() < 0.01, err.mean()
    assert err.max() < 0.08, err.max()


def test_occlusion_rejects_hidden_texels():
    """Only the front view: texels on the back must stay unseen."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    cam = make_views(corners, np.eye(3), ["front"], resolution=256)[0]
    pos, alpha = render_sphere(cam)
    tpos, tnrm, cov = sphere_uv_maps(64)
    res = project_views(tpos, tnrm, cov, [ViewSample(cam, paint(pos), pos, alpha)], 2 * np.sqrt(3))
    back = tpos[..., 1] > 0.1  # front camera sits at -Y
    assert not res.seen[back].any()
    front = tpos[..., 1] < -0.5
    assert res.seen[front].mean() > 0.95
