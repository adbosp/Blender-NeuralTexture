"""Camera placement and projection math (pure numpy, Blender conventions).

Blender cameras look down their local -Z axis with +Y up. Rendered images
are stored bottom-up, so pixel row 0 is the bottom of the frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Sequence

import numpy as np

# View name -> direction from the object centre TOWARDS the camera, expressed
# in the object's local frame (Blender: front view looks from -Y).
VIEW_DIRECTIONS: Dict[str, Sequence[float]] = {
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    "front_left": (-0.7071068, -0.7071068, 0.0),
    "front_right": (0.7071068, -0.7071068, 0.0),
    # Extra candidates used by the V2 view planner to reach occluded/unseen areas.
    "back_left": (-0.7071068, 0.7071068, 0.0),
    "back_right": (0.7071068, 0.7071068, 0.0),
    "top_front": (0.0, -0.7071068, 0.7071068),
    "top_back": (0.0, 0.7071068, 0.7071068),
    "top_left": (-0.7071068, 0.0, 0.7071068),
    "top_right": (0.7071068, 0.0, 0.7071068),
    "bottom_front": (0.0, -0.7071068, -0.7071068),
    "bottom_back": (0.0, 0.7071068, -0.7071068),
    "bottom_left": (-0.7071068, 0.0, -0.7071068),
    "bottom_right": (0.7071068, 0.0, -0.7071068),
}
DEFAULT_VIEWS = ["front", "back", "left", "right", "top", "bottom", "front_left", "front_right"]
CANDIDATE_VIEWS = list(VIEW_DIRECTIONS.keys())


@dataclass
class CameraModel:
    name: str
    matrix_world: List[List[float]]  # 4x4, camera -> world
    width: int
    height: int
    ortho: bool = True
    ortho_scale: float = 2.0
    lens: float = 50.0
    sensor_width: float = 36.0
    clip_start: float = 0.01
    clip_end: float = 100.0
    extra: dict = field(default_factory=dict)

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(**d)

    # -- geometry -----------------------------------------------------------
    @property
    def mat(self) -> np.ndarray:
        return np.asarray(self.matrix_world, dtype=np.float64)

    @property
    def location(self) -> np.ndarray:
        return self.mat[:3, 3]

    @property
    def forward(self) -> np.ndarray:
        """Unit viewing direction in world space (camera -Z)."""
        f = -self.mat[:3, 2]
        return f / np.linalg.norm(f)

    def _frame_size(self):
        """Width/height of the frame at unit depth (persp) or absolute (ortho)."""
        span = self.ortho_scale if self.ortho else self.sensor_width / self.lens
        if self.width >= self.height:
            return span, span * self.height / self.width
        return span * self.width / self.height, span

    def world_to_camera(self, pts: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(self.mat)
        p = np.asarray(pts, dtype=np.float64)
        return p @ inv[:3, :3].T + inv[:3, 3]

    def world_to_pixel(self, pts: np.ndarray):
        """Project world points -> (px, py, depth).

        px/py are continuous pixel coordinates (pixel centres at integers,
        py bottom-up). depth is the positive distance along the view axis.
        """
        pc = self.world_to_camera(pts)
        depth = -pc[..., 2]
        fw, fh = self._frame_size()
        if self.ortho:
            nx = pc[..., 0] / fw
            ny = pc[..., 1] / fh
        else:
            d = np.maximum(depth, 1e-9)
            nx = pc[..., 0] / d / fw
            ny = pc[..., 1] / d / fh
        px = (nx + 0.5) * self.width - 0.5
        py = (ny + 0.5) * self.height - 0.5
        return px.astype(np.float32), py.astype(np.float32), depth.astype(np.float32)

    def view_dirs_to(self, pts: np.ndarray) -> np.ndarray:
        """Unit vectors from each world point towards the camera."""
        if self.ortho:
            d = -self.forward
            return np.broadcast_to(d, np.shape(pts)).astype(np.float32)
        v = self.location - np.asarray(pts, dtype=np.float64)
        return (v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)).astype(np.float32)

    def pixel_world_size(self, depth) -> np.ndarray:
        fw, _ = self._frame_size()
        if self.ortho:
            return np.full(np.shape(depth), fw / self.width, dtype=np.float32)
        return (np.asarray(depth) * fw / self.width).astype(np.float32)


def look_at(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Camera->world matrix for a Blender camera at ``eye`` looking at ``target``."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    up = np.asarray(up_hint, float)
    if abs(np.dot(f, up / np.linalg.norm(up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, f)
    m = np.eye(4)
    m[:3, 0] = right
    m[:3, 1] = cam_up
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m


def make_views(corners_world: np.ndarray, object_rotation: np.ndarray,
               names: Iterable[str] = DEFAULT_VIEWS, resolution: int = 1024,
               ortho: bool = True, margin: float = 1.06, lens: float = 50.0,
               sensor_width: float = 36.0) -> List[CameraModel]:
    """Build tightly-framed cameras around an object's bounding box.

    ``corners_world``: (8, 3) world-space bounding box corners (or any point
    cloud). ``object_rotation``: 3x3 rotation of the object so that views
    follow the object's own front axis even if it is rotated in the scene.
    """
    pts = np.asarray(corners_world, dtype=np.float64)
    center = pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(pts - center, axis=1))) or 1.0
    rot = np.asarray(object_rotation, dtype=np.float64)
    # Remove scale / shear from the rotation.
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    obj_up = rot @ np.array([0.0, 0.0, 1.0])
    obj_fwd = rot @ np.array([0.0, 1.0, 0.0])

    cams: List[CameraModel] = []
    for name in names:
        if name not in VIEW_DIRECTIONS:
            raise ValueError(f"Unknown view '{name}'. Known: {sorted(VIEW_DIRECTIONS)}")
        d = rot @ np.asarray(VIEW_DIRECTIONS[name], dtype=np.float64)
        d /= np.linalg.norm(d)
        up_hint = obj_up if abs(np.dot(d, obj_up)) < 0.95 else obj_fwd * (1 if d @ obj_up > 0 else -1)

        # Frame tightly: project points onto the provisional camera plane.
        dist = radius * 4.0  # provisional; perspective distance refined below
        m = look_at(center + d * dist, center, up_hint)
        right, cup = m[:3, 0], m[:3, 1]
        rel = pts - center
        xs, ys = rel @ right, rel @ cup
        cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
        span = max(xs.max() - xs.min(), ys.max() - ys.min()) * margin
        span = max(span, 1e-4)
        shifted = center + right * cx + cup * cy
        if not ortho:
            half_fov = np.arctan(sensor_width / (2 * lens))
            dist = (span / 2) / np.tan(half_fov) + radius
        m = look_at(shifted + d * dist, shifted, up_hint)
        cams.append(CameraModel(
            name=name, matrix_world=m.tolist(), width=resolution, height=resolution,
            ortho=ortho, ortho_scale=float(span), lens=lens, sensor_width=sensor_width,
            clip_start=max(1e-3, dist - radius * 1.5), clip_end=dist + radius * 1.5,
        ))
    return cams
