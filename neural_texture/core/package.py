"""Guidance package: everything the (pure-Python) synthesis needs, on disk.

Written once by Blender (guidance stage); read by synthesis, region
regeneration and map rebuilding — which therefore run without Blender
(from the add-on's worker thread, the CLI, or the MCP server).

<out>/package.json
<out>/uv/{position,normal,tangent,bitangent,coverage,material_id,ao,curvature}.npy
<out>/existing/{basecolor,normal,roughness,metallic,ao}.npy      (optional)
<out>/views/<name>/{depth,normal,albedo,mask,alpha}.png           (model controls)
<out>/views/<name>/{position,alpha,uv}.npy                        (projection data)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from . import pngio
from .camera import CameraModel
from .pbr import ExistingMaps
from .projection import ViewSample

PACKAGE_VERSION = 2


def _save(out_dir: str, rel: str, arr: np.ndarray) -> str:
    path = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    return rel


class PackageWriter:
    """Incremental writer: views are flushed to disk as they are rendered, so
    memory stays bounded to one view regardless of how many candidates exist."""

    def __init__(self, out_dir: str, object_name: str, bbox_min, bbox_size, uv_map: str = "",
                 default_views: Optional[List[str]] = None):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        bbox_size = np.asarray(bbox_size, np.float64)
        self.man = {"version": PACKAGE_VERSION, "object": object_name,
                    "bbox_min": list(map(float, bbox_min)), "bbox_size": bbox_size.tolist(),
                    "bbox_diagonal": float(np.linalg.norm(bbox_size)),
                    "uv": {"resolution": 0, "uv_map": uv_map, "files": {}},
                    "existing": {"files": {}}, "slots": [], "views": {},
                    "default_views": default_views or []}

    def add_view(self, name: str, camera: CameraModel, position: np.ndarray, alpha: np.ndarray,
                 uv: np.ndarray, controls: Dict[str, np.ndarray], weight: float = 1.0):
        vd = f"views/{name}"
        os.makedirs(os.path.join(self.dir, vd), exist_ok=True)
        files = {}
        for k, img in controls.items():
            files[k] = f"{vd}/{k}.png"
            pngio.write_png(os.path.join(self.dir, files[k]), img)
        arrays = {"position": _save(self.dir, f"{vd}/position.npy", position.astype(np.float32)),
                  "alpha": _save(self.dir, f"{vd}/alpha.npy", (alpha > 0.5)),
                  "uv": _save(self.dir, f"{vd}/uv.npy", uv[..., :2].astype(np.float32))}
        self.man["views"][name] = {"camera": camera.to_dict(), "weight": float(weight),
                                   "files": files, "arrays": arrays,
                                   "coverage": float((alpha > 0.5).mean())}

    def set_uv(self, uv: Dict[str, np.ndarray]):
        self.man["uv"]["resolution"] = int(uv["coverage"].shape[0])
        for k, a in uv.items():
            if a is None:
                continue
            dtype = {"coverage": bool, "material_id": np.int16}.get(k, np.float32)
            self.man["uv"]["files"][k] = _save(self.dir, f"uv/{k}.npy", np.asarray(a).astype(dtype))

    def set_existing(self, maps: Dict[str, np.ndarray], meta: Optional[Dict] = None):
        self.man["existing"].update(meta or {})
        for k, a in maps.items():
            if a is not None:
                self.man["existing"]["files"][k] = _save(self.dir, f"existing/{k}.npy", np.asarray(a, np.float32))

    def set_slots(self, slots: List[Dict]):
        self.man["slots"] = slots

    def finish(self) -> Dict:
        if not self.man["uv"]["files"]:
            raise ValueError("package has no UV data")
        tmp = os.path.join(self.dir, "package.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.man, f, indent=2)
        os.replace(tmp, os.path.join(self.dir, "package.json"))
        return self.man


def write_package(out_dir: str, *, object_name: str, bbox_min, bbox_size, uv: Dict[str, np.ndarray],
                  views: Dict[str, Dict], existing: Optional[Dict[str, np.ndarray]] = None,
                  existing_meta: Optional[Dict] = None, slots: Optional[List[Dict]] = None,
                  uv_map: str = "", default_views: Optional[List[str]] = None) -> Dict:
    """One-shot writer. views[name] = {"camera", "position", "alpha", "uv", "controls", "weight"}."""
    w = PackageWriter(out_dir, object_name, bbox_min, bbox_size, uv_map, default_views)
    for name, v in views.items():
        w.add_view(name, v["camera"], v["position"], v["alpha"], v["uv"], v.get("controls", {}),
                   v.get("weight", 1.0))
    w.set_uv(uv)
    w.set_existing(existing or {}, existing_meta)
    w.set_slots(slots or [])
    return w.finish()


class Package:
    def __init__(self, out_dir: str):
        self.dir = out_dir
        with open(os.path.join(out_dir, "package.json"), encoding="utf-8") as f:
            self.man = json.load(f)
        if self.man.get("version") != PACKAGE_VERSION:
            raise ValueError(f"package version {self.man.get('version')} != {PACKAGE_VERSION}; re-run guidance")
        self._uv: Dict[str, np.ndarray] = {}

    # -- paths ----------------------------------------------------------------
    def path(self, rel: str) -> str:
        return os.path.join(self.dir, rel)

    @property
    def views(self) -> List[str]:
        return list(self.man["views"])

    @property
    def bbox_diag(self) -> float:
        return float(self.man["bbox_diagonal"])

    @property
    def resolution(self) -> int:
        return int(self.man["uv"]["resolution"])

    # -- uv ---------------------------------------------------------------------
    def uv(self, key: str) -> Optional[np.ndarray]:
        if key not in self._uv:
            rel = self.man["uv"]["files"].get(key)
            self._uv[key] = np.load(self.path(rel)) if rel else None
        return self._uv[key]

    def existing(self) -> ExistingMaps:
        files = self.man.get("existing", {}).get("files", {})
        load = lambda k: np.load(self.path(files[k])) if k in files else None  # noqa: E731
        meta = self.man.get("existing", {})
        return ExistingMaps(basecolor=load("basecolor"), normal=load("normal"), roughness=load("roughness"),
                            metallic=load("metallic"), ao=load("ao"),
                            normal_flip_green=bool(meta.get("normal_flip_green", False)),
                            default_roughness=float(meta.get("default_roughness", 0.5)),
                            default_metallic=float(meta.get("default_metallic", 0.0)))

    # -- views ----------------------------------------------------------------
    def camera(self, name: str) -> CameraModel:
        return CameraModel.from_dict(self.man["views"][name]["camera"])

    def view_sample(self, name: str, with_uv: bool = True) -> ViewSample:
        v = self.man["views"][name]
        a = v["arrays"]
        return ViewSample(camera=self.camera(name), image=None,
                          position=np.load(self.path(a["position"])),
                          alpha=np.load(self.path(a["alpha"])).astype(np.float32),
                          weight_scale=float(v.get("weight", 1.0)),
                          uv=np.load(self.path(a["uv"])) if with_uv else None, name=name)

    def control(self, name: str, key: str) -> Optional[np.ndarray]:
        rel = self.man["views"][name]["files"].get(key)
        return pngio.read_png(self.path(rel))[..., :3] if rel and os.path.exists(self.path(rel)) else None

    def control_path(self, name: str, key: str) -> Optional[str]:
        rel = self.man["views"][name]["files"].get(key)
        return self.path(rel) if rel else None

    def view_dir(self, name: str) -> str:
        d = self.path(f"views/{name}")
        os.makedirs(d, exist_ok=True)
        return d
