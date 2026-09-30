"""Temporary, isolated working scene (spec section 8: never modify the user's scene).

All helper data (proxy mesh, cameras, pass materials, bake images) lives in a
dedicated temporary Scene and is deleted on exit. The user's objects are never
linked into it; a static copy of the evaluated mesh is used instead, so the
original mesh, rig, weights and UVs are untouched.
"""

from __future__ import annotations

import bpy

_COLLECTIONS = (
    (bpy.types.Object, "objects"),
    (bpy.types.Mesh, "meshes"),
    (bpy.types.Camera, "cameras"),
    (bpy.types.Light, "lights"),
    (bpy.types.Material, "materials"),
    (bpy.types.Image, "images"),
    (bpy.types.NodeTree, "node_groups"),
    (bpy.types.World, "worlds"),
    (bpy.types.Collection, "collections"),
)


class TempScope:
    def __init__(self, name: str = "NT_Temp", keep: bool = False):
        self.name = name
        self.keep = keep
        self.scene: bpy.types.Scene | None = None
        self._owned: list = []

    # ------------------------------------------------------------------
    def __enter__(self) -> "TempScope":
        self.scene = bpy.data.scenes.new(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.keep:
            print(f"[NeuralTexture] keeping temp scene '{self.scene.name}' for debugging")
            return False
        self.cleanup()
        return False

    # ------------------------------------------------------------------
    def own(self, idblock):
        """Register a datablock for deletion when the scope closes."""
        self._owned.append(idblock)
        return idblock

    def link(self, obj: bpy.types.Object) -> bpy.types.Object:
        self.scene.collection.objects.link(obj)
        return self.own(obj)

    def cleanup(self):
        # Objects first so meshes/cameras lose their users.
        ordered = sorted(self._owned, key=lambda d: 0 if isinstance(d, bpy.types.Object) else 1)
        for idb in ordered:
            try:
                name = idb.name  # raises ReferenceError if already freed
            except ReferenceError:
                continue
            for typ, coll in _COLLECTIONS:
                if isinstance(idb, typ):
                    data = getattr(bpy.data, coll)
                    if name in data and data[name] == idb:
                        if typ is bpy.types.Object:
                            data.remove(idb, do_unlink=True)
                        else:
                            data.remove(idb)
                    break
        self._owned.clear()
        if self.scene is not None:
            try:
                bpy.data.scenes.remove(self.scene)
            except ReferenceError:
                pass
            self.scene = None
