"""Crash recovery (V6): remove temp data left behind if Blender crashed or a
job was killed mid-render. Runs at register time and after every file load.
Only touches datablocks with the add-on's reserved prefixes and no fake user.
"""

from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

TEMP_SCENES = ("NT_Guidance", "NT_UVBake", "NT_Temp")
TEMP_PREFIXES = {
    "objects": ("NT_proxy", "NT_cam_"),
    "cameras": ("NT_cam_",),
    "materials": ("NT_pass_", "NT_id_", "NT_albedo_"),
    "images": ("NT_bake_",),
}


def cleanup_orphans(verbose: bool = True) -> int:
    removed = 0
    for sc in list(bpy.data.scenes):
        if sc.name.startswith(TEMP_SCENES) and len(bpy.data.scenes) > 1:
            bpy.data.scenes.remove(sc)
            removed += 1
    for coll, prefixes in TEMP_PREFIXES.items():
        data = getattr(bpy.data, coll)
        for idb in list(data):
            if idb.name.startswith(prefixes) and not idb.use_fake_user:
                if coll == "objects":
                    if any(not s.name.startswith(TEMP_SCENES) for s in idb.users_scene):
                        continue  # linked into a real scene: not ours to delete
                    data.remove(idb, do_unlink=True)
                elif idb.users == 0:
                    data.remove(idb)
                else:
                    continue
                removed += 1
    for me in list(bpy.data.meshes):
        if me.users == 0 and me.name.startswith("NT_proxy"):
            bpy.data.meshes.remove(me)
            removed += 1
    if removed and verbose:
        print(f"[NeuralTexture] cleaned up {removed} leftover temporary datablock(s)")
    return removed


@persistent
def _on_load(_dummy=None):
    try:
        cleanup_orphans()
    except Exception as e:  # never break file loading
        print(f"[NeuralTexture] cleanup skipped: {e}")


def register():
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)
    bpy.app.timers.register(lambda: (_on_load(), None)[1], first_interval=1.0)


def unregister():
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
