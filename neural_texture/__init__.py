"""Neural Texture — structure-preserving neural appearance for existing meshes.

Existing geometry, UVs and maps are the structural guidance; a neural image
model (via ComfyUI) only synthesises surface appearance. Views are generated
sequentially with consistency inpainting, accumulated in UV space with a
confidence map, and baked into a material-aware PBR set. Geometry is never
modified.
"""

bl_info = {
    "name": "Neural Texture",
    "author": "Glor Games",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar > Neural Texture",
    "description": "Regenerate/enhance the appearance of an existing mesh with geometry-guided AI",
    "category": "Material",
}

try:  # Allow importing the pure-numpy core outside Blender (tests, MCP server, tooling).
    import bpy  # noqa: F401
    _HAS_BPY = True
except ImportError:  # pragma: no cover
    _HAS_BPY = False


def register():
    from . import properties, operators, ui
    from .blender import cleanup, compat
    if not compat.version_ok():
        print(f"[NeuralTexture] Blender {compat.version_str()} is older than 4.2; some features may fail.")
    properties.register()
    operators.register()
    ui.register()
    cleanup.register()


def unregister():
    from . import properties, operators, ui
    from .blender import cleanup, live_server, runner
    job = runner.current()
    if job is not None and job.running:
        job.cancel()
    live_server.stop()
    cleanup.unregister()
    ui.unregister()
    operators.unregister()
    properties.unregister()
