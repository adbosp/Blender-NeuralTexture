"""Import every bpy-dependent module against a stub bpy (catches syntax/name errors).

This is a smoke test only; real behaviour needs Blender (see README 'Testing in Blender').
Runs in a subprocess so the stub never leaks into other tests.
"""
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUB = textwrap.dedent(r'''
import sys, types

class _Auto(type):
    def __getattr__(cls, name):
        t = _Auto(name, (object,), {})
        setattr(cls, name, t)
        return t

class _Any:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Any()
    def __getattr__(self, n): return _Any()

def mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m

class types_ns(metaclass=_Auto): pass
registered = []
bpy = mod("bpy", types=types_ns, data=_Any(), context=_Any(), ops=_Any(), app=_Any(), path=_Any(),
          utils=types.SimpleNamespace(register_class=registered.append, unregister_class=lambda c: None))
prop = lambda *a, **k: ("prop", k)
mod("bpy.props", **{n: prop for n in ("BoolProperty", "CollectionProperty", "EnumProperty",
    "FloatProperty", "IntProperty", "PointerProperty", "StringProperty")})
bpy.props = sys.modules["bpy.props"]
mod("bpy_extras"); mod("bpy_extras.io_utils", ImportHelper=type("ImportHelper", (), {}))
mod("mathutils", Matrix=_Any)
mod("bpy.app"); mod("bpy.app.handlers", persistent=lambda f: f, load_post=[])
bpy.app = types.SimpleNamespace(handlers=sys.modules["bpy.app.handlers"], version=(4, 2, 0), background=True,
                                timers=_Any(), version_string="4.2.0")
sys.argv = ["blender"]

sys.path.insert(0, ROOT)
import importlib
import neural_texture
for m in ("properties", "operators", "ui", "pipeline", "blender.scope", "blender.imageio",
          "blender.shaders", "blender.inspect_asset", "blender.guidance", "blender.uv_bake",
          "blender.material_build", "blender.runner", "blender.cleanup", "blender.compat",
          "blender.live_server", "bridge.generate", "bridge.workflow", "bridge.comfyui",
          "bridge.comfy_generator", "bridge.compat", "cli"):
    importlib.import_module("neural_texture." + m)
neural_texture.register()
names = sorted(c.__dict__.get("bl_idname", c.__name__) for c in registered)
assert len(names) == 28, names
# every operator referenced in the UI must exist
import re, pathlib
ui = pathlib.Path(ROOT, "neural_texture", "ui.py").read_text()
used = set(re.findall(r'operator\("([\w.]+)"', ui))
assert used <= set(names), used - set(names)
# every settings property referenced in the UI must be declared
props = set(neural_texture.properties.NT_Settings.__annotations__)
refs = set(re.findall(r'prop\(s, "(\w+)"', ui))
for grp in re.findall(r'for p in \(([^)]*)\)', ui, re.S):
    refs |= set(grp.replace('"', '').replace('\n', ' ').split(','))
refs = {r.strip() for r in refs}
refs.discard("")
assert refs <= props, refs - props
print("OK", len(names))
''')


def test_bpy_modules_import_with_stub():
    r = subprocess.run([sys.executable, "-c", f"ROOT={ROOT!r}\n" + STUB], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK 28" in r.stdout
