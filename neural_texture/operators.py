"""Blender operators (V4 UX). Long work runs through blender.runner.JobRunner:
the UI stays responsive, progress is shown in the panel, Cancel stops it."""

from __future__ import annotations

import os
import traceback

import bpy
import numpy as np
from bpy_extras.io_utils import ImportHelper

from . import pipeline
from .blender import compat, material_build, runner
from .bridge.comfyui import ComfyClient, ComfyError
from .core import synth


def _obj(context):
    obj = context.active_object
    return obj if obj is not None and obj.type == "MESH" else None


def _settings(context):
    return context.scene.neural_texture


def _busy() -> bool:
    r = runner.current()
    return r is not None and r.running


def _done_callback(scene_name: str):
    def cb(job):
        sc = bpy.data.scenes.get(scene_name)
        if sc is None:
            return
        s = sc.neural_texture
        if job.status == "done":
            s.last_status = f"{job.name}: done"
        elif job.status == "cancelled":
            s.last_status = f"{job.name}: cancelled (progress saved; Generate resumes)"
        else:
            s.last_status = f"{job.name}: {job.error}"
        _refresh_history(sc)
    return cb


def _refresh_history(scene):
    s = scene.neural_texture
    obj = bpy.context.view_layer.objects.active if bpy.context.view_layer else None
    s.history.clear()
    if obj is None or obj.type != "MESH":
        return
    out = pipeline.output_dir(s.to_job(obj), obj)
    for h in reversed(synth.history(out)):
        it = s.history.add()
        it.run_id, it.label, it.ok = h["run_id"], h["label"], bool(h.get("ok"))


class _NeedsMesh:
    @classmethod
    def poll(cls, context):
        return _obj(context) is not None and not _busy()


def _start(op, context, job):
    try:
        job.start()
    except RuntimeError as e:
        op.report({"ERROR"}, str(e))
        return {"CANCELLED"}
    _settings(context).last_status = f"{job.name}: started"
    return {"FINISHED"}


def _guard(op, context, fn):
    s = _settings(context)
    try:
        return fn()
    except pipeline.StructuralWarning as e:
        s.last_status = "Needs geometry editing, not textures: " + str(e)
        op.report({"WARNING"}, s.last_status)
    except (ComfyError, RuntimeError, FileNotFoundError, ValueError) as e:
        traceback.print_exc()
        s.last_status = f"Error: {e}"
        op.report({"ERROR"}, str(e)[:500])
    return {"CANCELLED"}


# ----------------------------------------------------------------------------- main
class NT_OT_generate(_NeedsMesh, bpy.types.Operator):
    bl_idname = "neural_texture.generate"
    bl_label = "Generate Texture Set"
    bl_description = "Render guidance, generate consistent views with AI, bake a PBR set (resumes if interrupted)"
    force_guidance: bpy.props.BoolProperty(default=False, options={"SKIP_SAVE"})
    fresh: bpy.props.BoolProperty(name="Start Over", default=False, options={"SKIP_SAVE"})

    def execute(self, context):
        s, obj = _settings(context), _obj(context)
        return _guard(self, context, lambda: _start(self, context, pipeline.build_generate_job(
            obj, s.to_job(obj), s.generator, self.force_guidance, resume=not self.fresh,
            on_done=_done_callback(context.scene.name))))


class NT_OT_cancel(bpy.types.Operator):
    bl_idname = "neural_texture.cancel"
    bl_label = "Cancel"
    bl_description = "Stop the running job (completed views are kept)"

    @classmethod
    def poll(cls, context):
        return _busy()

    def execute(self, context):
        runner.current().cancel()
        return {"FINISHED"}


class NT_OT_analyze(_NeedsMesh, bpy.types.Operator):
    bl_idname = "neural_texture.analyze"
    bl_label = "Analyze"
    bl_description = "Inspect mesh, UVs, existing maps and the prompt (read-only)"

    def execute(self, context):
        s, obj = _settings(context), _obj(context)

        def run():
            cfg = s.to_job(obj)
            r = pipeline.stage_analyze(obj, cfg, pipeline.output_dir(cfg, obj))
            it = r["intent"]
            msg = (f"{r['mesh']['triangles']} tris, {len(r['warnings'])} warning(s); "
                   f"material: {it.get('material_class') or 'auto'}, mode: {it.get('mode')}")
            if r["requires_geometry_mode"]:
                msg += " | GEOMETRY EDIT REQUESTED: " + "; ".join(w["clause"] for w in it["geometry_warnings"])
            s.last_status = msg
            self.report({"WARNING" if r["warnings"] or r["requires_geometry_mode"] else "INFO"}, msg)
            return {"FINISHED"}
        return _guard(self, context, run)


class NT_OT_rebuild(_NeedsMesh, bpy.types.Operator):
    bl_idname = "neural_texture.rebuild"
    bl_label = "Rebuild Maps"
    bl_description = "Re-bake PBR maps with the current sliders without regenerating (seconds)"

    def execute(self, context):
        s, obj = _settings(context), _obj(context)
        return _guard(self, context, lambda: _start(self, context, pipeline.build_rebuild_job(
            obj, s.to_job(obj), on_done=_done_callback(context.scene.name))))


# ----------------------------------------------------------------------------- fix regions
class NT_OT_paint_mask(_NeedsMesh, bpy.types.Operator):
    bl_idname = "neural_texture.paint_mask"
    bl_label = "Paint Fix Mask"
    bl_description = "Paint white over areas to regenerate (texture paint on a separate mask image)"

    def execute(self, context):
        s, obj = _settings(context), _obj(context)
        res = int(s.texture_resolution)
        name = pipeline.mask_image_name(obj)
        img = bpy.data.images.get(name)
        if img is None or tuple(img.size) != (res, res):
            if img is not None:
                bpy.data.images.remove(img)
            img = bpy.data.images.new(name, res, res, alpha=False)
            img.generated_color = (0, 0, 0, 1)
            compat.set_non_color(img)
        me = obj.data
        render_uv = next((l for l in me.uv_layers if l.active_render), None)
        if render_uv is not None and me.uv_layers.active != render_uv:
            self.report({"WARNING"}, f"Painting uses the active UV map '{me.uv_layers.active.name}', "
                                     f"textures use '{render_uv.name}'. Make them the same for accurate masks.")
        bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
        ip = context.tool_settings.image_paint
        ip.mode = "IMAGE"
        ip.canvas = img
        compat.brush_color_white(context)
        s.last_status = "Paint white where the texture should be regenerated, then 'Regenerate Masked'."
        return {"FINISHED"}


class NT_OT_clear_mask(_NeedsMesh, bpy.types.Operator):
    bl_idname = "neural_texture.clear_mask"
    bl_label = "Clear Mask"

    def execute(self, context):
        img = bpy.data.images.get(pipeline.mask_image_name(_obj(context)))
        if img is not None:
            px = np.zeros(len(img.pixels), np.float32)
            px[3::4] = 1.0
            img.pixels.foreach_set(px)
            img.update()
        return {"FINISHED"}


class NT_OT_regenerate_masked(_NeedsMesh, bpy.types.Operator):
    bl_idname = "neural_texture.regenerate_masked"
    bl_label = "Regenerate Masked"
    bl_description = "Regenerate only the painted area, consistent with the rest"

    def execute(self, context):
        s, obj = _settings(context), _obj(context)
        mask = pipeline.read_mask_image(obj)
        if mask is None or mask.max() < 0.5:
            self.report({"WARNING"}, "Paint a mask first (white = regenerate).")
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        return _guard(self, context, lambda: _start(self, context, pipeline.build_regenerate_job(
            obj, s.to_job(obj), mask, s.generator, on_done=_done_callback(context.scene.name))))


class NT_OT_regenerate_low_conf(_NeedsMesh, bpy.types.Operator):
    bl_idname = "neural_texture.regenerate_low_confidence"
    bl_label = "Fix Low Confidence"
    bl_description = "Regenerate texels where views disagreed or coverage was thin"

    def execute(self, context):
        s, obj = _settings(context), _obj(context)

        def run():
            cfg = s.to_job(obj)
            out = pipeline.output_dir(cfg, obj)
            mask = synth.low_confidence_mask(out, cfg, s.low_conf_threshold)
            if mask.sum() == 0:
                s.last_status = "No low-confidence texels found."
                return {"FINISHED"}
            return _start(self, context, pipeline.build_regenerate_job(
                obj, cfg, mask.astype(np.float32), s.generator, on_done=_done_callback(context.scene.name)))
        return _guard(self, context, run)


# ----------------------------------------------------------------------------- result / history
class NT_OT_before_after(_NeedsMesh, bpy.types.Operator):
    bl_idname = "neural_texture.before_after"
    bl_label = "Before / After"
    bl_description = "Toggle between the original materials and the generated one"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = _obj(context)
        if material_build.ORIGINAL_KEY in obj:
            material_build.restore_materials(obj)
            _settings(context).last_status = "Showing: BEFORE (original materials)"
            return {"FINISHED"}
        mat = bpy.data.materials.get(f"NT_{obj.name}")
        if mat is None:
            self.report({"WARNING"}, "No generated material yet.")
            return {"CANCELLED"}
        material_build.assign_material(obj, mat)
        _settings(context).last_status = "Showing: AFTER (generated material)"
        return {"FINISHED"}


class NT_OT_history_refresh(bpy.types.Operator):
    bl_idname = "neural_texture.history_refresh"
    bl_label = "Refresh History"

    def execute(self, context):
        _refresh_history(context.scene)
        return {"FINISHED"}


class NT_OT_history_restore(_NeedsMesh, bpy.types.Operator):
    bl_idname = "neural_texture.history_restore"
    bl_label = "Use This Version"
    bl_description = "Make the selected history version the current material"

    def execute(self, context):
        s, obj = _settings(context), _obj(context)
        if not s.history or s.history_index >= len(s.history):
            return {"CANCELLED"}
        cfg = s.to_job(obj)
        out = pipeline.output_dir(cfg, obj)
        synth.set_latest(out, s.history[s.history_index].run_id)
        pipeline.apply_material(obj, out, cfg)
        s.last_status = f"Restored version {s.history[s.history_index].run_id}"
        return {"FINISHED"}


class NT_OT_show_map(_NeedsMesh, bpy.types.Operator):
    bl_idname = "neural_texture.show_map"
    bl_label = "Show Map"
    bl_description = "Open a generated map (e.g. confidence, preview) in an Image Editor"
    key: bpy.props.StringProperty(default="confidence")

    def execute(self, context):
        s, obj = _settings(context), _obj(context)
        out = pipeline.output_dir(s.to_job(obj), obj)
        res = synth.latest_result(out)
        if not res:
            return {"CANCELLED"}
        path = os.path.join(res["history_dir"], "preview.png") if self.key == "preview" \
            else os.path.join(out, res["textures"].get(self.key, ""))
        if not os.path.isfile(path):
            self.report({"WARNING"}, f"No '{self.key}' map in the latest result.")
            return {"CANCELLED"}
        img = bpy.data.images.load(path, check_existing=True)
        img.reload()
        for area in context.screen.areas:
            if area.type == "IMAGE_EDITOR":
                area.spaces.active.image = img
                break
        else:
            self.report({"INFO"}, f"Loaded '{img.name}'. Open an Image Editor to view it.")
        return {"FINISHED"}


class NT_OT_restore_materials(bpy.types.Operator):
    bl_idname = "neural_texture.restore_materials"
    bl_label = "Restore Original Materials"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        o = _obj(context)
        return o is not None and material_build.ORIGINAL_KEY in o

    def execute(self, context):
        material_build.restore_materials(_obj(context))
        return {"FINISHED"}


# ----------------------------------------------------------------------------- setup
class NT_OT_check_server(bpy.types.Operator):
    bl_idname = "neural_texture.check_server"
    bl_label = "Check ComfyUI"
    bl_description = "Ping ComfyUI, detect installed models/nodes and fill the settings"

    def execute(self, context):
        s = _settings(context)
        try:
            from .bridge import compat as ccompat
            cli = ComfyClient(s.server_url, timeout=5)
            vram = cli.vram()
            m = s.to_job(context.active_object).model
            if s.auto_detect:
                rep = ccompat.autodetect(m, cli.object_info(), vram.get("total_gb", 0))
                s.checkpoint, s.controlnet = m.checkpoint, m.controlnet
                s.controlnet_is_union, s.controlnet_normal = m.controlnet_is_union, m.controlnet_normal
                s.use_ipadapter, s.normal_estimator = m.use_ipadapter, m.normal_estimator
                warn = " | ".join(rep.get("warnings", []))
            else:
                warn = ""
            s.last_status = f"ComfyUI OK ({vram.get('name', '?')}, {vram.get('total_gb', 0):.0f} GB)" + \
                (f" | {warn}" if warn else "")
            self.report({"INFO"}, s.last_status[:300])
        except ComfyError as e:
            s.last_status = str(e)
            self.report({"ERROR"}, str(e))
        return {"FINISHED"}


class NT_OT_add_reference(bpy.types.Operator, ImportHelper):
    bl_idname = "neural_texture.add_reference"
    bl_label = "Add Reference Image"
    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg;*.webp;*.tif;*.tiff;*.exr",
                                          options={"HIDDEN"})
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    directory: bpy.props.StringProperty(subtype="DIR_PATH")

    def execute(self, context):
        refs = _settings(context).references
        names = [f.name for f in self.files if f.name] or [os.path.basename(self.filepath)]
        for n in names:
            refs.add().filepath = os.path.join(self.directory or os.path.dirname(self.filepath), n)
        return {"FINISHED"}


class NT_OT_remove_reference(bpy.types.Operator):
    bl_idname = "neural_texture.remove_reference"
    bl_label = "Remove Reference"
    index: bpy.props.IntProperty()

    def execute(self, context):
        _settings(context).references.remove(self.index)
        return {"FINISHED"}


class NT_OT_open_output(_NeedsMesh, bpy.types.Operator):
    bl_idname = "neural_texture.open_output"
    bl_label = "Open Output Folder"

    def execute(self, context):
        obj = _obj(context)
        bpy.ops.wm.path_open(filepath=pipeline.output_dir(_settings(context).to_job(obj), obj))
        return {"FINISHED"}


class NT_OT_agent_bridge(bpy.types.Operator):
    bl_idname = "neural_texture.agent_bridge"
    bl_label = "Agent Bridge"
    bl_description = "Start/stop the local socket the Claude/Codex MCP server uses to drive this Blender"

    def execute(self, context):
        from .blender import live_server
        if live_server.is_running():
            live_server.stop()
            _settings(context).last_status = "Agent bridge stopped"
        else:
            port = live_server.start()
            _settings(context).last_status = f"Agent bridge listening on 127.0.0.1:{port}"
        return {"FINISHED"}


CLASSES = (NT_OT_generate, NT_OT_cancel, NT_OT_analyze, NT_OT_rebuild, NT_OT_paint_mask, NT_OT_clear_mask,
           NT_OT_regenerate_masked, NT_OT_regenerate_low_conf, NT_OT_before_after, NT_OT_history_refresh,
           NT_OT_history_restore, NT_OT_show_map, NT_OT_restore_materials, NT_OT_check_server,
           NT_OT_add_reference, NT_OT_remove_reference, NT_OT_open_output, NT_OT_agent_bridge)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
