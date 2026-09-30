"""Sidebar UI: 3D Viewport > N panel > Neural Texture (V4)."""

from __future__ import annotations

import os

import bpy

from .blender import compat, material_build, runner
from .core.camera import DEFAULT_VIEWS


def _wrap(text, width=44):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width and line:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out[:8]


class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Neural Texture"


class NT_UL_history(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index=0):
        row = layout.row(align=True)
        row.label(text=item.run_id, icon="CHECKMARK" if item.ok else "ERROR")
        row.label(text=item.label)


class NT_PT_main(_Base, bpy.types.Panel):
    bl_label = "Neural Texture"
    bl_idname = "NT_PT_main"

    def draw(self, context):
        s = context.scene.neural_texture
        obj = context.active_object
        col = self.layout.column()
        if obj is None or obj.type != "MESH":
            col.label(text="Select a mesh object", icon="INFO")
        else:
            col.label(text=obj.name, icon="MESH_DATA")

        col.prop(s, "preset", text="")
        col.prop(s, "prompt", text="", icon="TEXT")
        box = col.box()
        row = box.row()
        row.label(text="Reference Images", icon="IMAGE_DATA")
        row.operator("neural_texture.add_reference", text="", icon="ADD")
        for i, r in enumerate(s.references):
            rr = box.row(align=True)
            rr.prop(r, "filepath", text="")
            rr.operator("neural_texture.remove_reference", text="", icon="X").index = i

        job = runner.current()
        col.separator()
        if job is not None and job.running:
            if compat.has_progress_widget():
                col.progress(factor=job.progress, type="BAR", text=job.message[:48] or job.name)
            else:
                col.label(text=f"{job.progress:.0%}  {job.message[:40]}")
            col.operator("neural_texture.cancel", icon="CANCEL")
        else:
            big = col.row()
            big.scale_y = 1.6
            big.operator("neural_texture.generate", icon="SHADING_TEXTURE")
            row = col.row(align=True)
            row.operator("neural_texture.analyze", icon="VIEWZOOM")
            row.operator("neural_texture.rebuild", icon="FILE_REFRESH")

        if s.last_status:
            b = col.box()
            for line in _wrap(s.last_status):
                b.label(text=line)


class NT_PT_look(_Base, bpy.types.Panel):
    bl_label = "Look"
    bl_parent_id = "NT_PT_main"

    def draw(self, context):
        s = context.scene.neural_texture
        col = self.layout.column(align=True)
        col.prop(s, "material_class")
        col.separator()
        for p in ("creativity", "reference_weight", "detail_strength", "keep_colors", "consistency",
                  "roughness_bias", "delight"):
            col.prop(s, p, slider=True)
        col.label(text="Tip: change Detail/Keep Colours, then Rebuild Maps", icon="INFO")


class NT_PT_result(_Base, bpy.types.Panel):
    bl_label = "Result & History"
    bl_parent_id = "NT_PT_main"

    def draw(self, context):
        s = context.scene.neural_texture
        obj = context.active_object
        col = self.layout.column()
        showing_after = obj is not None and material_build.ORIGINAL_KEY in obj
        row = col.row(align=True)
        row.operator("neural_texture.before_after", text="Showing: After" if showing_after else "Showing: Before",
                     icon="HIDE_OFF" if showing_after else "HIDE_ON", depress=showing_after)
        row.operator("neural_texture.show_map", text="", icon="IMAGE").key = "preview"
        row.operator("neural_texture.show_map", text="", icon="MOD_MASK").key = "confidence"
        row = col.row(align=True)
        row.label(text="Versions")
        row.operator("neural_texture.history_refresh", text="", icon="FILE_REFRESH")
        col.template_list("NT_UL_history", "", s, "history", s, "history_index", rows=3)
        col.operator("neural_texture.history_restore", icon="RECOVER_LAST")
        col.operator("neural_texture.open_output", icon="FILE_FOLDER")


class NT_PT_fix(_Base, bpy.types.Panel):
    bl_label = "Fix Areas"
    bl_parent_id = "NT_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.neural_texture
        col = self.layout.column(align=True)
        row = col.row(align=True)
        row.operator("neural_texture.paint_mask", icon="BRUSH_DATA")
        row.operator("neural_texture.clear_mask", text="", icon="TRASH")
        col.operator("neural_texture.regenerate_masked", icon="SELECT_SUBTRACT")
        col.separator()
        col.prop(s, "low_conf_threshold", slider=True)
        col.operator("neural_texture.regenerate_low_confidence", icon="ERROR")


class NT_PT_advanced(_Base, bpy.types.Panel):
    bl_label = "Advanced"
    bl_parent_id = "NT_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.neural_texture
        col = self.layout.column()
        col.prop(s, "generator")
        box = col.box()
        box.label(text="Views", icon="CAMERA_DATA")
        grid = box.grid_flow(columns=2, align=True)
        for v in DEFAULT_VIEWS:
            grid.prop(s, f"view_{v}")
        box.prop(s, "use_planner")
        box.prop(s, "max_views")
        box.prop(s, "orthographic")
        box = col.box()
        box.label(text="Quality", icon="SETTINGS")
        for p in ("view_resolution", "texture_resolution", "ao_samples", "uv_map", "use_gpu", "output_dir",
                  "assign_material", "keep_temp_scene"):
            box.prop(s, p)


class NT_PT_model(_Base, bpy.types.Panel):
    bl_label = "ComfyUI"
    bl_parent_id = "NT_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.neural_texture
        col = self.layout.column()
        row = col.row(align=True)
        row.prop(s, "server_url", text="")
        row.operator("neural_texture.check_server", text="", icon="URL")
        col.prop(s, "auto_detect")
        for p in ("checkpoint", "controlnet", "controlnet_is_union", "controlnet_normal", "depth_strength",
                  "normal_strength", "use_ipadapter", "normal_estimator", "steps", "cfg", "seed",
                  "custom_workflow"):
            col.prop(s, p)
        col.separator()
        col.operator("neural_texture.agent_bridge", icon="LINKED")


CLASSES = (NT_UL_history, NT_PT_main, NT_PT_look, NT_PT_result, NT_PT_fix, NT_PT_advanced, NT_PT_model)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
