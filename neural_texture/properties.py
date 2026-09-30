"""Add-on settings stored on the Scene (UI state only; no geometry data)."""

from __future__ import annotations

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)

from .core.camera import DEFAULT_VIEWS
from .core.config import JobConfig, ModelConfig
from .core.materials import CLASSES
from .core.presets import PRESETS

_RES = [(str(r), str(r), "") for r in (512, 768, 1024, 1536, 2048, 4096)]
_PRESETS = [("none", "Custom", "Use the sliders as they are")] + \
    [(k, v["label"], v["description"]) for k, v in PRESETS.items()]
_MATERIALS = [("auto", "Auto (from prompt/names)", "")] + \
    [(k, k.replace("_", " ").title(), "") for k in CLASSES if k != "generic"]
_GENERATORS = [("comfyui", "ComfyUI", "Generate with the configured ComfyUI server"),
               ("identity", "No AI (existing look)", "Re-project the existing albedo; tests the pipeline"),
               ("folder", "Images from folder", "Use views/<view>/generated.png produced elsewhere")]


def _apply_preset(self, context):
    if self.preset == "none":
        return
    vals = PRESETS[self.preset]["values"]
    mapping = {"model.denoise": "creativity", "model.ipadapter_weight": "reference_weight",
               "pbr.preserve_existing_color": "keep_colors", "detail_strength": "detail_strength",
               "view_resolution": "view_resolution", "texture_resolution": "texture_resolution",
               "synth.max_views": "max_views", "model.steps": "steps", "ao_samples": "ao_samples"}
    for key, prop in mapping.items():
        if key in vals:
            v = vals[key]
            setattr(self, prop, str(v) if prop.endswith("resolution") else v)
    if "intent.material_class" in vals:
        self.material_class = vals["intent.material_class"]


class NT_Reference(bpy.types.PropertyGroup):
    filepath: StringProperty(name="Image", subtype="FILE_PATH")


class NT_HistoryItem(bpy.types.PropertyGroup):
    run_id: StringProperty()
    label: StringProperty()
    ok: BoolProperty()


class NT_Settings(bpy.types.PropertyGroup):
    # Inputs
    prompt: StringProperty(name="Prompt", description="What should the surface look like? (EN / VI)")
    references: CollectionProperty(type=NT_Reference)
    preset: EnumProperty(name="Preset", items=_PRESETS, default="none", update=_apply_preset)
    material_class: EnumProperty(name="Material", items=_MATERIALS, default="auto")
    generator: EnumProperty(name="Generator", items=_GENERATORS, default="comfyui")
    output_dir: StringProperty(name="Output", subtype="DIR_PATH", default="//neural_texture_out")

    # Simple sliders
    creativity: FloatProperty(name="Creativity", default=0.75, min=0.05, max=1.0, subtype="FACTOR",
                              description="Low = enhance existing look, high = new look")
    reference_weight: FloatProperty(name="Reference", default=0.7, min=0.0, max=1.5, subtype="FACTOR",
                                    description="How strongly reference images drive the look")
    detail_strength: FloatProperty(name="Detail", default=0.6, min=0.0, max=1.0, subtype="FACTOR",
                                   description="Strength of generated detail normal (macro normal is always kept)")
    keep_colors: FloatProperty(name="Keep Colours", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
                               description="Keep the existing large-scale colour/tone")
    consistency: FloatProperty(name="Consistency", default=0.8, min=0.0, max=1.0, subtype="FACTOR",
                               description="How strongly later views must agree with earlier ones")
    roughness_bias: FloatProperty(name="Roughness Bias", default=0.0, min=-0.4, max=0.4)
    delight: FloatProperty(name="Remove Lighting", default=0.6, min=0.0, max=1.0, subtype="FACTOR")

    # Views / quality
    view_front: BoolProperty(name="Front", default=True)
    view_back: BoolProperty(name="Back", default=True)
    view_left: BoolProperty(name="Left", default=True)
    view_right: BoolProperty(name="Right", default=True)
    view_top: BoolProperty(name="Top", default=True)
    view_bottom: BoolProperty(name="Bottom", default=True)
    view_front_left: BoolProperty(name="Front Left", default=True)
    view_front_right: BoolProperty(name="Front Right", default=True)
    use_planner: BoolProperty(name="Smart View Planning", default=True,
                              description="Render extra candidate views and pick the best order automatically")
    max_views: IntProperty(name="Max Generated Views", default=10, min=1, max=18)
    view_resolution: EnumProperty(name="View Res", items=_RES, default="1024")
    texture_resolution: EnumProperty(name="Texture Res", items=_RES, default="2048")
    orthographic: BoolProperty(name="Orthographic Views", default=True)
    uv_map: StringProperty(name="UV Map", description="Empty = active render UV map")
    use_gpu: BoolProperty(name="GPU Render", default=False)
    ao_samples: IntProperty(name="AO Samples", default=32, min=1, max=512)

    # Model
    server_url: StringProperty(name="ComfyUI URL", default=ModelConfig.server_url)
    checkpoint: StringProperty(name="Checkpoint", default=ModelConfig.checkpoint)
    controlnet: StringProperty(name="ControlNet", default=ModelConfig.controlnet)
    controlnet_is_union: BoolProperty(name="Union ControlNet", default=True)
    controlnet_normal: StringProperty(name="Normal ControlNet")
    depth_strength: FloatProperty(name="Depth Guidance", default=ModelConfig.depth_strength, min=0, max=2)
    normal_strength: FloatProperty(name="Normal Guidance", default=ModelConfig.normal_strength, min=0, max=2)
    use_ipadapter: BoolProperty(name="IPAdapter (reference)", default=True)
    normal_estimator: StringProperty(name="Normal Estimator", default="auto",
                                     description="'auto', 'off' or a ComfyUI node class name")
    auto_detect: BoolProperty(name="Auto-detect Models", default=True)
    steps: IntProperty(name="Steps", default=ModelConfig.steps, min=1, max=150)
    cfg: FloatProperty(name="CFG", default=ModelConfig.cfg, min=0, max=30)
    seed: IntProperty(name="Seed", default=ModelConfig.seed, min=0)
    custom_workflow: StringProperty(name="Custom Workflow", subtype="FILE_PATH")

    # Output / state
    assign_material: BoolProperty(name="Assign Result", default=True,
                                  description="Show the result on the object (originals are recorded; use Before/After)")
    keep_temp_scene: BoolProperty(name="Keep Temp Scene (debug)", default=False)
    show_advanced: BoolProperty(name="Advanced", default=False)
    history: CollectionProperty(type=NT_HistoryItem)
    history_index: IntProperty(default=0)
    low_conf_threshold: FloatProperty(name="Low Confidence", default=0.35, min=0.05, max=0.95)
    last_status: StringProperty(default="")

    def to_job(self, obj) -> JobConfig:
        views = [v for v in DEFAULT_VIEWS if getattr(self, f"view_{v}")]
        m = ModelConfig(
            server_url=self.server_url, checkpoint=self.checkpoint, controlnet=self.controlnet,
            controlnet_is_union=self.controlnet_is_union, controlnet_normal=self.controlnet_normal,
            depth_strength=self.depth_strength, normal_strength=self.normal_strength,
            use_ipadapter=self.use_ipadapter, ipadapter_weight=self.reference_weight,
            denoise=self.creativity, consistency_denoise=min(1.0, 0.6 + 0.4 * self.creativity),
            steps=self.steps, cfg=self.cfg, seed=self.seed, normal_estimator=self.normal_estimator,
            auto_detect_models=self.auto_detect,
            custom_workflow=bpy.path.abspath(self.custom_workflow) if self.custom_workflow else "")
        cfg = JobConfig(
            object_name=obj.name if obj else "", output_dir=self.output_dir, prompt=self.prompt,
            reference_images=[bpy.path.abspath(r.filepath) for r in self.references if r.filepath],
            views=views or ["front"], use_planner=self.use_planner,
            view_resolution=int(self.view_resolution), texture_resolution=int(self.texture_resolution),
            orthographic=self.orthographic, uv_map=self.uv_map, use_gpu=self.use_gpu,
            ao_samples=self.ao_samples, keep_temp_scene=self.keep_temp_scene,
            synth={"max_views": self.max_views, "lock": 0.9 * self.consistency,
                   "harmonize_strength": self.consistency},
            pbr={"preserve_existing_color": self.keep_colors, "delight_amount": self.delight},
            model=m, assign_material=self.assign_material,
            preset="" if self.preset == "none" else self.preset)
        cfg.intent_overrides["detail_strength"] = self.detail_strength
        if self.roughness_bias:
            cfg.intent_overrides["roughness_offset"] = self.roughness_bias
        if self.material_class != "auto":
            cfg.intent_overrides["material_class"] = self.material_class
        if cfg.preset:
            from .core.presets import apply_preset
            apply_preset(cfg, cfg.preset)
            # sliders win over preset values the user has since changed
            cfg.model.denoise = self.creativity
            cfg.intent_overrides["detail_strength"] = self.detail_strength
            cfg.pbr["preserve_existing_color"] = self.keep_colors
        return cfg


CLASSES_ = (NT_Reference, NT_HistoryItem, NT_Settings)


def register():
    for c in CLASSES_:
        bpy.utils.register_class(c)
    bpy.types.Scene.neural_texture = PointerProperty(type=NT_Settings)


def unregister():
    del bpy.types.Scene.neural_texture
    for c in reversed(CLASSES_):
        bpy.utils.unregister_class(c)
