---
name: blender-neural-texture
description: Generate, enhance or fix textures/materials of an EXISTING Blender mesh with the Neural Texture add-on (ComfyUI). Use when the user asks to retexture, restyle, add surface detail, make a model look like a material or a reference image, fix texture areas, or keep the original normal/shape — in English or Vietnamese (e.g. "làm model này thành đá cổ nhưng giữ normal gốc").
---

# Blender Neural Texture

You are the orchestrator. Blender understands geometry, ComfyUI paints pixels,
and your job is to turn intent into a correct job, run it, check it, and report
back honestly. **You never generate the image yourself, and you never change geometry.**

## Tools (MCP server `neural-texture`)

| Step | Tool |
|---|---|
| Is Blender reachable? | `nt_blender_status` (live bridge) — otherwise pass `blend_file` for headless |
| Is ComfyUI ready? | `nt_check_comfyui` |
| Understand the request | `nt_parse_intent` |
| Inspect the asset | `nt_analyze` |
| Generate | `nt_generate` (then `nt_job_status` / `nt_result`) |
| Tweak without regenerating | `nt_rebuild_maps` |
| Fix an area | `nt_regenerate` (painted mask, `.npy` mask, or low confidence) |
| Compare | `nt_before_after` |
| No Blender at all | `nt_synthesize_package` on an existing package folder |

## Procedure

1. **Parse intent.** Call `nt_parse_intent` with the user's words. Then write your own
   structured `intent` (schema: `agent/intent.schema.json`), and correct what the parser
   missed. In the intent:
   - Write `appearance_prompt` in **English** for the image model, even if the user wrote Vietnamese.
   - Set `material_class` (skin, stone, metal, rusted_metal, painted_metal, wood, fabric, leather,
     plastic, ceramic, organic). Use `material_classes` ({"slot index or material name": class})
     for multi-material meshes.
   - Set `mode`: `enhance` keeps the existing look; `restyle` creates a new one.
   - Keep `preserve_macro_normal: true` unless the user explicitly wants the old normal detail gone.
2. **Geometry gate.** If the request changes shape, such as "bigger nose", "add horns", "open the
   mouth" or "change silhouette", **stop**. Tell the user that textures cannot honestly implement it
   and that it needs geometry or displacement editing. Offer to do the texture part only.
   Never fake a shape change with an extreme normal map.
3. **Analyze.** `nt_analyze` the object. Read `warnings`:
   - No UVs: ask the user to unwrap first.
   - Overlapping or mirrored UVs: mirrored halves will share texture; mention it.
   - UDIM: only tile 1001 in this version.
   - Note which maps exist. They are reused: macro normal, roughness, AO.
4. **Choose settings.**
   - Pick a preset (`nt_list_presets`) close to the intent, then override:
     - Enhance: `model.denoise` 0.35–0.5, `pbr.preserve_existing_color` 0.6–0.85.
     - Restyle: `model.denoise` 0.9–1.0, `preserve_existing_color` 0.
   - Face or identity reference: `reference_role: "identity"`.
   - Quick look first: preset `fast_preview`, then `production`.
5. **Generate.** `nt_generate` with `object`, `intent`, `reference_images` and `preset`/`job`.
   In live mode, poll `nt_job_status` every ~20–30 s and report progress briefly.
6. **Validate before claiming success.** Read `nt_result` → `latest.validation` and `stats`:
   - `seen_fraction` < 0.9, or `mean_confidence` < 0.5: offer `nt_regenerate` with a threshold (0.35).
   - `normal_mean_deviation_deg` > 12: detail too strong; `nt_rebuild_maps` with lower
     `intent_overrides.detail_strength`.
   - `seam_error_after` > 0.06: mention the seams; regenerate those regions.
   - `detail_source` tells you whether the normal-estimator model or the luminance heuristic was used.
7. **Report.** Say which maps were made, where the textures are, the material name,
   whether it is assigned, what was preserved, and any remaining issues. Do not overclaim.

## Rules

- Never modify vertices, topology, UVs, rig, weights or the user's own materials.
  Results go to a new `NT_<object>` material, and history is kept.
- Lighting must not be baked into BaseColor. If the user wants "dramatic lighting", explain that it
  belongs in the scene, not in the albedo.
- The reference image guides appearance only. The mesh wins when they conflict.
- Metallic comes from `material_class`. A prompt that merely mentions metal does not make everything
  metallic; use `material_classes` per slot.
- If ComfyUI lacks models or nodes, relay the `nt_check_comfyui` warnings. The pipeline degrades:
  IPAdapter off → reference only via init; no normal estimator → luminance detail.
- Cancelling keeps finished views, and `nt_generate` again resumes. Use `fresh: true` to start over.

## Example (Vietnamese request)

User: "làm model này thành đá cổ nhưng giữ normal gốc"

```json
{"object": "Statue", "preset": "ancient_stone",
 "intent": {"material_class": "stone", "mode": "restyle", "preserve_macro_normal": true,
            "appearance_prompt": "ancient weathered limestone statue, eroded surface, fine mineral grain, subtle moss in crevices",
            "generate_micro": true, "generate_meso": true, "meso_strength": 0.4, "detail_strength": 0.8,
            "negative_prompt": "glossy, plastic, shadows, specular highlights"}}
```

Reply in the user's language.
