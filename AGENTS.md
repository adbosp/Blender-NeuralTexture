# Blender Neural Texture: agent instructions

This repo is a Blender add-on plus an AI pipeline. It regenerates or enhances the **appearance** of an
**existing** mesh. Geometry is the structural guidance and is never modified.

## When a user asks for texture work

Use the skill `agent/skills/blender-neural-texture/SKILL.md` and the MCP server
`mcp_server/neural_texture_mcp.py`. Register it once:

```bash
claude mcp add neural-texture -e NT_BLENDER=/path/to/blender -- python mcp_server/neural_texture_mcp.py
```

## Hard rules (from the project spec)

1. The LLM orchestrates. ComfyUI generates the pixels, and Blender handles geometry, passes and materials.
2. Keep the macro normal: `FinalNormal = RNM(existing, generated_detail, strength)`. Never replace it blindly.
3. Do not modify vertices, silhouette, topology, rig, weights or UVs. Refuse shape changes; they need
   geometry mode (see `geometry_warnings`).
4. Keep lighting out of BaseColor.
5. Reference images are appearance guidance. When they conflict with the mesh, the mesh wins.
6. Do not discard existing maps. Reuse them for fallback, macro normal, roughness and AO.
7. Leave the user's scene untouched. Use temporary scenes and clean them up.

## Code map

- `neural_texture/core/`: pure numpy; testable without Blender.
  - `synth.py`: sequential consistent generation, resume, region regeneration, finalize/history.
  - `accumulate.py`: UV accumulation and the confidence map.
  - `planner.py`: view selection.
  - `consistency.py`: inpaint masks and colour harmonization.
  - `detail.py`: detail-normal sources, unblend, Frankot–Chellappa.
  - `pbr.py`: material-aware PBR maps.
  - `seams.py`: UV seam refinement.
  - `materials.py`: material classes (EN/VI).
  - `intent.py`: prompt parser (EN/VI) and the geometry gate.
- `neural_texture/bridge/`: ComfyUI client, graph builders, model auto-detection and graph validation (`compat.py`),
  `comfy_generator.py` (OOM retry, cancel, normal estimator, atlas inpaint).
- `neural_texture/blender/`: guidance export (`guidance.py`, `uv_bake.py`), job runner, live agent bridge,
  crash cleanup and version compatibility.
- `neural_texture/pipeline.py`, `operators.py`, `ui.py`, `properties.py`: the Blender integration.
- Tests: run `python tests/run_tests.py` (no Blender needed). The real-Blender checks are
  `tests/blender_smoke.py` and `tests/blender_asset_suite.py`.

Keep `core/` free of `bpy`. Anything that runs in the worker thread must not touch `bpy`.
