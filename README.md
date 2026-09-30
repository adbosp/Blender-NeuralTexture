# Neural Texture 0.2: Blender add-on + AI pipeline

Neural Texture uses AI to regenerate or enhance how an **existing** mesh looks. Geometry, UVs, the normal map and the other maps are the structure the AI has to follow. ComfyUI paints the look. The mesh itself is never modified.

```
Blender (main thread)                          Pure Python (worker thread / CLI / MCP)
─────────────────────                          ───────────────────────────────────────
analyze ─► guidance package ──────────────►  plan views ─► for each view:
  18 candidate views: position, UV,             reproject current texture ─► init + inpaint mask
  depth, normal, albedo, mask                   ComfyUI (SDXL + ControlNet depth/normal + IPAdapter)
  UV bakes: position, normal, tangent,          colour harmonisation ─► (normal estimator)
  bitangent, material id, AO, curvature         UV accumulation (+confidence)
  existing maps (resampled)                   UV-space inpaint of unseen texels
                                              material-aware PBR ─► seam refinement ─► history/<run>
material  ◄──────────────────────────────────  latest.json
```

## What each version adds

| | Feature | Where |
|---|---|---|
| **V2** | Sequential generation: each view is inpainted to match what earlier views already painted (reprojected through a UV pass). Uses a latent noise mask and per-view colour harmonisation. | `core/synth.py`, `core/consistency.py` |
| | UV accumulation weighted by visibility, facing angle and distance from the silhouette. Texels that earlier views already painted with confidence resist being overwritten. | `core/accumulate.py` |
| | View planner that picks from 18 candidate views, preferring views that overlap what's already painted | `core/planner.py` |
| | Unseen texels: extra candidate views, then UV-space inpainting in ComfyUI, then the existing albedo, then pull-push fill | `synth._uv_inpaint` |
| | Confidence map: how much weight a texel got combined with how well the views agree. Used for blending, validation and "Fix low confidence". | `accumulate.confidence_map` |
| | Detail-normal sources: `luminance`, `model` (a ComfyUI normal estimator whose axes are calibrated automatically against the true geometry normals), `height` and `hybrid`. | `core/detail.py` |
| | The original normal is kept and detail is added: `RNM(existing, detail)`. An unblend step removes large-scale shapes from the model's normals so only the detail bands remain. | `core/normals.py`, `core/detail.py` |
| **V3** | BaseColor: de-lighting, AO de-lighting, highlight suppression, clamping to a plausible albedo range per material, soft blending to the existing texture where confidence is low | `core/pbr.py` |
| | Detail Height (Frankot–Chellappa integration), Detail Normal, and an ORM map (AO, roughness, metallic) | `core/detail.py`, `core/pbr.py` |
| | Roughness driven by material priors plus cavities, AO and curvature. Metallic assigned per slot from the material class; rusted metal is split by colour. | `core/materials.py`, `core/pbr.py` |
| | Material-aware generation: 11 material classes recognised from English/Vietnamese text or material names, with per-slot overrides | `core/materials.py` |
| | Seam refinement: texels on opposite sides of a UV seam that share a 3D position are matched and the colour correction is spread into both sides | `core/seams.py` |
| **V4** | Cleaner panel: presets, simple sliders, a progress bar, Cancel, Before/After, version history, paint-a-mask then regenerate, fix low confidence, confidence/preview viewer | `ui.py`, `operators.py` |
| | Non-blocking runner: renders run one tick at a time on the main thread, AI work runs in a worker thread. Cancelling keeps finished views and the next run resumes. | `blender/runner.py` |
| | Rebuild maps (seconds, no regeneration) after changing Detail, Keep Colours, etc. | `synth.finalize` |
| **V5** | MCP server (stdio, no dependencies), plus a live Agent Bridge to a running Blender and a headless mode | `mcp_server/`, `blender/live_server.py` |
| | `SKILL.md`, `CLAUDE.md`/`AGENTS.md`, intent JSON schema | `agent/` |
| | Vietnamese prompts, e.g. *"làm model này thành đá cổ nhưng giữ normal gốc"* → stone, restyle, macro normal kept. Shape-change requests in Vietnamese are refused. | `core/intent.py` |
| **V6** | Crash handling: atomic state saves, resume, `error.json`, and cleanup of leftover temporary data when Blender starts or loads a file | `core/synth.py`, `blender/cleanup.py` |
| | VRAM: VRAM check, out-of-memory detection, `/free`, then a retry at lower resolution | `bridge/comfy_generator.py` |
| | Models detected automatically from `/object_info`, including SDXL checkpoints, union or separate ControlNets, IPAdapter and the normal estimator | `bridge/compat.py` |
| | Workflow compatibility: graphs are completed with the server's defaults, invalid options and missing nodes produce clear errors, and custom API-format workflows with `{{placeholders}}` are supported | `bridge/compat.py`, `bridge/workflow.py` |
| | Blender 4.2 LTS through 5.x (colour-space names, `use_nodes`, brush API, GPU selection) | `blender/compat.py` |
| | Packaging (extension zip and agent zip), benchmark, and an 11-asset test suite in real Blender | `tools/`, `tests/blender_asset_suite.py` |

## Install

1. Run `python tools/build.py`. It produces `dist/neural_texture-0.2.0.zip`; install it in Blender with Preferences › Get Extensions › Install from Disk. Pass `--blender <path> --install` to validate and install it in one step.
2. ComfyUI needs an SDXL checkpoint and a ControlNet: `controlnet-union-sdxl`, or separate depth and normal models. Optional:
   - **ComfyUI_IPAdapter_plus**, for reference images.
   - **comfyui_controlnet_aux**, for the DSINE/BAE normal estimator.

   Click *Check ComfyUI* in the panel to detect and fill everything in automatically.
3. For agents: `claude mcp add neural-texture -e NT_BLENDER=/path/to/blender -- python mcp_server/neural_texture_mcp.py`. Then turn on the *Agent Bridge* in the panel to drive the Blender that's already open.

## Using it in Blender (N panel › Neural Texture)

1. Pick a preset, type a prompt (English or Vietnamese), add reference images if you have them, then click **Generate Texture Set**. Watch the progress bar; you can Cancel.
2. Use **Before/After** to compare, and **Versions** to go back to an earlier version.
3. Adjust Detail or Keep Colours, then click **Rebuild Maps** (seconds).
4. If an area looks wrong: **Paint Fix Mask** → paint white over it → **Regenerate Masked**. Or use **Fix Low Confidence**.

## Headless use / agents

```bash
blender -b scene.blend --python neural_texture/cli.py -- --job job.json --stage guidance
python -m neural_texture.headless synth   out/Head --job job.json          # no Blender needed
python -m neural_texture.headless regenerate out/Head --threshold 0.35
blender -b scene.blend --python neural_texture/cli.py -- --job job.json --stage material --save
python -m neural_texture.headless intent "làm model này thành đá cổ nhưng giữ normal gốc"
```

Exit codes: 0 ok · 2 bad arguments · 3 needs geometry editing · 4 error (`error.json`) · 5 cancelled.

## Tests and benchmark

```bash
python tests/run_tests.py                                           # 56 tests, no Blender or GPU
blender -b --factory-startup --python tests/blender_smoke.py        # real Blender round trip
blender -b --factory-startup --python tests/blender_asset_suite.py  # 11 asset types
python tools/benchmark.py --res 256 512 1024
```

The pure-Python tests use analytic scenes (sphere, cube with 6 UV islands) and a fake "model" that adds lighting and colour drift the way diffusion models do. They check:

- The full synthesis run, including a check that consistency reduces error (about 25%).
- Planner coverage above 97%.
- Seam error reduced by more than 90% on the cube.
- Resuming after a cancel.
- Regenerating only the masked texels.
- Automatic calibration of normal-map axes.
- A mock ComfyUI with out-of-memory recovery, auto-detection, cancel and interrupt.
- The MCP protocol, both live-bridge and pure-Python.
- The JobRunner.

Benchmark (CPU, instant fake model): a 1024 texture with 10 views synthesises in about 14 s plus about 6 s to build maps. Real time is dominated by ComfyUI, at roughly 5–20 s per view.

## Known limitations

- There is one texture set: the UV map covering the most triangles, and tile 1001 only for UDIM layouts.
- Mirrored or overlapping UVs share texels, so the result on mirrored halves will be symmetric.
- The ControlNet and IPAdapter node names follow common packs. Anything else goes through a custom API workflow.
- Building maps at 4K takes about 1–2 minutes on CPU.
- **None of the Blender-side code (rendering, baking, UI, bridge) could be run where it was written.** It needs `tests/blender_smoke.py` and `tests/blender_asset_suite.py` on a real machine.
