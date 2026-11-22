# ComfyUI-StepwiseTileRefine

Fork of Blakeem/ComfyUI-ContextAnchoredTileRefine (GPL-3.0). Read the code for how things work;
this file holds only the rules the code cannot tell you. Upstream's long guide is in git history.

## Quality rules (override convenience)

1. Image quality first. Never resize, resample or apply any lossy operation to a tile: tiles are
   cut, sampled and pasted back at their native size.
2. Never re-diffuse finished pixels that survive into the output (it compounds grit).
3. Hide seams by conditioning or by stepping tiles together, never by a wide blend of two
   independent refinements (ghosting) or a double hard paste of one strip.
4. The owner judges seams and detail by eye. Metrics are not a verdict.

## Code rules

- Registered nodes are only `StepwiseTileUpscale` and `StepwiseTileRefineAnyVAE`
  (`stepwise_nodes.py`). Upstream's other nodes stay in the code, unregistered.
- New widgets go LAST. The frontend restores `widgets_values` by position, so a widget inserted
  mid-list silently shifts every saved workflow. If you must reorder, update `workflows/` and tell
  the owner.
- Module scopes stay comfy-free (lazy `import comfy...` inside functions). `grid.py`,
  `geometry.py` and `stepwise_plan.py` stay stdlib-only. Subprocess tests pin this.
- Patch comfy model objects on the INSTANCE and restore in `finally`, never on the class: core
  caches models across the session.
- `refine_image` in `sampling.py` is the one place the nodes meet. Put cross-cutting behaviour
  there, not in a node.
- Match ComfyUI core naming and comment style, and comment only non-obvious constraints.

## Gates (both green before any commit)

Run from the repo root:

- `COMFYUI_ROOT=../ComfyUI ../venv/Scripts/python.exe -m pytest tests -m "not gpu" --deselect tests/test_tags_bench_export.py`
  (that test needs an external repo and fails without changes)
- `../venv/Scripts/ruff.exe check .` (ruff 0.16.2, config in `pyproject.toml`, no formatter)

## Running and releasing

- One GPU sampling job at a time. Check that `nvidia-smi` is idle first: a second job spills into
  Windows sysmem fallback and can crash silently.
- Never run MOSS-Image with `--use-sage-attention`: it gives wrong images.
- Do not commit or push without the owner's go-ahead.
