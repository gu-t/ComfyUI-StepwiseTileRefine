# ComfyUI-StepwiseTileRefine

Fork of Blakeem/ComfyUI-ContextAnchoredTileRefine (GPL-3.0). This file holds only the rules the code
cannot tell you.

## Quality rules

1. Never resize, resample or apply any other lossy operation to a tile: tiles are cut, sampled and
   pasted back at their native size.
2. Hide seams by conditioning or by stepping tiles together, never by a wide blend of two
   independent refinements (it ghosts).
3. The owner judges seams and detail by eye. Metrics are not a verdict.

## Code rules

- New widgets go LAST. The frontend restores `widgets_values` by position, so a widget inserted
  mid-list silently shifts every saved workflow. If you must reorder, update `workflows/` and tell
  the owner.
- Patch comfy model objects on the INSTANCE and restore in `finally`, never on the class: core
  caches models across the session.
- Upstream code that no registered node uses stays in the package until the owner decides
  otherwise.
