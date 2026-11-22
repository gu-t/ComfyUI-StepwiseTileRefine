# ComfyUI-StepwiseTileRefine, a fork of Blakeem/ComfyUI-ContextAnchoredTileRefine (GPL-3.0).
# Only the fork's own nodes are registered, under their own class names and category, so this
# package can be installed beside the original. The upstream nodes' code stays in the package
# (the fork's nodes build on it) but is not registered here; install the original for those.
from .context_anchored_tile_refine.noise_level import NoiseLevelScheduler
from .context_anchored_tile_refine.stepwise_nodes import StepwiseTileUpscale, TileRefineAnyVAE

NODE_CLASS_MAPPINGS = {
    "StepwiseTileUpscale": StepwiseTileUpscale,
    "StepwiseTileRefineAnyVAE": TileRefineAnyVAE,
    "StepwiseNoiseLevelScheduler": NoiseLevelScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StepwiseTileUpscale": "Stepwise Tile Upscale",
    "StepwiseTileRefineAnyVAE": "Tile Refine (Any VAE)",
    "StepwiseNoiseLevelScheduler": "Noise Level Scheduler",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
