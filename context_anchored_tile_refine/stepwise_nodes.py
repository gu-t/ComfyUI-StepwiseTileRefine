"""The fork's two production nodes: Tile Refine (Any VAE) and Stepwise Tile Upscale.

Both run the upstream base node's raster engine (sampling.refine_image) with a latent geometry
read from the VAE (geometry.py), so they work with VAEs whose spatial compression is not 8x,
such as the FLUX.2 VAE that MOSS-Image uses (16x, 128 channels). Class names and category are
the fork's own, so the package installs beside the original Context-Anchored Tile Refine.

Module scope is comfy-free like node.py: comfy is imported lazily inside functions."""
import logging
from typing import ClassVar

from .node import MAX_RESOLUTION, ContextAnchoredTileRefine, _normalize_mask, _require_geometry, _validate_image

CATEGORY = "image/upscaling/stepwise"

# Auto tile cap (sampled crop, rings included) by model family. Everything not listed is a
# ~1 MP model (SDXL, SD3, Flux, Chroma, Z-Image, Qwen-Image, Krea, MOSS-Image, ...).
DEFAULT_TILE = 1024
FAMILY_TILE = {
    "SD15": 768, "SD15_instructpix2pix": 768, "SD20": 768, "SD21UnclipL": 768, "SD21UnclipH": 768,
    "SD_X4Upscaler": 768, "SVD_img2vid": 768,
}


def _latent_geometry_inputs():
    return {
        "latent_scale": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "advanced": True, "tooltip": "Pixels per latent cell. 0 reads it from the VAE (8 for SD/SDXL/Flux.1, 16 for the FLUX.2 VAE used by MOSS-Image). Only set it if the VAE reports it wrong."}),
        "pixel_multiple": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, "advanced": True, "tooltip": "Pixel grid every canvas, tile and ring lands on, a multiple of the latent scale. 0 uses the latent scale. Raise it for a DiT that patchifies the latent, e.g. 32 for a 16x VAE with patch size 2."}),
    }


def resolve_run_geometry(vae, latent_scale=0, pixel_multiple=0):
    from .geometry import resolve_geometry

    return resolve_geometry(vae, latent_scale, pixel_multiple)


def auto_tile_cap(guider):
    # The model family's default tile cap, read off the guider's model config.
    patcher = getattr(guider, "model_patcher", None)
    model = getattr(patcher, "model", None)
    config = getattr(model, "model_config", None)
    return FAMILY_TILE.get(type(config).__name__, DEFAULT_TILE)


class TileRefineAnyVAE(ContextAnchoredTileRefine):
    """The upstream base node (Context-Anchored Tile Refine) with the latent scale read from the
    VAE instead of assumed to be 8. Behaviour on an 8x VAE is unchanged."""

    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        types["optional"] = {**types.get("optional", {}), **_latent_geometry_inputs()}
        return types

    CATEGORY = CATEGORY
    SEARCH_ALIASES: ClassVar[list[str]] = ["tiled refine", "tile refine", "any vae", "moss", "flux2", "16x vae"]
    DESCRIPTION = "Refines an already upscaled image tile by tile. Works with any VAE latent scale (8x, 16x, ...)."

    def refine(self, image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=None, latent_scale=0, pixel_multiple=0):
        _validate_image(image)
        _require_geometry(max_tile_width, max_tile_height, context_anchor, context_overlap)
        if mask is not None:
            mask = _normalize_mask(mask, image)
        from . import sampling

        geometry = resolve_run_geometry(vae, latent_scale, pixel_multiple)
        return (sampling.refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=mask, geometry=geometry),)


def check_stepwise(max_tile_width=None, max_tile_height=None, context_anchor=None, context_overlap=None):
    for name, value in (("max_tile_width", max_tile_width), ("max_tile_height", max_tile_height)):
        if value is None or value == 0:
            continue
        if value < 256 or value > MAX_RESOLUTION:
            return f"{name} must be 0 (auto) or between 256 and {MAX_RESOLUTION}, got {value}"
    for name, value in (("context_anchor", context_anchor), ("context_overlap", context_overlap)):
        if value is not None and not 0 <= value <= 512:
            return f"{name} must be between 0 and 512, got {value}"
    return True


def enlarge(image, width, height, upscale_model=None):
    # Pixel-space enlargement of the whole image to exactly width x height: the optional
    # upscale model first, then one lanczos onto the exact size. A same-size lanczos is skipped
    # (upscale.py: it is an 8-bit round trip, a loss rather than an identity).
    from . import upscale

    current = image
    if upscale_model is not None:
        current = upscale._upscale_with_model(upscale_model, current)
    if int(current.shape[2]) == width and int(current.shape[1]) == height:
        return current
    return upscale._lanczos_resize(current, width, height)


class StepwiseTileUpscale:
    """Upscales in several steps. Each step enlarges in pixel space and runs a tile refine, with
    denoise falling from the first step to the last. The final size is exact."""

    @classmethod
    def INPUT_TYPES(s):
        import comfy.samplers

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to upscale."}),
                "guider": ("GUIDER", {"tooltip": "The guider that denoises each tile, e.g. CFGGuider."}),
                "sampler": ("SAMPLER", {"tooltip": "The sampler used for every tile."}),
                "noise": ("NOISE", {"tooltip": "Noise is drawn once per step for the entire image and sliced for each tile."}),
                "vae": ("VAE", {"tooltip": "The VAE that encodes and decodes each tile. Its latent scale is read automatically."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "How the noise levels are spaced. Each step's sigmas are built like BasicScheduler with that step's denoise."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000, "tooltip": "Sampler steps per tile at every upscale step."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.05, "max": 16.0, "step": 0.01, "tooltip": "Total enlargement. The final size is round(input x upscale_by) on each axis, exactly."}),
                "upscale_steps": ("INT", {"default": 0, "min": 0, "max": 16, "tooltip": "Number of enlarge + refine steps. 0 picks the fewest steps whose factor is at most auto_step_factor."}),
                "max_tile_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 16, "tooltip": "Largest width the model sees per tile, rings included. 0 picks a size for the model family (1024 for SDXL/Flux/MOSS-class models, 768 for SD1.x/2.x)."}),
                "max_tile_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 16, "tooltip": "Largest height the model sees per tile, rings included. 0 picks a size for the model family."}),
                "context_anchor": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Minimum frozen context ring around each tile. Rounded up to whole latent cells and the pixel grid."}),
                "context_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Minimum overlap band shared and blended between neighboring tiles. Rounded up to whole latent cells."}),
                "denoise_first": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise of the first step."}),
                "denoise_last": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise of the last step. Steps in between are interpolated linearly."}),
                "plan_only": ("BOOLEAN", {"default": False, "tooltip": "Only compute and output the plan, without sampling. The image passes through unchanged."}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional. Used for the first step only, then lanczos to that step's exact size. Later steps use lanczos."}),
                "aspect_tolerance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1, "advanced": True, "tooltip": "Percent of aspect-ratio deviation an intermediate step may take to land on a size that splits into equal tiles. 0 keeps the closest exact enlargement (the last tile may then be narrower). The final step is always exact."}),
                "auto_step_factor": ("FLOAT", {"default": 1.5, "min": 1.05, "max": 8.0, "step": 0.05, "advanced": True, "tooltip": "Largest per-step factor when upscale_steps is 0."}),
                **_latent_geometry_inputs(),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "step_images", "plan")
    OUTPUT_IS_LIST = (False, True, False)
    OUTPUT_TOOLTIPS = (
        "The final image at exactly round(input x upscale_by).",
        "The result of every step, for inspection (a list, since the sizes differ).",
        "The plan as Markdown: sizes, factors, aspect deviation, grids, tile and sampled sizes, rings, denoise and enlargement per step.",
    )
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = "Upscales in several enlarge + tile refine steps and plans the steps itself. Works with any VAE latent scale."
    SEARCH_ALIASES: ClassVar[list[str]] = ["stepwise upscale", "multi step upscale", "tiled upscale", "moss upscale"]

    @classmethod
    def VALIDATE_INPUTS(s, max_tile_width=None, max_tile_height=None, context_anchor=None, context_overlap=None):
        return check_stepwise(max_tile_width, max_tile_height, context_anchor, context_overlap)

    @staticmethod
    def build_plan(image, guider, vae, upscale_by, upscale_steps, max_tile_width, max_tile_height, context_anchor,
                   context_overlap, denoise_first, denoise_last, upscale_model=None, aspect_tolerance=1.0,
                   auto_step_factor=1.5, latent_scale=0, pixel_multiple=0):
        from . import stepwise_plan

        geometry = resolve_run_geometry(vae, latent_scale, pixel_multiple)
        auto = auto_tile_cap(guider)
        cap_w = max_tile_width or auto
        cap_h = max_tile_height or auto
        model_scale = None
        if upscale_model is not None:
            model_scale = float(getattr(upscale_model, "scale", 1.0))
        return stepwise_plan.plan_upscale(
            int(image.shape[2]), int(image.shape[1]), float(upscale_by), int(upscale_steps), geometry, cap_w, cap_h,
            int(context_anchor), int(context_overlap), float(denoise_first), float(denoise_last),
            aspect_tolerance=float(aspect_tolerance) / 100.0, upscale_model_scale=model_scale,
            auto_step_factor=float(auto_step_factor))

    def upscale(self, image, guider, sampler, noise, vae, scheduler, steps, upscale_by, upscale_steps, max_tile_width,
                max_tile_height, context_anchor, context_overlap, denoise_first, denoise_last, plan_only,
                upscale_model=None, aspect_tolerance=1.0, auto_step_factor=1.5, latent_scale=0, pixel_multiple=0):
        _validate_image(image)
        message = check_stepwise(max_tile_width, max_tile_height, context_anchor, context_overlap)
        if message is not True:
            raise ValueError(message)
        from . import sampling, stepwise_plan, upscale

        plan = self.build_plan(image, guider, vae, upscale_by, upscale_steps, max_tile_width, max_tile_height,
                               context_anchor, context_overlap, denoise_first, denoise_last, upscale_model,
                               aspect_tolerance, auto_step_factor, latent_scale, pixel_multiple)
        text = stepwise_plan.format_plan(plan, steps)
        logging.info("Stepwise Tile Upscale:\n%s", text)
        if plan_only:
            return (image, [image], text)

        import comfy.model_management

        current = image[..., :3] if image.shape[-1] > 3 else image
        results = []
        for step in plan.steps:
            comfy.model_management.throw_exception_if_processing_interrupted()
            logging.info("Stepwise Tile Upscale: step %d/%d, %dx%d -> %dx%d, denoise %.3f, %d tiles",
                         step.index, len(plan.steps), step.src_width, step.src_height, step.width, step.height,
                         step.denoise, step.tiles)
            current = enlarge(current, step.width, step.height, upscale_model if step.use_model else None)
            sigmas = upscale.build_sigmas(guider.model_patcher, scheduler, steps, step.denoise)
            current = sampling.refine_image(
                current, guider, sampler, sigmas, vae, noise, plan.cap_width, plan.cap_height,
                plan.context_anchor, plan.context_overlap, geometry=plan.geometry)
            current = current.to(comfy.model_management.intermediate_device())
            if (int(current.shape[2]), int(current.shape[1])) != (step.width, step.height):
                raise RuntimeError(
                    f"Stepwise Tile Upscale: step {step.index} produced {int(current.shape[2])}x{int(current.shape[1])}, "
                    f"expected {step.width}x{step.height}")
            results.append(current)
        return (current, results, text)
