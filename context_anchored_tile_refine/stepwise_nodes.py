"""Two of the fork's three registered nodes: Tile Refine (Any VAE) and Stepwise Tile Upscale (the third,
Noise Level Scheduler, is in noise_level.py).

Both run the upstream base node's raster engine (sampling.refine_image) with a latent geometry
read from the VAE (geometry.py), so they work with VAEs whose spatial compression is not 8x,
such as the FLUX.2 VAE that MOSS-Image uses (16x, 128 channels). Class names and category are
the fork's own, so the package installs beside the original Context-Anchored Tile Refine.

Module scope is comfy-free like node.py: comfy is imported lazily inside functions."""
import logging
from typing import ClassVar

from .all_at_once import SIZE_FULL, TILE_BY_TILE, TILE_MODE_TOOLTIP, TILE_MODES, TILE_SIZE_TOOLTIP, TILE_SIZES
from .node import MAX_RESOLUTION, ContextAnchoredTileRefine, _normalize_mask, _validate_image
from .noise_level import DENOISE_MODES, NOISE_LEVEL
from .preview import GUESS, PREVIEW_MODE_TOOLTIP, PREVIEW_MODES
from .stepwise_plan import (
    DEFAULT_DOWNSCALE_METHOD,
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_UPSCALE_METHOD,
    DOWNSCALE_METHODS,
    OUTPUT_SIZES,
    OVERSAMPLE_CONSTANT,
    OVERSAMPLE_MODES,
    UPSCALE_METHODS,
)

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


TILE_OVERLAP_TOOLTIP = (
    "Band where neighbouring tiles overlap. tile by tile: diffused from both sides and cross-faded into the "
    "finished neighbour afterwards. all at once: both neighbours' answers are weighted together at every step. "
    "Unused in all at once, shifting. Rounded up to whole latent cells.")
TILE_PADDING_TOOLTIP = (
    "Context around each tile, outside the overlap band, that the model sees and that is thrown away afterwards. "
    "tile by tile: frozen (finished neighbours, or the source). all at once modes: live, at the same noise level. "
    "Rounded up to whole latent cells and the pixel grid.")


def _tile_inputs(max_tile):
    # The tile group, in the order a tile is built: how the tiles run, how big a window is, then the
    # overlap band around the core and the padding around that. Both nodes share it. The widget lists
    # were reordered and renamed 2026-11-23 while unreleased (the example workflow was updated with
    # them). From here on, append new widgets last: the frontend restores widgets_values by position.
    return {
        "tile_mode": (list(TILE_MODES), {"default": TILE_BY_TILE, "tooltip": TILE_MODE_TOOLTIP}),
        "tile_size": (list(TILE_SIZES), {"default": SIZE_FULL, "tooltip": TILE_SIZE_TOOLTIP}),
        **max_tile,
        "tile_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": TILE_OVERLAP_TOOLTIP}),
        "tile_padding": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": TILE_PADDING_TOOLTIP}),
    }


def _preview_inputs():
    # Appended last on both nodes (the frontend restores widgets_values by position).
    return {"preview_mode": (list(PREVIEW_MODES), {"default": GUESS, "tooltip": PREVIEW_MODE_TOOLTIP})}


def _widget_names(message):
    # Upstream's checks name the engine's arguments. The fork's nodes call them tile_padding and
    # tile_overlap.
    if not isinstance(message, str):
        return message
    return message.replace("context_anchor", "tile_padding").replace("context_overlap", "tile_overlap")


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
        up = super().INPUT_TYPES()
        req = up["required"]
        return {
            "required": {
                **{name: req[name] for name in ("image", "guider", "sampler", "sigmas", "noise", "vae")},
                **_tile_inputs({name: req[name] for name in ("max_tile_width", "max_tile_height")}),
            },
            "optional": {**up.get("optional", {}), **_latent_geometry_inputs(), **_preview_inputs()},
        }

    @classmethod
    def VALIDATE_INPUTS(s, max_tile_width=None, max_tile_height=None, tile_padding=None, tile_overlap=None):
        return check_tiles(max_tile_width, max_tile_height, tile_padding, tile_overlap)

    CATEGORY = CATEGORY
    SEARCH_ALIASES: ClassVar[list[str]] = ["tiled refine", "tile refine", "any vae", "moss", "flux2", "16x vae"]
    DESCRIPTION = "Refines an already upscaled image tile by tile. Works with any VAE latent scale (8x, 16x, ...)."

    def refine(self, image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, tile_padding, tile_overlap, mask=None, latent_scale=0, pixel_multiple=0, tile_mode=TILE_BY_TILE, tile_size=SIZE_FULL, preview_mode=GUESS):
        _validate_image(image)
        message = check_tiles(max_tile_width, max_tile_height, tile_padding, tile_overlap)
        if message is not True:
            raise ValueError(message)
        if mask is not None:
            mask = _normalize_mask(mask, image)
        from . import grid, sampling

        geometry = resolve_run_geometry(vae, latent_scale, pixel_multiple)
        height, width = int(image.shape[1]), int(image.shape[2])
        if mask is None and (width % geometry.multiple or height % geometry.multiple):
            logging.warning(
                "Tile Refine (Any VAE): the image is %dx%d, off the %d px grid, so its right and bottom edges are "
                "padded with a mirrored strip during denoise. The model may misread the outermost pixels. Feed a "
                "size that is a multiple of %d px to avoid it.", width, height, geometry.multiple, geometry.multiple)
        try:
            return (sampling.refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, tile_padding, tile_overlap, mask=mask, geometry=geometry, tile_mode=tile_mode, tile_size=tile_size, preview_mode=preview_mode),)
        except grid.GridConfigError as e:
            raise ValueError(_widget_names(str(e))) from e


def check_tiles(max_tile_width=None, max_tile_height=None, tile_padding=None, tile_overlap=None, auto=False):
    # The fork's tile widget rule, for both nodes: no multiple-of-8 demand, since the rings snap up
    # to the geometry (geometry.snap_rings) and the grid solver takes any cap. `auto` allows 0 for
    # the cap (Stepwise picks one for the model family). Returns True or the message.
    for name, value in (("max_tile_width", max_tile_width), ("max_tile_height", max_tile_height)):
        if value is None or (auto and value == 0):
            continue
        if value < 256 or value > MAX_RESOLUTION:
            low = "0 (auto) or between 256" if auto else "between 256"
            return f"{name} must be {low} and {MAX_RESOLUTION}, got {value}"
    for name, value in (("tile_padding", tile_padding), ("tile_overlap", tile_overlap)):
        if value is not None and not 0 <= value <= 512:
            return f"{name} must be between 0 and 512, got {value}"
    return True


def enlarge(image, width, height, upscale_model=None, method="lanczos", virtual=None):
    # Pixel-space resample of the whole image to exactly width x height: the optional upscale
    # model first, then one resample onto the exact size with `method` (resample.py), or onto
    # the centre of the fractional size `virtual`. A same-size resample is skipped.
    from . import resample, upscale

    current = image
    if upscale_model is not None:
        current = upscale._upscale_with_model(upscale_model, current)
    return resample.resize(current, width, height, method, virtual)


def centre_crop(image, width, height):
    # The centre width x height of a [B,H,W,C] image (the left and top margins take the odd
    # pixel, as stepwise_plan.FinalCrop does). A same-size crop is the input.
    y0 = (int(image.shape[1]) - height) // 2
    x0 = (int(image.shape[2]) - width) // 2
    if (x0, y0) == (0, 0) and (int(image.shape[2]), int(image.shape[1])) == (width, height):
        return image
    return image[:, y0:y0 + height, x0:x0 + width, :]


def step_noise(noise, index):
    # The NOISE for pass `index` (1-based): the caller's object for pass 1, a copy with seed +
    # index - 1 after. Noise is drawn from the seed and the canvas shape, so two passes at the
    # same size would otherwise get identical noise, which overcooks the image (measured on MOSS
    # 2026-11-21: same-seed passes reproduce each other's invented detail exactly). A NOISE
    # without a seed is passed through.
    if index == 1 or not isinstance(getattr(noise, "seed", None), int):
        return noise
    import copy

    shifted = copy.copy(noise)
    shifted.seed = (noise.seed + index - 1) % (1 << 64)
    return shifted


class StepwiseTileUpscale:
    """Upscales in several steps. Each step enlarges in pixel space and runs a tile refine, with
    strength (start noise level or denoise) falling from the first step to the last. Every step
    samples a size on the pixel grid; output_size says how the output lands on the target."""

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
                # Size: how big the output gets and in how many passes.
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.01, "tooltip": "Output size: the final image is round(input x upscale_by) on each axis (output_size says how it lands there). 1 keeps the input size (a pure repair run with oversample)."}),
                "output_size": (list(OUTPUT_SIZES), {"default": DEFAULT_OUTPUT_SIZE, "tooltip": "How the output lands on round(input x upscale_by) when that is not a multiple of the pixel grid (16 px for MOSS). Every step before it is always on the grid. exact, crop edges: the last step runs a few px larger without stretching and is centre-cropped, so a few px of picture are lost at the edges. exact, pad edges during denoise: the right and bottom edges are padded with a mirrored strip while the last step denoises and cut off after; nothing mirrored is left in the image, but the model may misread the outermost pixels. stretch image: the output is the grid size nearest the target, each axis a few px larger or smaller, so the aspect changes slightly; the plan says by how much. With oversample the final resample hits the exact size, so only stretch image changes anything there."}),
                "upscale_steps": ("INT", {"default": 0, "min": 0, "max": 16, "tooltip": "Number of refine passes. 0 picks the fewest whose growth per pass is at most auto_step_factor."}),
                "auto_step_factor": ("FLOAT", {"default": 1.5, "min": 1.05, "max": 8.0, "step": 0.05, "tooltip": "Largest growth per pass when upscale_steps is 0 (auto). Unused when upscale_steps is set."}),
                "oversample": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.01, "tooltip": "Runs the refine passes this much larger than the output track, then shrinks onto the exact output size after the last pass. The largest pass is input x upscale_by x oversample. Enlarging makes small objects bigger in pixels, so the passes can redraw them. 1 is off."}),
                "oversample_mode": (list(OVERSAMPLE_MODES), {"default": OVERSAMPLE_CONSTANT, "tooltip": "constant: the first pass jumps to the oversample and every pass keeps it (enlarge, refine, shrink back each pass; at upscale_by 1 every pass runs at one size). ramp up: the passes grow evenly to input x upscale_by x oversample and reach it on the last pass."}),
                "upscale_method": (list(UPSCALE_METHODS), {"default": DEFAULT_UPSCALE_METHOD, "tooltip": "Filter for every enlargement, soft to sharp. lanczos is Lanczos-3 in float. The bicubic variants are the Mitchell-Netravali family: B-spline is the softest with no ringing, Catmull-Rom the sharpest."}),
                "downscale_method": (list(DOWNSCALE_METHODS), {"default": DEFAULT_DOWNSCALE_METHOD, "tooltip": "Filter for the shrink onto the output size after the last pass (the resamples between passes always enlarge). area is the exact mean of the covered pixels and never rings. Every filter is widened by the shrink factor, so all source pixels count however large the reduction."}),
                # Strength: how much each pass redraws, and the schedule it samples.
                "denoise_mode": (list(DENOISE_MODES), {"default": NOISE_LEVEL, "tooltip": "noise level: denoise_first/last are the share of noise each step starts from (0..1, the flow sigma; sigma / (1 + sigma) on SD/SDXL-type models), the same for every scheduler. denoise: they are BasicScheduler denoise values, whose start noise depends on the scheduler and the model's shift. The plan shows both for every step."}),
                "denoise_first": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001, "tooltip": "Strength of the first step, read as set by denoise_mode: the start noise level (default) or a BasicScheduler denoise."}),
                "denoise_last": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001, "tooltip": "Strength of the last step. Steps in between are interpolated linearly per step, and the step sizes grow by a constant factor, so the strength falls by the same amount per factor of size."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "How the noise levels are spaced. In denoise mode each step's sigmas are built like BasicScheduler; in noise level mode the schedule is cut so it starts at that step's noise level."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000, "tooltip": "Sampler steps per tile at every upscale step."}),
                # Tiles.
                **_tile_inputs({
                    "max_tile_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 16, "tooltip": "Largest width the model sees per tile, overlap and padding included. 0 picks a size for the model family (1024 for SDXL/Flux/MOSS-class models, 768 for SD1.x/2.x)."}),
                    "max_tile_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 16, "tooltip": "Largest height the model sees per tile, overlap and padding included. 0 picks a size for the model family."}),
                }),
                "plan_only": ("BOOLEAN", {"default": False, "tooltip": "Only compute and output the plan, without sampling. The image passes through unchanged."}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional. Used for the first step only, then upscale_method to that step's exact size."}),
                **_latent_geometry_inputs(),
                **_preview_inputs(),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "step_images", "plan")
    OUTPUT_IS_LIST = (False, True, False)
    OUTPUT_TOOLTIPS = (
        "The final image at round(input x upscale_by), or the grid size nearest it (output_size).",
        "The result of every step, for inspection (a list, since the sizes differ).",
        "The plan as plain text: sizes, factors, aspect deviation, tiles and what the model sees, rings, noise and enlargement per step.",
    )
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = "Upscales in several enlarge + tile refine steps and plans the steps itself. Works with any VAE latent scale."
    SEARCH_ALIASES: ClassVar[list[str]] = ["stepwise upscale", "multi step upscale", "tiled upscale", "moss upscale"]

    @classmethod
    def VALIDATE_INPUTS(s, max_tile_width=None, max_tile_height=None, tile_padding=None, tile_overlap=None):
        return check_tiles(max_tile_width, max_tile_height, tile_padding, tile_overlap, auto=True)

    @staticmethod
    def build_plan(image, guider, vae, upscale_by, upscale_steps, max_tile_width, max_tile_height, tile_padding,
                   tile_overlap, denoise_first, denoise_last, upscale_model=None, output_size=DEFAULT_OUTPUT_SIZE,
                   auto_step_factor=1.5, latent_scale=0, pixel_multiple=0, oversample=1.0,
                   oversample_mode=OVERSAMPLE_CONSTANT, upscale_method=DEFAULT_UPSCALE_METHOD,
                   downscale_method=DEFAULT_DOWNSCALE_METHOD, tile_mode=TILE_BY_TILE):
        from . import all_at_once, stepwise_plan

        geometry = resolve_run_geometry(vae, latent_scale, pixel_multiple)
        auto = auto_tile_cap(guider)
        cap_w = max_tile_width or auto
        cap_h = max_tile_height or auto
        if tile_mode == all_at_once.SHIFTING:
            # The shifting grid has no overlap band, so tile_overlap must not reach the plan's fixed
            # grid solve, where a band the run never uses could fail the plan. Its own check first.
            all_at_once.tile_cells(cap_w, cap_h, tile_padding, geometry)
            tile_overlap = 0
        model_scale = None
        if upscale_model is not None:
            model_scale = float(getattr(upscale_model, "scale", 1.0))
        return stepwise_plan.plan_upscale(
            int(image.shape[2]), int(image.shape[1]), float(upscale_by), int(upscale_steps), geometry, cap_w, cap_h,
            int(tile_padding), int(tile_overlap), float(denoise_first), float(denoise_last),
            upscale_model_scale=model_scale,
            auto_step_factor=float(auto_step_factor), oversample=float(oversample), oversample_mode=oversample_mode,
            upscale_method=upscale_method, downscale_method=downscale_method,
            fixed_grid=tile_mode != all_at_once.SHIFTING, output_size=output_size)

    def upscale(self, image, guider, sampler, noise, vae, scheduler, steps, upscale_by, upscale_steps, max_tile_width,
                max_tile_height, tile_padding, tile_overlap, denoise_first, denoise_last, plan_only,
                upscale_model=None, auto_step_factor=1.5, latent_scale=0, pixel_multiple=0,
                denoise_mode=NOISE_LEVEL, oversample=1.0, oversample_mode=OVERSAMPLE_CONSTANT,
                upscale_method=DEFAULT_UPSCALE_METHOD, downscale_method=DEFAULT_DOWNSCALE_METHOD,
                tile_mode=TILE_BY_TILE, tile_size=SIZE_FULL, preview_mode=GUESS, output_size=DEFAULT_OUTPUT_SIZE):
        _validate_image(image)
        message = check_tiles(max_tile_width, max_tile_height, tile_padding, tile_overlap, auto=True)
        if message is not True:
            raise ValueError(message)
        if float(upscale_by) < 1.0:
            raise ValueError(f"upscale_by must be at least 1, got {upscale_by}")
        from . import grid, noise_level, resample, sampling, stepwise_plan

        try:
            plan = self.build_plan(image, guider, vae, upscale_by, upscale_steps, max_tile_width, max_tile_height,
                                   tile_padding, tile_overlap, denoise_first, denoise_last, upscale_model,
                                   output_size, auto_step_factor, latent_scale, pixel_multiple, oversample,
                                   oversample_mode, upscale_method, downscale_method, tile_mode)
        except grid.GridConfigError as e:
            raise ValueError(_widget_names(str(e))) from e
        schedules = [noise_level.build_schedule(guider.model_patcher, scheduler, steps, s.denoise, denoise_mode)
                     for s in plan.steps]
        label = "Noise level" if denoise_mode == NOISE_LEVEL else "Denoise"
        from . import all_at_once

        shifting = all_at_once.shifting_tiles(plan, tile_size, tile_padding) if tile_mode == all_at_once.SHIFTING else None
        text = stepwise_plan.format_plan(plan, steps, value_label=label, shifting=shifting, schedules=schedules,
                                         tile_mode=tile_mode, tile_size=tile_size, scheduler=scheduler)
        logging.info("Stepwise Tile Upscale:\n%s", text)
        if plan_only:
            return (image, [image], text)

        import comfy.model_management

        current = image[..., :3] if image.shape[-1] > 3 else image
        results = []
        for step, schedule in zip(plan.steps, schedules, strict=True):
            comfy.model_management.throw_exception_if_processing_interrupted()
            logging.info("Stepwise Tile Upscale: step %d/%d, %dx%d -> %dx%d, start noise %.3f (denoise %.4f), %d tiles",
                         step.index, len(plan.steps), step.src_width, step.src_height, step.width, step.height,
                         schedule.start_level, schedule.denoise, step.tiles)
            current = enlarge(current, step.width, step.height, upscale_model if step.use_model else None, step.method,
                              step.enlarged if step.enlarge_width else None)
            current = sampling.refine_image(
                current, guider, sampler, schedule.sigmas, vae, step_noise(noise, step.index), plan.cap_width,
                plan.cap_height, tile_padding if tile_mode == all_at_once.SHIFTING else plan.context_anchor, plan.context_overlap,
                geometry=plan.geometry, tile_mode=tile_mode, tile_size=tile_size, preview_mode=preview_mode)
            current = current.to(comfy.model_management.intermediate_device())
            if (int(current.shape[2]), int(current.shape[1])) != (step.width, step.height):
                raise RuntimeError(
                    f"Stepwise Tile Upscale: step {step.index} produced {int(current.shape[2])}x{int(current.shape[1])}, "
                    f"expected {step.width}x{step.height}")
            results.append(current)
        if isinstance(plan.final, stepwise_plan.FinalCrop):
            # A crop is no pass of its own: the last step's image is shown cropped.
            current = centre_crop(current, plan.final.width, plan.final.height)
            results[-1] = current
        elif plan.final is not None:
            current = resample.resize(current, plan.final.width, plan.final.height, plan.final.method)
            results.append(current)
        return (current, results, text)
