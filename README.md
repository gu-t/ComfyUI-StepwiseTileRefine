# Stepwise Tile Refine

ComfyUI nodes for tiled upscaling that work with **any VAE latent scale**, including the 16x FLUX.2 VAE that **MOSS-Image** uses, and a node that upscales in **several steps** and plans those steps itself.

> **Fork notice.** This is a fork of [Blakeem/ComfyUI-ContextAnchoredTileRefine](https://github.com/Blakeem/ComfyUI-ContextAnchoredTileRefine) (GPL-3.0). The tile engine (context rings, shared noise, directional feather, minimum-error seam cut, DC match) is upstream's work. What this fork changes is listed [below](#what-the-fork-changes). The upstream README is kept in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).
>
> The fork is maintained when time allows and comes with no support commitment.

## Contents

- [Installation](#installation)
- [The nodes](#the-nodes)
  - [Stepwise Tile Upscale](#stepwise-tile-upscale)
  - [Tile Refine (Any VAE)](#tile-refine-any-vae)
- [How the steps are planned](#how-the-steps-are-planned)
- [MOSS-Image settings](#moss-image-settings)
- [Other models](#other-models)
- [What the fork changes](#what-the-fork-changes)
- [Known limitations](#known-limitations)
- [Tests](#tests)
- [License](#license)

## Installation

Clone into your ComfyUI `custom_nodes` folder and restart ComfyUI:

```
git clone https://github.com/gu-t/ComfyUI-StepwiseTileRefine
```

The nodes need nothing beyond ComfyUI itself (Python 3.12+). The package uses its own class names and the category `image/upscaling/stepwise`, so it can be installed next to the original Context-Anchored Tile Refine.

## The nodes

Both nodes are wired like `SamplerCustomAdvanced`: a `GUIDER` (for example `CFGGuider`), a `SAMPLER` (`KSamplerSelect`), a `NOISE` (`RandomNoise`) and the `VAE`.

### Stepwise Tile Upscale

Upscales in several steps. Each step enlarges the whole image in pixel space and then refines it tile by tile, with the denoise falling from the first step to the last.

| Input | What it does |
|---|---|
| `image`, `guider`, `sampler`, `noise`, `vae` | As for `SamplerCustomAdvanced`. The VAE's latent scale is read automatically. |
| `scheduler`, `steps` | Each step's sigmas are built like `BasicScheduler` with that step's denoise. |
| `upscale_by` | Total enlargement. The final size is `round(input × upscale_by)` on each axis, exactly. |
| `upscale_steps` | Number of enlarge + refine steps. `0` picks the fewest steps whose factor stays at or under `auto_step_factor` (1.5), e.g. 2 steps for x2 and 3 for x3. |
| `max_tile_width` / `max_tile_height` | Largest size the model sees per tile, rings included. `0` picks 1024 for SDXL/Flux/MOSS-class models and 768 for SD1.x/2.x. |
| `context_anchor` / `context_overlap` | Minimum ring widths (frozen context, and the band shared and blended between neighbours). They are rounded up to whole latent cells and the pixel grid. |
| `denoise_first` / `denoise_last` | Denoise of the first and the last step. The steps in between are interpolated linearly. |
| `plan_only` | Computes and outputs the plan without sampling, so you can check it before a long run. |
| `upscale_model` (optional) | Used for the first step only, then lanczos to that step's exact size. Later steps use lanczos, so a 4x model never runs on a large intermediate. |
| `aspect_tolerance` (advanced) | How much aspect-ratio deviation (percent) an intermediate step may take to get tiles of equal size. `0` keeps the closest exact enlargement. The final step is always exact. |
| `auto_step_factor`, `latent_scale`, `pixel_multiple` (advanced) | See [How the steps are planned](#how-the-steps-are-planned) and [Other models](#other-models). |

Outputs: the final `image`, `step_images` (the result of every step, as a list since the sizes differ) and `plan`, a Markdown table that is also written to the log. Connect `plan` to *Preview as Text* in Markdown mode.

Example plan, MOSS-Image, 1280x960 x3 in 3 steps:

| Step | Size | Factor x / y | Aspect dev | Tiles | Core px (w x h) | Sampled px (w x h) | Denoise | Enlarge | Size choice |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1280x960 -> 1824x1376 | 1.425 / 1.433 | -0.58% | 2x2 = 4 | 912 x 688 | 976 x 752 | 0.450 | lanczos | even tiles |
| 2 | 1824x1376 -> 2688x2016 | 1.474 / 1.465 | +0.00% | 3x3 = 9 | 896 x 672 | 1024/960 x 800/736 | 0.335 | lanczos | even tiles |
| 3 | 2688x2016 -> 3840x2880 | 1.429 / 1.429 | +0.00% | 5x4 = 20 | 768 x 720 | 896/832 x 848/784 | 0.220 | lanczos | final (exact) |

### Tile Refine (Any VAE)

Upstream's base node (*Context-Anchored Tile Refine*) with the latent scale read from the VAE instead of assumed to be 8. Upscale the image first and feed it in; denoise is set through the sigmas (`BasicScheduler`'s `denoise`). An optional `mask` refines only the masked region. On an 8x VAE it gives byte-identical output to the upstream node.

## How the steps are planned

1. The final size is `round(source × factor)`. There is no rounding to 64 and no stretch.
2. Each step enlarges by about `factor^(1/steps)`. The intermediate sizes are a trade-off between an even factor per step, the right aspect ratio, and sizes where each axis splits into a whole number of **equal** tiles whose cores and sampled crops are multiples of the pixel grid and fit under the tile cap. Within `aspect_tolerance` the planner picks such a size. Otherwise it takes a grid-multiple size, and with tolerance 0 the closest exact enlargement (reflect-padded to the grid and cropped back, so the last tile of an axis can be narrower). The plan shows the factor and the aspect deviation of every step.
3. Tiles are never resampled. A tile is cut from the canvas, sampled and pasted back at its native size.
4. Denoise goes linearly from `denoise_first` to `denoise_last`, applied through the sigmas like `BasicScheduler`.

## MOSS-Image settings

What the tests in this repository used, and a good starting point:

- Diffusion model: `moss_image_sft_int8_convrot` (Load Diffusion Model), text encoder `qwen3.5_9b_w4a8_gptq` (Load CLIP, type `moss_image`), VAE `flux2-vae`.
- `CFGGuider` cfg 3, sampler `er_sde`, scheduler `simple`, 25 steps. Shift 5 is set by the model automatically.
- Max tile `0` (auto = 1024, about 1 MP per tile, within MOSS's 1-2 MP sweet spot), `context_anchor` 32, `context_overlap` 32.
- Denoise 0.45 -> 0.22 for a 2-3 step upscale.
- The positive prompt reaches every tile. Objects it names (say "seagulls") get drawn into any tile with a vaguely matching shape, more so at higher denoise. For upscaling, prefer a prompt that describes the style, medium and lighting rather than the objects, or lower the denoise (0.35 -> 0.18 cut the colour drift of the 3-step test by a fifth).
- Do not start ComfyUI with `--use-sage-attention`: it produces broken images with MOSS-Image.
- MOSS's own guide advises against latent upscaling. These nodes enlarge in pixel space and re-encode, as recommended.

On an RTX 4090 one 1 MP tile takes about 5 s at 25 steps (int8 model, cfg 3).

### Measured with MOSS-Image

Same seed, prompt, sampler and denoise per step, against Ultimate SD Upscale (custom sample, lanczos, tile 1024 + padding 64, chained once per step). Harness: [tests-moss/](tests-moss/).

| Case | Steps | Tiles | Time | Final size | USDU final size | USDU time |
|---|---|---|---|---|---|---|
| 1280x960 x3 | 1 | 20 | 100 s | 3840x2880 | 3840x2880 | 98 s |
| 1280x960 x3 | 3 | 33 | 163 s | 3840x2880 | 3840x**2888** | 177 s |
| 1280x976 x3 | 3 | 33 | 167 s | 3840x2928 | 3840x2928 | 177 s |
| 2000x2000 x2 | 2 (auto) | 41 | 205 s | 4000x4000 | **4008x4008** | 203 s |
| 1344x768 x2 | 2 (auto) | 10 | 50 s | 2688x1536 | **2696**x1536 | 80 s |

- The final size was exact in every case. USDU missed it in three of five (stretched to a multiple of 8 after each pass).
- No seam line stood out from the image's own edges in any output (`tests-moss/analyze.py` ranks every row and column against its neighbourhood).
- MOSS img2img brightens slightly at these denoise values (+1 to +6 /255 over a 2-3 step chain). USDU shows the same shift, so it comes from the model, not the tiling.
- A faint 16 px pattern on the latent grid is a property of the FLUX.2 VAE and MOSS, not of the tiles: at x1.5 / denoise 0.45 the grid-periodicity score was 1.028 for lanczos only, 1.042 after a plain VAE round trip, 1.066 for one untiled whole-image MOSS pass, and 1.049 for this node's tiled pass. USDU scores lower because it resamples every tile, which blurs that pattern along with the detail.

## Other models

Nothing model-specific is hard-coded. Everything that depends on the model or the VAE has an automatic default that can be overridden:

| Value | Default | Override |
|---|---|---|
| Latent scale (pixels per latent cell) | read from the VAE (`vae.spacial_compression_encode()`): 8 for SD/SDXL/Flux.1/Z-Image, 16 for FLUX.2/MOSS-Image | `latent_scale` |
| Pixel grid for canvas, tiles and rings | the latent scale | `pixel_multiple`, e.g. 32 for a 16x VAE with a DiT patch size of 2 (MiniMax H3), or 16 for Flux.1 if you want patch-exact tiles |
| Tile cap | 1024, or 768 for SD1.x/2.x | `max_tile_width` / `max_tile_height` |
| Ring widths | your minimums, rounded up to whole latent cells and the grid | `context_anchor` / `context_overlap` |

## What the fork changes

- **Any latent scale.** Upstream assumed 8x in about 30 places (`// 8`, `scale=8`, padding to /8, the grid's `multiple=8`, the mask snap). A `LatentGeometry` (latent cell size + pixel grid, `geometry.py`) is now read from the VAE and threaded through the base node's whole raster path: canvas padding, grid solve, the noise canvas and its per-tile slices, denoise masks, region gates and the mask bounding box. Rings are snapped onto whole latent cells. With an 8x VAE the code path is unchanged, and upstream's test suite passes untouched.
- **Stepwise Tile Upscale**, a new multi-step node with its own planner (`stepwise_plan.py`, pure Python).
- **Tile Refine (Any VAE)**, the base node with `latent_scale` / `pixel_multiple` overrides.
- Only these two nodes are registered, under their own class names. Upstream's VL nodes and tile test nodes stay in the code (they were built for Krea 2's Qwen3-VL encoder and are not adapted to non-8x VAEs) but are not registered.
- No runtime dependency: `logit-classifier` was only needed by the VL nodes and is now a test extra.

## Known limitations

- The VL nodes from upstream are not registered and still assume an 8x VAE.
- Tiles are refined one after another in raster order (upstream's base engine). A seam is hidden by context rings, a thin directional feather, a minimum-error cut and a DC match, not prevented. Very smooth gradients may need a wider `context_overlap`.
- On the final step, and on intermediate steps the planner could not snap within `aspect_tolerance`, the last tile of an axis can be narrower than the others.
- `plan_only` still runs the nodes upstream of it (model and text encoder loading), since ComfyUI executes the inputs first.
- An upscale model runs only on step 1. A fixed 4x model on a large source can still produce a big image there; the plan warns above 40 MP.
- Randomly shifted tile grids per step (TiledKSampler-style) are not implemented yet. The grid code is kept separate from the planner so that such a tile mode can be added later.

## Tests

```
pip install pytest logit-classifier
COMFYUI_ROOT=/path/to/ComfyUI python -m pytest tests -m "not gpu"
```

`tests/test_stepwise.py` covers the fork: the geometry, the planner's invariants (exact final size, no remainder tiles, every size on the grid and under the cap, no resampling), and both nodes end to end with fake VAEs at 8x (bit-identical to upstream), 16x with 128 channels (FLUX.2 / MOSS-Image) and a 32 px grid (MiniMax H3). The rest is upstream's suite.

## License

[GPLv3](LICENSE). Original work © Blakeem, [ComfyUI-ContextAnchoredTileRefine](https://github.com/Blakeem/ComfyUI-ContextAnchoredTileRefine). Changes in this fork are released under the same license.
