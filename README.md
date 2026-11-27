# Stepwise Tile Refine

ComfyUI nodes for tiled upscaling that work with **any VAE latent scale**, including the 16x FLUX.2 VAE that **MOSS-Image** uses, and a node that upscales in **several passes** and plans those passes itself.

> **Fork notice.** This is a fork of [Blakeem/ComfyUI-ContextAnchoredTileRefine](https://github.com/Blakeem/ComfyUI-ContextAnchoredTileRefine) (GPL-3.0). The tile engine (context rings, shared noise, directional feather, minimum-error seam cut, DC match) is upstream's work. What this fork changes is listed [below](#what-the-fork-changes). The upstream README is kept in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).
>
> The fork is maintained when time allows and comes with no support commitment.

## Contents

- [Installation](#installation)
- [The nodes](#the-nodes)
  - [Stepwise Tile Upscale](#stepwise-tile-upscale)
  - [Tile Refine (Any VAE)](#tile-refine-any-vae)
- [Tile modes](#tile-modes)
- [How the passes are planned](#how-the-passes-are-planned)
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

Upscales in several passes. Each pass enlarges the whole image in pixel space and then refines it tile by tile, with the strength (start noise level, or denoise) falling from the first pass to the last. An axis with less than a grid cell to grow keeps its length for a pass, so with many small passes a pass can refine at the same size ([How the passes are planned](#how-the-passes-are-planned)). "Steps" always means sampler steps.

| Input | What it does |
|---|---|
| `image`, `guider`, `sampler`, `noise`, `vae` | As for `SamplerCustomAdvanced`. The VAE's latent scale is read automatically. |
| `upscale_by` | Output size: the final image is `round(input × upscale_by)` on each axis (`output_size` says how it lands there). `1` keeps the input size. |
| `output_size` | How the output lands on `round(input × upscale_by)` when that is not a multiple of the pixel grid (16 px for MOSS). `exact, crop edges` (default): the last pass runs a few px larger without stretching and is centre-cropped. `exact, pad edges during denoise`: the right and bottom edges are padded with a mirrored strip while the last pass denoises and cut off after; the model may misread the outermost pixels. `stretch image`: the output is a grid size near the target that keeps the aspect. The rounding of the shorter axis is matched on the longer one, so both are off by about the same percentage (2147×344 becomes 2208×352 on a 32 px grid) and the aspect changes only slightly. The plan says by how much. With `oversample` the final resample hits the exact size anyway. |
| `passes` | Number of enlarge + refine passes. `0` picks the fewest whose growth per pass stays at or under `auto_pass_factor`. Called `upscale_steps` before 2026-11-27. |
| `auto_pass_factor` | Largest growth per pass when `passes` is `0` (default 1.5, so 2 passes for x2 and 3 for x3). Unused when `passes` is set. Called `auto_step_factor` before 2026-11-27. |
| `oversample` | Runs the refine passes this much larger than the output track and shrinks onto the exact output size after the last pass. The largest pass is `input × upscale_by × oversample`. `1` is off. See [The repair loop](#the-repair-loop). |
| `oversample_mode` | `constant`: the first pass jumps to the oversample and every pass keeps it. `ramp up`: the passes grow evenly and reach it on the last pass. |
| `upscale_method` / `downscale_method` | The filters for enlarging and for the shrink after the last pass, soft to sharp: `bilinear` (enlarge only), `area` (shrink only), `bicubic (B-spline)`, `bicubic (mitchell)`, `bicubic (catmull-rom)`, `lanczos`. Defaults `lanczos` and `area`. |
| `denoise_mode` | `noise level` (default): `denoise_first` / `denoise_last` are the share of noise each pass starts from, the same for every scheduler. `denoise`: they are `BasicScheduler` denoise values. The plan shows both for every pass. |
| `denoise_first` / `denoise_last` | Strength of the first and the last pass, three decimals. The passes in between are interpolated linearly per pass. Since every pass enlarges by about the same factor, that is the same drop per factor of size. |
| `scheduler`, `steps` | How each pass's noise levels are spaced, and how many sampler steps each tile gets. See [Noise level or denoise](#noise-level-or-denoise). |
| `steps_mode` | How each pass gets its steps; the plan shows how many every pass samples. `same step size` (default): `steps` (default 50) is a full generation from pure noise, and every pass samples the part of that schedule below its own start, rounded up to whole steps, so every pass follows that schedule's step sizes and a pass with little noise is quick. In `noise level` mode that part is scaled down to start exactly at the level, which shrinks its steps a little (none is longer than the full schedule's). On MOSS (shift 5, `simple`) a pass from noise level 0.40 gets 6 of 50 steps. `proportional to noise`: every pass samples `steps` x its start noise level (or denoise), rounded up, so the noise falls by the same amount per step on average. `steps per pass`: every pass samples `steps`. No pass repeats a sigma: `simple`, `beta` and `ddim_uniform` pick from the model's 1000 timesteps and have few at low noise (on MOSS about 48 below 0.2), so a pass may sample fewer steps than asked rather than start higher than its level, and the plan says why. |
| `tile_mode` | How the tiles are run: `tile by tile` (default), `all at once` or `all at once, shifting`. See [Tile modes](#tile-modes). |
| `tile_size` | `always full` (default): every window the model sees is grown to exactly `max_tile`. `allow smaller`: windows keep their bare size, so edge tiles can be smaller. |
| `max_tile_width` / `max_tile_height` | Largest size the model sees per tile, overlap and padding included. `0` picks 1024 for SDXL/Flux/MOSS-class models and 768 for SD1.x/2.x. |
| `tile_overlap` | The band where neighbouring tiles overlap and are blended, outward from each core. Rounded up to whole latent cells. Unused in `all at once, shifting`. |
| `tile_padding` | Context outside the overlap band that the model sees and that is thrown away afterwards: frozen in `tile by tile`, live in the `all at once` modes. Rounded up to whole latent cells and the pixel grid. |
| `plan_only` | Computes and outputs the plan without sampling, so you can check it before a long run. |
| `upscale_model` (optional) | Used for the first pass only, then `upscale_method` to that pass's exact size. Later passes use `upscale_method` only, so a 4x model never runs on a large intermediate. |
| `latent_scale`, `pixel_multiple` (advanced) | See [Other models](#other-models). |

Outputs: the final `image`, `pass_images` (the result of every pass, as a list since the sizes differ) and `plan`, plain text that is also written to the log when the run starts. Connect `plan` to *Preview as Text*. Node outputs reach other nodes only when the whole node is done, so set `plan_only` to see the plan before a long run.

While it runs, the node shows one progress bar over the whole run, sized in advance from the plan's tile steps, with a status line under it: `pass 1/3 · upscale model`, then `pass 2/3 · tile 5/12 · step 14/18` in `tile by tile` (each tile takes all its steps) or `pass 2/3 · step 14/18 · pos · tile 3/6` in the `all at once` modes (each step goes through all tiles). Tile Refine (Any VAE) shows the same bar and line, without the pass, over the whole batch.

With cfg above 1 the tiles of the `all at once` modes go round twice per step, `pos` and then `neg`: core asks the model for the positive and the negative prompt's answer as two calls whenever it cannot batch them, which on MOSS-Image is whenever the two prompts have different token counts. It costs no time: measured on MOSS-Image, one batched call took exactly as long as the two. When they are batched the line shows neither word and the tiles go round once.

Example plan, MOSS-Image (shift 5), 1280x960 x3, the defaults (auto passes, noise level 0.35 -> 0.25, 50 steps, same step size):

```text
STEPWISE TILE UPSCALE PLAN
Source 1280x960, x3 -> 3840x2880, 3 passes.
Steps: same step size, each pass's share of one 50-step schedule over the whole noise range (simple).
Tile mode: tile by tile, always full. Max tile 1024x1024 px, rings included.
Rings: tile_overlap 32 px, tile_padding 32 px. Pixel grid 16 px.

Pass 1, noise level 0.350 (~ denoise 0.097), 5 steps
  enlarge  1280x960 -> 1856x1392    x1.450, lanczos
  denoise  1856x1392                2x2 tiles, core 928x704/688

Pass 2, noise level 0.300 (~ denoise 0.079), 4 steps
  enlarge  1856x1392 -> 2672x2000   x1.440 / x1.437, lanczos, aspect +0.20%
  denoise  2672x2000                3x3 tiles, core 896/880x672/656

Pass 3, noise level 0.250 (~ denoise 0.063), 4 steps
  enlarge  2672x2000 -> 3840x2880   x1.437 / x1.440, lanczos
  denoise  3840x2880                5x4 tiles, core 768x720

Output 3840x2880. Total 33 tiles, 136 tile steps (tiles x steps).
```

### Noise Level Scheduler

`BasicScheduler` with the start noise level as its setting instead of denoise (category `sampling/custom_sampling/schedulers`). It outputs `SIGMAS` for any sampler node, plus an `info` string with the denoise it equals. Use it with *Tile Refine (Any VAE)*, `SamplerCustom` or other upscalers.

### Tile Refine (Any VAE)

Upstream's base node (*Context-Anchored Tile Refine*) with the latent scale read from the VAE instead of assumed to be 8. Upscale the image first and feed it in; denoise is set through the sigmas (`BasicScheduler`'s `denoise`). An optional `mask` refines only the masked region (tile by tile only). It has the same `tile_mode`, `tile_size`, `tile_overlap` and `tile_padding` widgets as the Stepwise node (upstream calls the last two `context_overlap` and `context_anchor`). With `tile by tile` and `tile_size` `allow smaller` on an 8x VAE it gives byte-identical output to the upstream node.

## Tile modes

`tile_mode` picks how a refine pass runs its tiles. Both nodes have it, with the same options.

| Mode | How it runs | The seam |
|---|---|---|
| `tile by tile` | The tiles run one after another in raster order. Each tile is denoised to the end with the full schedule before the next starts. Its ring is frozen: finished neighbours show their finished pixels, the rest show the source. The overlap band is cross-faded into the finished neighbour afterwards. | Hidden by the tile continuing its finished neighbours. |
| `all at once` | The same tiles as tile by tile, but all of them take every sampler step together. In the overlap bands both neighbours' answers are weighted together at every step. The ring sits at the same noise level as the rest of the image and is only context. Nothing is frozen. | The seams stay in place, but both sides share one band at every step. |
| `all at once, shifting` | All tiles take every step together on a grid that is shifted by a random x/y offset (drawn from the seed) before every step. The core is `max_tile` minus `tile_padding` on both sides, cores meet edge to edge and `tile_overlap` is unused. | The seams move every step, so none builds up. |

The two `all at once` modes put the tiling inside the model call: the sampler runs one ordinary sampling run over the whole image, and every model evaluation is split into tile windows and put back together. So every sampler works, including ones with history and per-step (ancestral / SDE) noise, and that noise is drawn for the whole image. Windows that come out identical run once. Measured on MOSS-Image at 8 MP with a single-subject prompt at noise level 0.95: `tile by tile` showed clear seams, `all at once` at most faint ones, and `all at once, shifting` none, at about 1.8x the time of the other two.

`tile_size` applies to all three modes. `always full` grows every window to exactly `max_tile` (inward at the image edge): in `tile by tile` and `all at once` the ring grows and the core and band stay the same, and in `all at once, shifting` a thin edge tile slides inward. `allow smaller` keeps the bare windows (cheaper).

## How the passes are planned

1. The final size is `round(source × upscale_by)`, landed on as `output_size` says.
2. Each pass enlarges by about `factor^(1/passes)`. Every intermediate size is a multiple of the pixel grid (the VAE encodes whole latent cells, so nothing is padded with a mirrored strip): one near the ideal size that weighs size against aspect deviation. Each axis follows its own curve: it never shrinks, never passes the canvas the last pass denoises, and keeps its length for a pass while its ideal has not reached the next grid cell. A pass where neither axis grows refines at the same size with its own seed. On a 16 px grid with automatic passes the aspect stays within about 0.9 % and each pass's factor within about 2.5 % of the even factor, worst cases on 0.5 MP sources. With many small passes both can be off by up to a grid cell of the shorter side (about 1.2 % and 4 % there). The last tile of an axis can be narrower than the others; with `tile_size` `always full` the model still sees a full window. The plan shows the factor and the aspect deviation of every pass.
3. Tiles are never resampled. A tile is cut from the canvas, sampled and pasted back at its native size.
4. The strength goes linearly per pass from `denoise_first` to `denoise_last`, as a noise level or a `BasicScheduler` denoise (`denoise_mode`).

## The repair loop

Enlarging makes small objects bigger in pixels, and the tiles then redraw them well. That repairs AI deformities in small objects. `oversample` uses this: the refine passes run `oversample` times larger than the output track, and one resample after the last pass brings the image onto the exact output size. The largest pass is the last, at `input × upscale_by × oversample`.

- `constant`: pass *i* runs at `upscale_by^(i/passes) × oversample`. Each pass is "enlarge by U, refine, shrink by D" with U = `upscale_by^(1/passes) × oversample` and D = `oversample`. Between two passes the shrink and the next enlargement are one resample, which tested on MOSS-Image as good as doing them apart and less soft. At `upscale_by` 1 every pass runs at one size.
- `ramp up`: pass *i* runs at `(upscale_by × oversample)^(i/passes)`, growing evenly to the largest size.

Set `denoise_first` = `denoise_last` for a constant noise level. Recipes, 5 passes:

| Recipe | upscale_by | oversample | oversample_mode |
|---|---|---|---|
| enlarge x1.5, refine, shrink back, five times | 1 | 1.5 | constant |
| grow slowly to x1.5, then shrink back once | 1 | 1.5 | ramp up |
| x2 with a repair in every pass (about x1.49 up, /1.3 down per pass) | 2 | 1.3 | constant |
| x2 through x3, area down at the end | 2 | 1.5 | ramp up |

Each pass draws its noise on seed + pass − 1, so passes at the same size never reuse each other's noise. Same-seed passes reproduce each other's invented detail and overcook the image.

Every shrink widens its filter by the shrink factor, so all source pixels count however large the reduction ("precise" resampling). `area` is the exact mean of the covered pixels and never rings. The bicubic variants are the Mitchell–Netravali family: B-spline is the softest with no ringing, Catmull-Rom (PIL's "bicubic") the sharpest. `lanczos` is Lanczos-3, the same filter as core's lanczos but without its round trip through 8 bits. Every filter runs in float32, so the passes never round the image to 0–255.

## Noise level or denoise

`BasicScheduler` turns a denoise value into sigmas by scheduling `int(steps / denoise)` steps and keeping the last `steps + 1`. Where that tail starts depends on the scheduler and on the model's shift, so the same denoise means very different amounts of noise. On MOSS-Image (flow, shift 5):

| denoise | 0.05 | 0.10 | 0.22 | 0.30 | 0.45 | 0.60 |
|---|---|---|---|---|---|---|
| start noise, `simple` | 0.21 | 0.36 | 0.59 | 0.68 | 0.81 | 0.89 |
| start noise, `beta` | 0.11 | 0.18 | 0.46 | 0.61 | - | - |
| start noise, `karras` | 0.007 | 0.010 | 0.023 | 0.037 | 0.086 | 0.19 |

`karras` ignores the model's shift, so its low denoise values barely sample at all on flow models.

In `noise level` mode the node searches for the schedule length whose tail starts nearest the requested level, then cuts the tail like `BasicScheduler`. The level is the flow sigma (share of noise in the sample, 0..1); on SD/SDXL-type models it is `sigma / (1 + sigma)`, so it means about the same amount of noise there. The schedule length is a whole number, so most schedulers hit the level to within about 0.005. Where no length does, the node takes the nearest tail that starts above the level and scales it down to start exactly at it, so no sampler step gets longer. That happens with `ddim_uniform`, which has only about 1000 / steps distinct tails (0.40 at 50 steps landed on 0.357 before), and with `karras`, `exponential` and `kl_optimal` near full noise or with few steps. A scaled tail matches no `denoise` setting, so the plan shows its denoise with "~". The same level does not make two different models change an image by the same amount.

Measured on MOSS-Image, at the same start noise `simple`, `beta`, `euler`, `dpmpp_2m` and `er_sde` gave nearly the same result, and cfg 2 to 4.5 made no measurable difference at these levels. The start noise is what matters.

## MOSS-Image settings

What the tests in this repository used, and a good starting point:

- Diffusion model: `moss_image_sft_int8_convrot` (Load Diffusion Model), text encoder `qwen3.5_9b_w4a8_gptq` (Load CLIP, type `moss_image`), VAE `flux2-vae`.
- `CFGGuider` cfg 3, sampler `er_sde`, scheduler `simple`, 25 steps. Shift 5 is set by the model automatically.
- Max tile `0` (auto = 1024, about 1 MP per tile, within MOSS's 1-2 MP sweet spot), `tile_overlap` 32, `tile_padding` 32.
- Noise level 0.35 -> 0.25 (the default). A ground-truth sweep (a MOSS image downscaled, upscaled back in one pass and compared with the original) found about 0.21-0.27 per x1.5 pass and 0.36-0.40 per x2 pass the most that sharpens without changing content. From about 0.47 (denoise 0.15) MOSS starts inventing things (an object in a dark window, redrawn wires), and from about 0.68 (denoise 0.3) skin turns waxy and texture painterly. The tests below ran at the older default, denoise 0.45 -> 0.22 (start noise 0.81 -> 0.59).
- The positive prompt reaches every tile. Objects it names (say "seagulls") get drawn into any tile with a vaguely matching shape, more so at higher denoise. For upscaling, prefer a prompt that describes the style, medium and lighting rather than the objects, or lower the denoise (0.35 -> 0.18 cut the colour drift of the 3-pass test by a fifth).
- Do not start ComfyUI with `--use-sage-attention`: it produces broken images with MOSS-Image.
- MOSS's own guide advises against latent upscaling. These nodes enlarge in pixel space and re-encode, as recommended.

On an RTX 4090 one 1 MP tile takes about 5 s at 25 steps (int8 model, cfg 3).

### Measured with MOSS-Image

Same seed, prompt, sampler and denoise per pass, against Ultimate SD Upscale (custom sample, lanczos, tile 1024 + padding 64, chained once per pass). Harness: [tests-moss/](tests-moss/).

| Case | Passes | Tiles | Time | Final size | USDU final size | USDU time |
|---|---|---|---|---|---|---|
| 1280x960 x3 | 1 | 20 | 100 s | 3840x2880 | 3840x2880 | 98 s |
| 1280x960 x3 | 3 | 33 | 163 s | 3840x2880 | 3840x**2888** | 177 s |
| 1280x976 x3 | 3 | 33 | 167 s | 3840x2928 | 3840x2928 | 177 s |
| 2000x2000 x2 | 2 (auto) | 41 | 205 s | 4000x4000 | **4008x4008** | 203 s |
| 1344x768 x2 | 2 (auto) | 10 | 50 s | 2688x1536 | **2696**x1536 | 80 s |

- The final size was exact in every case. USDU missed it in three of five (stretched to a multiple of 8 after each pass).
- No seam line stood out from the image's own edges in any output (`tests-moss/analyze.py` ranks every row and column against its neighbourhood).
- MOSS img2img brightens slightly at these denoise values (+1 to +6 /255 over a 2-3 pass chain). USDU shows the same shift, so it comes from the model, not the tiling.
- A faint 16 px pattern on the latent grid is a property of the FLUX.2 VAE and MOSS, not of the tiles: at x1.5 / denoise 0.45 the grid-periodicity score was 1.028 for lanczos only, 1.042 after a plain VAE round trip, 1.066 for one untiled whole-image MOSS pass, and 1.049 for this node's tiled pass. USDU scores lower because it resamples every tile, which blurs that pattern along with the detail.

## Other models

Nothing model-specific is hard-coded. Everything that depends on the model or the VAE has an automatic default that can be overridden:

| Value | Default | Override |
|---|---|---|
| Latent scale (pixels per latent cell) | read from the VAE (`vae.spacial_compression_encode()`): 8 for SD/SDXL/Flux.1/Z-Image, 16 for FLUX.2/MOSS-Image | `latent_scale` |
| Pixel grid for canvas, tiles and rings | the latent scale | `pixel_multiple`, e.g. 32 for a 16x VAE with a DiT patch size of 2 (MiniMax H3), or 16 for Flux.1 if you want patch-exact tiles |
| Tile cap | 1024, or 768 for SD1.x/2.x | `max_tile_width` / `max_tile_height` |
| Ring widths | your minimums, rounded up to whole latent cells and the grid | `tile_overlap` / `tile_padding` |

## What the fork changes

- **Any latent scale.** Upstream assumed 8x in about 30 places (`// 8`, `scale=8`, padding to /8, the grid's `multiple=8`, the mask snap). A `LatentGeometry` (latent cell size + pixel grid, `geometry.py`) is now read from the VAE and threaded through the base node's whole raster path: canvas padding, grid solve, the noise canvas and its per-tile slices, denoise masks, region gates and the mask bounding box. Rings are snapped onto whole latent cells. With an 8x VAE the code path is unchanged, and upstream's test suite passes untouched.
- **Stepwise Tile Upscale**, a new multi-pass node with its own planner (`stepwise_plan.py`, pure Python).
- **Tile Refine (Any VAE)**, the base node with `latent_scale` / `pixel_multiple` overrides.
- **Tile modes** (`all_at_once.py`): `all at once` and `all at once, shifting` next to upstream's raster engine, and `tile_size` `always full`. See [Tile modes](#tile-modes).
- Only these two nodes are registered, under their own class names. Upstream's VL nodes and tile test nodes stay in the code (they were built for Krea 2's Qwen3-VL encoder and are not adapted to non-8x VAEs) but are not registered.
- No runtime dependency: `logit-classifier` was only needed by the VL nodes and is now a test extra.

## Known limitations

- The VL nodes from upstream are not registered and still assume an 8x VAE.
- In `tile by tile`, tiles are refined one after another in raster order (upstream's base engine). A seam is hidden by context rings, a thin directional feather, a minimum-error cut and a DC match, not prevented. Very smooth gradients may need a wider `context_overlap`.
- `plan_only` still runs the nodes upstream of it (model and text encoder loading), since ComfyUI executes the inputs first.
- An upscale model runs only on pass 1. A fixed 4x model on a large source can still produce a big image there; the plan warns above 40 MP.
- The `all at once` modes take no `mask` and no ControlNet. They need a guider that keeps its options in `model_options` (CFGGuider and core's guiders do); with any other guider the run stops with an error after the first step instead of running the model untiled.

## Tests

```
pip install pytest logit-classifier
COMFYUI_ROOT=/path/to/ComfyUI python -m pytest tests -m "not gpu"
```

`tests/test_stepwise.py` covers the fork: the geometry, the planner's invariants (exact final size, no remainder tiles, every size on the grid and under the cap, no resampling), and both nodes end to end with fake VAEs at 8x (bit-identical to upstream), 16x with 128 channels (FLUX.2 / MOSS-Image) and a 32 px grid (MiniMax H3). The rest is upstream's suite.

## License

[GPLv3](LICENSE). Original work © Blakeem, [ComfyUI-ContextAnchoredTileRefine](https://github.com/Blakeem/ComfyUI-ContextAnchoredTileRefine). Changes in this fork are released under the same license.
