"""The all at once tile modes: the whole canvas is denoised together, one model call at a time, with
the tiling INSIDE the model call. torch + stdlib at module scope, comfy lazy.

The sampler runs ONE ordinary `guider.sample()` over the whole canvas latent (MultiDiffusion /
Tiled Diffusion style). A `model_function_wrapper` splits every `model(x, sigma)` into tile
windows, runs each window, and assembles the canvas-sized answer. The sampler never sees a tile,
so every sampler works, multistep history and per-step (ancestral / SDE) noise included, and that
noise is drawn for the whole canvas, so it has no seams either. In every window the ring around
what the tile writes sits at the same noise level as the rest of the canvas (nothing is frozen)
and is thrown away.

Two grids:

    all at once            the planner's fixed grid, the same tiles as tile by tile. A tile writes
                           its core plus the tile_overlap band on every neighbour side, and in
                           the bands the neighbours' answers are weighted together smoothly at
                           every call, so the two sides of a seam share one band every step.
    all at once, shifting  a grid of max_tile minus the ring on both sides, shifted by a random
                           x/y offset drawn from (seed, sigma) at every sampler step. The cores
                           butt and cover every cell exactly once, with no band: the seams move.
                           Every call at one sigma (cond and uncond, which MOSS runs separately)
                           shares one grid, and a run replays from its seed.

tile_size: `always full` grows every window to max_tile (inward at the canvas edge), so the model
always samples its usual size on real picture; `allow smaller` keeps the bare windows. Windows that
come out identical run once, and every core in them is written from that one answer."""
import contextlib
import copy
import hashlib
import logging
import random
import struct
from dataclasses import dataclass, field

import torch

from . import grid

TILE_BY_TILE = "tile by tile"
ALL_AT_ONCE = "all at once"
SHIFTING = "all at once, shifting"
TILE_MODES = (TILE_BY_TILE, ALL_AT_ONCE, SHIFTING)

SIZE_FULL = "always full"
SIZE_SMALLER = "allow smaller"
TILE_SIZES = (SIZE_FULL, SIZE_SMALLER)

TILE_MODE_TOOLTIP = (
    "tile by tile: the tiles run one after another in raster order, and each is denoised to the end with the "
    "full schedule before the next starts. Its ring is frozen: finished neighbours show their finished pixels, "
    "the rest show the source, and the overlap band is cross-faded into the finished neighbour afterwards. "
    "all at once: the same tiles, but all of them take every sampler step together. In the overlap bands both "
    "neighbours' answers are weighted together at every step, and the ring sits at the same noise level and is "
    "only context. Nothing is frozen. "
    "all at once, shifting: all tiles take every step together on a grid that is shifted randomly before every "
    "step, so the seams move. The core is max_tile minus tile_padding on both sides, cores meet edge to edge "
    "and tile_overlap is unused.")
TILE_SIZE_TOOLTIP = (
    "always full: every window the model sees is grown to exactly max_tile (inward at the image edge). In tile "
    "by tile and all at once the ring grows and the core and band stay the same. In all at once, shifting a "
    "thin edge tile slides inward. "
    "allow smaller: windows keep their bare size, so edge tiles can be smaller (cheaper).")


# --- geometry (cells) ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Span:
    """One tile along one axis, in latent cells: the core it writes [core0, core1) and the window
    the model sees [crop0, crop1), always inside the canvas."""
    core0: int
    core1: int
    crop0: int
    crop1: int


def tile_cells(max_tile_width, max_tile_height, ring_px, geometry):
    """Shifting grid: (core_h, core_w, ring) in latent cells. The ring rounds up to the pixel grid,
    the cap down to it, and the core is the cap minus the ring on both sides."""
    ring_px = geometry.round_up(int(ring_px))
    cores = []
    for name, cap in (("max_tile_height", max_tile_height), ("max_tile_width", max_tile_width)):
        core_px = geometry.round_down(int(cap)) - 2 * ring_px
        if core_px < geometry.multiple:
            raise ValueError(
                f"All at once, shifting: {name} {cap} leaves no core inside a {ring_px} px ring on both sides. "
                f"Raise {name} or lower tile_padding.")
        cores.append(core_px // geometry.scale)
    return cores[0], cores[1], ring_px // geometry.scale


def axis_spans(length, core, ring, offset, tile_size, unit=1):
    """Shifting grid: the tiles along one axis of `length` cells for a grid shifted by `offset`
    cells (0 <= offset < core). An axis that fits one window is one span with no seam at all."""
    if tile_size not in TILE_SIZES:
        raise ValueError(f"unknown tile_size {tile_size!r}, expected one of {TILE_SIZES}")
    full = core + 2 * ring
    if length <= full:
        return [Span(0, length, 0, length)]
    if core % unit or ring % unit or offset % unit or length % unit:
        raise ValueError(f"shifting spans off the {unit}-cell grid: length {length}, core {core}, ring {ring}, offset {offset}")
    spans = []
    for start in range(-offset, length, core):
        core0, core1 = max(start, 0), min(start + core, length)
        if core1 <= core0:
            continue
        crop0, crop1 = max(core0 - ring, 0), min(core1 + ring, length)
        if tile_size == SIZE_FULL:
            # Only an edge tile is short: slide its window inward to the full cap. The axis is
            # longer than the cap here, so the window always fits.
            if crop0 == 0:
                crop1 = max(crop1, full)
            if crop1 == length:
                crop0 = min(crop0, length - full)
        spans.append(Span(core0, core1, crop0, crop1))
    return spans


def grid_offsets(seed, sigma, core_h, core_w, unit=1):
    """Shifting grid: the (dy, dx) offset for the call at `sigma`, in cells on the unit grid. Drawn
    from blake2b(seed, sigma) so every call at one sigma shares a grid and a run replays."""
    digest = hashlib.blake2b(struct.pack("<Qd", int(seed) % (1 << 64), float(sigma)), digest_size=8).digest()
    rng = random.Random(int.from_bytes(digest, "little"))
    return unit * rng.randrange(core_h // unit), unit * rng.randrange(core_w // unit)


# --- windows ------------------------------------------------------------------------------------

@dataclass(eq=False)
class Write:
    """A canvas region [y0, y1) x [x0, x1) in cells filled from a window's answer. weight None is
    a hard write (shifting cores, which never overlap); else a [h, w] float weight (the fixed
    grid's bands), and the canvas answer is the weighted mean of every write covering a cell."""
    y0: int
    y1: int
    x0: int
    x1: int
    weight: torch.Tensor | None = None


@dataclass(eq=False)
class Window:
    y0: int
    y1: int
    x0: int
    x1: int
    writes: list = field(default_factory=list)


def merge_windows(pairs):
    """[(window rect, Write)] -> [Window], one per distinct rect in first-seen order, so a window
    two tiles share runs once and writes all their regions from the same answer."""
    out = {}
    for rect, write in pairs:
        if rect not in out:
            out[rect] = Window(*rect)
        out[rect].writes.append(write)
    return list(out.values())


def shifting_windows(height, width, core_h, core_w, ring, offsets, tile_size, unit=1):
    dy, dx = offsets
    rows = axis_spans(height, core_h, ring, dy, tile_size, unit)
    cols = axis_spans(width, core_w, ring, dx, tile_size, unit)
    return merge_windows(((r.crop0, r.crop1, c.crop0, c.crop1), Write(r.core0, r.core1, c.core0, c.core1))
                         for r in rows for c in cols)


def _smoothstep(t):
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _axis_weight(w0, w1, core0, core1, band, before, after):
    # 1-D weight over the write span [w0, w1): 1 in the core, a smooth ramp over the 2*band cells
    # centred on each core edge that has a neighbour, sampled at cell centres. A neighbour's ramp
    # over the same cells is the complement, so the two sum to 1 there.
    u = torch.arange(w0, w1, dtype=torch.float32) + 0.5
    w = torch.ones(w1 - w0)
    if band > 0 and before:
        w = w * _smoothstep((u - (core0 - band)) / (2 * band))
    if band > 0 and after:
        w = w * (1.0 - _smoothstep((u - (core1 - band)) / (2 * band)))
    return w


def fixed_windows(layout, scale):
    """The fixed grid (all at once) in cells: each tile's window is its crop_rect, and it writes its
    overlap_inner_rect (core + band on every neighbour side) with smooth weights in the bands."""
    band = layout.overlap // scale
    pairs = []
    for t in layout.tiles:
        crop, inner, core = t.crop_rect, t.overlap_inner_rect, t.core
        wy = _axis_weight(inner.y0 // scale, inner.y1 // scale, core.y0 // scale, core.y1 // scale, band, t.nb.top, t.nb.bottom)
        wx = _axis_weight(inner.x0 // scale, inner.x1 // scale, core.x0 // scale, core.x1 // scale, band, t.nb.left, t.nb.right)
        weight = None if band == 0 else wy[:, None] * wx[None, :]
        rect = (crop.y0 // scale, crop.y1 // scale, crop.x0 // scale, crop.x1 // scale)
        if band == 0:
            write = Write(core.y0 // scale, core.y1 // scale, core.x0 // scale, core.x1 // scale)
        else:
            write = Write(inner.y0 // scale, inner.y1 // scale, inner.x0 // scale, inner.x1 // scale, weight)
        pairs.append((rect, write))
    return merge_windows(pairs)


# --- the wrapper --------------------------------------------------------------------------------

class ShiftingPlanner:
    def __init__(self, core_h, core_w, ring, seed, tile_size=SIZE_FULL, unit=1):
        if tile_size not in TILE_SIZES:
            raise ValueError(f"unknown tile_size {tile_size!r}, expected one of {TILE_SIZES}")
        self.core_h, self.core_w, self.ring = int(core_h), int(core_w), int(ring)
        self.seed = int(seed) % (1 << 64)
        self.tile_size = tile_size
        self.unit = int(unit)

    def __call__(self, height, width, sigma):
        offsets = grid_offsets(self.seed, sigma, self.core_h, self.core_w, self.unit)
        windows = shifting_windows(height, width, self.core_h, self.core_w, self.ring, offsets, self.tile_size, self.unit)
        return offsets, windows


class FixedPlanner:
    def __init__(self, windows, height, width):
        self.windows, self.height, self.width = windows, height, width

    def __call__(self, height, width, sigma):
        if (height, width) != (self.height, self.width):
            raise RuntimeError(f"all at once: the model got a {width}x{height} latent, the grid is {self.width}x{self.height}")
        return (0, 0), self.windows


class TiledModelWrapper:
    """The `model_function_wrapper` that runs one model call as tile windows. `inner` is a wrapper
    already on the model, run per window in place of apply_model. `log`, when a list, records
    (sigma, dy, dx, windows) per call for tests. `check_interrupt` runs before every window, so a
    cancel lands within one window rather than one whole step. `calls` counts the model calls it
    has handled, which is how the run proves the guider actually used it."""

    def __init__(self, planner, inner=None, log=None, check_interrupt=None, hub=None):
        self.planner = planner
        self.inner = inner
        self.log = log
        self.check_interrupt = check_interrupt
        self.calls = 0
        # preview_mode (preview.py): the hub sees every window's cond answer, and `emit` sends what
        # it renders. Both None by default.
        self.hub = hub
        self.emit = None

    def __call__(self, apply_model, args):
        x, timestep, c = args["input"], args["timestep"], args["c"]
        if c.get("control") is not None:
            raise ValueError("All at once tile modes: ControlNet is not supported.")
        height, width = int(x.shape[-2]), int(x.shape[-1])
        sigma = float(timestep.reshape(-1)[0])
        (dy, dx), windows = self.planner(height, width, sigma)
        self.calls += 1
        if self.log is not None:
            self.log.append((sigma, dy, dx, len(windows)))
        out = wsum = None
        for win in windows:
            if self.check_interrupt is not None:
                self.check_interrupt()
            ys, xs = slice(win.y0, win.y1), slice(win.x0, win.x1)
            tile_c = {key: _slice_spatial(value, height, width, ys, xs) for key, value in c.items()}
            tile_x = x[..., ys, xs]
            if self.inner is not None:
                result = self.inner(apply_model, {"input": tile_x, "timestep": timestep, "c": tile_c,
                                                  "cond_or_uncond": args.get("cond_or_uncond")})
            else:
                result = apply_model(tile_x, timestep, **tile_c)
            if self.hub is not None:
                self._preview(win, tile_x, result, args.get("cond_or_uncond"), windows)
            if out is None:
                out = result.new_zeros((*result.shape[:-2], height, width))
                if any(w.weight is not None for v in windows for w in v.writes):
                    wsum = result.new_zeros((height, width))
            for w in win.writes:
                src = result[..., w.y0 - win.y0:w.y1 - win.y0, w.x0 - win.x0:w.x1 - win.x0]
                if w.weight is None:
                    out[..., w.y0:w.y1, w.x0:w.x1] = src
                    if wsum is not None:
                        wsum[w.y0:w.y1, w.x0:w.x1] = 1.0
                else:
                    weight = w.weight.to(device=result.device, dtype=result.dtype)
                    out[..., w.y0:w.y1, w.x0:w.x1] += src * weight
                    wsum[w.y0:w.y1, w.x0:w.x1] += weight
        if wsum is not None:
            out = out / wsum
        return out

    def _preview(self, win, tile_x, result, cond_or_uncond, windows):
        # The cond chunk of this window (the batch is the cond/uncond chunks concatenated); a call
        # with no cond chunk only records the windows, for the grid.
        chunks = list(cond_or_uncond or [0])
        if 0 not in chunks:
            self.hub.window(win, None, None, windows)
            return
        i = chunks.index(0)
        x_win = tile_x.chunk(len(chunks))[i]
        out_win = result.chunk(len(chunks))[i]
        shown = self.hub.window(win, x_win, out_win, windows)
        if shown is not None and self.emit is not None:
            self.emit(shown)


def _slice_spatial(value, height, width, ys, xs):
    # A conditioning tensor laid out on the canvas (c_concat, a denoise mask) takes the window;
    # text, pooled vectors, references and options pass through.
    if isinstance(value, torch.Tensor) and value.ndim >= 4 and tuple(value.shape[-2:]) == (height, width):
        return value[..., ys, xs]
    return value


@contextlib.contextmanager
def tile_memory_estimate(guider, cap_h, cap_w):
    """Core sizes the model load and the cond batching from the shape it is handed, which here is
    the whole canvas although every model call sees at most one window. Patch the model INSTANCE's
    memory_required to clamp the spatial dims to the window cap (cells) for the run, restored in
    finally (core caches model objects session-wide, so a class patch would leak)."""
    model = getattr(getattr(guider, "model_patcher", None), "model", None)
    original = getattr(model, "memory_required", None)
    if not callable(original):
        yield
        return
    had_own = "memory_required" in vars(model)
    saved = vars(model).get("memory_required")

    def clamped(input_shape, cond_shapes={}):  # noqa: B006, core's own signature
        shape = list(input_shape)
        if len(shape) >= 4:
            shape[-2] = min(int(shape[-2]), cap_h)
            shape[-1] = min(int(shape[-1]), cap_w)
        return original(shape, cond_shapes)

    model.memory_required = clamped
    try:
        yield
    finally:
        if had_own:
            model.memory_required = saved
        else:
            del model.memory_required


def wrapped_guider(guider, wrapper):
    """A shallow copy of the guider whose model_options carry the wrapper, so the caller's guider
    and model keep their own options. A wrapper already there runs inside, per window."""
    options = dict(getattr(guider, "model_options", None) or {})
    wrapper.inner = options.get("model_function_wrapper")
    options["model_function_wrapper"] = wrapper
    clone = copy.copy(guider)
    clone.model_options = options
    return clone


def fixed_layout(canvas_w, canvas_h, max_tile_width, max_tile_height, context_anchor, context_overlap, geometry, tile_size):
    layout = grid.solve_layout(canvas_w, canvas_h, max_tile_width, max_tile_height, context_anchor, context_overlap,
                               multiple=geometry.multiple)
    if tile_size == SIZE_FULL:
        layout = grid.grow_layout(layout, geometry.round_down(int(max_tile_width)),
                                  geometry.round_down(int(max_tile_height)), geometry.multiple)
    return layout


def refine_all_at_once(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor,
                       context_overlap, geometry, tile_mode=SHIFTING, tile_size=SIZE_FULL, batch_size=1, batch_index=0,
                       log=None, preview_mode=None):
    """One picture through an all at once mode: encode the whole canvas (core's VAE call, which
    tiles itself when its estimate or an OOM says so), sample it as one run with the tiling
    wrapper, decode the whole canvas, crop back. For the shifting grid context_anchor is the ring
    as given; for the fixed grid both rings are already snapped (geometry.snap_rings)."""
    import comfy.model_management

    from . import sampling

    if tile_mode not in (ALL_AT_ONCE, SHIFTING):
        raise ValueError(f"refine_all_at_once runs {ALL_AT_ONCE!r} or {SHIFTING!r}, got {tile_mode!r}")
    if tile_size not in TILE_SIZES:
        raise ValueError(f"unknown tile_size {tile_size!r}, expected one of {TILE_SIZES}")
    pixels = image[..., :3]
    padded, (height, width) = sampling.pad_image_to_multiple(pixels, geometry.multiple)
    batch, canvas_h, canvas_w = padded.shape[0], padded.shape[1], padded.shape[2]
    cells_h, cells_w = canvas_h // geometry.scale, canvas_w // geometry.scale
    seed = getattr(noise, "seed", None)
    seed = seed if isinstance(seed, int) else 0

    if tile_mode == SHIFTING:
        core_h, core_w, ring = tile_cells(max_tile_width, max_tile_height, context_anchor, geometry)
        planner = ShiftingPlanner(core_h, core_w, ring, seed, tile_size, geometry.multiple // geometry.scale)
        cap_h, cap_w = core_h + 2 * ring, core_w + 2 * ring
        _, first = planner(cells_h, cells_w, float(sigmas[0]))
        logging.info(
            "Tile Refine (%s): %dx%d px canvas, core %dx%d px, ring %d px, tile size %s, %d windows in the first step.",
            tile_mode, canvas_w, canvas_h, core_w * geometry.scale, core_h * geometry.scale, ring * geometry.scale,
            tile_size, len(first))
    else:
        layout = fixed_layout(canvas_w, canvas_h, max_tile_width, max_tile_height, context_anchor, context_overlap,
                              geometry, tile_size)
        windows = fixed_windows(layout, geometry.scale)
        planner = FixedPlanner(windows, cells_h, cells_w)
        cap_h = max(w.y1 - w.y0 for w in windows)
        cap_w = max(w.x1 - w.x0 for w in windows)
        logging.info(
            "Tile Refine (%s): %dx%d px canvas, %dx%d grid, %d windows per model call up to %dx%d px, band %d px, tile size %s.",
            tile_mode, canvas_w, canvas_h, layout.sol_x.n, layout.sol_y.n, len(windows), cap_w * geometry.scale,
            cap_h * geometry.scale, layout.overlap, tile_size)

    latent = sampling.encode_pixels(vae, padded)
    canvas_noise = sampling.build_canvas_noise(vae, noise, canvas_h, canvas_w, batch, batch_size, batch_index,
                                               latent_scale=geometry.scale)
    sampling.check_latent_layout(latent, canvas_noise, canvas_w, canvas_h, geometry.scale, "canvas")
    from . import preview

    hub = preview.make_hub(preview_mode, guider.model_patcher, (cells_h, cells_w), scale=geometry.scale)
    if hub is not None:
        hub.begin_canvas(latent)
    wrapper = TiledModelWrapper(planner, log=log,
                                check_interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
                                hub=hub)
    run_guider = wrapped_guider(guider, wrapper)
    with tile_memory_estimate(guider, cap_h, cap_w):
        samples = sampling.sample_latent(run_guider, sampler, sigmas, canvas_noise, seed, latent,
                                         callback=_guarded_callback(guider, sigmas, wrapper))
    _require_wrapper_used(wrapper)
    decoded = sampling.fold_decoded_frames(vae.decode(samples))
    decoded = decoded.to(comfy.model_management.intermediate_device())
    return sampling.crop_image_to(decoded, height, width)


def _require_wrapper_used(wrapper):
    # A guider that builds its model options anywhere but its own `model_options` never reaches the
    # wrapper, and the model would silently run on the whole canvas at once: wrong output, or an
    # out-of-memory at large sizes. Fail loudly instead.
    if wrapper.calls == 0:
        raise RuntimeError(
            "All at once tile modes: the guider never called the tiling wrapper, so the model ran on the whole "
            "image untiled. This guider does not use model_options['model_function_wrapper'] (CFGGuider and "
            "core's guiders do). Use tile by tile with this guider.")


def _guarded_callback(guider, sigmas, wrapper):
    # The sampler's per-step callback (core's latent preview) with the wrapper check after the first
    # step, so a guider that bypasses the wrapper stops after one step instead of at the end.
    # With a preview_mode hub on the wrapper the callback keeps its own progress bar, so the
    # wrapper's per-window previews (one tile at a time) land on the step the run is at.
    import latent_preview

    steps = sigmas.shape[-1] - 1
    if wrapper.hub is None:
        core_preview = latent_preview.prepare_callback(guider.model_patcher, steps)

        def callback(step, x0, x, total_steps):
            _require_wrapper_used(wrapper)
            core_preview(step, x0, x, total_steps)

        return callback

    import comfy.utils

    pbar = comfy.utils.ProgressBar(steps)
    at = {"step": 0}
    wrapper.emit = lambda shown: pbar.update_absolute(at["step"], steps, shown)

    def callback(step, x0, x, total_steps):
        _require_wrapper_used(wrapper)
        at["step"] = step + 1
        pbar.update_absolute(step + 1, total_steps, wrapper.hub.canvas_step(x0, x))

    return callback


# --- the Stepwise plan --------------------------------------------------------------------------

def windows_per_call(width, height, core_h, core_w, ring, geometry, tile_size=SIZE_FULL):
    """Shifting grid: (fewest, most) distinct windows one model call runs on a width x height px
    image over all offsets."""
    # Windows merge by (row window, column window), so the distinct count is the product of the
    # distinct window counts per axis, each extreme over that axis's own offsets.
    unit = geometry.multiple // geometry.scale
    low = high = 1
    for length, core in ((geometry.round_up(height) // geometry.scale, core_h),
                         (geometry.round_up(width) // geometry.scale, core_w)):
        n = [len({(s.crop0, s.crop1) for s in axis_spans(length, core, ring, off, tile_size, unit)})
             for off in range(0, core, unit)]
        low, high = low * min(n), high * max(n)
    return low, high


def shifting_tiles(plan, tile_size, context_anchor):
    """The shifting grid's core, ring and windows per model call on every step of a Stepwise plan,
    for stepwise_plan.format_plan."""
    from .stepwise_plan import ShiftingTiles

    g = plan.geometry
    core_h, core_w, ring = tile_cells(plan.cap_width, plan.cap_height, context_anchor, g)
    windows = tuple(windows_per_call(s.width, s.height, core_h, core_w, ring, g, tile_size) for s in plan.steps)
    return ShiftingTiles(core_w * g.scale, core_h * g.scale, ring * g.scale, windows)
