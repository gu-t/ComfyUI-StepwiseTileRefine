"""Planning for the Stepwise Tile Upscale node: the sizes, grids, rings and denoise of every step.
Stdlib only (no torch, no comfy), so the whole plan is testable without a GPU or a model.

A step is "enlarge in pixel space, then tile refine". Every size a pass samples is a multiple of
the pixel grid, because the VAE encodes whole latent cells and a canvas off the grid would have
to be reflect-padded: the model takes the mirrored strip for picture and may draw something
else from it. An intermediate takes the grid size nearest its ideal, source * factor ** (i /
steps), weighing size against aspect deviation (_grid_size). On a 16 px grid that keeps the
aspect within about 0.85 % and every step's factor within about 2.4 % of the even factor, worst
cases on 0.5 MP sources; the deviation shrinks as the image grows.

The output is round(source * factor) per axis. When it is off the grid and the last pass lands on
it, `output_size` says how:

    exact, crop edges                the last pass samples the grid size just above the target,
                                     enlarged without stretch, and is centre-cropped onto it:
                                     a few px of picture are lost at the edges
    exact, pad edges during denoise  the last pass samples the target, reflect-padded to the grid
                                     and cropped back
    stretch image                    the output is the grid size nearest the target, so the
                                     aspect changes slightly (the plan warns by how much)

With oversample the last pass is not the target, and the final resample reaches it exactly.

OVERSAMPLE (the repair loop). `factor` is always the output size. `oversample` (>= 1) runs the
passes that much larger, and ONE final resample after the last pass brings the image onto the
exact target, so the largest pass is source * factor * oversample. Two ways to get there:

    constant    pass i at source * factor ** (i / steps) * oversample: the first pass jumps to
                the oversample and every pass keeps it. Per pass that is the enlarge-by-U, refine,
                shrink-by-D loop with U = factor ** (1 / steps) * oversample and D = oversample,
                the shrink and the next enlargement merged into one resample (tested as good as
                the two apart, and less soft). factor 1 refines at one size every pass.
    ramp up     pass i at source * (factor * oversample) ** (i / steps): the passes grow evenly
                and reach the oversample on the last pass.

oversample 1 is the plain upscale in both modes and plans exactly as it always did."""
import math
from dataclasses import dataclass, field

from . import grid
from .geometry import LatentGeometry, snap_rings

# Auto step count: the fewest steps whose per-step factor stays at or under this.
DEFAULT_AUTO_STEP_FACTOR = 1.5
# The grid size search: cells either side of the ideal length, and the weight of the aspect
# deviation against the size deviation.
SEARCH_CELLS = 2
ASPECT_WEIGHT = 4.0
# An upscale model output above this many megapixels gets a warning in the plan.
MODEL_OUTPUT_WARN_MP = 40.0

SIZE_GRID = "grid"
SIZE_FINAL = "final"
SIZE_SAME = "same size"

OUTPUT_CROP = "exact, crop edges"
OUTPUT_PAD = "exact, pad edges during denoise"
OUTPUT_STRETCH = "stretch image"
OUTPUT_SIZES = (OUTPUT_CROP, OUTPUT_PAD, OUTPUT_STRETCH)
DEFAULT_OUTPUT_SIZE = OUTPUT_CROP

# The resample methods (resample.py), in the order the widgets list them: the simple filters,
# then the BC-cubic family soft to sharp, then lanczos. Area enlarges like nearest neighbour,
# and bilinear adds nothing between B-spline and Mitchell when shrinking.
UPSCALE_METHODS = ("bilinear", "bicubic (B-spline)", "bicubic (mitchell)", "bicubic (catmull-rom)", "lanczos")
DOWNSCALE_METHODS = ("area", "bicubic (B-spline)", "bicubic (mitchell)", "bicubic (catmull-rom)", "lanczos")
DEFAULT_UPSCALE_METHOD = "lanczos"
OVERSAMPLE_CONSTANT = "constant"
OVERSAMPLE_RAMP = "ramp up"
OVERSAMPLE_MODES = (OVERSAMPLE_CONSTANT, OVERSAMPLE_RAMP)
DEFAULT_DOWNSCALE_METHOD = "area"


def resample_method(src_w, src_h, width, height, upscale_method, downscale_method):
    # A resample that shrinks both axes (or keeps one) uses the downscale method, anything that
    # enlarges an axis the upscale method.
    if width <= src_w and height <= src_h:
        return downscale_method
    return upscale_method


@dataclass(frozen=True)
class FinalResample:
    src_width: int
    src_height: int
    width: int
    height: int
    method: str


@dataclass(frozen=True)
class FinalCrop:
    # The centre crop of the last pass onto the exact target (output_size "exact, crop edges").
    src_width: int
    src_height: int
    width: int
    height: int
    x0: int
    y0: int


@dataclass(frozen=True)
class AxisGrid:
    length: int        # canvas length the grid is solved on (the size padded to the pixel grid)
    n: int
    base: int
    last: int
    overhead: int

    @property
    def even(self):
        return self.last == self.base

    def sampled(self):
        # Every distinct sampled crop length along this axis, largest first.
        r = self.overhead if self.n == 2 else self.overhead // 2 if self.n >= 3 else 0
        if self.n == 1:
            return (self.length,)
        sizes = {self.base + r, self.last + r}
        if self.n >= 3:
            sizes.add(self.base + 2 * r)
        return tuple(sorted(sizes, reverse=True))


@dataclass(frozen=True)
class StepPlan:
    index: int                 # 1-based
    src_width: int
    src_height: int
    width: int
    height: int
    ideal_width: float
    ideal_height: float
    grid_x: AxisGrid
    grid_y: AxisGrid
    denoise: float
    upscale: str               # human-readable enlargement method
    use_model: bool
    size_choice: str
    warnings: tuple = ()
    method: str = DEFAULT_UPSCALE_METHOD   # the resample onto this step's size (after the model, if any)
    # output_size "exact, crop edges": the fractional size the step enlarges to, of which it keeps
    # the centre width x height (resample.resize `virtual`). 0 when it resamples onto its size.
    enlarge_width: float = 0.0
    enlarge_height: float = 0.0
    model_width: int = 0       # the upscale model's output, 0 without a model
    model_height: int = 0

    @property
    def enlarged(self):
        return (self.enlarge_width or self.width, self.enlarge_height or self.height)

    @property
    def factor_x(self):
        return self.enlarged[0] / self.src_width

    @property
    def factor_y(self):
        return self.enlarged[1] / self.src_height

    @property
    def tiles(self):
        return self.grid_x.n * self.grid_y.n


@dataclass(frozen=True)
class ShiftingTiles:
    # What a run on the shifting grid tiles with, for the plan text: core and ring in px, and the
    # (fewest, most) windows per model call on each step.
    core_width: int
    core_height: int
    ring: int
    windows: tuple


@dataclass(frozen=True)
class UpscalePlan:
    src_width: int
    src_height: int
    width: int
    height: int
    factor: float
    geometry: LatentGeometry
    cap_width: int
    cap_height: int
    context_anchor: int
    context_overlap: int
    steps: tuple
    notes: tuple = field(default=())
    oversample: float = 1.0
    oversample_mode: str = OVERSAMPLE_CONSTANT
    # After the last pass: a FinalResample (oversample), a FinalCrop (exact, crop edges), or None
    # when that pass is the output.
    final: object = None
    fixed_grid: bool = True       # False: the run tiles on its own grid, and grid_x / grid_y are unused
    output_size: str = DEFAULT_OUTPUT_SIZE
    warnings: tuple = ()          # about the plan as a whole; each step has its own
    rings_asked: tuple = None     # (tile_padding, tile_overlap) as given, before snap_rings

    def aspect_deviation(self, width, height):
        return (width / height) / (self.src_width / self.src_height) - 1.0

    @property
    def total_tiles(self):
        return sum(step.tiles for step in self.steps)


def auto_step_count(factor, max_step_factor=DEFAULT_AUTO_STEP_FACTOR):
    # The fewest steps whose per-step factor is at most max_step_factor. A factor at or below 1
    # (a pure refine, or a downscale) is one step.
    if factor <= 1.0 or max_step_factor <= 1.0:
        return 1
    return max(1, math.ceil(math.log(factor) / math.log(max_step_factor) - 1e-9))


def interpolate_denoise(start, end, steps):
    # Linear from start (step 1) to end (last step). One step uses start.
    if steps <= 1:
        return (float(start),)
    return tuple(start + (end - start) * i / (steps - 1) for i in range(steps))


def _axis(length, cap, anchor, overlap, geometry, axis):
    padded = geometry.round_up(length)
    sol = grid.solve_axis(padded, cap, anchor, overlap, multiple=geometry.multiple, axis=axis)
    return AxisGrid(length=padded, n=sol.n, base=sol.base, last=sol.last, overhead=sol.overhead)


def _grid_size(ideal_w, ideal_h, src_aspect, geometry, min_w=1, min_h=1):
    # The grid size near (ideal_w, ideal_h) that best trades size deviation against aspect
    # deviation, each axis at least its minimum.
    m = geometry.multiple

    def lengths(ideal, low):
        centre = round(ideal / m)
        first = max(1, -(-low // m), centre - SEARCH_CELLS)
        return range(first * m, (max(first, centre) + SEARCH_CELLS) * m + 1, m)

    def cost(size):
        w, h = size
        aspect = abs(math.log((w / h) / src_aspect))
        return abs(math.log(w / ideal_w)) + abs(math.log(h / ideal_h)) + ASPECT_WEIGHT * aspect, w * h

    return min(((w, h) for w in lengths(ideal_w, min_w) for h in lengths(ideal_h, min_h)), key=cost)


def _crop_canvas(src_w, src_h, target_w, target_h, geometry):
    # output_size "exact, crop edges": the grid size just above the target, and the fractional
    # size the source enlarges to, by one exact factor on both axes, to cover it. Returns
    # (enlarge w, h, canvas w, h), the enlargement 0 when the target is on the grid.
    canvas_w, canvas_h = geometry.round_up(target_w), geometry.round_up(target_h)
    if (canvas_w, canvas_h) == (target_w, target_h):
        return 0.0, 0.0, target_w, target_h
    k = max(canvas_w / src_w, canvas_h / src_h)
    enlarge_w, enlarge_h = max(float(canvas_w), src_w * k), max(float(canvas_h), src_h * k)
    if abs(enlarge_w - canvas_w) < 1e-6 and abs(enlarge_h - canvas_h) < 1e-6:
        # The source's aspect fits the grid size exactly: a plain resample onto it.
        return 0.0, 0.0, canvas_w, canvas_h
    return enlarge_w, enlarge_h, canvas_w, canvas_h


def size_change_text(exact_w, exact_h, width, height):
    # How much each axis that moved grew or shrank, e.g. "height -4 px (-0.25%)".
    parts = [f"{name} {new - old:+d} px ({(new / old - 1.0) * 100:+.2f}%)"
             for name, old, new in (("width", exact_w, width), ("height", exact_h, height)) if new != old]
    return ", ".join(parts)


def plan_upscale(src_width, src_height, factor, steps, geometry, cap_width, cap_height,
                 context_anchor, context_overlap, denoise_start, denoise_end,
                 upscale_model_scale=None,
                 auto_step_factor=DEFAULT_AUTO_STEP_FACTOR, oversample=1.0, oversample_mode=OVERSAMPLE_CONSTANT,
                 upscale_method=DEFAULT_UPSCALE_METHOD, downscale_method=DEFAULT_DOWNSCALE_METHOD,
                 fixed_grid=True, output_size=DEFAULT_OUTPUT_SIZE):
    """The full plan. `steps` 0 picks the count automatically (auto_step_count). The rings are
    the minimums the caller asked for, snapped onto the geometry (geometry.snap_rings), and are
    the same on every step. `upscale_model_scale` is the connected upscale model's fixed factor,
    or None for none; the model runs on step 1 only. `oversample` and `oversample_mode` are the
    repair loop and `output_size` the landing on the target (module docstring); the two methods
    only name the resamples here. `fixed_grid` False is a run that does not tile on the solved
    grid (the shifting tile mode), which only changes the notes."""
    if src_width < 1 or src_height < 1:
        raise ValueError(f"source size must be positive, got {src_width}x{src_height}")
    if not factor > 0:
        raise ValueError(f"factor must be positive, got {factor}")
    if not oversample >= 1.0:
        raise ValueError(f"oversample must be at least 1, got {oversample:g}")
    if oversample_mode not in OVERSAMPLE_MODES:
        raise ValueError(f"oversample_mode must be one of {list(OVERSAMPLE_MODES)}, got {oversample_mode!r}")
    for name, value, allowed in (("upscale_method", upscale_method, UPSCALE_METHODS),
                                 ("downscale_method", downscale_method, DOWNSCALE_METHODS)):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")
    if output_size not in OUTPUT_SIZES:
        raise ValueError(f"output_size must be one of {list(OUTPUT_SIZES)}, got {output_size!r}")
    src_aspect = src_width / src_height
    exact_w = max(1, round(src_width * factor))
    exact_h = max(1, round(src_height * factor))
    plan_warnings = []
    if output_size == OUTPUT_STRETCH:
        target_w, target_h = _grid_size(src_width * factor, src_height * factor, src_aspect, geometry)
        if (target_w, target_h) != (exact_w, exact_h):
            plan_warnings.append(f"Output {target_w}x{target_h} instead of {exact_w}x{exact_h}: "
                                 f"{size_change_text(exact_w, exact_h, target_w, target_h)}")
    else:
        target_w, target_h = exact_w, exact_h
    n_steps = int(steps) if steps and steps > 0 else auto_step_count(factor * oversample, auto_step_factor)
    anchor, overlap = snap_rings(context_anchor, context_overlap, geometry)
    denoises = interpolate_denoise(denoise_start, denoise_end, n_steps)
    ramp = oversample_mode == OVERSAMPLE_RAMP
    plain = oversample == 1.0
    # The size a pass that lands on the target samples, and what it enlarges to first.
    if plain and output_size == OUTPUT_CROP:
        enlarge_w, enlarge_h, canvas_w, canvas_h = _crop_canvas(src_width, src_height, target_w, target_h, geometry)
    else:
        enlarge_w, enlarge_h, canvas_w, canvas_h = 0.0, 0.0, target_w, target_h
    # Growth from one pass to the next. Only without oversampling does the last pass land on the
    # target itself; otherwise the final resample follows it.
    growth = (factor * oversample if ramp else factor) ** (1.0 / n_steps)
    notes = []

    plan_steps = []
    prev_w, prev_h = src_width, src_height
    for i in range(1, n_steps + 1):
        scale = (factor * oversample) ** (i / n_steps) if ramp else factor ** (i / n_steps) * oversample
        ideal_w, ideal_h = src_width * scale, src_height * scale
        to_w = to_h = 0.0
        if plain and (factor <= 1.0 or i == n_steps):
            width, height, choice = canvas_w, canvas_h, SIZE_FINAL
            if (prev_w, prev_h) != (canvas_w, canvas_h):
                to_w, to_h = enlarge_w, enlarge_h
        elif i > 1 and abs(growth - 1.0) < 1e-9:
            # No growth between passes (constant oversample at factor 1): the pass keeps the
            # previous size exactly, since a near-identity resample would only soften it.
            width, height, choice = prev_w, prev_h, SIZE_SAME
        else:
            width, height = _grid_size(ideal_w, ideal_h, src_aspect, geometry, prev_w + 1, prev_h + 1)
            choice = SIZE_GRID
        gx = _axis(width, cap_width, anchor, overlap, geometry, "width")
        gy = _axis(height, cap_height, anchor, overlap, geometry, "height")

        warnings = []
        use_model = upscale_model_scale is not None and i == 1
        mw = mh = 0
        if use_model:
            mw, mh = round(prev_w * upscale_model_scale), round(prev_h * upscale_model_scale)
            method = resample_method(mw, mh, width, height, upscale_method, downscale_method)
            if (mw, mh) == (width, height) and not to_w:
                upscale = f"model x{upscale_model_scale:g}"
            else:
                upscale = f"model x{upscale_model_scale:g} ({mw}x{mh}) + {method}"
            if mw * mh / 1e6 > MODEL_OUTPUT_WARN_MP:
                warnings.append(f"upscale model output {mw}x{mh} is {mw * mh / 1e6:.1f} MP")
        else:
            method = resample_method(prev_w, prev_h, width, height, upscale_method, downscale_method)
            upscale = "none (same size)" if (width, height) == (prev_w, prev_h) and not to_w else method
        if gx.length != width or gy.length != height:
            warnings.append(f"mirrored edge during denoise, {_pad_text(width, height, gx.length, gy.length)}")
        if denoises[i - 1] <= 0.0:
            warnings.append("denoise 0: enlarge only, no sampling")
        plan_steps.append(StepPlan(
            index=i, src_width=prev_w, src_height=prev_h, width=width, height=height,
            ideal_width=ideal_w, ideal_height=ideal_h, grid_x=gx, grid_y=gy,
            denoise=float(denoises[i - 1]), upscale=upscale, use_model=use_model,
            size_choice=choice, warnings=tuple(warnings), method=method, enlarge_width=to_w, enlarge_height=to_h,
            model_width=mw, model_height=mh))
        prev_w, prev_h = width, height

    final = None
    if (prev_w, prev_h) == (target_w, target_h):
        pass
    elif plain:
        final = FinalCrop(prev_w, prev_h, target_w, target_h, (prev_w - target_w) // 2, (prev_h - target_h) // 2)
    else:
        final = FinalResample(prev_w, prev_h, target_w, target_h,
                              resample_method(prev_w, prev_h, target_w, target_h, upscale_method, downscale_method))
    return UpscalePlan(
        src_width=src_width, src_height=src_height, width=target_w, height=target_h, factor=factor,
        geometry=geometry, cap_width=cap_width, cap_height=cap_height, context_anchor=anchor,
        context_overlap=overlap, steps=tuple(plan_steps), notes=tuple(notes), oversample=float(oversample),
        oversample_mode=oversample_mode, final=final, fixed_grid=bool(fixed_grid), output_size=output_size,
        warnings=tuple(plan_warnings), rings_asked=(context_anchor, context_overlap))


def _pad_text(width, height, canvas_w, canvas_h):
    sides = [f"{n} px at the {side}" for n, side in ((canvas_w - width, "right"), (canvas_h - height, "bottom")) if n]
    return " and ".join(sides)


def _fmt_axis_sizes(gx, gy):
    def one(g):
        return f"{g.base}" if g.even or g.n == 1 else f"{g.base}/{g.last}"
    return f"{one(gx)}x{one(gy)}"


def _fmt_span(low, high):
    return f"{low}" if low == high else f"{low} to {high}"


def _size(w, h):
    # A size with one decimal where it is fractional: 2901.3x1632.
    def one(v):
        text = f"{v:.1f}"
        return text[:-2] if text.endswith(".0") else text
    return f"{one(w)}x{one(h)}"


def _factors(fx, fy):
    # "x1.511", or "x1.444 / x1.439" when the two axes differ at the precision shown.
    x, y = f"x{fx:.3f}", f"x{fy:.3f}"
    return x if x == y else f"{x} / {y}"


def _op(verb, sizes, detail=""):
    # One operation of a step: what is done, the size before -> after, and its details.
    return f"  {verb:<8} {sizes:<24} {detail}".rstrip()


def _resample_op(plan, in_w, in_h, s):
    # The resample onto the step's size, or None when there is none.
    if (in_w, in_h) == (s.width, s.height) and not s.enlarge_width:
        return None
    ew, eh = s.enlarged
    verb = "shrink" if ew <= in_w and eh <= in_h else "enlarge"
    detail = f"{_factors(ew / in_w, eh / in_h)}, {s.method}"
    # Name the stretch only where it exceeds what whole pixels can hold.
    deviation = plan.aspect_deviation(ew, eh)
    if abs(deviation) > 0.5 / ew + 0.5 / eh:
        detail += f", aspect {deviation * 100:+.2f}%"
    if s.enlarge_width:
        detail += f", centre of {_size(ew, eh)}"
    return _op(verb, f"{in_w}x{in_h} -> {s.width}x{s.height}", detail)


def _step_ops(plan, s, shifting):
    # Every operation of one step, in the order it runs.
    ops = []
    in_w, in_h = s.src_width, s.src_height
    if s.use_model:
        ops.append(_op("model", f"{in_w}x{in_h} -> {s.model_width}x{s.model_height}",
                       "upscale " + s.upscale.split(" (")[0]))
        in_w, in_h = s.model_width, s.model_height
    resample_op = _resample_op(plan, in_w, in_h, s)
    if resample_op:
        ops.append(resample_op)
    canvas = f"{s.grid_x.length}x{s.grid_y.length}"
    padded = (s.grid_x.length, s.grid_y.length) != (s.width, s.height)
    if padded:
        ops.append(_op("pad", f"{s.width}x{s.height} -> {canvas}", "mirrored"))
    if shifting is None:
        tiles = f"{s.grid_x.n}x{s.grid_y.n} tiles, core {_fmt_axis_sizes(s.grid_x, s.grid_y)}"
    else:
        low, high = shifting.windows[s.index - 1]
        tiles = f"{_fmt_span(low, high)} windows per model call"
    ops.append(_op("denoise", canvas, tiles))
    if padded:
        ops.append(_op("crop", f"{canvas} -> {s.width}x{s.height}"))
    if s is plan.steps[-1] and isinstance(plan.final, FinalCrop):
        f = plan.final
        ops.append(_op("crop", f"{f.src_width}x{f.src_height} -> {f.width}x{f.height}"))
    return ops


def _step_head(s, value_label, schedule):
    # "Step 1, noise level 0.450 (= denoise 0.140)": the setting, and what it equals in the other
    # denoise mode when the run's schedule is known.
    head = f"Step {s.index}, {value_label.lower()} {s.denoise:.3f}"
    if schedule is None:
        return head
    if schedule.total == 0:
        return head + ", no sampling"
    if value_label.lower() == "noise level":
        return head + f" (= denoise {schedule.denoise:.3f})"
    return head + f" (= noise level {schedule.start_level:.3f})"


def format_plan(plan, sampler_steps=None, value_label="Denoise", shifting=None, schedules=None, tile_mode=None,
                tile_size=None, scheduler=None):
    """The plan as plain text (it is read in a text preview and the log, where Markdown shows raw): a
    header, every step's operations in the order they run, then warnings and notes. `shifting`
    (ShiftingTiles) replaces the fixed grid's tiles with the shifting grid's windows, for a plan
    made with fixed_grid False. `schedules` (noise_level.Schedule per step) adds what each step's
    strength equals in the other denoise mode; the tile mode, tile size and scheduler only name
    the run's settings."""
    g = plan.geometry
    n = len(plan.steps)
    asked = (max(1, round(plan.src_width * plan.factor)), max(1, round(plan.src_height * plan.factor)))
    head = f"Source {plan.src_width}x{plan.src_height}, x{plan.factor:g} -> {asked[0]}x{asked[1]}"
    if plan.oversample != 1.0:
        head += f", oversample x{plan.oversample:g} {plan.oversample_mode}"
    head += f", {n} step{'s' if n != 1 else ''}."
    if plan.oversample == 1.0 or plan.output_size == OUTPUT_STRETCH:
        head += f" Output size: {plan.output_size}."
    # What the model sees at most: the cap on the pixel grid.
    cap = f"{g.round_down(plan.cap_width)}x{g.round_down(plan.cap_height)} px"
    asked_anchor, asked_overlap = plan.rings_asked or (plan.context_anchor, plan.context_overlap)
    if tile_mode is None:
        tiles = f"Max tile {cap}, rings included."
    else:
        tiles = f"Tile mode: {tile_mode}, {tile_size}. Max tile {cap}, rings included."
    grid_text = f"Pixel grid {g.multiple} px" + (f" (latent cell {g.scale} px)." if g.multiple != g.scale else ".")
    def ring(name, value, asked):
        return f"{name} {value} px" + (f" (rounded from {asked})" if asked != value else "")

    if shifting is not None:
        rings = (f"Ring: {ring('tile_padding', plan.context_anchor, asked_anchor)}, core {shifting.core_width}x"
                 f"{shifting.core_height} px (tile_overlap is not used).")
    elif plan.fixed_grid:
        rings = (f"Rings: {ring('tile_overlap', plan.context_overlap, asked_overlap)}, "
                 f"{ring('tile_padding', plan.context_anchor, asked_anchor)}.")
    else:
        rings = f"Ring: tile_padding {plan.context_anchor} px (tile_overlap is not used)."
    lines = ["STEPWISE TILE UPSCALE PLAN", head, tiles, f"{rings} {grid_text}"]
    for i, s in enumerate(plan.steps):
        schedule = schedules[i] if schedules else None
        lines += ["", _step_head(s, value_label, schedule), *_step_ops(plan, s, shifting)]
    f = plan.final
    if isinstance(f, FinalResample):
        verb = "shrink" if f.width <= f.src_width and f.height <= f.src_height else "resize"
        lines += ["", "Final", _op(verb, f"{f.src_width}x{f.src_height} -> {f.width}x{f.height}",
                                   f"{_factors(f.width / f.src_width, f.height / f.src_height)}, {f.method}")]
    if shifting is None:
        total = f"Total {plan.total_tiles} tiles"
    else:
        total = f"Total {_fmt_span(sum(w[0] for w in shifting.windows), sum(w[1] for w in shifting.windows))} windows"
    if sampler_steps:
        total += f" x {sampler_steps} sampler steps"
        if scheduler:
            total += f" ({scheduler})"
    lines += ["", f"Output {plan.width}x{plan.height}. {total}."]
    warn = [f"  {w}" for w in plan.warnings] + [f"  Step {s.index}: {w}" for s in plan.steps for w in s.warnings]
    if warn:
        lines += ["", "WARNINGS", *warn]
    if plan.notes:
        lines += ["", "NOTES", *(f"  {n}" for n in plan.notes)]
    return "\n".join(lines)
