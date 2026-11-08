"""Planning for the Stepwise Tile Upscale node: the sizes, grids, rings and denoise of every step.
Stdlib only (no torch, no comfy), so the whole plan is testable without a GPU or a model.

A step is "enlarge in pixel space, then tile refine". The final size is always exact,
round(source * factor) per axis, with no rounding to 64 and no stretch. The intermediate sizes
trade three goals against each other:

    even factor     every step enlarges by about factor ** (1 / steps)
    aspect ratio    the intermediate keeps the source's aspect ratio
    nice sizes      each axis splits into a whole number of EQUAL tiles whose cores and sampled
                    crops are multiples of the pixel grid and fit under the tile cap

`aspect_tolerance` is the one knob: the largest aspect deviation an intermediate may take to
reach a nice size. 0 asks for the closest exact enlargement instead, and the refine then
reflect-pads to the grid and crops back, so the last tile of an axis may be narrower."""
import math
from dataclasses import dataclass, field

from . import grid
from .geometry import LatentGeometry, snap_rings

# Auto step count: the fewest steps whose per-step factor stays at or under this.
DEFAULT_AUTO_STEP_FACTOR = 1.5
# How far (as a fraction of the ideal size) an intermediate axis may move to reach a nice size.
SEARCH_WINDOW = 0.06
# An upscale model output above this many megapixels gets a warning in the plan.
MODEL_OUTPUT_WARN_MP = 40.0

SIZE_EVEN = "even tiles"
SIZE_GRID = "grid multiple"
SIZE_EXACT = "exact (padded)"
SIZE_FINAL = "final (exact)"


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

    @property
    def factor_x(self):
        return self.width / self.src_width

    @property
    def factor_y(self):
        return self.height / self.src_height

    @property
    def tiles(self):
        return self.grid_x.n * self.grid_y.n


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
    aspect_tolerance: float
    steps: tuple
    notes: tuple = field(default=())

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


def _candidates(ideal, cap, anchor, overlap, geometry, axis, floor):
    # Grid-multiple lengths within SEARCH_WINDOW of `ideal`, each with its solved grid.
    lo = max(floor, geometry.round_up(math.floor(ideal * (1.0 - SEARCH_WINDOW))))
    hi = geometry.round_down(math.ceil(ideal * (1.0 + SEARCH_WINDOW)))
    out = []
    for length in range(lo, hi + 1, geometry.multiple):
        out.append((length, _axis(length, cap, anchor, overlap, geometry, axis)))
    return out


def _choose_size(ideal_w, ideal_h, src_aspect, prev_w, prev_h, cap_w, cap_h, anchor, overlap, geometry, tolerance):
    # Returns (width, height, size_choice) for one intermediate step.
    exact = (max(prev_w + 1, round(ideal_w)), max(prev_h + 1, round(ideal_h)))
    if tolerance <= 0.0:
        return (*exact, SIZE_EXACT)

    floor_w = geometry.round_up(prev_w + 1)
    floor_h = geometry.round_up(prev_h + 1)
    cand_w = _candidates(ideal_w, cap_w, anchor, overlap, geometry, "width", floor_w)
    cand_h = _candidates(ideal_h, cap_h, anchor, overlap, geometry, "height", floor_h)

    def cost(w, h):
        aspect = math.log((w / h) / src_aspect)
        return abs(math.log(w / ideal_w)) + abs(math.log(h / ideal_h)) + 4.0 * abs(aspect)

    best = {SIZE_EVEN: None, SIZE_GRID: None}
    for w, gx in cand_w:
        for h, gy in cand_h:
            if abs((w / h) / src_aspect - 1.0) > tolerance + 1e-12:
                continue
            key = (cost(w, h), gx.n * gy.n, w * h)
            tier = SIZE_EVEN if gx.even and gy.even else SIZE_GRID
            if best[tier] is None or key < best[tier][0]:
                best[tier] = (key, w, h)
    for tier in (SIZE_EVEN, SIZE_GRID):
        if best[tier] is not None:
            _, w, h = best[tier]
            return w, h, tier
    return (*exact, SIZE_EXACT)


def plan_upscale(src_width, src_height, factor, steps, geometry, cap_width, cap_height,
                 context_anchor, context_overlap, denoise_start, denoise_end,
                 aspect_tolerance=0.01, upscale_model_scale=None,
                 auto_step_factor=DEFAULT_AUTO_STEP_FACTOR):
    """The full plan. `steps` 0 picks the count automatically (auto_step_count). The rings are
    the minimums the caller asked for, snapped onto the geometry (geometry.snap_rings), and are
    the same on every step. `upscale_model_scale` is the connected upscale model's fixed factor,
    or None for lanczos only; the model runs on step 1 only."""
    if src_width < 1 or src_height < 1:
        raise ValueError(f"source size must be positive, got {src_width}x{src_height}")
    if not factor > 0:
        raise ValueError(f"factor must be positive, got {factor}")
    target_w = max(1, round(src_width * factor))
    target_h = max(1, round(src_height * factor))
    n_steps = int(steps) if steps and steps > 0 else auto_step_count(factor, auto_step_factor)
    anchor, overlap = snap_rings(context_anchor, context_overlap, geometry)
    denoises = interpolate_denoise(denoise_start, denoise_end, n_steps)
    src_aspect = src_width / src_height
    notes = []
    if (anchor, overlap) != (context_anchor, context_overlap):
        notes.append(f"rings rounded up to anchor {anchor} / overlap {overlap} px for the {geometry.multiple} px grid")
    if n_steps > 1 and factor <= 1.0:
        notes.append("factor <= 1: every step after the first refines at the same size")

    plan_steps = []
    prev_w, prev_h = src_width, src_height
    for i in range(1, n_steps + 1):
        scale = factor ** (i / n_steps)
        ideal_w, ideal_h = src_width * scale, src_height * scale
        if i == n_steps or factor <= 1.0:
            width, height, choice = target_w, target_h, SIZE_FINAL
        else:
            width, height, choice = _choose_size(ideal_w, ideal_h, src_aspect, prev_w, prev_h, cap_width,
                                                 cap_height, anchor, overlap, geometry, aspect_tolerance)
        gx = _axis(width, cap_width, anchor, overlap, geometry, "width")
        gy = _axis(height, cap_height, anchor, overlap, geometry, "height")

        warnings = []
        use_model = upscale_model_scale is not None and i == 1
        if use_model:
            mw, mh = round(prev_w * upscale_model_scale), round(prev_h * upscale_model_scale)
            if (mw, mh) == (width, height):
                upscale = f"model x{upscale_model_scale:g}"
            else:
                upscale = f"model x{upscale_model_scale:g} ({mw}x{mh}) + lanczos"
            if mw * mh / 1e6 > MODEL_OUTPUT_WARN_MP:
                warnings.append(f"upscale model output {mw}x{mh} is {mw * mh / 1e6:.1f} MP")
        elif (width, height) == (prev_w, prev_h):
            upscale = "none (same size)"
        else:
            upscale = "lanczos"
        for g, name in ((gx, "column"), (gy, "row")):
            if not g.even and g.n > 1:
                warnings.append(f"last {name} core {g.last} px vs {g.base} px")
        if gx.length != width or gy.length != height:
            warnings.append(f"reflect-padded to {gx.length}x{gy.length} and cropped back")
        if denoises[i - 1] <= 0.0:
            warnings.append("denoise 0: enlarge only, no sampling")
        plan_steps.append(StepPlan(
            index=i, src_width=prev_w, src_height=prev_h, width=width, height=height,
            ideal_width=ideal_w, ideal_height=ideal_h, grid_x=gx, grid_y=gy,
            denoise=float(denoises[i - 1]), upscale=upscale, use_model=use_model,
            size_choice=choice, warnings=tuple(warnings)))
        prev_w, prev_h = width, height

    return UpscalePlan(
        src_width=src_width, src_height=src_height, width=target_w, height=target_h, factor=factor,
        geometry=geometry, cap_width=cap_width, cap_height=cap_height, context_anchor=anchor,
        context_overlap=overlap, aspect_tolerance=aspect_tolerance, steps=tuple(plan_steps),
        notes=tuple(notes))


def _fmt_axis_sizes(gx, gy):
    def one(g):
        return f"{g.base}" if g.even or g.n == 1 else f"{g.base}/{g.last}"
    return f"{one(gx)} x {one(gy)}"


def _fmt_sampled(gx, gy):
    return f"{'/'.join(str(s) for s in gx.sampled())} x {'/'.join(str(s) for s in gy.sampled())}"


def format_plan(plan, sampler_steps=None):
    """The plan as Markdown: a header, one table row per step, then warnings and notes."""
    g = plan.geometry
    lines = [
        "## Stepwise Tile Upscale plan",
        "",
        f"- Source {plan.src_width}x{plan.src_height} -> target {plan.width}x{plan.height} "
        f"(x{plan.factor:g}), {len(plan.steps)} step{'s' if len(plan.steps) != 1 else ''}",
        f"- Latent cell {g.scale} px, pixel grid {g.multiple} px, max tile {plan.cap_width}x{plan.cap_height} px "
        f"(sampled crop incl. rings)",
        f"- Rings: context_anchor {plan.context_anchor} px, context_overlap {plan.context_overlap} px; "
        f"aspect tolerance {plan.aspect_tolerance * 100:g}%",
        "",
        "| Step | Size | Factor x / y | Aspect dev | Tiles | Core px (w x h) | Sampled px (w x h) | Denoise | Enlarge | Size choice |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in plan.steps:
        aspect = plan.aspect_deviation(s.width, s.height) * 100
        lines.append(
            f"| {s.index} | {s.src_width}x{s.src_height} -> {s.width}x{s.height} | {s.factor_x:.3f} / {s.factor_y:.3f} "
            f"| {aspect:+.2f}% | {s.grid_x.n}x{s.grid_y.n} = {s.tiles} | {_fmt_axis_sizes(s.grid_x, s.grid_y)} "
            f"| {_fmt_sampled(s.grid_x, s.grid_y)} | {s.denoise:.3f} | {s.upscale} | {s.size_choice} |")
    total = f"- Total {plan.total_tiles} tiles"
    if sampler_steps:
        total += f" x {sampler_steps} sampler steps"
    lines += ["", total]
    warn = [f"- Step {s.index}: {w}" for s in plan.steps for w in s.warnings]
    if warn:
        lines += ["", "### Warnings", *warn]
    if plan.notes:
        lines += ["", "### Notes", *(f"- {n}" for n in plan.notes)]
    return "\n".join(lines)
