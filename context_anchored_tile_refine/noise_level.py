"""Start noise level <-> sampler schedule: the "noise level" denoise mode and its scheduler node.

A denoise value means a different start noise for every scheduler and model shift.
BasicScheduler schedules int(steps / denoise) steps and keeps the last steps + 1, so the start
sigma lands wherever that longer schedule happens to be. On MOSS-Image (flow, shift 5) denoise
0.22 starts at sigma 0.59 with "simple", 0.46 with "beta" and 0.02 with "karras". The noise
level mode takes the start noise instead and finds the schedule length that starts there, so
one value means the same noise for every scheduler.

The level is the share of noise in the noisy sample, 0..1: the sigma itself on flow models
(x = (1 - s) x0 + s n), and sigma / (1 + sigma) on the eps / v family (x = x0 + sigma n, the
same mix up to scale), so a level means about the same amount of noise on both. The same level
does not make two different models change an image by the same amount.

Stdlib only at module scope; torch and comfy are imported lazily."""
import math
from dataclasses import dataclass
from typing import ClassVar

NOISE_LEVEL = "noise level"
DENOISE = "denoise"
DENOISE_MODES = (NOISE_LEVEL, DENOISE)

# How the passes of a Stepwise run get their steps (build_schedules), fewest to most steps at low noise.
SAME_STEP_SIZE = "same step size"
PROPORTIONAL = "proportional to noise"
STEPS_PER_PASS = "steps per pass"
STEPS_MODES = (SAME_STEP_SIZE, PROPORTIONAL, STEPS_PER_PASS)

# Longest schedule the search tries, as a multiple of the sampler steps (denoise 0.001).
MAX_TOTAL_FACTOR = 1000
# Most guesses solve_total refines before it brackets the answer by galloping.
GUESS_ROUNDS = 4


def level_to_sigma(level, flow):
    if flow:
        return float(level)
    return math.inf if level >= 1.0 else float(level) / (1.0 - float(level))


def sigma_to_level(sigma, flow):
    if flow:
        return float(sigma)
    return float(sigma) / (1.0 + float(sigma))


def solve_total(start_sigma, steps, target, max_total, guess=None):
    """The schedule length whose first kept sigma is closest to `target`.

    start_sigma(total) is the first of the last steps + 1 sigmas of a `total` step schedule. It
    falls with total for every core scheduler, but not strictly everywhere: beta drops repeated
    sigmas itself and then rises a little in places (0.222 -> 0.232 at length 210 for 25 steps),
    and linear_quadratic wobbles. Lengths run from `steps` (the full schedule, denoise 1) to
    `max_total`; a target above the full schedule's start gives the full schedule, one below the
    longest schedule's start gives the longest.

    Where start falls monotonically, the answer is the one length lo with start(lo) > target >=
    start(lo + 1) (or lo + 1), so any search that brackets it finds the same one. Where it does
    not, there can be several such crossings and the search finds one of them, whose start is
    then a hair from the very nearest. A schedule costs its length to build (normal and
    sgm_uniform loop in Python, about 20 us a sigma), so the search stays near the answer instead
    of bisecting down from max_total: guess(n), when given, estimates the answer from the length
    n schedule; the search gallops from the last guess to bracket the answer and bisects inside.
    Only when the guesses are far off (more than twice the last one) does it try max_total."""
    if start_sigma(steps) <= target:
        return steps
    top = max(steps, int(max_total))
    lo, hi = steps, None  # start(lo) > target; start(hi) <= target once one is found
    n = steps
    for _ in range(GUESS_ROUNDS):
        g = guess(n) if guess is not None else None
        if g is None or not math.isfinite(g):
            break
        g = min(top if hi is None else hi - 1, max(lo + 1, round(g)))  # inside the open bracket
        if g == n or (hi is not None and hi - lo <= 1):
            break
        n = g
        if start_sigma(n) > target:
            lo = n
        else:
            hi = n
    step = 1
    if hi is None:  # gallop up from lo (the last guess, or steps)
        far = 2 * lo
        while hi is None:
            m = lo + step
            if m >= top or m > far:
                if start_sigma(top) > target:
                    return top
                hi = top
            elif start_sigma(m) > target:
                # x4: a scheduler with too few distinct sigmas gets to max_total in few probes
                lo, step = m, step * 4
            else:
                hi = m
    elif n == hi:  # the last guess is at or below the target: gallop down from it
        while hi - lo > 1:
            m = hi - step
            if m <= lo:
                break
            if start_sigma(m) > target:
                lo = m
                break
            hi, step = m, step * 2
    else:  # the last guess is above the target: gallop up from it toward hi
        while hi - lo > 1:
            m = lo + step
            if m >= hi:
                break
            if start_sigma(m) > target:
                lo, step = m, step * 2
            else:
                hi = m
                break
    while hi - lo > 1:  # start(lo) > target >= start(hi)
        mid = (lo + hi) // 2
        if start_sigma(mid) > target:
            lo = mid
        else:
            hi = mid
    return lo if abs(start_sigma(lo) - target) < abs(start_sigma(hi) - target) else hi


@dataclass(frozen=True)
class Schedule:
    """One run's sigmas and what they amount to."""
    mode: str
    value: float          # the requested denoise or noise level
    sigmas: object        # tensor, steps + 1 values, or empty for "no sampling"
    total: int            # length of the schedule the sigmas were cut from, 0 when empty
    start_sigma: float
    start_level: float
    # The BasicScheduler denoise of these sigmas: steps / total, total the length BasicScheduler
    # asks the scheduler for. A pass with fewer steps than asked and build_schedules' same step
    # size passes give their share of the schedule they were cut from instead, which a denoise
    # setting may not reproduce; `exact` says whether it does.
    denoise: float
    # The steps asked for (build_schedule); more than `steps` when the scheduler has too few
    # distinct sigmas. 0 where the count is a share of one schedule (same step size).
    asked: int = 0
    # True when BasicScheduler with `steps` and `denoise` gives exactly these sigmas (always in
    # DENOISE mode, where the plan shows the noise level instead).
    exact: bool = True

    @property
    def steps(self):
        return max(0, self.sigmas.numel() - 1)


def is_flow(model_sampling):
    import comfy.model_sampling

    return isinstance(model_sampling, comfy.model_sampling.CONST)


def drop_repeats(sigmas):
    """The sigmas without repeats in a row. Schedulers that pick from the model's timestep table
    (simple, ddim_uniform) repeat entries when asked for more steps than the table has there; a
    repeat is a step of length 0, a model call that changes nothing."""
    import torch

    return torch.unique_consecutive(sigmas) if sigmas.numel() > 1 else sigmas


def is_basic(raw, sigmas, denoise):
    """Whether BasicScheduler with len(sigmas) - 1 steps and `denoise` gives exactly `sigmas`:
    it asks the scheduler for int(steps / denoise) steps (raw(n), as calculate_sigmas returns
    them) and keeps the last steps + 1."""
    import torch

    k = sigmas.numel() - 1
    if k < 1 or denoise <= 0.0:
        return False
    f = raw(int(k / denoise))
    return f.numel() > k and torch.equal(f[-(k + 1):], sigmas)


def build_schedule(model_patcher, scheduler, steps, value, mode=NOISE_LEVEL):
    """Sigmas for one sampling run, never with a sigma repeated. DENOISE mode is BasicScheduler
    with its repeats dropped; NOISE_LEVEL mode cuts the same kind of tail from the schedule length
    whose start is nearest the level.

    A scheduler that picks from the model's 1000 timesteps (simple, beta, ddim_uniform) has few
    distinct sigmas at low noise (on MOSS about 48 below 0.2), and no length gives more. A NOISE_LEVEL
    pass that cannot cut `steps` distinct steps below its level then takes the ones there are,
    from the sigma nearest the level, rather than start higher; `asked` keeps the count asked for."""
    import comfy.samplers
    import torch

    from . import upscale

    if mode not in DENOISE_MODES:
        raise ValueError(f"denoise_mode must be one of {list(DENOISE_MODES)}, got {mode!r}")
    model_sampling = model_patcher.get_model_object("model_sampling")
    flow = is_flow(model_sampling)
    steps = int(steps)
    value = float(value)

    if mode == DENOISE:
        sigmas = drop_repeats(upscale.build_sigmas(model_patcher, scheduler, steps, value))
        total = 0 if value <= 0.0 else (steps if value >= 1.0 else int(steps / value))
        denoise = steps / total if total else 0.0
    elif value <= 0.0:
        sigmas, total = torch.FloatTensor([]), 0
    else:
        raw_cache, cache = {}, {}

        def raw(n):
            if n not in raw_cache:
                raw_cache[n] = comfy.samplers.calculate_sigmas(model_sampling, scheduler, n).cpu()
            return raw_cache[n]

        def full(n):
            if n not in cache:
                cache[n] = drop_repeats(raw(n))
            return cache[n]

        def start(n):
            # Without its repeats (beta drops them itself) a length can give fewer than steps + 1
            # sigmas. Such a schedule counts as starting above any level, so the search moves on
            # to longer ones.
            f = full(n)
            return float(f[-(steps + 1)]) if f.numel() > steps else math.inf

        def guess(n):
            # The length whose start lands on the target if the length n schedule kept its shape
            # when stretched: the target sits at share p of its steps, and a length m schedule
            # starts at share (m - steps) / m, so m = steps / (1 - p).
            f = full(n)
            k = f.numel() - 1
            j = int((f[:-1] > target).sum())  # f[:j] are above the target
            if j == 0 or k < 1:
                return None
            if j >= k:  # below the last sigma before 0: call it halfway to 0
                p = (k - 0.5) / k
            else:
                above, below = float(f[j - 1]), float(f[j])
                p = (j - 1 + (above - target) / (above - below)) / k
            return steps / (1.0 - p)

        target = level_to_sigma(value, flow)
        longest = steps * MAX_TOTAL_FACTOR
        n = solve_total(start, steps, target, longest, guess)
        sigmas, fewer = full(n)[-(steps + 1):], None
        if float(sigmas[0]) > target and n == longest:
            # No length has `steps` distinct sigmas from the level down, so the start lands above
            # it. The longest schedule's tail from the sigma nearest the level has fewer steps and
            # starts where asked; it wins when it starts nearer. Below the longest (n < longest)
            # length n + 1 has steps + 1 distinct sigmas at or below the level, and no core
            # scheduler has fewer of them at a greater length, so the longest schedule's tail has
            # at least `steps` steps and is never taken: it is not built.
            f = full(longest)
            j = int((f[:-1] >= target).sum())  # f[:j] are at or above the target
            # The nearer of f[j - 1] and f[j], never the final 0.
            below_nearer = j < f.numel() - 1 and abs(float(f[j]) - target) < abs(float(f[j - 1]) - target)
            tail = f[j if j == 0 or below_nearer else j - 1:]
            if tail.numel() - 1 < steps and abs(float(tail[0]) - target) < abs(float(sigmas[0]) - target):
                sigmas, fewer = tail, f
        if fewer is None:
            # BasicScheduler's denoise for length n. Its own count of what the scheduler returns
            # does not matter: beta and ddim_uniform return fewer than n (232 for 233, say), and
            # steps / 232 would make BasicScheduler ask for 232 and start elsewhere.
            total, denoise = n, steps / n
            if int(steps / denoise) < n:  # float: steps / (steps / n) can land a hair below n
                denoise = math.nextafter(denoise, 0.0)
        else:
            # The share of the schedule the fewer steps were cut from, repeats dropped; no
            # denoise setting gives these sigmas, since BasicScheduler keeps `steps` of them.
            total = fewer.numel() - 1
            denoise = (sigmas.numel() - 1) / total

    if sigmas.numel() < 2:
        return Schedule(mode, value, sigmas, 0, 0.0, 0.0, 0.0)
    first = float(sigmas[0])
    exact = mode == DENOISE or is_basic(raw, sigmas, denoise)
    return Schedule(mode, value, sigmas, total, first, sigma_to_level(first, flow), denoise, steps, exact)


def build_schedules(model_patcher, scheduler, steps, values, mode=NOISE_LEVEL, steps_mode=SAME_STEP_SIZE):
    """Sigmas for every pass of a run, one Schedule per value.

    STEPS_PER_PASS: build_schedule per pass, `steps` steps each.

    PROPORTIONAL: build_schedule per pass with ceil(steps * value) steps, the value being the start
    noise level or the denoise, so every pass falls by the same noise per step on average (in
    NOISE_LEVEL mode) and keeps the scheduler's own shape within the pass.

    SAME_STEP_SIZE: `steps` are the steps of a full generation (denoise 1), and every pass samples
    the tail of that one schedule below its own start, so a pass with less noise takes fewer steps
    and no pass depends on the others' values. NOISE_LEVEL: the tail starts at the last sigma at or
    above the level and the whole tail is scaled down to start exactly at it, so a level between
    two sigmas (2.4 steps of the schedule, say) gets the count rounded up (3), every step of the
    tail shrinks by the same factor (down to the next sigma below over the one above, at most) and
    none is longer than the schedule's. A level on a schedule sigma keeps the tail as it is.
    DENOISE: the last ceil(steps * denoise) steps, as they are.

    Repeated sigmas are dropped (beta drops them itself, simple and ddim_uniform repeat when asked
    for more than the model's 1000 timesteps), so the share is counted on the schedule without
    them, not on `steps`. Either way a pass gets at least 1 step, and a value of 0 samples
    nothing."""
    import torch

    if steps_mode not in STEPS_MODES:
        raise ValueError(f"steps_mode must be one of {list(STEPS_MODES)}, got {steps_mode!r}")
    if mode not in DENOISE_MODES:
        raise ValueError(f"denoise_mode must be one of {list(DENOISE_MODES)}, got {mode!r}")
    values = [float(v) for v in values]
    if steps_mode == STEPS_PER_PASS:
        return [build_schedule(model_patcher, scheduler, steps, v, mode) for v in values]
    if steps_mode == PROPORTIONAL:
        return [build_schedule(model_patcher, scheduler, min(steps, max(1, math.ceil(steps * v - 1e-9))), v, mode)
                for v in values]

    import comfy.samplers

    model_sampling = model_patcher.get_model_object("model_sampling")
    flow = is_flow(model_sampling)

    def raw(n):
        return comfy.samplers.calculate_sigmas(model_sampling, scheduler, n).cpu()

    full = drop_repeats(raw(int(steps)))
    total = full.numel() - 1  # the steps the scheduler returned, repeats dropped
    schedules = []
    for v in values:
        if v <= 0.0:
            schedules.append(Schedule(mode, v, torch.FloatTensor([]), 0, 0.0, 0.0, 0.0))
            continue
        if mode == DENOISE:
            n = min(total, max(1, math.ceil(total * v - 1e-9)))
            sigmas = full[-(n + 1):]
            share = n / total
        else:
            target = level_to_sigma(v, flow)
            # Sigmas fall along the schedule, so those at or above the target are a prefix of it.
            # The tolerance keeps a level that is a schedule sigma up to float error from adding a
            # step.
            above = int((full[:total] >= target * (1.0 - 1e-6)).sum())
            if above == 0:  # above the schedule's top: the whole schedule
                sigmas, share = full, 1.0
            else:
                j = above - 1
                top, below = float(full[j]), float(full[j + 1])
                sigmas = full[j:]
                share = (total - j - 1) / total
                if top > target:
                    sigmas = sigmas * (target / top)
                    share += (target - below) / (top - below) / total
                else:
                    share += 1.0 / total
        first = float(sigmas[0])
        # share: the part of the full schedule the pass covers. In DENOISE mode it is the denoise
        # itself (rounded to whole steps); in NOISE_LEVEL mode it is fractional (2.4 / 25 in the
        # example above), so it only approximates a denoise setting, which samples whole steps.
        # A scaled tail is never BasicScheduler's; one that is not may be, and is_basic checks.
        exact = mode == DENOISE or is_basic(raw, sigmas, share)
        schedules.append(Schedule(mode, v, sigmas, total, first, sigma_to_level(first, flow), share, exact=exact))
    return schedules


class NoiseLevelScheduler:
    """BasicScheduler with the start noise level as the setting instead of denoise."""

    @classmethod
    def INPUT_TYPES(s):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "noise_level": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001, "tooltip": "Share of noise the sampling starts from, 0..1 (the flow sigma; sigma / (1 + sigma) on SD/SDXL-type models). 1 is a full generation, 0 samples nothing. Means the same start noise for every scheduler, unlike denoise."}),
            },
        }

    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "info")
    OUTPUT_TOOLTIPS = (
        "The last steps + 1 sigmas of the schedule length whose first kept sigma is nearest the noise level.",
        "The start sigma reached and the BasicScheduler denoise that gives the same sigmas.",
    )
    SEARCH_ALIASES: ClassVar[list[str]] = ["noise level", "sigma scheduler", "start noise", "denoise by noise"]
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"
    DESCRIPTION = "Builds sigmas that start at a given noise level, for any scheduler. info shows the denoise it equals."

    def get_sigmas(self, model, scheduler, steps, noise_level):
        schedule = build_schedule(model, scheduler, steps, noise_level, NOISE_LEVEL)
        if schedule.total == 0:
            info = "noise level 0: empty sigmas, nothing is sampled"
        else:
            info = (f"noise level {noise_level:.3f} -> start sigma {schedule.start_sigma:.4g} "
                    f"(level {schedule.start_level:.3f}), same as denoise {schedule.denoise:.4f} "
                    f"with {scheduler}, {schedule.steps} steps")
        return (schedule.sigmas, info)
