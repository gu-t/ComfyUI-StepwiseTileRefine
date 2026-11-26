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


def level_to_sigma(level, flow):
    if flow:
        return float(level)
    return math.inf if level >= 1.0 else float(level) / (1.0 - float(level))


def sigma_to_level(sigma, flow):
    if flow:
        return float(sigma)
    return float(sigma) / (1.0 + float(sigma))


def solve_total(start_sigma, steps, target, max_total):
    """The schedule length whose first kept sigma is closest to `target`.

    start_sigma(total) is the first of the last steps + 1 sigmas of a `total` step schedule,
    non-increasing in total for every core scheduler. Lengths run from `steps` (the full
    schedule, denoise 1) to `max_total`; a target above the full schedule's start gives the
    full schedule, one below the longest schedule's start gives the longest."""
    if start_sigma(steps) <= target:
        return steps
    lo, hi = steps, max(steps, int(max_total))
    if start_sigma(hi) > target:
        return hi
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
    denoise: float        # the BasicScheduler denoise that gives the same sigmas (steps / total)

    @property
    def steps(self):
        return max(0, self.sigmas.numel() - 1)


def is_flow(model_sampling):
    import comfy.model_sampling

    return isinstance(model_sampling, comfy.model_sampling.CONST)


def build_schedule(model_patcher, scheduler, steps, value, mode=NOISE_LEVEL):
    """Sigmas for one sampling run. DENOISE mode is BasicScheduler exactly; NOISE_LEVEL mode
    cuts the same kind of tail from the schedule length whose start is nearest the level."""
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
        sigmas = upscale.build_sigmas(model_patcher, scheduler, steps, value)
        total = 0 if value <= 0.0 else (steps if value >= 1.0 else int(steps / value))
    elif value <= 0.0:
        sigmas, total = torch.FloatTensor([]), 0
    else:
        cache = {}

        def full(n):
            if n not in cache:
                cache[n] = comfy.samplers.calculate_sigmas(model_sampling, scheduler, n).cpu()
            return cache[n]

        def start(n):
            return float(full(n)[-(steps + 1)])

        total = solve_total(start, steps, level_to_sigma(value, flow), steps * MAX_TOTAL_FACTOR)
        sigmas = full(total)[-(steps + 1):]

    if sigmas.numel() < 2:
        return Schedule(mode, value, sigmas, 0, 0.0, 0.0, 0.0)
    first = float(sigmas[0])
    return Schedule(mode, value, sigmas, total, first, sigma_to_level(first, flow), steps / total)


def build_schedules(model_patcher, scheduler, steps, values, mode=NOISE_LEVEL, steps_mode=SAME_STEP_SIZE):
    """Sigmas for every pass of a run, one Schedule per value.

    STEPS_PER_PASS: build_schedule per pass, `steps` steps each.

    PROPORTIONAL: build_schedule per pass with ceil(steps * value) steps, the value being the start
    noise level or the denoise, so every pass falls by the same noise per step on average (in
    NOISE_LEVEL mode) and keeps the scheduler's own shape within the pass.

    SAME_STEP_SIZE: `steps` are the steps of a full generation (denoise 1), and every pass samples
    the tail of that one schedule below its own start, so all passes share its spacing, a pass with
    less noise takes fewer steps, and no pass depends on the others' values. NOISE_LEVEL: the tail
    starts at the last sigma at or above the level and is scaled down to start exactly at it, so a
    level between two sigmas (2.4 steps of the schedule, say) gets the count rounded up (3) and no
    step is longer than the schedule's. DENOISE: the last ceil(steps * denoise) steps, as they are.

    Either way a pass gets at least 1 step, and a value of 0 samples nothing."""
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
    total = int(steps)
    full = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total).cpu()
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
        # share: the part of the full schedule the pass covers, the BasicScheduler denoise of its
        # start (in NOISE_LEVEL mode fractional, 2.4 / 25 in the example above).
        schedules.append(Schedule(mode, v, sigmas, total, first, sigma_to_level(first, flow), share))
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
                    f"with {scheduler}, {steps} steps")
        return (schedule.sigmas, info)
