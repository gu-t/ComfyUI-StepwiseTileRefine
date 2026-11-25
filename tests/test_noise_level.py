"""The noise level denoise mode (noise_level.py) and its use in the Stepwise node."""
import sys
from types import SimpleNamespace

import pytest
import torch

from context_anchored_tile_refine import noise_level
from context_anchored_tile_refine.noise_level import DENOISE, NOISE_LEVEL, build_schedule, solve_total


def flow_simple(n, shift=5.0):
    # core's simple_scheduler over ModelSamplingDiscreteFlow (1000 timesteps, time_snr_shift).
    t = torch.arange(1, 1001, dtype=torch.float64) / 1000
    table = shift * t / (1 + (shift - 1) * t)
    ss = len(table) / n
    return torch.tensor([float(table[-(1 + int(x * ss))]) for x in range(n)] + [0.0])


def test_level_sigma_round_trip():
    for level in (0.0, 0.1, 0.35, 0.9):
        assert noise_level.level_to_sigma(level, True) == level
        assert noise_level.sigma_to_level(noise_level.level_to_sigma(level, False), False) == pytest.approx(level)
    assert noise_level.level_to_sigma(1.0, False) == float("inf")
    assert noise_level.sigma_to_level(14.6, False) == pytest.approx(14.6 / 15.6)


@pytest.mark.parametrize("denoise", [0.05, 0.1, 0.22, 0.3, 0.6])
def test_solve_total_finds_the_basic_scheduler_length(denoise):
    # The start sigma BasicScheduler gives for `denoise` is found again from the sigma alone.
    steps = 25
    target = float(flow_simple(int(steps / denoise))[-(steps + 1)])
    total = solve_total(lambda n: float(flow_simple(n)[-(steps + 1)]), steps, target, steps * 1000)
    assert float(flow_simple(total)[-(steps + 1)]) == pytest.approx(target, abs=1e-9)


def test_solve_total_clamps_to_the_ends():
    def start(n):
        return 10.0 / n

    assert solve_total(start, 10, 5.0, 10_000) == 10       # above the full schedule: denoise 1
    assert solve_total(start, 10, 1e-6, 10_000) == 10_000  # below the longest: the longest
    assert solve_total(start, 10, 0.1, 10_000) == 100


def _patcher(flow):
    import comfy.model_sampling

    ms = comfy.model_sampling.CONST() if flow else object()
    return SimpleNamespace(get_model_object=lambda name: ms)


def test_build_schedule_noise_level_mode(comfy_stubs, monkeypatch):
    monkeypatch.setattr(sys.modules["comfy.samplers"], "calculate_sigmas", lambda ms, name, n: flow_simple(n))
    s = build_schedule(_patcher(True), "simple", 25, 0.3, NOISE_LEVEL)
    assert s.sigmas.numel() == 26
    assert s.start_level == pytest.approx(0.3, abs=0.005)
    assert s.start_sigma == float(s.sigmas[0])
    assert s.denoise == pytest.approx(25 / s.total)
    assert 0.07 < s.denoise < 0.09   # MOSS shift 5: start noise 0.3 is denoise ~0.08 with simple


def test_build_schedule_denoise_mode_is_basic_scheduler(comfy_stubs, monkeypatch):
    monkeypatch.setattr(sys.modules["comfy.samplers"], "calculate_sigmas", lambda ms, name, n: flow_simple(n))
    s = build_schedule(_patcher(True), "simple", 25, 0.22, DENOISE)
    assert s.total == int(25 / 0.22)
    assert torch.equal(s.sigmas, flow_simple(int(25 / 0.22))[-26:])
    assert s.start_level == pytest.approx(0.588, abs=0.001)


@pytest.mark.parametrize("mode", [NOISE_LEVEL, DENOISE])
def test_build_schedule_zero_samples_nothing(comfy_stubs, mode):
    s = build_schedule(_patcher(True), "simple", 25, 0.0, mode)
    assert s.sigmas.numel() == 0 and s.total == 0


def test_build_schedule_rejects_unknown_mode(comfy_stubs):
    with pytest.raises(ValueError, match="denoise_mode"):
        build_schedule(_patcher(True), "simple", 25, 0.3, "sigma")


def test_eps_models_use_the_noise_share(comfy_stubs, monkeypatch):
    # SD/SDXL-type sigmas run to ~14.6; level 0.5 is sigma 1.
    sigmas = torch.cat([torch.linspace(14.6, 0.03, 2000, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)])

    def calc(ms, name, n):
        idx = torch.linspace(0, 1999, n).round().long()
        return torch.cat([sigmas[idx], torch.zeros(1, dtype=torch.float64)])

    monkeypatch.setattr(sys.modules["comfy.samplers"], "calculate_sigmas", calc)
    s = build_schedule(_patcher(False), "normal", 20, 0.5, NOISE_LEVEL)
    assert s.start_sigma == pytest.approx(1.0, abs=0.02)
    assert s.start_level == pytest.approx(0.5, abs=0.005)


def test_scheduler_node(comfy_stubs, monkeypatch):
    monkeypatch.setattr(sys.modules["comfy.samplers"], "calculate_sigmas", lambda ms, name, n: flow_simple(n))
    sigmas, info = noise_level.NoiseLevelScheduler().get_sigmas(_patcher(True), "simple", 25, 0.25)
    assert sigmas.numel() == 26 and float(sigmas[0]) == pytest.approx(0.25, abs=0.005)
    assert "same as denoise" in info


@pytest.mark.comfy
@pytest.mark.parametrize("scheduler", ["simple", "beta", "karras", "normal", "exponential", "sgm_uniform",
                                       "kl_optimal", "linear_quadratic"])
@pytest.mark.parametrize("level", [0.2, 0.35, 0.6])
def test_real_core_schedulers_start_at_the_level(comfy_env, scheduler, level):
    # Real core schedulers over a MOSS-like flow model (shift 5): every one starts at the
    # schedule length nearest the level. Near full noise a schedule length is a coarse step
    # for karras / exponential, so the tight bound is held only at upscale levels.
    import comfy.model_sampling

    class FlowShift5(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
        pass

    ms = FlowShift5(SimpleNamespace(sampling_settings={"shift": 5.0, "multiplier": 1000}))
    patcher = SimpleNamespace(get_model_object=lambda name: ms)
    s = build_schedule(patcher, scheduler, 25, level, NOISE_LEVEL)
    assert s.sigmas.numel() == 26
    if level <= 0.35:
        assert s.start_level == pytest.approx(level, abs=0.02)
    import comfy.samplers

    def start(n):
        return float(comfy.samplers.calculate_sigmas(ms, scheduler, n)[-26])

    # linear_quadratic is not quite monotonic in the length, so the bisection may stop a hair
    # (< 0.001) from the very nearest; that is far below anything visible.
    assert abs(s.start_sigma - level) <= min(abs(start(n) - level) for n in (s.total - 1, s.total + 1) if n >= 25) + 1e-3
    assert all(float(a) >= float(b) for a, b in zip(s.sigmas[:-1], s.sigmas[1:], strict=True))
