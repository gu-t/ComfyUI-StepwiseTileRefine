"""The fork nodes' one progress bar (run_progress.py): a value that never goes down, a total fixed
before the run that the value reaches exactly, and a status line sent only when it changes, in
every tile mode, over several passes and a batch. Without a run bar the engines report as before."""
import sys
import types

import pytest
import torch
from test_all_at_once import WrapperGuider
from test_preview import FakePreviewer
from test_stepwise import ScaleVAE, ZeroNoise, _smooth_image, _stepwise_guider
from test_tiling import RecordingPreviewer
from test_upscale import FakeUpscaleModel

from context_anchored_tile_refine import all_at_once, run_progress, sampling
from context_anchored_tile_refine.all_at_once import ALL_AT_ONCE, SHIFTING, TILE_BY_TILE, TILE_MODES
from context_anchored_tile_refine.stepwise_nodes import StepwiseTileUpscale, TileRefineAnyVAE

SIGMAS = torch.linspace(1.0, 0.0, 5)


@pytest.fixture()
def status(monkeypatch):
    """A recording PromptServer installed as the `server` module. Returns the (text, node_id) sent."""
    sent = []

    class RecordingPromptServer:
        def send_progress_text(self, text, node_id, sid=None):
            sent.append((text, node_id))

    module = types.ModuleType("server")
    module.PromptServer = RecordingPromptServer
    RecordingPromptServer.instance = RecordingPromptServer()
    monkeypatch.setitem(sys.modules, "server", module)
    return sent


@pytest.fixture()
def counting_tiled_scale(comfy_stubs):
    # The stub tiled_scale runs the model in one go; this one also ticks the bar once per tile, as
    # core's does.
    utils = sys.modules["comfy.utils"]
    plain = utils.tiled_scale

    def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, pbar=None, **kwargs):
        if pbar is not None:
            for _ in range(utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)):
                pbar.update(1)
        return plain(samples, function, tile_x=tile_x, tile_y=tile_y, overlap=overlap, **kwargs)

    utils.tiled_scale = tiled_scale
    return comfy_stubs


def _the_one_bar(recorded):
    # Exactly one ProgressBar for the run; every update on its own total, never going down, ending
    # on the total.
    bars = recorded["progress_bars"]
    assert len(bars) == 1, [b.total for b in bars]
    bar = bars[0]
    values = [v for v, _, _ in bar.updates]
    assert {t for _, t, _ in bar.updates} == {bar.total}
    assert values == sorted(values)
    assert values[-1] == bar.total
    return bar


# --- the bar on its own ----------------------------------------------------------------------------

def test_parts_split_the_bar_by_weight_and_end_exactly(comfy_stubs):
    run = run_progress.RunProgress([2000, 0, 1000])
    first, empty, last = run.part(0), run.part(1), run.part(2)
    assert run.total == 3000 and run.scale == 1
    first.update_absolute(5, 10)
    assert run.value == 1000
    first.update_absolute(10, 10)
    empty.update_absolute(0, 0)
    assert run.value == 2000
    last.update_absolute(1, 3)
    assert run.value == 2333
    last.update_absolute(3, 3)
    assert run.value == run.total


def test_a_light_run_is_drawn_in_finer_units(comfy_stubs):
    run = run_progress.RunProgress([4, 1.2])
    assert run.scale == 167 and run.total == (4 + 2) * 167
    part = run.part(0)
    for i in range(1, 13):
        before = run.value
        part.update_absolute(i, 12)
        assert run.value > before


def test_the_value_never_goes_down_and_a_restart_holds_it(comfy_stubs):
    run = run_progress.RunProgress([10, 10])
    part = run.part(0)
    part.update_absolute(8, 10)
    held = run.value
    part.update_absolute(0, 20)  # an OOM retry: the same part again at a new size
    part.update(4)
    assert run.value == held
    part.update(16)
    assert run.value == run.ends[0]
    run.part(0).update_absolute(1, 10)
    assert run.value == run.ends[0]


def test_a_preview_is_sent_even_when_the_value_stays(comfy_stubs):
    run = run_progress.RunProgress([10])
    with run:
        part = run.part(0)
        part.update_absolute(5, 10)
        part.update_absolute(5, 10, preview="image")
    bar = comfy_stubs["progress_bars"][0]
    assert bar.updates[-2] == (run.total // 2, run.total, "image")


def test_the_status_is_sent_once_per_change_and_ends_with_the_outcome(comfy_stubs, status):
    with run_progress.RunProgress([10], node_id="7") as run:
        part = run.part(0, "step 1/2")
        part.note("tile 1/3")
        part.note("tile 1/3")
        part.note("tile 2/3")
    assert [t for t, _ in status[:2]] == ["step 1/2 · tile 1/3", "step 1/2 · tile 2/3"]
    assert status[-1][0].startswith("done in ") and {n for _, n in status} == {"7"}
    with pytest.raises(RuntimeError), run_progress.RunProgress([10], node_id="7"):
        raise RuntimeError("boom")
    assert status[-1] == ("stopped", "7")


def test_no_status_without_a_node_id_or_a_server(comfy_stubs, monkeypatch):
    module = types.ModuleType("server")

    class PromptServer:
        def send_progress_text(self, text, node_id, sid=None):
            raise AssertionError("nothing may be sent")

    module.PromptServer = PromptServer
    monkeypatch.setitem(sys.modules, "server", module)
    run_progress.send_status("7", "headless: no instance")
    PromptServer.instance = None
    run_progress.send_status("7", "headless: None")
    PromptServer.instance = PromptServer()
    run_progress.send_status(None, "no node id")


# --- the nodes -------------------------------------------------------------------------------------

@pytest.mark.parametrize("tile_mode", TILE_MODES)
@pytest.mark.parametrize("pictures", [1, 2])
def test_stepwise_fills_one_bar_over_every_pass(counting_tiled_scale, status, tile_mode, pictures):
    image = torch.cat([_smooth_image(120, 160)] * pictures)
    guider = _stepwise_guider() if tile_mode == TILE_BY_TILE else WrapperGuider()
    final, _, _ = StepwiseTileUpscale().upscale(
        image, guider, object(), ZeroNoise(), ScaleVAE(16, 128), "simple", 6, 3.0, 3, 256, 256, 32, 32, 0.5, 0.2,
        False, upscale_model=FakeUpscaleModel(scale=2), tile_mode=tile_mode, unique_id="12")
    assert final.shape[0] == pictures
    bar = _the_one_bar(counting_tiled_scale)
    texts = [t for t, _ in status]
    assert texts[0] == "step 1/3 · upscale model"
    assert texts[-1].startswith("done in ")
    assert len(texts) == len(set(texts))
    picture = "image 2/2 · " if pictures == 2 else ""
    if tile_mode == TILE_BY_TILE:
        assert any(t.startswith(f"step 3/3 · {picture}tile ") for t in texts)
    else:
        assert any(t.startswith(f"step 3/3 · {picture}sampler step ") for t in texts)
    # Every refine pass moves the bar: one value per unit the engine reports.
    assert len({v for v, _, _ in bar.updates}) > 3 * pictures


def test_stepwise_weights_follow_the_plan_text(comfy_stubs):
    from context_anchored_tile_refine import noise_level
    from context_anchored_tile_refine.stepwise_nodes import run_bars

    image = _smooth_image(120, 160)
    plan = StepwiseTileUpscale.build_plan(image, _stepwise_guider(), ScaleVAE(16, 128), 3.0, 3, 256, 256, 32, 32,
                                          0.5, 0.2, upscale_model=FakeUpscaleModel(scale=2))
    schedules = [noise_level.Schedule("noise level", 0.5, torch.arange(n + 1.0), 10, 1.0, 0.5, 0.5) for n in (5, 4, 0)]
    run, parts = run_bars(plan, schedules, pictures=2, upscale_model=object())
    tiles = [s.tiles for s in plan.steps]
    units = [1, tiles[0] * 5 * 2, tiles[1] * 4 * 2, 0]  # 0.1 x 1 model tile x 2 pictures, rounded up
    assert run.ends == [sum(units[:i + 1]) * run.scale for i in range(4)]
    assert parts[0][0] is not None and parts[1][0] is None and parts[2][0] is None
    assert parts[1][1].label == "step 2/3"


@pytest.mark.parametrize("tile_mode", TILE_MODES)
@pytest.mark.parametrize("pictures", [1, 3])
def test_tile_refine_fills_one_bar_over_the_batch(comfy_stubs, status, tile_mode, pictures):
    image = torch.cat([_smooth_image(600, 744)] * pictures)
    guider = _stepwise_guider() if tile_mode == TILE_BY_TILE else WrapperGuider()
    comfy_stubs["previewer"] = RecordingPreviewer()
    TileRefineAnyVAE().refine(image, guider, object(), SIGMAS, ScaleVAE(16, 128), ZeroNoise(), 256, 256, 16, 32,
                              tile_mode=tile_mode, unique_id="3")
    bar = _the_one_bar(comfy_stubs)
    # Core's x0 preview still rides on every step, now on the run bar.
    assert sum(1 for _, _, p in bar.updates if p is not None) == 4 * (len(guider.calls) if tile_mode == TILE_BY_TILE
                                                                       else pictures)
    texts = [t for t, _ in status]
    if pictures > 1:
        assert any(t.startswith(f"image {pictures}/{pictures} · ") for t in texts)
    assert comfy_stubs["prepare_callback_args"] is None


@pytest.mark.parametrize("tile_mode", [ALL_AT_ONCE, SHIFTING])
def test_a_preview_mode_rides_on_the_run_bar(comfy_stubs, tile_mode):
    # preview_mode's per-window previews land on the step the run is at, on the same bar.
    comfy_stubs["previewer"] = FakePreviewer()
    TileRefineAnyVAE().refine(_smooth_image(600, 744), WrapperGuider(), object(), SIGMAS, ScaleVAE(16, 128),
                              ZeroNoise(), 256, 256, 16, 32, tile_mode=tile_mode, preview_mode="guess, one tile at a time")
    bar = _the_one_bar(comfy_stubs)
    assert sum(1 for _, _, p in bar.updates if p is not None) > 4


def test_a_zero_step_refine_still_fills_the_bar(comfy_stubs):
    TileRefineAnyVAE().refine(_smooth_image(64, 64), _stepwise_guider(), object(), torch.tensor([0.3]),
                              ScaleVAE(16, 128), ZeroNoise(), 256, 256, 16, 32)
    assert _the_one_bar(comfy_stubs).total == 1


# --- without a run bar, as before ---------------------------------------------------------------

def test_tile_by_tile_without_a_bar_keeps_its_own(comfy_stubs):
    image = _smooth_image(600, 744)
    guider = _stepwise_guider()
    sampling.refine_image(image, guider, object(), SIGMAS, ScaleVAE(16, 128), ZeroNoise(), 256, 256, 16, 32)
    (bar,) = comfy_stubs["progress_bars"]
    assert bar.total == 4 * len(guider.calls)
    assert [v for v, _, _ in bar.updates] == list(range(1, bar.total + 1))


@pytest.mark.parametrize("tile_mode", [ALL_AT_ONCE, SHIFTING])
def test_all_at_once_without_a_bar_uses_cores_preview_callback(comfy_stubs, tile_mode):
    image = torch.cat([_smooth_image(600, 744)] * 2)
    sampling.refine_image(image, WrapperGuider(), object(), SIGMAS, ScaleVAE(16, 128), ZeroNoise(), 256, 256, 16, 32,
                          tile_mode=tile_mode)
    assert comfy_stubs["progress_bars"] == []
    assert comfy_stubs["prepare_callback_args"][1] == 4
    assert comfy_stubs["callback_calls"] == [(s, 4) for s in range(4)] * 2


def test_the_preview_hub_without_a_bar_keeps_its_own_per_picture(comfy_stubs):
    comfy_stubs["previewer"] = FakePreviewer()
    image = torch.cat([_smooth_image(600, 744)] * 2)
    sampling.refine_image(image, WrapperGuider(), object(), SIGMAS, ScaleVAE(16, 128), ZeroNoise(), 256, 256, 16, 32,
                          tile_mode=ALL_AT_ONCE, preview_mode="guess, one tile at a time")
    bars = comfy_stubs["progress_bars"]
    assert [b.total for b in bars] == [4, 4]
    assert all(v <= 4 and t == 4 for b in bars for v, t, _ in b.updates)


def test_the_upscale_model_without_a_bar_keeps_its_own(counting_tiled_scale):
    from context_anchored_tile_refine import upscale

    upscale._upscale_with_model(FakeUpscaleModel(scale=2), _smooth_image(600, 744))
    (bar,) = counting_tiled_scale["progress_bars"]
    assert bar.total == 4


def test_guarded_callback_without_a_bar_or_hub_is_cores(comfy_stubs):
    wrapper = all_at_once.TiledModelWrapper(lambda h, w, s: None)
    wrapper.calls = 1
    all_at_once._guarded_callback(_stepwise_guider(), SIGMAS, wrapper)(0, None, None, 4)
    assert comfy_stubs["callback_calls"] == [(0, 4)]
    assert comfy_stubs["progress_bars"] == []
