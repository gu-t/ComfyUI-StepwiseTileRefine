"""The tile modes (all_at_once.py, grid.grow_layout): span and window geometry, the seed/sigma grid
offsets, the model_function_wrapper on fake models (shared windows run once, the fixed grid's bands
weighted together), tile_size "always full" in all three modes, the memory-estimate patch, and both
fork nodes end to end with a fake guider that drives the wrapper like core's sampler would.
No GPU and no model."""
import itertools
from types import SimpleNamespace

import pytest
import torch
from test_stepwise import ScaleVAE, ZeroNoise, _smooth_image, _stepwise_guider

from context_anchored_tile_refine import all_at_once, grid, sampling
from context_anchored_tile_refine.all_at_once import (
    ALL_AT_ONCE,
    SHIFTING,
    SIZE_FULL,
    SIZE_SMALLER,
    TILE_BY_TILE,
    TILE_SIZES,
    FixedPlanner,
    ShiftingPlanner,
    TiledModelWrapper,
    axis_spans,
    fixed_windows,
    grid_offsets,
    shifting_windows,
    tile_cells,
)
from context_anchored_tile_refine.geometry import LatentGeometry
from context_anchored_tile_refine.stepwise_nodes import StepwiseTileUpscale, TileRefineAnyVAE

SIGMAS = torch.linspace(1.0, 0.0, 5)
MOSS = LatentGeometry(16, 16)


# --- shifting spans -----------------------------------------------------------------------------

@pytest.mark.parametrize("tile_size", TILE_SIZES)
@pytest.mark.parametrize("length, core, ring, unit", [(40, 12, 0, 1), (40, 12, 3, 1), (37, 10, 2, 1), (64, 16, 4, 2),
                                                      (100, 30, 5, 1), (9, 4, 1, 1)])
def test_cores_cover_every_cell_exactly_once(tile_size, length, core, ring, unit):
    for offset in range(0, core, unit):
        if length % unit:
            continue
        spans = axis_spans(length, core, ring, offset, tile_size, unit)
        hits = torch.zeros(length, dtype=torch.int64)
        for s in spans:
            assert 0 <= s.core0 < s.core1 <= length
            assert s.crop0 <= s.core0 and s.core1 <= s.crop1
            assert s.core1 - s.core0 <= core
            hits[s.core0:s.core1] += 1
            assert s.crop0 >= 0 and s.crop1 <= length
            ringed = (max(s.core0 - ring, 0), min(s.core1 + ring, length))
            if tile_size == SIZE_SMALLER:
                assert (s.crop0, s.crop1) == ringed
            elif length > core + 2 * ring:
                # Always full: every window is the whole cap, and holds at least the ringed crop.
                assert s.crop1 - s.crop0 == core + 2 * ring
                assert s.crop0 <= ringed[0] and s.crop1 >= ringed[1]
                if ringed[0] > 0 and ringed[1] < length:
                    assert (s.crop0, s.crop1) == ringed
        assert torch.equal(hits, torch.ones(length, dtype=torch.int64)), (offset, spans)


def test_an_axis_that_fits_one_tile_has_no_seam():
    for offset in range(8):
        assert axis_spans(20, 8, 6, offset, SIZE_SMALLER) == [all_at_once.Span(0, 20, 0, 20)]


def test_the_offset_grid_needs_one_more_tile():
    assert len(axis_spans(40, 20, 2, 0, SIZE_SMALLER)) == 2
    assert len(axis_spans(40, 20, 2, 5, SIZE_SMALLER)) == 3


def test_always_full_slides_an_edge_window_inward():
    # length 40, core 12, ring 2, offset 10: cores 0-2, 2-14, 14-26, 26-38, 38-40.
    spans = axis_spans(40, 12, 2, 10, SIZE_FULL)
    assert [(s.core0, s.core1) for s in spans] == [(0, 2), (2, 14), (14, 26), (26, 38), (38, 40)]
    assert [(s.crop0, s.crop1) for s in spans] == [(0, 16), (0, 16), (12, 28), (24, 40), (24, 40)]


def test_tile_cells_subtracts_the_ring_and_snaps_to_the_grid():
    assert tile_cells(1024, 768, 64, MOSS) == (40, 56, 4)
    assert tile_cells(1030, 1024, 20, MOSS) == (60, 60, 2)  # ring 20 -> 32, cap 1030 -> 1024
    assert tile_cells(1024, 1024, 0, MOSS) == (64, 64, 0)
    with pytest.raises(ValueError, match="no core"):
        tile_cells(256, 256, 128, MOSS)


# --- offsets ------------------------------------------------------------------------------------

def test_grid_offsets_replay_from_seed_and_sigma():
    sigmas = torch.linspace(1, 0.01, 25).tolist()
    a = [grid_offsets(42, s, 20, 30) for s in sigmas]
    b = [grid_offsets(42, s, 20, 30) for s in sigmas]
    c = [grid_offsets(43, s, 20, 30) for s in sigmas]
    assert a == b and a != c
    assert len(set(a)) > 15
    assert all(0 <= dy < 20 and 0 <= dx < 30 for dy, dx in a)


def test_grid_offsets_stay_on_the_unit_grid():
    for s in torch.linspace(1, 0.01, 40).tolist():
        dy, dx = grid_offsets(7, s, 16, 24, unit=2)
        assert dy % 2 == 0 and dx % 2 == 0 and dy < 16 and dx < 24


# --- shared windows -----------------------------------------------------------------------------

def test_two_cores_with_one_window_run_once():
    # rt_city 1664 px at 16 px cells, max_tile 1024, ring 64: core 56 cells. dx 832 px = 52 cells gives
    # the cores 0-64 px and 64-960 px, and always full gives both the window 0-1024 px.
    core_h, core_w, ring = tile_cells(1024, 1024, 64, MOSS)
    windows = shifting_windows(1, 104, 1, core_w, ring, (0, 52), SIZE_FULL)
    assert [(w.x0, w.x1) for w in windows] == [(0, 64), (40, 104)]
    assert [[(r.x0, r.x1) for r in w.writes] for w in windows] == [[(0, 4), (4, 60)], [(60, 104)]]
    calls = []
    wrapper = TiledModelWrapper(ShiftingPlanner(core_h, core_w, ring, seed=0))
    wrapper.planner = lambda h, w, s: ((0, 52), shifting_windows(h, w, core_h, core_w, ring, (0, 52), SIZE_FULL))
    x = torch.randn(1, 3, 1, 104)
    out = wrapper(lambda x, t, **c: calls.append(tuple(x.shape)) or x * 3.0, {"input": x, "timestep": torch.tensor([0.5]), "c": {}})
    assert torch.equal(out, x * 3.0) and len(calls) == 2


# --- the wrapper, shifting ------------------------------------------------------------------------

def _local_apply(sizes):
    def apply_model(x, t, **c):
        sizes.append(tuple(x.shape[-2:]))
        out = x * 2.0
        if "c_concat" in c:
            out = out + c["c_concat"]
        return out
    return apply_model


@pytest.mark.parametrize("tile_size", TILE_SIZES)
@pytest.mark.parametrize("ring", [0, 2, 3])
def test_a_local_model_through_the_shifting_wrapper_is_the_whole_canvas_call(tile_size, ring):
    x = torch.randn(2, 5, 23, 31)
    concat = torch.randn(2, 1, 23, 31).expand(2, 5, 23, 31).contiguous()
    text = torch.randn(2, 7, 9)
    wrapper = TiledModelWrapper(ShiftingPlanner(8, 10, ring, seed=5, tile_size=tile_size), log=[])
    for sigma in (0.9, 0.5, 0.1):
        sizes = []
        out = wrapper(_local_apply(sizes), {"input": x, "timestep": torch.full((2,), sigma),
                                            "c": {"c_concat": concat, "c_crossattn": text, "transformer_options": {}},
                                            "cond_or_uncond": [0, 1]})
        assert torch.equal(out, x * 2.0 + concat)
        assert all(h <= 8 + 2 * ring and w <= 10 + 2 * ring for h, w in sizes)
        if tile_size == SIZE_FULL:
            assert set(sizes) == {(8 + 2 * ring, 10 + 2 * ring)}
    assert len({(dy, dx) for _, dy, dx, _ in wrapper.log}) > 1


def test_cond_and_uncond_calls_at_one_sigma_share_the_grid():
    log = []
    wrapper = TiledModelWrapper(ShiftingPlanner(8, 8, 1, seed=9), log=log)
    x = torch.randn(1, 3, 30, 30)
    for sigma in (0.8, 0.8, 0.4, 0.4):
        wrapper(lambda x, t, **c: x, {"input": x, "timestep": torch.tensor([sigma]), "c": {}})
    assert log[0][1:] == log[1][1:] and log[2][1:] == log[3][1:]


def test_a_model_wrapper_already_on_the_model_runs_per_window():
    calls = []

    def inner(apply_model, args):
        calls.append(tuple(args["input"].shape[-2:]))
        return apply_model(args["input"], args["timestep"], **args["c"])

    guider = SimpleNamespace(model_options={"model_function_wrapper": inner, "other": 1})
    wrapper = TiledModelWrapper(ShiftingPlanner(8, 8, 0, seed=1))
    run = all_at_once.wrapped_guider(guider, wrapper)
    assert guider.model_options["model_function_wrapper"] is inner
    assert run.model_options["model_function_wrapper"] is wrapper and run.model_options["other"] == 1
    x = torch.randn(1, 2, 20, 20)
    out = run.model_options["model_function_wrapper"](lambda x, t, **c: x + 1, {"input": x, "timestep": torch.tensor([0.5]), "c": {}})
    assert torch.equal(out, x + 1) and len(calls) >= 4


def test_controlnet_is_rejected_in_the_wrapper():
    wrapper = TiledModelWrapper(ShiftingPlanner(8, 8, 0, seed=1))
    with pytest.raises(ValueError, match="ControlNet"):
        wrapper(lambda x, t, **c: x, {"input": torch.zeros(1, 1, 9, 9), "timestep": torch.tensor([0.5]),
                                      "c": {"control": object()}})


# --- the fixed grid (all at once) ---------------------------------------------------------------

FIXED_CASES = [
    pytest.param(1664, 1248, 1024, 64, 32, 16, id="rt_city-2x2"),
    pytest.param(2816, 2816, 1024, 64, 32, 16, id="8mp-3x3"),
    pytest.param(1664, 1248, 1024, 64, 0, 16, id="no-band"),
    pytest.param(640, 480, 256, 16, 16, 8, id="8x-3x2"),
]


@pytest.mark.parametrize("w, h, cap, anchor, overlap, scale", FIXED_CASES)
@pytest.mark.parametrize("tile_size", TILE_SIZES)
def test_fixed_weights_partition_the_canvas(w, h, cap, anchor, overlap, scale, tile_size):
    g = LatentGeometry(scale, scale)
    layout = all_at_once.fixed_layout(w, h, cap, cap, anchor, overlap, g, tile_size)
    windows = fixed_windows(layout, scale)
    total = torch.zeros(h // scale, w // scale)
    for win in windows:
        assert (win.x1 - win.x0) * scale <= cap and (win.y1 - win.y0) * scale <= cap
        if tile_size == SIZE_FULL:
            assert (win.x1 - win.x0) * scale == min(cap, w) and (win.y1 - win.y0) * scale == min(cap, h)
        for wr in win.writes:
            assert win.y0 <= wr.y0 < wr.y1 <= win.y1 and win.x0 <= wr.x0 < wr.x1 <= win.x1
            total[wr.y0:wr.y1, wr.x0:wr.x1] += 1.0 if wr.weight is None else wr.weight
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_a_local_model_through_the_fixed_wrapper_is_the_whole_canvas_call():
    layout = all_at_once.fixed_layout(1664, 1248, 1024, 1024, 64, 32, MOSS, SIZE_FULL)
    windows = fixed_windows(layout, 16)
    x = torch.randn(1, 4, 78, 104)
    sizes = []
    out = TiledModelWrapper(FixedPlanner(windows, 78, 104))(_local_apply(sizes), {"input": x, "timestep": torch.tensor([0.5]), "c": {}})
    assert torch.allclose(out, x * 2.0, atol=1e-5)
    assert sizes == [(64, 64)] * 4


def test_fixed_bands_blend_the_neighbours_smoothly():
    # Each window answers a constant, its own index: cores keep it, bands ramp monotonically between.
    layout = all_at_once.fixed_layout(1664, 1248, 1024, 1024, 64, 32, MOSS, SIZE_SMALLER)
    windows = fixed_windows(layout, 16)
    ids = iter(range(len(windows)))
    out = TiledModelWrapper(FixedPlanner(windows, 78, 104))(
        lambda x, t, **c: torch.full_like(x, float(next(ids))), {"input": torch.zeros(1, 1, 78, 104), "timestep": torch.tensor([0.5]), "c": {}})
    row = out[0, 0, 10]  # inside the top row of tiles: 0 on the left, 1 on the right, seam at cell 52
    assert torch.all(row[:50] == 0) and torch.all(row[54:] == 1)
    band = row[50:54]
    assert torch.all(band[1:] > band[:-1]) and 0 < float(band[0]) < 0.5 < float(band[-1]) < 1


def test_fixed_planner_refuses_another_canvas():
    planner = FixedPlanner([], 10, 12)
    with pytest.raises(RuntimeError, match="grid"):
        planner(10, 13, 0.5)


# --- always full on the planner's grid -------------------------------------------------------------

def test_grow_layout_matches_the_rt_city_example():
    # Tile 4 of rt_city (1664x1248, max_tile 1024, anchor 64, overlap 32) sees 928x720 bare, 1024x1024 grown.
    layout = grid.solve_layout(1664, 1248, 1024, 1024, 64, 32, multiple=16)
    t4 = layout.tiles[3]
    assert (t4.crop_rect.x0, t4.crop_rect.y0, t4.crop_rect.x1, t4.crop_rect.y1) == (736, 528, 1664, 1248)
    grown = grid.grow_layout(layout, 1024, 1024, 16)
    g4 = grown.tiles[3]
    assert (g4.crop_rect.x0, g4.crop_rect.y0, g4.crop_rect.x1, g4.crop_rect.y1) == (640, 224, 1664, 1248)
    for bare, big in zip(layout.tiles, grown.tiles, strict=True):
        assert (big.sampled_w, big.sampled_h) == (1024, 1024)
        assert big.core == bare.core and big.overlap_inner_rect == bare.overlap_inner_rect and big.paste_rect == bare.paste_rect
        assert big.crop_rect.x0 <= bare.crop_rect.x0 and big.crop_rect.x1 >= bare.crop_rect.x1
        assert big.crop_rect.y0 <= bare.crop_rect.y0 and big.crop_rect.y1 >= bare.crop_rect.y1
    assert grown.total_sampled_px == 4 * 1024 * 1024


def test_grow_span_stays_on_the_grid_and_inside():
    for a, b, length, cap in [(0, 100, 300, 256), (200, 300, 300, 256), (96, 200, 300, 256), (0, 50, 60, 256)]:
        x0, x1 = grid.grow_span(a, b, length, cap, 16)
        assert 0 <= x0 <= a and b <= x1 <= length and x1 - x0 == max(min(cap, length), b - a)


@pytest.mark.parametrize("tile_size", TILE_SIZES)
def test_tile_by_tile_always_full_samples_max_tile_windows(comfy_stubs, tile_size):
    from test_stepwise import CheckingGuider

    image = _smooth_image(600, 744)
    vae = ScaleVAE(16, 128)
    guider = CheckingGuider()
    (out,) = TileRefineAnyVAE().refine(image, guider, object(), SIGMAS, vae, ZeroNoise(), 512, 512, 32, 32,
                                       tile_size=tile_size)
    assert out.shape == image.shape
    if tile_size == SIZE_FULL:
        assert set(vae.encode_shapes) == {(512, 512)}
    else:
        assert any(shape != (512, 512) for shape in vae.encode_shapes)


# --- memory -------------------------------------------------------------------------------------

def test_memory_estimate_is_clamped_to_the_tile_and_restored():
    class Model:
        def memory_required(self, input_shape, cond_shapes={}):  # noqa: B006
            return tuple(input_shape)

    model = Model()
    guider = SimpleNamespace(model_patcher=SimpleNamespace(model=model))
    with all_at_once.tile_memory_estimate(guider, 40, 50):
        assert model.memory_required([2, 128, 177, 30]) == (2, 128, 40, 30)
    assert "memory_required" not in vars(model)
    assert model.memory_required([2, 128, 177, 30]) == (2, 128, 177, 30)


# --- end to end ---------------------------------------------------------------------------------

class WrapperGuider:
    """Fake guider that samples like core: every sigma calls model_options' wrapper (cond and
    uncond as two calls) around a per-cell model that answers zeros."""

    def __init__(self):
        base = _stepwise_guider()
        self.model_patcher = base.model_patcher
        self.model_options = {}
        self.calls = []
        self.tile_sizes = []

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        assert noise.shape == latent_image.shape
        wrapper = self.model_options["model_function_wrapper"]
        self.calls.append({"seed": seed, "shape": tuple(latent_image.shape)})

        def apply_model(x, t, **c):
            self.tile_sizes.append(tuple(x.shape[-2:]))
            return torch.zeros_like(x)

        x = latent_image
        steps = len(sigmas) - 1
        for i, sigma in enumerate(sigmas[:-1].tolist()):
            for _ in range(2):
                x = x + wrapper(apply_model, {"input": x, "timestep": torch.tensor([sigma]), "c": {}, "cond_or_uncond": [0]})
            if callback is not None:
                callback(i, x, x, steps)
        return x


class BypassGuider(WrapperGuider):
    """A guider that keeps its options somewhere else: it never calls model_options' wrapper and
    runs the model on the whole canvas, which is what the run must refuse."""

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        self.calls.append({"seed": seed, "shape": tuple(latent_image.shape)})
        self.tile_sizes.append(tuple(latent_image.shape[-2:]))
        callback(0, latent_image, latent_image, len(sigmas) - 1)
        self.tile_sizes.append("past the first step")
        return latent_image


@pytest.mark.parametrize("tile_mode", [ALL_AT_ONCE, SHIFTING])
def test_a_guider_that_skips_the_wrapper_stops_after_the_first_step(comfy_stubs, tile_mode):
    guider = BypassGuider()
    with pytest.raises(RuntimeError, match="never called the tiling wrapper"):
        TileRefineAnyVAE().refine(_smooth_image(600, 744), guider, object(), SIGMAS, ScaleVAE(16, 128), ZeroNoise(),
                                  256, 256, 16, 32, tile_mode=tile_mode)
    assert "past the first step" not in guider.tile_sizes


def test_the_interrupt_check_runs_before_every_window():
    checks = []
    windows = shifting_windows(30, 30, 8, 8, 1, (3, 5), SIZE_FULL)
    wrapper = TiledModelWrapper(lambda h, w, s: ((3, 5), windows), check_interrupt=lambda: checks.append(1))
    wrapper(lambda x, t, **c: x, {"input": torch.zeros(1, 1, 30, 30), "timestep": torch.tensor([0.5]), "c": {}})
    assert len(checks) == len(windows) > 1 and wrapper.calls == 1


@pytest.mark.parametrize("tile_mode", [ALL_AT_ONCE, SHIFTING])
def test_all_at_once_modes_refine_a_batch_one_picture_at_a_time(comfy_stubs, tile_mode):
    image = torch.cat([_smooth_image(600, 744), 1.0 - _smooth_image(600, 744)], dim=0)
    vae = ScaleVAE(16, 128)
    guider = WrapperGuider()
    (out,) = TileRefineAnyVAE().refine(image, guider, object(), SIGMAS, vae, ZeroNoise(), 256, 256, 16, 32,
                                       tile_mode=tile_mode)
    assert out.shape == image.shape
    assert [c["shape"][0] for c in guider.calls] == [1, 1]
    for b in range(2):
        expected = vae.decode(vae.encode(sampling.pad_image_to_multiple(image[b:b + 1], 16)[0]))[:, :600, :744]
        assert torch.allclose(out[b:b + 1], expected, atol=1e-6)


def test_tile_by_tile_always_full_with_a_mask_keeps_the_background(comfy_stubs):
    from test_stepwise import CheckingGuider

    image = _smooth_image(600, 744)
    mask = torch.zeros(1, 600, 744)
    mask[:, 100:500, 150:650] = 1.0
    vae = ScaleVAE(16, 128)
    (out,) = TileRefineAnyVAE().refine(image, CheckingGuider(), object(), SIGMAS, vae, ZeroNoise(), 256, 256, 32, 32,
                                       mask=mask, tile_size=SIZE_FULL)
    assert out.shape == image.shape
    outside = mask[0] < 0.5
    outside[95:505, 145:655] = False  # the 1 px anti-aliased edge and its neighbourhood
    assert torch.equal(out[0][outside], image[0][outside])
    assert vae.encode_shapes and all(w <= 256 and h <= 256 and w % 16 == 0 and h % 16 == 0 for w, h in vae.encode_shapes)
    assert (256, 256) in vae.encode_shapes


@pytest.mark.parametrize("tile_mode", [ALL_AT_ONCE, SHIFTING])
@pytest.mark.parametrize("tile_size", TILE_SIZES)
@pytest.mark.parametrize("scale, channels, multiple", [(8, 3, 0), (16, 128, 0), (16, 128, 32)])
def test_tile_refine_node_all_at_once_end_to_end(comfy_stubs, tile_mode, tile_size, scale, channels, multiple):
    image = _smooth_image(600, 744)
    vae = ScaleVAE(scale, channels, require=multiple or scale)
    guider = WrapperGuider()
    (out,) = TileRefineAnyVAE().refine(image, guider, object(), SIGMAS, vae, ZeroNoise(), 256, 256, 16, 32,
                                       pixel_multiple=multiple, tile_mode=tile_mode, tile_size=tile_size)
    grid_px = multiple or scale
    expected = vae.decode(vae.encode(sampling.pad_image_to_multiple(image, grid_px)[0]))[:, :600, :744]
    assert torch.allclose(out, expected, atol=1e-6)
    assert guider.model_options == {}
    assert len(guider.calls) == 1 and guider.calls[0]["seed"] == ZeroNoise.seed
    cap = 256 // scale
    assert guider.tile_sizes and all(h <= cap and w <= cap for h, w in guider.tile_sizes)
    unit = grid_px // scale
    assert all(h % unit == 0 and w % unit == 0 for h, w in guider.tile_sizes)
    if tile_size == SIZE_FULL:
        assert set(guider.tile_sizes) == {(cap, cap)}


@pytest.mark.parametrize("tile_mode", [ALL_AT_ONCE, SHIFTING])
def test_all_at_once_modes_reject_a_mask(comfy_stubs, tile_mode):
    image = _smooth_image(64, 64)
    with pytest.raises(ValueError, match="mask"):
        TileRefineAnyVAE().refine(image, WrapperGuider(), object(), SIGMAS, ScaleVAE(8), ZeroNoise(), 256, 256, 16, 16,
                                  mask=torch.ones(1, 64, 64), tile_mode=tile_mode)


@pytest.mark.parametrize("tile_mode, heading", [(ALL_AT_ONCE, "Tile mode: all at once, always full"),
                                                (SHIFTING, "windows per model call"),
                                                (TILE_BY_TILE, "Tile mode: tile by tile, always full")])
def test_stepwise_node_tile_modes_end_to_end(comfy_stubs, tile_mode, heading):
    image = _smooth_image(120, 160)
    guider = WrapperGuider() if tile_mode != TILE_BY_TILE else _stepwise_guider()
    final, pass_images, text = StepwiseTileUpscale().upscale(
        image, guider, object(), ZeroNoise(), ScaleVAE(16, 128), "simple", 4, 3.0, 3, 256, 256, 32, 32, 0.5, 0.2,
        False, tile_mode=tile_mode, tile_size=SIZE_FULL)
    assert (final.shape[2], final.shape[1]) == (480, 360)
    assert len(pass_images) == 3 and heading in text
    if tile_mode != TILE_BY_TILE:
        assert len(guider.calls) == 3
        assert max(max(s) for s in guider.tile_sizes) == 256 // 16
        # Every pass draws its noise (and its shifting grid) from its own seed (seed + pass - 1).
        assert [c["seed"] for c in guider.calls] == [ZeroNoise.seed, ZeroNoise.seed + 1, ZeroNoise.seed + 2]


STEPWISE_ORDER = [
    "image", "guider", "sampler", "noise", "vae",
    "upscale_by", "output_size", "passes", "auto_pass_factor", "oversample", "oversample_mode",
    "upscale_method", "downscale_method",
    "denoise_mode", "denoise_first", "denoise_last", "scheduler", "steps", "steps_mode",
    "tile_mode", "tile_size", "max_tile_width", "max_tile_height", "tile_overlap", "tile_padding",
    "plan_only",
    "upscale_model", "latent_scale", "pixel_multiple", "preview_mode",
]
ANY_VAE_ORDER = [
    "image", "guider", "sampler", "sigmas", "noise", "vae",
    "tile_mode", "tile_size", "max_tile_width", "max_tile_height", "tile_overlap", "tile_padding",
    "mask", "latent_scale", "pixel_multiple", "preview_mode",
]


@pytest.mark.parametrize("node, order", [(StepwiseTileUpscale, STEPWISE_ORDER), (TileRefineAnyVAE, ANY_VAE_ORDER)])
def test_the_widget_order_is_pinned(comfy_stubs, node, order):
    # The frontend restores widgets_values by position, so this order is what every saved workflow
    # (and workflows/) depends on. Append new widgets last.
    types = node.INPUT_TYPES()
    assert list(itertools.chain(types["required"], types.get("optional", {}))) == order
    assert types["required"]["tile_mode"][0] == [TILE_BY_TILE, ALL_AT_ONCE, SHIFTING]
    assert types["required"]["tile_mode"][1]["default"] == TILE_BY_TILE
    assert types["required"]["tile_size"][0] == [SIZE_FULL, SIZE_SMALLER]
    assert types["required"]["tile_size"][1]["default"] == SIZE_FULL


def test_the_example_workflow_matches_the_widget_order(comfy_stubs):
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "workflows" / "MOSS-Image stepwise upscale.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    node = next(n for n in workflow["nodes"] if n["type"] == "StepwiseTileUpscale")
    types = StepwiseTileUpscale.INPUT_TYPES()
    specs = {**types["required"], **types.get("optional", {})}
    widgets = [name for name in STEPWISE_ORDER if not isinstance(specs[name][0], str) or specs[name][0] in
               ("INT", "FLOAT", "BOOLEAN", "STRING")]
    assert len(node["widgets_values"]) == len(widgets)
    values = dict(zip(widgets, node["widgets_values"], strict=True))
    assert values["tile_mode"] == TILE_BY_TILE and values["scheduler"] == "simple" and values["tile_padding"] == 32
    assert values["steps_mode"] == "same step size" and values["steps"] == 50


def test_stepwise_shifting_ignores_tile_overlap_in_the_plan(comfy_stubs):
    # The shifting grid has no overlap band, so a tile_overlap the fixed grid cannot hold must not
    # fail the plan. Tile by tile uses the band, and there the same settings still fail.
    image = _smooth_image(120, 160)
    args = (ZeroNoise(), ScaleVAE(16, 128), "simple", 4, 3.0, 3, 256, 256, 32, 512, 0.5, 0.2, False)
    final, _, text = StepwiseTileUpscale().upscale(image, WrapperGuider(), object(), *args, tile_mode=SHIFTING)
    assert (final.shape[2], final.shape[1]) == (480, 360)
    assert "Ring: tile_padding 32 px, core 192x192 px (tile_overlap is not used)." in text
    with pytest.raises(ValueError, match="tile_overlap"):
        StepwiseTileUpscale().upscale(image, _stepwise_guider(), object(), *args, tile_mode=TILE_BY_TILE)


def test_stepwise_shifting_plan_text_shows_the_shifting_grid(comfy_stubs):
    # The shifting plan gives core and ring once and the windows per step, none of the fixed grid's
    # tiles. The same settings tile by tile still show the fixed grid.
    args = (_smooth_image(976, 1280), WrapperGuider(), object(), ZeroNoise(), ScaleVAE(16, 128), "simple", 25, 3.0, 3,
            1024, 1024, 32, 32, 0.45, 0.22, True)
    _, _, text = StepwiseTileUpscale().upscale(*args, tile_mode=SHIFTING)
    assert " tiles" not in text and "do not apply" not in text
    assert sum(line.startswith("Pass ") for line in text.splitlines()) == 3
    assert "Ring: tile_padding 32 px, core 960x960 px (tile_overlap is not used)." in text
    assert sum(line.startswith("  denoise ") and line.endswith("windows per model call")
               for line in text.splitlines()) == 3
    _, _, fixed = StepwiseTileUpscale().upscale(*args, tile_mode=TILE_BY_TILE)
    assert " tiles, core " in fixed and "Rings: tile_overlap 32 px, tile_padding 32 px." in fixed


@pytest.mark.parametrize("tile_mode, masked", [(TILE_BY_TILE, False), (TILE_BY_TILE, True), (ALL_AT_ONCE, False),
                                               (SHIFTING, False)])
def test_tile_refine_any_vae_takes_a_cap_and_rings_off_the_8_grid(comfy_stubs, tile_mode, masked):
    # Tile Refine (Any VAE) follows the fork's rule: any cap and ring in range. The rings snap up
    # to the geometry and every crop the model sees stays on the grid and under the cap.
    assert TileRefineAnyVAE.VALIDATE_INPUTS(1000, 1000, 20, 12) is True
    image = _smooth_image(1200, 1400)
    vae = ScaleVAE(16, 128)
    guider = _stepwise_guider() if tile_mode == TILE_BY_TILE else WrapperGuider()
    mask = None
    if masked:
        mask = torch.zeros(1, 1200, 1400)
        mask[:, 100:1100, 150:1250] = 1.0
    (out,) = TileRefineAnyVAE().refine(image, guider, object(), SIGMAS, vae, ZeroNoise(), 1000, 1000, 20, 12,
                                       mask=mask, tile_mode=tile_mode, tile_size=SIZE_FULL)
    assert out.shape == image.shape
    if tile_mode == TILE_BY_TILE:
        assert len(vae.encode_shapes) > 1
        assert all(w <= 1000 and h <= 1000 and w % 16 == 0 and h % 16 == 0 for w, h in vae.encode_shapes)
    else:
        assert guider.tile_sizes and all(h <= 1000 // 16 and w <= 1000 // 16 for h, w in guider.tile_sizes)


def test_stepwise_shifting_names_a_padding_that_leaves_no_core(comfy_stubs):
    with pytest.raises(ValueError, match="tile_padding"):
        StepwiseTileUpscale().upscale(_smooth_image(120, 160), WrapperGuider(), object(), ZeroNoise(),
                                      ScaleVAE(16, 128), "simple", 4, 2.0, 1, 256, 256, 128, 32, 0.5, 0.2, True,
                                      tile_mode=SHIFTING)


@pytest.mark.parametrize("tile_mode", [ALL_AT_ONCE, SHIFTING])
def test_all_at_once_modes_name_a_wrong_latent_scale(comfy_stubs, tile_mode):
    # A 16x VAE run as 8x: the canvas latent and its noise disagree, caught before sampling.
    guider = WrapperGuider()
    with pytest.raises(RuntimeError, match="latent_scale"):
        TileRefineAnyVAE().refine(_smooth_image(512, 512), guider, object(), SIGMAS, ScaleVAE(16, 128), ZeroNoise(),
                                  256, 256, 16, 32, latent_scale=8, tile_mode=tile_mode)
    assert guider.calls == []
