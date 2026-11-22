"""The fork's tests: latent geometry, the stepwise planner, and both fork nodes end to end with
fake VAEs at several latent scales (8x, 16x with 128 channels like FLUX.2 / MOSS-Image, and a
32 px grid on a 16x VAE like MiniMax H3). No GPU and no model."""
import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from test_tiling import GridGuider, GridNoise, GridVAE

from context_anchored_tile_refine import grid, sampling, stepwise_plan
from context_anchored_tile_refine.geometry import EIGHT, LatentGeometry, resolve_geometry, snap_rings, vae_latent_scale
from context_anchored_tile_refine.node import ContextAnchoredTileRefine
from context_anchored_tile_refine.stepwise_nodes import StepwiseTileUpscale, TileRefineAnyVAE, auto_tile_cap

SIGMAS = torch.linspace(1.0, 0.0, 5)


# --- fakes ---------------------------------------------------------------------------------

class ScaleVAE:
    """Position-preserving fake VAE with an arbitrary spatial compression and channel count.

    encode = stride-`scale` sample, the 3 colour channels first and the rest zero; decode = the
    first 3 channels, nearest-upsampled by `scale`. It raises on a crop that is not a multiple of
    its own `require` grid, like a real model with a patch size would misbehave."""

    def __init__(self, scale, channels=3, require=None):
        self.scale = scale
        self.latent_channels = channels
        self.require = require or scale
        self.encode_shapes = []

    def spacial_compression_encode(self):
        return self.scale

    def encode(self, pixels):
        h, w = pixels.shape[1], pixels.shape[2]
        if h % self.require or w % self.require:
            raise AssertionError(f"crop {w}x{h} is off the {self.require} px grid")
        self.encode_shapes.append((w, h))
        rgb = pixels[:, ::self.scale, ::self.scale, :3].movedim(-1, 1)
        if self.latent_channels == 3:
            return rgb
        pad = torch.zeros(rgb.shape[0], self.latent_channels - 3, rgb.shape[2], rgb.shape[3])
        return torch.cat([rgb, pad], dim=1)

    def decode(self, samples):
        rgb = samples[:, :3].movedim(1, -1)
        return rgb.repeat_interleave(self.scale, dim=1).repeat_interleave(self.scale, dim=2)


class CheckingGuider(GridGuider):
    """GridGuider that also asserts the noise, latent and denoise mask agree in shape and
    returns the latent unchanged (so the output is the encode/decode round trip)."""

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        assert noise.shape == latent_image.shape, (noise.shape, latent_image.shape)
        if denoise_mask is not None:
            assert tuple(denoise_mask.shape[-2:]) == tuple(latent_image.shape[-2:])
            assert set(denoise_mask.unique().tolist()) <= {0.0, 1.0}
        super().sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        return latent_image


class ZeroNoise:
    seed = 3

    def generate_noise(self, input_latent):
        return torch.zeros_like(input_latent["samples"])


def _smooth_image(h, w):
    ys = torch.linspace(0, 1, h)[:, None]
    xs = torch.linspace(0, 1, w)[None, :]
    return torch.stack([ys.expand(h, w), xs.expand(h, w), (ys * xs).expand(h, w)], dim=-1)[None]


GEOMETRY_CASES = [
    pytest.param(8, 3, 0, id="8x-sd"),
    pytest.param(16, 128, 0, id="16x-128ch-flux2-moss"),
    pytest.param(16, 128, 32, id="16x-grid32-minimax-h3"),
]


# --- geometry -------------------------------------------------------------------------------

def test_vae_latent_scale_reads_comfy_vae_api():
    assert vae_latent_scale(ScaleVAE(16)) == 16
    assert vae_latent_scale(SimpleNamespace(downscale_ratio=16)) == 16
    assert vae_latent_scale(SimpleNamespace(downscale_ratio=(lambda a: a, 8, 8))) == 8


@pytest.mark.parametrize("vae", [object(), MagicMock(), GridVAE(), SimpleNamespace(downscale_ratio=True)])
def test_vae_latent_scale_falls_back_to_eight(vae):
    assert vae_latent_scale(vae) == 8


def test_resolve_geometry_overrides_and_rejects_off_grid_multiple():
    assert resolve_geometry(ScaleVAE(16)) == LatentGeometry(16, 16)
    assert resolve_geometry(ScaleVAE(16), pixel_multiple=32) == LatentGeometry(16, 32)
    assert resolve_geometry(GridVAE(), latent_scale=16) == LatentGeometry(16, 16)
    with pytest.raises(ValueError, match="not a multiple"):
        resolve_geometry(ScaleVAE(16), pixel_multiple=24)


@pytest.mark.parametrize("anchor, overlap", list(itertools.product(range(0, 129, 8), range(0, 129, 8))))
def test_snap_rings_is_identity_at_eight(anchor, overlap):
    assert snap_rings(anchor, overlap, EIGHT) == (anchor, overlap)


@pytest.mark.parametrize("scale, multiple", [(16, 16), (16, 32), (8, 16), (32, 32)])
def test_snap_rings_lands_on_cells_and_grid(scale, multiple):
    g = LatentGeometry(scale, multiple)
    for anchor, overlap in itertools.product(range(0, 97, 8), range(0, 97, 8)):
        a, o = snap_rings(anchor, overlap, g)
        assert a >= anchor and o >= overlap
        assert o % scale == 0 and a % scale == 0
        assert (a + o) % multiple == 0
        assert o - overlap < scale and (a + o) - (anchor + overlap) < multiple + scale


# --- the planner ----------------------------------------------------------------------------

PLAN_CASES = [
    (1280, 960, 3.0, 3), (1280, 976, 3.0, 3), (2000, 2000, 2.0, 0), (1344, 768, 2.0, 0),
    (1280, 960, 3.0, 1), (1000, 700, 4.0, 0), (777, 555, 2.5, 2), (512, 512, 1.0, 1), (1920, 1080, 1.5, 0),
]
GEOMETRIES = [LatentGeometry(8, 8), LatentGeometry(16, 16), LatentGeometry(16, 32)]


@pytest.mark.parametrize("w, h, factor, steps", PLAN_CASES)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=["g8", "g16", "g32"])
@pytest.mark.parametrize("tolerance", [0.0, 0.01, 0.03])
def test_plan_invariants(w, h, factor, steps, geometry, tolerance):
    cap = 1024
    plan = stepwise_plan.plan_upscale(w, h, factor, steps, geometry, cap, cap, 32, 32, 0.45, 0.22, tolerance)
    m = geometry.multiple
    # Exact final size, no 64-rounding and no stretch.
    assert (plan.width, plan.height) == (round(w * factor), round(h * factor))
    assert (plan.steps[-1].width, plan.steps[-1].height) == (plan.width, plan.height)
    if steps:
        assert len(plan.steps) == steps
    prev = (w, h)
    for step in plan.steps:
        assert (step.src_width, step.src_height) == prev
        prev = (step.width, step.height)
        for g, length in ((step.grid_x, step.width), (step.grid_y, step.height)):
            # The canvas is the size padded onto the grid, never more than one grid step.
            assert g.length % m == 0 and 0 <= g.length - length < m
            # Cores tile the canvas exactly: n-1 equal bases plus a positive last core.
            assert g.base % m == 0 and 0 < g.last <= g.base
            assert (g.n - 1) * g.base + g.last == g.length
            # Every sampled crop is on the grid and under the cap.
            for size in g.sampled():
                assert size % m == 0 and size <= cap
            # The grid is the solver's own (minimal tile count).
            sol = grid.solve_axis(g.length, cap, plan.context_anchor, plan.context_overlap, multiple=m)
            assert (sol.n, sol.base) == (g.n, g.base)
        if step.index < len(plan.steps) and factor > 1:
            assert step.width > step.src_width and step.height > step.src_height
            if step.size_choice in (stepwise_plan.SIZE_EVEN, stepwise_plan.SIZE_GRID):
                assert step.width % m == 0 and step.height % m == 0
                assert abs(plan.aspect_deviation(step.width, step.height)) <= tolerance + 1e-9
            if step.size_choice == stepwise_plan.SIZE_EVEN:
                assert step.grid_x.even and step.grid_y.even
            if tolerance == 0.0:
                assert step.size_choice == stepwise_plan.SIZE_EXACT


def test_plan_example_from_the_brief():
    # 1280x960 x3 in 3 steps on MOSS (16 px grid): every intermediate splits into equal tiles.
    plan = stepwise_plan.plan_upscale(1280, 960, 3.0, 3, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.45, 0.22, 0.01)
    assert [(s.width, s.height) for s in plan.steps][-1] == (3840, 2880)
    assert all(s.size_choice == stepwise_plan.SIZE_EVEN for s in plan.steps[:-1])
    assert [round(s.denoise, 3) for s in plan.steps] == [0.45, 0.335, 0.22]
    factors = [s.factor_x for s in plan.steps]
    assert max(factors) / min(factors) < 1.05


def test_auto_step_count_and_denoise():
    assert stepwise_plan.auto_step_count(3.0) == 3
    assert stepwise_plan.auto_step_count(2.0) == 2
    assert stepwise_plan.auto_step_count(1.5) == 1
    assert stepwise_plan.auto_step_count(1.0) == 1
    assert stepwise_plan.auto_step_count(4.0, 2.0) == 2
    assert stepwise_plan.interpolate_denoise(0.5, 0.2, 1) == (0.5,)
    assert stepwise_plan.interpolate_denoise(0.6, 0.2, 3) == pytest.approx((0.6, 0.4, 0.2))


def test_plan_uses_the_upscale_model_on_step_one_only():
    plan = stepwise_plan.plan_upscale(1280, 960, 3.0, 3, EIGHT, 1024, 1024, 32, 32, 0.4, 0.2, 0.01, upscale_model_scale=4.0)
    assert [s.use_model for s in plan.steps] == [True, False, False]
    assert plan.steps[0].upscale.startswith("model x4")
    assert all(s.upscale == "lanczos" for s in plan.steps[1:])


def test_plan_text_lists_every_step():
    plan = stepwise_plan.plan_upscale(1280, 976, 3.0, 3, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.45, 0.22, 0.01)
    text = stepwise_plan.format_plan(plan, 25)
    assert "1280x976 -> target 3840x2928" in text
    assert text.count("\n| ") == 1 + len(plan.steps)
    assert "Latent cell 16 px" in text


def test_plan_reports_rounded_rings():
    plan = stepwise_plan.plan_upscale(1000, 1000, 2.0, 2, LatentGeometry(16, 32), 1024, 1024, 24, 8, 0.4, 0.2, 0.01)
    assert (plan.context_anchor, plan.context_overlap) == (48, 16)
    assert any("rings rounded up" in n for n in plan.notes)


# --- the raster engine at any latent scale ----------------------------------------------------

@pytest.mark.parametrize("scale, channels, multiple", GEOMETRY_CASES)
@pytest.mark.parametrize("size", [(240, 176), (250, 190), (512, 384)])
@pytest.mark.parametrize("rings", [(0, 0), (16, 16), (32, 32), (8, 24)])
def test_refine_runs_on_any_latent_scale(comfy_stubs, scale, channels, multiple, size, rings):
    h, w = size
    g = LatentGeometry(scale, multiple or scale)
    vae = ScaleVAE(scale, channels, require=g.multiple)
    guider = CheckingGuider()
    image = _smooth_image(h, w)
    out = sampling.refine_image(image, guider, object(), SIGMAS, vae, ZeroNoise(), 192, 192, *rings,
                                geometry=resolve_geometry(vae, 0, multiple))
    assert out.shape == image.shape
    assert len(guider.calls) == len(vae.encode_shapes) >= 1
    anchor, overlap = snap_rings(*rings, g)
    for cw, ch in vae.encode_shapes:
        assert cw % g.multiple == 0 and ch % g.multiple == 0
        assert cw <= 192 and ch <= 192
    # With a pass-through sampler the output is the VAE round trip of the input, so every pixel
    # landed where it came from (no shift, no resample).
    padded, _ = sampling.pad_image_to_multiple(image, g.multiple)
    expected = padded[:, ::scale, ::scale, :].repeat_interleave(scale, 1).repeat_interleave(scale, 2)[:, :h, :w]
    if anchor == 0 and overlap == 0:
        assert torch.equal(out, expected)
    else:
        assert torch.allclose(out, expected, atol=1e-6)


def test_eight_x_geometry_on_a_sixteen_x_vae_is_the_upstream_crash(comfy_stubs):
    # Pins the bug the fork fixes: forcing upstream's /8 geometry on a 16x VAE fails fast.
    with pytest.raises(RuntimeError, match="latent shape"):
        sampling.refine_image(_smooth_image(256, 256), CheckingGuider(), object(), SIGMAS, ScaleVAE(16, 128),
                              ZeroNoise(), 128, 128, 16, 16, geometry=EIGHT)


@pytest.mark.parametrize("scale, channels, multiple", GEOMETRY_CASES)
def test_masked_refine_on_any_latent_scale(comfy_stubs, scale, channels, multiple):
    g = LatentGeometry(scale, multiple or scale)
    vae = ScaleVAE(scale, channels, require=g.multiple)
    image = _smooth_image(250, 300)
    mask = torch.zeros(1, 250, 300)
    mask[:, 90:170, 100:230] = 1.0
    out = sampling.refine_image(image, CheckingGuider(), object(), SIGMAS, vae, ZeroNoise(), 256, 256, 32, 32,
                                mask=mask, geometry=g)
    assert out.shape == image.shape
    # Far outside the mask the input survives byte for byte.
    assert torch.equal(out[:, :40], image[:, :40])


@pytest.mark.parametrize("size, rings", [((84, 76), (0, 0)), ((160, 96), (8, 16)), ((200, 120), (16, 32))])
def test_any_vae_node_is_bit_identical_to_upstream_at_eight(comfy_stubs, size, rings):
    # Auto geometry, explicit EIGHT and an explicit 8x scale all give the same bytes on a small
    # cap that forces real multi-tile grids (refine_image sits below the widget minimum of 256).
    h, w = size
    image = torch.rand(1, h, w, 3, generator=torch.Generator().manual_seed(5))
    outs = [
        sampling.refine_image(image, GridGuider(), object(), SIGMAS, GridVAE(), GridNoise(), 128, 128, *rings, geometry=geometry)
        for geometry in (None, EIGHT, resolve_geometry(GridVAE(), latent_scale=8))
    ]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


def test_any_vae_node_refine_method_matches_upstream_node(comfy_stubs):
    image = torch.rand(1, 600, 520, 3, generator=torch.Generator().manual_seed(9))
    up = ContextAnchoredTileRefine().refine(image, GridGuider(), object(), SIGMAS, GridVAE(), GridNoise(), 256, 256, 32, 32)
    # tile_size "allow smaller" is upstream's bare windows; the fork's default "always full" grows them.
    fork = TileRefineAnyVAE().refine(image, GridGuider(), object(), SIGMAS, GridVAE(), GridNoise(), 256, 256, 32, 32,
                                     tile_size="allow smaller")
    assert torch.equal(up[0], fork[0])


def test_any_vae_node_inputs_extend_upstream():
    upstream = ContextAnchoredTileRefine.INPUT_TYPES()
    fork = TileRefineAnyVAE.INPUT_TYPES()
    assert list(fork["required"]) == list(upstream["required"])
    assert {"mask", "latent_scale", "pixel_multiple"} <= set(fork["optional"])


# --- the stepwise node end to end -------------------------------------------------------------

def _stepwise_guider():
    guider = CheckingGuider()
    guider.model_patcher.get_model_object = lambda name: object()
    guider.model_patcher.model = SimpleNamespace(latent_format=object(), model_config=SimpleNamespace())
    return guider


@pytest.mark.parametrize("scale, channels, multiple", GEOMETRY_CASES)
@pytest.mark.parametrize("src, factor, steps", [((160, 120), 3.0, 3), ((160, 122), 3.0, 3), ((200, 200), 2.0, 0),
                                                ((168, 96), 2.0, 2), ((160, 120), 3.0, 1)])
def test_stepwise_node_end_to_end(comfy_stubs, scale, channels, multiple, src, factor, steps):
    w, h = src
    vae = ScaleVAE(scale, channels, require=multiple or scale)
    image = _smooth_image(h, w)
    node = StepwiseTileUpscale()
    final, step_images, text = node.upscale(
        image, _stepwise_guider(), object(), ZeroNoise(), vae, "simple", 4, factor, steps, 256, 256, 32, 32,
        0.5, 0.2, False, pixel_multiple=multiple)
    assert (final.shape[2], final.shape[1]) == (round(w * factor), round(h * factor))
    assert step_images[-1] is final
    if steps:
        assert len(step_images) == steps
    assert "Stepwise Tile Upscale plan" in text
    for cw, ch in vae.encode_shapes:
        assert cw % (multiple or scale) == 0 and ch % (multiple or scale) == 0 and cw <= 256 and ch <= 256


def test_stepwise_plan_only_samples_nothing(comfy_stubs):
    vae = ScaleVAE(16, 128)
    image = _smooth_image(120, 160)
    guider = _stepwise_guider()
    final, steps, text = StepwiseTileUpscale().upscale(
        image, guider, object(), ZeroNoise(), vae, "simple", 4, 3.0, 3, 0, 0, 32, 32, 0.45, 0.22, True)
    assert final is image and steps == [image]
    assert guider.calls == [] and vae.encode_shapes == []
    assert "max tile 1024x1024" in text


def test_stepwise_denoise_drives_the_sigmas(comfy_stubs):
    # build_sigmas schedules int(steps / denoise) and keeps the last steps + 1, like BasicScheduler.
    StepwiseTileUpscale().upscale(_smooth_image(120, 160), _stepwise_guider(), object(), ZeroNoise(), ScaleVAE(16, 128),
                                  "simple", 10, 2.0, 2, 256, 256, 32, 32, 0.5, 0.25, False,
                                  denoise_mode="denoise")
    totals = [call[2] for call in comfy_stubs["calculate_sigmas_calls"]]
    assert totals == [20, 40]


def test_auto_tile_cap_by_family():
    def guider_for(name):
        config = type(name, (), {})()
        return SimpleNamespace(model_patcher=SimpleNamespace(model=SimpleNamespace(model_config=config)))

    assert auto_tile_cap(guider_for("SD15")) == 768
    assert auto_tile_cap(guider_for("SDXL")) == 1024
    assert auto_tile_cap(guider_for("MOSSImage")) == 1024
    assert auto_tile_cap(object()) == 1024


def test_validate_inputs():
    assert StepwiseTileUpscale.VALIDATE_INPUTS(0, 0, 32, 32) is True
    assert StepwiseTileUpscale.VALIDATE_INPUTS(1024, 768, 0, 0) is True
    assert isinstance(StepwiseTileUpscale.VALIDATE_INPUTS(100, 0, 32, 32), str)
    assert isinstance(StepwiseTileUpscale.VALIDATE_INPUTS(0, 0, 600, 32), str)


# --- the repair loop: oversample, oversample_mode, the resample methods -----------------------

def test_plan_constant_oversample_at_upscale_by_one_refines_at_one_size():
    # K: pass 1 enlarges by the oversample, every later pass keeps that size exactly, and the
    # final resample shrinks back onto the source size.
    plan = stepwise_plan.plan_upscale(1152, 864, 1.0, 5, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.7, 0.7, 0.01,
                                      oversample=1.5)
    assert (plan.width, plan.height) == (1152, 864)
    first = (plan.steps[0].width, plan.steps[0].height)
    assert abs(first[0] / 1152 - 1.5) < 0.07
    assert all((s.width, s.height) == first for s in plan.steps)
    assert all(s.size_choice == stepwise_plan.SIZE_SAME for s in plan.steps[1:])
    assert [s.upscale for s in plan.steps[1:]] == ["none (same size)"] * 4
    assert plan.final == stepwise_plan.FinalResample(*first, 1152, 864, "area")
    assert any("same size" in n for n in plan.notes)


def test_plan_ramp_up_grows_to_the_oversample():
    # G: the passes grow evenly to x1.5 and the final resample brings the image back to its
    # exact source size.
    plan = stepwise_plan.plan_upscale(1152, 864, 1.0, 5, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.7, 0.7, 0.01,
                                      oversample=1.5, oversample_mode="ramp up", downscale_method="bicubic (mitchell)")
    sizes = [(s.width, s.height) for s in plan.steps]
    assert all(b[0] > a[0] and b[1] > a[1] for a, b in itertools.pairwise([(1152, 864), *sizes]))
    assert abs(sizes[-1][0] / 1152 - 1.5) < 0.07
    assert (plan.final.width, plan.final.height, plan.final.method) == (1152, 864, "bicubic (mitchell)")
    assert (plan.width, plan.height) == (1152, 864)


@pytest.mark.parametrize("mode", ["constant", "ramp up"])
def test_plan_oversampled_upscale(mode):
    # x2 out, oversample 1.3: the largest pass is the last at about x2.6, then one shrink onto
    # exactly round(source x 2).
    plan = stepwise_plan.plan_upscale(1000, 750, 2.0, 5, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.7, 0.5, 0.0,
                                      oversample=1.3, oversample_mode=mode, upscale_method="bicubic (catmull-rom)")
    assert (plan.width, plan.height) == (2000, 1500)
    if mode == "constant":
        ideal = [1000 * 2.0 ** (i / 5) * 1.3 for i in range(1, 6)]
    else:
        ideal = [1000 * 2.6 ** (i / 5) for i in range(1, 6)]
    assert [s.width for s in plan.steps] == [round(x) for x in ideal]
    assert max(s.width for s in plan.steps) == plan.steps[-1].width == 2600
    assert all(s.method == "bicubic (catmull-rom)" for s in plan.steps)
    assert plan.final == stepwise_plan.FinalResample(2600, 1950, 2000, 1500, "area")
    text = stepwise_plan.format_plan(plan, 12)
    assert f"oversample x1.3 {mode}" in text and "| final |" in text and "2600x1950" in text


def test_plan_constant_oversample_is_the_enlarge_refine_shrink_loop():
    # Per pass enlarge by U = 2 ** (1/5) * 1.3, shrink by D = 1.3, the two merged between passes.
    plan = stepwise_plan.plan_upscale(1000, 750, 2.0, 5, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.7, 0.5, 0.0,
                                      oversample=1.3)
    text = stepwise_plan.format_plan(plan)
    assert f"enlarge x{2 ** 0.2 * 1.3:.4f}, refine, shrink /1.3" in text


def test_plan_plain_upscale_has_no_final_resample():
    plan = stepwise_plan.plan_upscale(1280, 960, 3.0, 3, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.45, 0.22, 0.01)
    assert plan.final is None and plan.oversample == 1.0
    assert "| final |" not in stepwise_plan.format_plan(plan)
    ramp = stepwise_plan.plan_upscale(1280, 960, 3.0, 3, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.45, 0.22,
                                      0.01, oversample_mode="ramp up")
    assert ramp.steps == plan.steps


def test_plan_auto_steps_count_the_oversample():
    plan = stepwise_plan.plan_upscale(1000, 750, 2.0, 0, EIGHT, 1024, 1024, 32, 32, 0.5, 0.5, 0.01, oversample=1.5)
    assert len(plan.steps) == stepwise_plan.auto_step_count(3.0) == 3


@pytest.mark.parametrize("kwargs, match", [
    ({"oversample": 0.9}, "oversample"), ({"oversample_mode": "jump"}, "oversample_mode"),
    ({"upscale_method": "area"}, "upscale_method"), ({"downscale_method": "bilinear"}, "downscale_method"),
])
def test_plan_rejects_bad_repair_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        stepwise_plan.plan_upscale(800, 600, 2.0, 2, EIGHT, 1024, 1024, 32, 32, 0.5, 0.5, 0.01, **kwargs)


def test_widget_method_lists_match_the_resampler(comfy_stubs):
    from context_anchored_tile_refine import resample

    assert set(stepwise_plan.UPSCALE_METHODS) | set(stepwise_plan.DOWNSCALE_METHODS) == set(resample.METHODS)
    types = StepwiseTileUpscale.INPUT_TYPES()["required"]
    assert types["upscale_method"][0] == list(stepwise_plan.UPSCALE_METHODS)
    assert types["downscale_method"][0] == list(stepwise_plan.DOWNSCALE_METHODS)
    assert types["oversample_mode"][0] == list(stepwise_plan.OVERSAMPLE_MODES)
    assert types["upscale_by"][1]["min"] == 1.0 and types["oversample"][1]["min"] == 1.0


def test_resample_area_is_the_block_mean():
    from context_anchored_tile_refine import resample

    image = torch.rand(1, 96, 132, 3, generator=torch.Generator().manual_seed(1))
    out = resample.resize(image, 44, 32, resample.AREA)
    assert torch.allclose(out, image.reshape(1, 32, 3, 44, 3, 3).mean((2, 4)), atol=1e-6)


@pytest.mark.parametrize("size", [(87, 65), (40, 30), (13, 9), (200, 150)])
def test_resample_matches_pil_for_catmull_rom_and_bilinear(size):
    # PIL's float-mode BICUBIC (Keys a = -0.5, widened when shrinking) and BILINEAR are the same
    # filters, so both directions must agree to float precision.
    import numpy as np
    from PIL import Image

    from context_anchored_tile_refine import resample

    image = torch.rand(1, 97, 131, 3, generator=torch.Generator().manual_seed(2))
    w, h = size
    for method, pil_filter in ((resample.CATMULL_ROM, Image.Resampling.BICUBIC),
                               (resample.BILINEAR, Image.Resampling.BILINEAR)):
        ref = np.stack([np.asarray(Image.fromarray(image[0, ..., c].numpy(), "F").resize((w, h), pil_filter))
                        for c in range(3)], -1)
        out = resample.resize(image, w, h, method)[0].numpy()
        assert np.abs(out - np.clip(ref, 0, 1)).max() < 1e-5, method


@pytest.mark.parametrize("method", ["area", "bilinear", "bicubic (B-spline)", "bicubic (mitchell)", "bicubic (catmull-rom)"])
@pytest.mark.parametrize("size", [(23, 17), (140, 100), (70, 51)])
def test_resample_keeps_a_flat_image_flat(method, size):
    from context_anchored_tile_refine import resample

    out = resample.resize(torch.full((1, 50, 70, 3), 0.37), *size, method)
    assert out.shape == (1, size[1], size[0], 3)
    assert torch.allclose(out, torch.full_like(out, 0.37), atol=1e-6)


def test_resample_same_size_is_the_input():
    from context_anchored_tile_refine import resample

    image = torch.rand(1, 20, 30, 3)
    assert resample.resize(image, 30, 20, resample.MITCHELL) is image


def test_step_noise_offsets_the_seed_per_pass():
    from context_anchored_tile_refine.stepwise_nodes import step_noise

    noise = ZeroNoise()
    assert step_noise(noise, 1) is noise
    assert [step_noise(noise, i).seed for i in (2, 3)] == [4, 5]
    assert noise.seed == 3
    top = SimpleNamespace(seed=(1 << 64) - 1)
    assert step_noise(top, 2).seed == 0
    seedless = object()
    assert step_noise(seedless, 4) is seedless


class SeedNoise(ZeroNoise):
    def __init__(self, log):
        self.log = log

    def generate_noise(self, input_latent):
        self.log.append(self.seed)
        return super().generate_noise(input_latent)


@pytest.mark.parametrize("upscale_by, oversample, mode", [(1.0, 1.5, "constant"), (1.0, 1.5, "ramp up"),
                                                       (2.0, 1.3, "constant")])
def test_stepwise_node_repair_loop_end_to_end(comfy_stubs, upscale_by, oversample, mode):
    image = _smooth_image(120, 160)
    log = []
    final, step_images, text = StepwiseTileUpscale().upscale(
        image, _stepwise_guider(), object(), SeedNoise(log), ScaleVAE(16, 128), "simple", 4, upscale_by, 3, 256, 256,
        32, 32, 0.7, 0.7, False, oversample=oversample, oversample_mode=mode,
        downscale_method="bicubic (catmull-rom)")
    assert (final.shape[2], final.shape[1]) == (round(160 * upscale_by), round(120 * upscale_by))
    assert len(step_images) == 4 and step_images[-1] is final
    assert "| final |" in text
    # Each pass draws its noise on its own seed.
    assert sorted(set(log)) == [3, 4, 5]
