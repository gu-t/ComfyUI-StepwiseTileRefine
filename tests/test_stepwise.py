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
    # A small factor in many steps: an intermediate can reach the last pass's canvas size.
    (1000, 667, 1.02, 3), (1001, 667, 1.0, 2),
]
GEOMETRIES = [LatentGeometry(8, 8), LatentGeometry(16, 16), LatentGeometry(16, 32)]


@pytest.mark.parametrize("w, h, factor, steps", PLAN_CASES)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=["g8", "g16", "g32"])
@pytest.mark.parametrize("output_size", stepwise_plan.OUTPUT_SIZES)
def test_plan_invariants(w, h, factor, steps, geometry, output_size):
    cap = 1024
    plan = stepwise_plan.plan_upscale(w, h, factor, steps, geometry, cap, cap, 32, 32, 0.45, 0.22,
                                      output_size=output_size)
    m = geometry.multiple
    exact = (round(w * factor), round(h * factor))
    if output_size == stepwise_plan.OUTPUT_STRETCH:
        # The nearest grid size, and a warning whenever it is not the exact target.
        assert plan.width % m == 0 and plan.height % m == 0
        assert abs(plan.width - exact[0]) <= 2 * m and abs(plan.height - exact[1]) <= 2 * m
        assert bool(plan.warnings) == ((plan.width, plan.height) != exact)
    else:
        assert (plan.width, plan.height) == exact
    if steps:
        assert len(plan.steps) == steps
    last = plan.steps[-1]
    if output_size == stepwise_plan.OUTPUT_PAD:
        assert (last.width, last.height) == exact and plan.final is None
    else:
        # Every pass samples a size on the grid, so nothing is padded.
        assert all(s.width % m == 0 and s.height % m == 0 for s in plan.steps)
        assert not any("padded" in warning for s in plan.steps for warning in s.warnings)
    if output_size == stepwise_plan.OUTPUT_CROP and (last.width, last.height) != exact:
        # The last pass is the grid size just above the target, enlarged without stretch and cut
        # down onto the target.
        assert 0 <= last.width - exact[0] < m and 0 <= last.height - exact[1] < m
        assert plan.final == stepwise_plan.FinalCrop(last.width, last.height, *exact, (last.width - exact[0]) // 2,
                                                     (last.height - exact[1]) // 2)
        # The first pass that lands on the canvas (every pass at upscale_by 1) enlarges onto the
        # source's aspect; the passes after it keep that size.
        landing = next(s for s in plan.steps if s.size_choice == stepwise_plan.SIZE_FINAL)
        ew, eh = landing.enlarged
        assert ew >= last.width and eh >= last.height
        assert abs(plan.aspect_deviation(ew, eh)) < 1.0 / min(ew, eh)
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
            assert step.size_choice == stepwise_plan.SIZE_GRID
            # Within one grid cell of the shorter side in aspect.
            assert abs(plan.aspect_deviation(step.width, step.height)) <= m / min(step.width, step.height)


def test_plan_crop_without_a_fractional_enlargement():
    # 1664x1248 x1.3 is 2163x1622; 4:3 fits 2176x1632 exactly, so the step resamples straight onto it
    # and only the final crop remains.
    plan = stepwise_plan.plan_upscale(1664, 1248, 1.3, 1, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.35, 0.35)
    step = plan.steps[0]
    assert (step.width, step.height, step.enlarge_width) == (2176, 1632, 0.0)
    assert plan.final == stepwise_plan.FinalCrop(2176, 1632, 2163, 1622, 6, 5)
    assert "centre of" not in stepwise_plan.format_plan(plan)


def test_plan_crop_corrects_an_intermediate_at_the_canvas_size():
    # 1000x667 x1.02 in 3 steps: step 2 is forced up to 1024x688, the canvas of the 1020x680
    # target, with the grid size's -0.73 % aspect. The last pass still enlarges onto the source's
    # aspect (an enlargement, so the upscale filter) instead of cropping the stretched intermediate.
    plan = stepwise_plan.plan_upscale(1000, 667, 1.02, 3, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.3, 0.2)
    mid, last = plan.steps[1], plan.steps[2]
    assert (mid.width, mid.height) == (last.width, last.height) == (1024, 688)
    assert last.enlarged == pytest.approx((1000 * 688 / 667, 688.0))
    assert last.method == stepwise_plan.DEFAULT_UPSCALE_METHOD
    assert plan.final == stepwise_plan.FinalCrop(1024, 688, 1020, 680, 2, 4)
    assert "  enlarge  1024x688 -> 1024x688" in stepwise_plan.format_plan(plan)


def test_plan_crop_after_the_canvas_is_reached_does_not_enlarge_again():
    # upscale_by 1, 1001x667: step 1 enlarges onto 1008x672 and every later pass keeps it as is.
    plan = stepwise_plan.plan_upscale(1001, 667, 1.0, 3, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.3, 0.2)
    assert plan.steps[0].enlarge_width > 0
    assert all(s.enlarge_width == 0 and s.upscale == "none (same size)" for s in plan.steps[1:])


@pytest.mark.parametrize("size, warned", [((1001, 667), True), ((1008, 672), False)])
def test_plan_warns_that_a_crop_at_upscale_by_one_resamples(size, warned):
    plan = stepwise_plan.plan_upscale(*size, 1.0, 1, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.3, 0.3)
    text = stepwise_plan.format_plan(plan)
    assert ("Width 1001 and height 667 are not multiples of 16 px, so the image is enlarged x1.007 and cropped "
            "back, which softens it slightly. output_size \"exact, pad edges during denoise\" avoids the resample "
            "and keeps every edge pixel." in text) == warned
    assert "Total 1 tile." in text
    assert "Total 1 tile x 1 sampler step." in stepwise_plan.format_plan(plan, sampler_steps=1)
    assert ("Output size:" in text) == warned
    pad = stepwise_plan.plan_upscale(*size, 1.0, 1, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.3, 0.3,
                                     output_size=stepwise_plan.OUTPUT_PAD)
    assert not pad.warnings


def test_plan_crop_warning_names_only_the_axis_off_the_grid():
    # 1008x667: the width is on the 16 px grid, so only the height is named.
    plan = stepwise_plan.plan_upscale(1008, 667, 1.0, 1, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.3, 0.3)
    assert plan.warnings[0].startswith("Height 667 is not a multiple of 16 px, so the image is enlarged")


def test_plan_text_shows_the_shifting_grids_own_ring():
    # On an 8 px latent with a 32 px grid, snap_rings turns tile_padding 32 + tile_overlap 8 into a
    # 56 px anchor. The shifting grid ignores the overlap and runs a 32 px ring, and says so.
    from context_anchored_tile_refine import all_at_once

    plan = stepwise_plan.plan_upscale(1000, 667, 2.0, 2, LatentGeometry(8, 32), 1024, 1024, 32, 8, 0.4, 0.2,
                                      fixed_grid=False)
    assert plan.context_anchor == 56
    shifting = all_at_once.shifting_tiles(plan, all_at_once.SIZE_FULL, 32)
    text = stepwise_plan.format_plan(plan, shifting=shifting)
    assert "Ring: tile_padding 32 px, core 960x960 px (tile_overlap is not used)." in text


def test_plan_example_from_the_brief():
    # 1280x960 x3 in 3 steps on MOSS (16 px grid): every pass on the grid, the factor even.
    plan = stepwise_plan.plan_upscale(1280, 960, 3.0, 3, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.45, 0.22)
    assert [(s.width, s.height) for s in plan.steps][-1] == (3840, 2880)
    assert all(s.width % 16 == 0 and s.height % 16 == 0 for s in plan.steps)
    assert [round(s.denoise, 3) for s in plan.steps] == [0.45, 0.335, 0.22]
    factors = [s.factor_x for s in plan.steps]
    assert max(factors) / min(factors) < 1.02


@pytest.mark.parametrize("output_size, ops, warning", [
    # 1920x1080 x1.5 is 2880x1620, and 1620 is off the 16 px grid.
    (stepwise_plan.OUTPUT_CROP, [
        "  enlarge  1920x1080 -> 2880x1632   x1.511, lanczos, centre of 2901.3x1632",
        "  denoise  2880x1632                4x2 tiles, core 720x816",
        "  crop     2880x1632 -> 2880x1620",
    ], None),
    (stepwise_plan.OUTPUT_PAD, [
        "  enlarge  1920x1080 -> 2880x1620   x1.500, lanczos",
        "  pad      2880x1620 -> 2880x1632   mirrored",
        "  denoise  2880x1632                4x2 tiles, core 720x816",
        "  crop     2880x1632 -> 2880x1620",
    ], "Step 1: mirrored edge during denoise, 12 px at the bottom"),
    (stepwise_plan.OUTPUT_STRETCH, [
        "  enlarge  1920x1080 -> 2880x1616   x1.500 / x1.496, lanczos, aspect +0.25%",
        "  denoise  2880x1616                4x2 tiles, core 720x816/800",
    ], "Output 2880x1616 instead of 2880x1620: height -4 px (-0.25%)"),
])
def test_plan_output_size(output_size, ops, warning):
    # The plan lists every operation of a step in the order it runs.
    plan = stepwise_plan.plan_upscale(1920, 1080, 1.5, 1, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.4, 0.4,
                                      output_size=output_size)
    text = stepwise_plan.format_plan(plan, value_label="Noise level")
    assert f"Source 1920x1080, x1.5 -> 2880x1620, 1 step. Output size: {output_size}." in text
    lines = text.splitlines()
    first = lines.index("Step 1, noise level 0.400") + 1
    assert lines[first:first + len(ops)] == ops
    assert f"Output {plan.width}x{plan.height}." in text
    if output_size == stepwise_plan.OUTPUT_CROP:
        # One exact factor on both axes: the centre of 2901.33x1632, then cut onto 2880x1620.
        assert plan.steps[0].enlarged == pytest.approx((1920 * 1632 / 1080, 1632.0))
        assert plan.final == stepwise_plan.FinalCrop(2880, 1632, 2880, 1620, 0, 6)
    if warning is None:
        assert "WARNINGS" not in text
    else:
        assert warning in text


def test_auto_step_count_and_denoise():
    assert stepwise_plan.auto_step_count(3.0) == 3
    assert stepwise_plan.auto_step_count(2.0) == 2
    assert stepwise_plan.auto_step_count(1.5) == 1
    assert stepwise_plan.auto_step_count(1.0) == 1
    assert stepwise_plan.auto_step_count(4.0, 2.0) == 2
    assert stepwise_plan.interpolate_denoise(0.5, 0.2, 1) == (0.5,)
    assert stepwise_plan.interpolate_denoise(0.6, 0.2, 3) == pytest.approx((0.6, 0.4, 0.2))


def test_plan_uses_the_upscale_model_on_step_one_only():
    plan = stepwise_plan.plan_upscale(1280, 960, 3.0, 3, EIGHT, 1024, 1024, 32, 32, 0.4, 0.2, upscale_model_scale=4.0)
    assert [s.use_model for s in plan.steps] == [True, False, False]
    assert plan.steps[0].upscale.startswith("model x4")
    assert all(s.upscale == "lanczos" for s in plan.steps[1:])


def test_plan_text_lists_every_step():
    plan = stepwise_plan.plan_upscale(1280, 976, 3.0, 3, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.45, 0.22)
    text = stepwise_plan.format_plan(plan, 25)
    assert "Source 1280x976, x3 -> 3840x2928" in text
    assert sum(line.startswith("Step ") for line in text.splitlines()) == len(plan.steps)
    assert "|" not in text and "#" not in text
    assert "Pixel grid 16 px." in text


def test_plan_names_the_tile_the_model_sees_on_the_grid():
    # A cap off the 16 px grid: every window lands on the grid below it.
    plan = stepwise_plan.plan_upscale(1000, 750, 2.0, 2, LatentGeometry(16, 16), 1000, 1000, 32, 32, 0.4, 0.2)
    text = stepwise_plan.format_plan(plan, tile_mode="tile by tile", tile_size="always full")
    assert "Tile mode: tile by tile, always full. Max tile 992x992 px, rings included." in text


def test_plan_reports_rounded_rings():
    plan = stepwise_plan.plan_upscale(1000, 1000, 2.0, 2, LatentGeometry(16, 32), 1024, 1024, 24, 8, 0.4, 0.2)
    assert (plan.context_anchor, plan.context_overlap) == (48, 16)
    text = stepwise_plan.format_plan(plan)
    assert "Rings: tile_overlap 16 px (rounded from 8), tile_padding 48 px (rounded from 24)." in text
    assert "NOTES" not in text


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


def test_any_vae_node_inputs_rename_upstreams_rings():
    # The fork's widget names: tile_padding is upstream's context_anchor, tile_overlap its
    # context_overlap. The sockets and the tile cap keep upstream's specs.
    upstream = ContextAnchoredTileRefine.INPUT_TYPES()
    fork = TileRefineAnyVAE.INPUT_TYPES()
    renamed = {"context_anchor": "tile_padding", "context_overlap": "tile_overlap"}
    assert {renamed.get(n, n) for n in upstream["required"]} <= set(fork["required"])
    for name in ("image", "guider", "sampler", "sigmas", "vae", "noise", "max_tile_width", "max_tile_height"):
        assert fork["required"][name] == upstream["required"][name]
    assert {"mask", "latent_scale", "pixel_multiple"} <= set(fork["optional"])
    # The fork's own rule (stepwise_nodes.check_tiles): no multiple-of-8 demand, and no auto cap.
    assert TileRefineAnyVAE.VALIDATE_INPUTS(1000, 1000, 12, 20) is True
    assert TileRefineAnyVAE.VALIDATE_INPUTS(tile_padding=600) == "tile_padding must be between 0 and 512, got 600"
    assert TileRefineAnyVAE.VALIDATE_INPUTS(max_tile_width=0) == "max_tile_width must be between 256 and 16384, got 0"


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
    assert "STEPWISE TILE UPSCALE PLAN" in text
    for cw, ch in vae.encode_shapes:
        assert cw % (multiple or scale) == 0 and ch % (multiple or scale) == 0 and cw <= 256 and ch <= 256


@pytest.mark.parametrize("output_size, size", [(stepwise_plan.OUTPUT_CROP, (480, 366)),
                                               (stepwise_plan.OUTPUT_PAD, (480, 366)),
                                               (stepwise_plan.OUTPUT_STRETCH, (480, 368))])
def test_stepwise_node_output_size(comfy_stubs, output_size, size):
    # 160x122 x3 is 480x366, off the 16 px grid. Only pad edges hands the VAE a padded canvas
    # (the tiles of a padded canvas are on the grid too, so the encodes cannot tell): the other
    # two sample grid sizes on every step, which their step images show.
    vae = ScaleVAE(16, 128)
    final, step_images, text = StepwiseTileUpscale().upscale(
        _smooth_image(122, 160), _stepwise_guider(), object(), ZeroNoise(), vae, "simple", 4, 3.0, 2, 256, 256, 32, 32,
        0.5, 0.2, False, output_size=output_size)
    assert (final.shape[2], final.shape[1]) == size
    assert len(step_images) == 2 and step_images[-1] is final
    assert all(img.shape[1] % 16 == 0 and img.shape[2] % 16 == 0 for img in step_images[:-1])
    assert ("mirrored" in text) == (output_size == stepwise_plan.OUTPUT_PAD)


def test_stepwise_plan_only_samples_nothing(comfy_stubs):
    vae = ScaleVAE(16, 128)
    image = _smooth_image(120, 160)
    guider = _stepwise_guider()
    final, steps, text = StepwiseTileUpscale().upscale(
        image, guider, object(), ZeroNoise(), vae, "simple", 4, 3.0, 3, 0, 0, 32, 32, 0.45, 0.22, True)
    assert final is image and steps == [image]
    assert guider.calls == [] and vae.encode_shapes == []
    assert "Max tile 1024x1024 px" in text


def test_stepwise_denoise_drives_the_sigmas(comfy_stubs):
    # build_sigmas schedules int(steps / denoise) and keeps the last steps + 1, like BasicScheduler.
    StepwiseTileUpscale().upscale(_smooth_image(120, 160), _stepwise_guider(), object(), ZeroNoise(), ScaleVAE(16, 128),
                                  "simple", 10, 2.0, 2, 256, 256, 32, 32, 0.5, 0.25, False,
                                  denoise_mode="denoise", steps_mode="steps per pass")
    totals = [call[2] for call in comfy_stubs["calculate_sigmas_calls"]]
    assert totals == [20, 40]


# The stub schedule of length n is 0..n, so a schedule's tail names the length it was cut from.
# Denoise 0.5 / 0.25 on 10 steps: steps per pass cuts 10 steps from 20 and from 40; same step size
# takes the last ceil(10 x 0.5) = 5 and ceil(10 x 0.25) = 3 of the 10-step generation; proportional
# to noise cuts 5 steps from int(5 / 0.5) = 10 and 3 from int(3 / 0.25) = 12.
STEPS_MODE_CASES = {
    "steps per pass": ([range(10, 21), range(30, 41)], "Sampler steps: 10 per step (simple)."),
    "same step size": ([range(5, 11), range(7, 11)], "Sampler steps: same step size, each step's share of one "
                                                     "10-step schedule over the whole noise range (simple)."),
    "proportional to noise": ([range(5, 11), range(9, 13)], "Sampler steps: proportional to noise, 10 x each step's "
                                                            "denoise, rounded up (simple)."),
}


@pytest.mark.parametrize("steps_mode", list(STEPS_MODE_CASES))
def test_stepwise_steps_mode_drives_the_sigmas_and_the_plan(comfy_stubs, monkeypatch, steps_mode):
    from context_anchored_tile_refine import sampling

    seen = []
    real = sampling.refine_image

    def spy(image, guider, sampler, sigmas, *args, **kwargs):
        seen.append(sigmas.clone())
        return real(image, guider, sampler, sigmas, *args, **kwargs)

    monkeypatch.setattr(sampling, "refine_image", spy)
    _, _, text = StepwiseTileUpscale().upscale(
        _smooth_image(120, 160), _stepwise_guider(), object(), ZeroNoise(), ScaleVAE(16, 128), "simple", 10, 2.0, 2,
        256, 256, 32, 32, 0.5, 0.25, False, denoise_mode="denoise", steps_mode=steps_mode)
    ranges, header = STEPS_MODE_CASES[steps_mode]
    assert [s.tolist() for s in seen] == [list(r) for r in ranges]
    counts = tuple(len(r) - 1 for r in ranges)
    assert text.splitlines()[2] == header
    assert f", {counts[0]} sampler steps" in text.split("Step 1,")[1].splitlines()[0]
    assert f", {counts[1]} sampler steps" in text.split("Step 2,")[1].splitlines()[0]
    import re

    tiles = [int(m[1]) * int(m[2]) for m in re.finditer(r"(\d+)x(\d+) tiles, core", text)]
    assert len(tiles) == 2
    total = sum(n * c for n, c in zip(tiles, counts, strict=True))
    assert f", {total} tile steps (tiles x sampler steps)." in text


@pytest.mark.parametrize("steps_mode, sign", [("same step size", "~"), ("proportional to noise", "="),
                                              ("steps per pass", "=")])
def test_step_head_marks_a_fractional_same_step_size_denoise_as_about(steps_mode, sign):
    # A same step size pass in noise level mode covers a fractional share of the full schedule,
    # which no denoise setting reproduces; the other modes' denoise is a BasicScheduler value.
    from context_anchored_tile_refine.noise_level import Schedule
    from context_anchored_tile_refine.stepwise_plan import _step_head

    step = SimpleNamespace(index=1, denoise=0.31)
    schedule = Schedule("noise level", 0.31, torch.tensor([0.31, 0.2, 0.0]), 25, 0.31, 0.31, 0.0912)
    assert _step_head(step, "Noise level", schedule, steps_mode) == \
        f"Step 1, noise level 0.310 ({sign} denoise 0.091), 2 sampler steps"
    denoise = Schedule("denoise", 0.1, torch.tensor([0.4, 0.2, 0.0]), 25, 0.4, 0.4, 0.08)
    assert _step_head(step, "Denoise", denoise, steps_mode).endswith("(= noise level 0.400), 2 sampler steps")


def test_same_step_size_plan_names_the_schedule_length_returned():
    # beta drops repeated sigmas at many steps; the header names the schedule the steps were cut from.
    from context_anchored_tile_refine.noise_level import Schedule

    plan = stepwise_plan.plan_upscale(160, 120, 2.0, 2, LatentGeometry(16, 16), 256, 256, 32, 32, 0.5, 0.25)
    def schedule(total):
        return Schedule("noise level", 0.5, torch.tensor([0.5, 0.2, 0.0]), total, 0.5, 0.5, 0.01)

    text = stepwise_plan.format_plan(plan, 200, "Noise level", schedules=[schedule(199)] * 2, scheduler="beta",
                                     steps_mode="same step size")
    assert ("Sampler steps: same step size, each step's share of one 199-step schedule over the whole noise range "
            "(beta), 200 asked, repeats dropped.") in text
    text = stepwise_plan.format_plan(plan, 200, "Noise level", schedules=[schedule(200)] * 2, scheduler="beta",
                                     steps_mode="same step size")
    assert "one 200-step schedule over the whole noise range (beta)." in text


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
    assert StepwiseTileUpscale.VALIDATE_INPUTS(1000, 0, 12, 20) is True
    assert StepwiseTileUpscale.VALIDATE_INPUTS(100, 0, 32, 32) == "max_tile_width must be 0 (auto) or between 256 and 16384, got 100"
    assert isinstance(StepwiseTileUpscale.VALIDATE_INPUTS(0, 0, 600, 32), str)


# --- the shifting tile mode ----------------------------------------------------------------------

def test_plan_sizes_do_not_depend_on_the_tile_mode():
    args = (1280, 976, 3.0, 3, LatentGeometry(16, 16), 1024, 1024, 32, 0, 0.45, 0.22)
    fixed = stepwise_plan.plan_upscale(*args)
    shifting = stepwise_plan.plan_upscale(*args, fixed_grid=False)
    assert [(s.width, s.height) for s in shifting.steps] == [(s.width, s.height) for s in fixed.steps]
    assert fixed.fixed_grid and not shifting.fixed_grid
    assert all(s.width % 16 == 0 and s.height % 16 == 0 and not s.warnings for s in shifting.steps)


# --- the repair loop: oversample, oversample_mode, the resample methods -----------------------

def test_plan_constant_oversample_at_upscale_by_one_refines_at_one_size():
    # K: pass 1 enlarges by the oversample, every later pass keeps that size exactly, and the
    # final resample shrinks back onto the source size.
    plan = stepwise_plan.plan_upscale(1152, 864, 1.0, 5, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.7, 0.7,
                                      oversample=1.5)
    assert (plan.width, plan.height) == (1152, 864)
    first = (plan.steps[0].width, plan.steps[0].height)
    assert abs(first[0] / 1152 - 1.5) < 0.07
    assert all((s.width, s.height) == first for s in plan.steps)
    assert all(s.size_choice == stepwise_plan.SIZE_SAME for s in plan.steps[1:])
    assert [s.upscale for s in plan.steps[1:]] == ["none (same size)"] * 4
    assert plan.final == stepwise_plan.FinalResample(*first, 1152, 864, "area")


def test_plan_ramp_up_grows_to_the_oversample():
    # G: the passes grow evenly to x1.5 and the final resample brings the image back to its
    # exact source size.
    plan = stepwise_plan.plan_upscale(1152, 864, 1.0, 5, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.7, 0.7,
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
    plan = stepwise_plan.plan_upscale(1000, 750, 2.0, 5, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.7, 0.5,
                                      oversample=1.3, oversample_mode=mode, upscale_method="bicubic (catmull-rom)")
    assert (plan.width, plan.height) == (2000, 1500)
    if mode == "constant":
        ideal = [1000 * 2.0 ** (i / 5) * 1.3 for i in range(1, 6)]
    else:
        ideal = [1000 * 2.6 ** (i / 5) for i in range(1, 6)]
    assert all(s.width % 16 == 0 and abs(s.width - x) <= 16 for s, x in zip(plan.steps, ideal, strict=True))
    last = plan.steps[-1]
    assert max(s.width for s in plan.steps) == last.width
    assert all(s.method == "bicubic (catmull-rom)" for s in plan.steps)
    assert plan.final == stepwise_plan.FinalResample(last.width, last.height, 2000, 1500, "area")
    text = stepwise_plan.format_plan(plan, 12)
    assert f"oversample x1.3 {mode}" in text and "\nFinal\n" in text and f"{last.width}x{last.height} -> 2000x1500" in text


def test_plan_constant_oversample_is_the_enlarge_refine_shrink_loop():
    # Per pass enlarge by U = 2 ** (1/5) * 1.3, shrink by D = 1.3, the two merged between passes: the
    # first pass jumps by U, every later one grows by U / D = 2 ** (1/5), and one shrink /1.3 ends it.
    plan = stepwise_plan.plan_upscale(1000, 750, 2.0, 5, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.7, 0.5,
                                      oversample=1.3)
    widths = [1000] + [s.width for s in plan.steps]
    assert widths[1] / widths[0] == pytest.approx(2 ** 0.2 * 1.3, rel=0.02)
    assert all(b / a == pytest.approx(2 ** 0.2, rel=0.02) for a, b in itertools.pairwise(widths[1:]))
    text = stepwise_plan.format_plan(plan)
    assert f"shrink   {widths[-1]}x{plan.steps[-1].height} -> 2000x1500" in text


def test_plan_plain_upscale_has_no_final_resample():
    plan = stepwise_plan.plan_upscale(1280, 960, 3.0, 3, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.45, 0.22)
    assert plan.final is None and plan.oversample == 1.0
    assert "\nFinal\n" not in stepwise_plan.format_plan(plan)
    ramp = stepwise_plan.plan_upscale(1280, 960, 3.0, 3, LatentGeometry(16, 16), 1024, 1024, 32, 32, 0.45, 0.22,
                                      oversample_mode="ramp up")
    assert ramp.steps == plan.steps


def test_plan_auto_steps_count_the_oversample():
    plan = stepwise_plan.plan_upscale(1000, 750, 2.0, 0, EIGHT, 1024, 1024, 32, 32, 0.5, 0.5, oversample=1.5)
    assert len(plan.steps) == stepwise_plan.auto_step_count(3.0) == 3


@pytest.mark.parametrize("kwargs, match", [
    ({"oversample": 0.9}, "oversample"), ({"oversample_mode": "jump"}, "oversample_mode"),
    ({"upscale_method": "area"}, "upscale_method"), ({"downscale_method": "bilinear"}, "downscale_method"),
])
def test_plan_rejects_bad_repair_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        stepwise_plan.plan_upscale(800, 600, 2.0, 2, EIGHT, 1024, 1024, 32, 32, 0.5, 0.5, **kwargs)


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
def test_resample_matches_pil_float_filters(size):
    # PIL's float-mode BICUBIC (Keys a = -0.5, widened when shrinking), BILINEAR and LANCZOS are
    # the same filters, so both directions must agree to float precision.
    import numpy as np
    from PIL import Image

    from context_anchored_tile_refine import resample

    image = torch.rand(1, 97, 131, 3, generator=torch.Generator().manual_seed(2))
    w, h = size
    for method, pil_filter in ((resample.CATMULL_ROM, Image.Resampling.BICUBIC),
                               (resample.BILINEAR, Image.Resampling.BILINEAR),
                               (resample.LANCZOS, Image.Resampling.LANCZOS)):
        ref = np.stack([np.asarray(Image.fromarray(image[0, ..., c].numpy(), "F").resize((w, h), pil_filter))
                        for c in range(3)], -1)
        out = resample.resize(image, w, h, method)[0].numpy()
        assert np.abs(out - np.clip(ref, 0, 1)).max() < 1e-5, method


@pytest.mark.parametrize("size", [(87, 65), (13, 9), (200, 150), (393, 291)])
def test_resample_lanczos_matches_cores_8_bit_lanczos(comfy_stubs, size):
    # The float lanczos is core's PIL lanczos minus its 8-bit round trip: on an 8-bit input the
    # two differ by the output rounding plus PIL's fixed-point weights, about one level. PIL
    # also clips its intermediate (one axis done) to 0..255, so the levels stay mid-range where
    # that clip cannot bind.
    from context_anchored_tile_refine import resample, upscale

    levels = torch.randint(64, 192, (1, 97, 131, 3), generator=torch.Generator().manual_seed(3))
    image = levels.float() / 255.0
    w, h = size
    core = upscale._lanczos_resize(image, w, h)
    assert (resample.resize(image, w, h, resample.LANCZOS) - core).abs().max() <= 1.5 / 255.0


@pytest.mark.parametrize("method", ["area", "bilinear", "bicubic (B-spline)", "bicubic (mitchell)", "bicubic (catmull-rom)",
                                    "lanczos"])
@pytest.mark.parametrize("size", [(23, 17), (140, 100), (70, 51), (1, 1), (7, 3)])
def test_resample_keeps_a_flat_image_flat(method, size):
    from context_anchored_tile_refine import resample

    out = resample.resize(torch.full((1, 50, 70, 3), 0.37), *size, method)
    assert out.shape == (1, size[1], size[0], 3)
    assert torch.allclose(out, torch.full_like(out, 0.37), atol=1e-6)


def test_resample_same_size_is_the_input():
    from context_anchored_tile_refine import resample

    image = torch.rand(1, 20, 30, 3)
    assert resample.resize(image, 30, 20, resample.MITCHELL) is image


@pytest.mark.parametrize("method", ["area", "lanczos", "bicubic (mitchell)"])
def test_resample_virtual_keeps_the_centre_of_a_fractional_resize(method):
    from context_anchored_tile_refine import resample

    image = torch.rand(1, 20, 30, 3, generator=torch.Generator().manual_seed(4))
    # A whole-pixel virtual size is the plain resize, cropped around the centre.
    whole = resample.resize(image, 72, 48, method)
    window = resample.resize(image, 64, 44, method, virtual=(72.0, 48.0))
    assert torch.allclose(window, whole[:, 2:46, 4:68], atol=1e-6)
    # A fractional one: a flat image stays flat, and the window sits in the exact centre, so a
    # mirror-symmetric image comes out mirror-symmetric on both axes.
    flat = resample.resize(torch.full((1, 20, 30, 3), 0.25), 64, 44, method, virtual=(72.5, 48.3))
    assert torch.allclose(flat, torch.full_like(flat, 0.25), atol=1e-6)
    xs = (torch.arange(30) - 14.5).abs().view(1, 1, 30, 1)
    ys = (torch.arange(20) - 9.5).abs().view(1, 20, 1, 1)
    symmetric = ((xs + 2 * ys) / 40).expand(1, 20, 30, 3).contiguous()
    out = resample.resize(symmetric, 60, 40, method, virtual=(72.5, 48.3))
    assert torch.allclose(out, out.flip(2), atol=1e-6) and torch.allclose(out, out.flip(1), atol=1e-6)


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
    assert "\nFinal\n" in text
    # Each pass draws its noise on its own seed.
    assert sorted(set(log)) == [3, 4, 5]
