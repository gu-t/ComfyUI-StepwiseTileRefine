"""preview_mode (preview.py): parsing, the hub's renders on a fake previewer, and both engines feeding
it through the Tile Refine (Any VAE) node. No GPU and no model."""
import pytest
import torch
from PIL import Image
from test_all_at_once import WrapperGuider
from test_stepwise import CheckingGuider, ScaleVAE, ZeroNoise, _smooth_image

from context_anchored_tile_refine import preview
from context_anchored_tile_refine.all_at_once import ALL_AT_ONCE, SHIFTING
from context_anchored_tile_refine.preview import PREVIEW_MODES, PreviewHub, parse_mode
from context_anchored_tile_refine.stepwise_nodes import TileRefineAnyVAE

SIGMAS = torch.linspace(1.0, 0.0, 5)


def fake_decode(latent):
    # One pixel per latent cell, the first three channels as RGB in 0..1.
    rgb = latent[0, :3].clamp(0, 1).movedim(0, -1)
    return Image.fromarray((rgb * 255).round().to(torch.uint8).cpu().numpy())


class FakePreviewer:
    def __init__(self):
        self.decodes = 0

    def decode_latent_to_preview(self, latent):
        self.decodes += 1
        return fake_decode(latent)

    def decode_latent_to_preview_image(self, preview_format, latent):
        return ("JPEG", self.decode_latent_to_preview(latent), 512)


def test_parse_mode():
    assert parse_mode("guess") == ("guess", None, False)
    assert parse_mode("noisy, whole image") == ("noisy", "whole", False)
    assert parse_mode("change, one tile at a time") == ("change", "tile", False)
    assert parse_mode("grid") == ("guess", "whole", True)
    assert all(parse_mode(m) for m in PREVIEW_MODES)
    with pytest.raises(ValueError, match="preview_mode"):
        parse_mode("sparkles")


def test_the_default_mode_builds_no_hub():
    assert preview.make_hub("guess", object(), (4, 4)) is None
    assert preview.make_hub(None, object(), (4, 4)) is None


def test_change_is_black_where_nothing_changed():
    hub = PreviewHub("change", "tile", False, fake_decode, (8, 8))
    same = torch.rand(1, 3, 8, 8)
    hub.begin_canvas(same)
    shown = hub.window(_win(0, 8, 0, 8), same, same, [])
    assert shown[1].getextrema() == ((0, 0), (0, 0), (0, 0))
    moved = hub.window(_win(0, 8, 0, 8), same, (same + 0.1).clamp(0, 1), [])
    assert max(hi for _, hi in moved[1].getextrema()) > 0


def _win(y0, y1, x0, x1, writes=()):
    from context_anchored_tile_refine.all_at_once import Window, Write

    return Window(y0, y1, x0, x1, [Write(*w) for w in writes])


def test_the_whole_image_view_shows_the_canvas_and_the_grid():
    hub = PreviewHub("guess", "whole", True, fake_decode, (8, 12))
    canvas = torch.zeros(1, 3, 8, 12)
    hub.window(_win(0, 8, 0, 6, [(0, 8, 0, 6)]), None, None, [_win(0, 8, 0, 6, [(0, 8, 0, 6)])])
    shown = hub.canvas_step(canvas, canvas)
    img = shown[1]
    assert img.size == (12, 8)  # one pixel per cell from the fake decode
    colors = {img.getpixel((x, y)) for x in range(img.width) for y in range(img.height)}
    assert preview.CORE_COLOR in colors


def test_tile_by_tile_whole_image_keeps_finished_tiles(comfy_stubs):
    comfy_stubs["previewer"] = FakePreviewer()
    image = _smooth_image(600, 744)
    (out,) = TileRefineAnyVAE().refine(image, CheckingGuider(), object(), SIGMAS, ScaleVAE(16, 3), ZeroNoise(),
                                       256, 256, 16, 16, preview_mode="guess, whole image")
    assert out.shape == image.shape
    sent = [u[2] for bar in comfy_stubs["progress_bars"] for u in bar.updates if u[2] is not None]
    assert sent and all(p[0] == "JPEG" for p in sent)
    # The whole picture every time, never one tile.
    assert {p[1].size for p in sent} == {(744 // 16 + (1 if 744 % 16 else 0), 600 // 16 + (1 if 600 % 16 else 0))}


@pytest.mark.parametrize("tile_mode", [ALL_AT_ONCE, SHIFTING])
@pytest.mark.parametrize("mode", ["noisy, one tile at a time", "change, whole image", "grid"])
def test_all_at_once_modes_feed_the_preview(comfy_stubs, tile_mode, mode):
    comfy_stubs["previewer"] = FakePreviewer()
    image = _smooth_image(600, 744)
    (out,) = TileRefineAnyVAE().refine(image, WrapperGuider(), object(), SIGMAS, ScaleVAE(16, 3), ZeroNoise(),
                                       256, 256, 16, 32, tile_mode=tile_mode, preview_mode=mode)
    assert out.shape == image.shape
    sent = [u[2] for bar in comfy_stubs["progress_bars"] for u in bar.updates if u[2] is not None]
    assert sent
    sizes = {p[1].size for p in sent}
    if mode.endswith("one tile at a time"):
        assert all(w <= 16 and h <= 16 for w, h in sizes)  # 256 px windows at 16 px cells
    else:
        assert all(w > 16 for w, h in sizes)
