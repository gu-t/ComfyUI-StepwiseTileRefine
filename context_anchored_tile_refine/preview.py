"""preview_mode: what the node shows in its live preview while it samples (a test-phase widget).
torch at module scope; PIL, comfy and latent_preview lazy.

The default, "guess", is core's preview and the path every run took before this widget: the
model's guess of the finished picture (x0) per sampler step, of the tile in tile by tile and of
the whole image in the all at once modes. The other modes pick WHAT is shown and WHERE:

    what   guess   the model's guess of the finished picture (x0)
           noisy   the noisy latent the sampler is at (x), so the noise is seen leaving
           change  |guess - the pass's input|, amplified: where, and how much, is being redrawn
    where  whole image          the whole picture, tiles composited where they belong (in tile
                                by tile: finished tiles stay, the running one updates live)
           one tile at a time   one tile window per update (in the all at once modes: every
                                window of every model call, cond only)
    grid   the whole-image guess with every tile window (yellow) and core (cyan) outlined; in
           all at once, shifting the grid is seen moving step by step

A PreviewHub is built per refined picture by `make_hub` (None for the default, so the default
path is untouched) and fed by the two engines: `begin_tile` / `tile_step` from the raster path's
per-tile callback, `begin_canvas` / `window` / `canvas_step` from the all at once wrapper and its
step callback. Every feed returns a ComfyUI preview tuple or None (keep the last preview)."""
import torch

GUESS = "guess"
PREVIEW_MODES = (
    GUESS,
    "guess, whole image",
    "guess, one tile at a time",
    "noisy, whole image",
    "noisy, one tile at a time",
    "change, whole image",
    "change, one tile at a time",
    "grid",
)
PREVIEW_MODE_TOOLTIP = (
    "Test phase. What the live preview shows while the node samples. guess (default): the model's guess of the "
    "finished picture, as in core (per tile in tile by tile, the whole image in the all at once modes). noisy: the "
    "noisy latent, so the noise is seen leaving. change: the amplified difference from the pass's input, so you see "
    "where and how much is redrawn. whole image / one tile at a time: where it is shown. grid: the whole image with "
    "every tile window (yellow) and core (cyan) outlined.")

CHANGE_GAIN = 4
MAX_COMPOSITE = 1024
WINDOW_COLOR = (255, 220, 0)
CORE_COLOR = (0, 220, 255)


def parse_mode(mode):
    """(what, where, grid) for a preview_mode string."""
    if mode not in PREVIEW_MODES:
        raise ValueError(f"unknown preview_mode {mode!r}, expected one of {PREVIEW_MODES}")
    if mode == "grid":
        return "guess", "whole", True
    if mode == GUESS:
        return "guess", None, False
    what, where = (part.strip() for part in mode.split(","))
    return what, ("whole" if where == "whole image" else "tile"), False


def make_hub(mode, model_patcher, canvas_cells, pixels=None, layout=None, scale=8):
    """The hub for one picture, or None for the default mode or when previews are off."""
    if mode is None or mode == GUESS:
        return None
    what, where, grid = parse_mode(mode)
    import latent_preview

    previewer = latent_preview.get_previewer(model_patcher.load_device, model_patcher.model.latent_format)
    if previewer is None:
        return None
    process_in = getattr(model_patcher.model, "process_latent_in", None)
    return PreviewHub(what, where, grid, previewer.decode_latent_to_preview, canvas_cells, pixels=pixels,
                      layout=layout, scale=scale, process_in=process_in,
                      max_size=getattr(latent_preview, "MAX_PREVIEW_RESOLUTION", 512))


class PreviewHub:
    """Renders previews for one picture. `decode` turns a [1,C,h,w] latent into a PIL image;
    `canvas_cells` is (h, w) in latent cells; `pixels` ([1,H,W,3], optional) seeds the whole-image
    composite in tile by tile; `layout` (grid.Layout, px) gives tile by tile's grid; `process_in`
    maps a raw VAE latent into the model's space, where x0 and x live."""

    def __init__(self, what, where, grid, decode, canvas_cells, pixels=None, layout=None, scale=8, process_in=None,
                 max_size=512):
        self.what, self.where, self.grid = what, where, grid
        self.decode = decode
        self.cells_h, self.cells_w = int(canvas_cells[0]), int(canvas_cells[1])
        self.pixels = pixels
        self.layout = layout
        self.scale = int(scale)
        self.process_in = process_in
        self.max_size = max_size
        self.composite = None
        self.k = None  # composite pixels per latent cell
        self.tile = None  # (tile, source latent) of the running raster tile
        self.source = None  # the all at once canvas source latent (model space)
        self.windows = None  # the all at once windows of the latest model call

    # --- rendering --------------------------------------------------------------------------------

    def _image(self, x0, x, src):
        from PIL import ImageChops

        if self.what == "noisy":
            return self.decode(x[:1]).convert("RGB")
        guess = self.decode(x0[:1]).convert("RGB")
        if self.what == "guess" or src is None:
            return guess
        diff = ImageChops.difference(guess, self.decode(src[:1]).convert("RGB"))
        return diff.point(lambda v: min(255, v * CHANGE_GAIN))

    def _ensure_composite(self, img, cells_w):
        if self.composite is not None:
            return
        from PIL import Image

        r = img.width / max(1, cells_w)
        self.k = max(0.25, min(r, MAX_COMPOSITE / max(self.cells_h, self.cells_w)))
        size = (max(1, round(self.cells_w * self.k)), max(1, round(self.cells_h * self.k)))
        if self.what == "change" or self.pixels is None:
            self.composite = Image.new("RGB", size)
            return
        px = self.pixels[0, :, :, :3].movedim(-1, 0)[None].float()
        small = torch.nn.functional.interpolate(px, size=(size[1], size[0]), mode="area")[0]
        arr = (small.clamp(0, 1) * 255).round().to(torch.uint8).movedim(0, -1).cpu().numpy()
        self.composite = Image.fromarray(arr)

    def _box(self, y0, y1, x0, x1):
        k = self.k
        return (round(x0 * k), round(y0 * k), round(x1 * k), round(y1 * k))

    def _paste(self, img, win, region):
        # Paste the part of `img` (rendered over the window `win`, cells y0, y1, x0, x1) that covers
        # `region` (cells) into the composite.
        wy0, wy1, wx0, wx1 = win
        ry0, ry1, rx0, rx1 = region
        fx = img.width / max(1, wx1 - wx0)
        fy = img.height / max(1, wy1 - wy0)
        part = img.crop((round((rx0 - wx0) * fx), round((ry0 - wy0) * fy),
                         round((rx1 - wx0) * fx), round((ry1 - wy0) * fy)))
        box = self._box(ry0, ry1, rx0, rx1)
        if box[2] > box[0] and box[3] > box[1]:
            self.composite.paste(part.resize((box[2] - box[0], box[3] - box[1])), box[:2])

    def _with_grid(self, img, windows, cores, current=None):
        from PIL import ImageDraw

        out = img.copy()
        draw = ImageDraw.Draw(out)
        for rect in windows:
            box = self._box(*rect)
            draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=WINDOW_COLOR)
        for rect in cores:
            box = self._box(*rect)
            draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=CORE_COLOR)
        if current is not None:
            box = self._box(*current)
            draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=(255, 64, 64), width=2)
        return out

    def _out(self, img):
        return ("JPEG", img, self.max_size)

    # --- tile by tile ---------------------------------------------------------------------------

    def begin_tile(self, tile, source_latent):
        src = source_latent
        if self.what == "change" and self.process_in is not None:
            src = self.process_in(source_latent)
        self.tile = (tile, src)

    def tile_step(self, x0, x):
        tile, src = self.tile
        s = self.scale
        crop, paste = tile.crop_rect, tile.paste_rect
        win = (crop.y0 // s, crop.y1 // s, crop.x0 // s, crop.x1 // s)
        img = self._image(x0, x, src)
        if self.where == "tile":
            return self._out(img)
        self._ensure_composite(img, win[3] - win[2])
        self._paste(img, win, (paste.y0 // s, paste.y1 // s, paste.x0 // s, paste.x1 // s))
        if self.grid and self.layout is not None:
            rects = [(t.crop_rect.y0 // s, t.crop_rect.y1 // s, t.crop_rect.x0 // s, t.crop_rect.x1 // s)
                     for t in self.layout.tiles]
            cores = [(t.core.y0 // s, t.core.y1 // s, t.core.x0 // s, t.core.x1 // s) for t in self.layout.tiles]
            return self._out(self._with_grid(self.composite, rects, cores, current=win))
        return self._out(self.composite)

    # --- all at once ----------------------------------------------------------------------------

    def begin_canvas(self, source_latent):
        src = source_latent
        if self.process_in is not None:
            src = self.process_in(source_latent)
        self.source = src

    def window(self, win, x_window, out_window, windows):
        """One window of one model call (cond only). Returns a preview in the one-tile view."""
        self.windows = windows
        if self.where != "tile" or x_window is None:
            return None
        src = None if self.source is None else self.source[..., win.y0:win.y1, win.x0:win.x1]
        return self._out(self._image(out_window, x_window, src))

    def canvas_step(self, x0, x):
        """The whole canvas after one sampler step. Returns a preview in the whole-image view."""
        if self.where == "tile":
            return None
        img = self._image(x0, x, self.source)
        self._ensure_composite(img, self.cells_w)
        self.composite = img.resize(self.composite.size)
        if self.grid and self.windows:
            rects = [(w.y0, w.y1, w.x0, w.x1) for w in self.windows]
            cores = [(r.y0, r.y1, r.x0, r.x1) for w in self.windows for r in w.writes]
            return self._out(self._with_grid(self.composite, rects, cores))
        return self._out(self.composite)
