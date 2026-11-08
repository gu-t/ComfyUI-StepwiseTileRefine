"""Latent geometry: how canvas pixels map onto latent cells for the VAE in use. Stdlib only.

Two numbers describe it:

    scale     pixels per latent cell along each axis, the VAE's spatial compression
              (8 for SD/SDXL/Flux.1/Z-Image, 16 for the FLUX.2 VAE that MOSS-Image uses).
    multiple  the pixel granularity every canvas size, tile base and ring width lands on,
              a multiple of `scale`. It defaults to `scale`. A DiT that patchifies the latent
              (MiniMax H3: VAE 16x times patch 2) can ask for a coarser grid, 32 there.

Upstream hard-coded both as 8. EIGHT is that geometry, and every function that takes a
geometry defaults to it, so an 8x VAE runs the exact code path it always did."""
from dataclasses import dataclass

DEFAULT_LATENT_SCALE = 8


@dataclass(frozen=True)
class LatentGeometry:
    scale: int
    multiple: int

    def __post_init__(self):
        if not isinstance(self.scale, int) or self.scale < 1:
            raise ValueError(f"latent scale must be a positive integer, got {self.scale!r}")
        if not isinstance(self.multiple, int) or self.multiple < 1 or self.multiple % self.scale != 0:
            raise ValueError(
                f"pixel multiple {self.multiple!r} must be a positive multiple of the latent scale {self.scale}")

    def round_up(self, px):
        # Smallest multiple of `multiple` that is >= px.
        return -(-px // self.multiple) * self.multiple

    def round_down(self, px):
        return (px // self.multiple) * self.multiple


EIGHT = LatentGeometry(DEFAULT_LATENT_SCALE, DEFAULT_LATENT_SCALE)


def _as_scale(value):
    # bool is an int subclass and never a compression factor.
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def vae_latent_scale(vae):
    """The VAE's spatial compression in pixels per latent cell, or 8 when it cannot be read.

    ComfyUI's VAE exposes it as `spacial_compression_encode()` (sic), which reads
    `downscale_ratio` and takes the last entry of the tuple form video VAEs use. Anything that
    does not return a positive int (a test double, a VAE from another pack) falls back to
    reading `downscale_ratio` directly and then to 8, the value upstream always assumed."""
    method = getattr(vae, "spacial_compression_encode", None)
    if callable(method):
        try:
            scale = _as_scale(method())
        except Exception:
            scale = None
        if scale is not None:
            return scale
    ratio = getattr(vae, "downscale_ratio", None)
    if isinstance(ratio, (tuple, list)) and ratio:
        ratio = ratio[-1]
    return _as_scale(ratio) or DEFAULT_LATENT_SCALE


def resolve_geometry(vae, latent_scale=0, pixel_multiple=0):
    """The geometry a run uses. 0 means auto for either value: the scale is read off the VAE,
    and the multiple defaults to the scale. A given multiple that is not a multiple of the scale
    is an error rather than a silent round, since it names a model grid the VAE cannot hit."""
    scale = int(latent_scale) if latent_scale else vae_latent_scale(vae)
    multiple = int(pixel_multiple) if pixel_multiple else scale
    if multiple % scale != 0:
        raise ValueError(
            f"pixel_multiple {multiple} is not a multiple of the VAE's latent scale {scale}. "
            f"Use 0 (auto) or a multiple of {scale}.")
    return LatentGeometry(scale, multiple)


def snap_rings(context_anchor, context_overlap, geometry):
    """Ring widths on whole latent cells and on the pixel grid.

    The diffused band edge (core + overlap) must sit on a latent cell boundary, so the overlap
    rounds up to `scale`. The sampled crop (core + overlap + anchor) must keep the crop size on
    `multiple`, so the anchor then rounds up until overlap + anchor is a multiple of `multiple`.
    Values already on the grid come back unchanged, which is every legal value at 8x."""
    overlap = -(-int(context_overlap) // geometry.scale) * geometry.scale
    ring = geometry.round_up(int(context_anchor) + overlap)
    anchor = ring - overlap
    return anchor, overlap
