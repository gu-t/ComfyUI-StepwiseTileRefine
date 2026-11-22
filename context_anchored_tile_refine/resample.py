"""Whole-image pixel resampling for the Stepwise Tile Upscale node: every method its upscale_method
and downscale_method widgets offer, in float, on an NHWC IMAGE.

"lanczos" is core's lanczos (upscale._lanczos_resize, an antialiased PIL resize with an 8-bit
round trip), unchanged so a lanczos run keeps the bytes it always had. Every other method is a
separable filter applied here in float32. Each output pixel is a normalized weighted sum of
source pixels ("taps"), and taps beyond the image edge are dropped and the rest renormalized,
as PIL does. Shrinking widens the kernel by the scale factor, so every source pixel contributes
however large the reduction ("precise" resampling, as in VirtualDub's precise bicubic), and the
result has no aliasing from skipped pixels.

    area      exact area average: each output pixel is the mean of the source area it covers,
              with fractional weights at the edges (OpenCV INTER_AREA). Non-negative weights, so
              no ringing. Offered for downscaling only (enlarging with it is nearest neighbour
              with blended seams).
    bilinear  triangle kernel. Offered for enlarging only.
    bicubic   the Mitchell-Netravali BC family: (B, C) = (1, 0) B-spline (softest, no ringing),
              (1/3, 1/3) Mitchell, (0, 1/2) Catmull-Rom (PIL's "bicubic").

torch + stdlib at module scope; upscale.py (and so PIL) is imported only for lanczos."""
import math

import torch

LANCZOS = "lanczos"
AREA = "area"
BILINEAR = "bilinear"
B_SPLINE = "bicubic (B-spline)"
MITCHELL = "bicubic (mitchell)"
CATMULL_ROM = "bicubic (catmull-rom)"


def _bc_cubic(b, c):
    def kernel(x):
        x = x.abs()
        x2 = x * x
        x3 = x2 * x
        near = ((12 - 9 * b - 6 * c) * x3 + (-18 + 12 * b + 6 * c) * x2 + (6 - 2 * b)) / 6
        far = ((-b - 6 * c) * x3 + (6 * b + 30 * c) * x2 + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) / 6
        return torch.where(x < 1, near, torch.where(x < 2, far, torch.zeros_like(x)))
    return kernel


def _triangle(x):
    return (1.0 - x.abs()).clamp(min=0.0)


# name -> (kernel, support in source pixels at scale 1)
KERNELS = {
    BILINEAR: (_triangle, 1.0),
    B_SPLINE: (_bc_cubic(1.0, 0.0), 2.0),
    MITCHELL: (_bc_cubic(1.0 / 3.0, 1.0 / 3.0), 2.0),
    CATMULL_ROM: (_bc_cubic(0.0, 0.5), 2.0),
}
METHODS = (LANCZOS, AREA, *KERNELS)


def axis_taps(n_in, n_out, method):
    """(index, weight) for one axis, both [n_out, taps]: output pixel o is
    sum_k source[index[o, k]] * weight[o, k], and every row of weights sums to 1."""
    scale = n_in / n_out
    out = torch.arange(n_out, dtype=torch.float64)[:, None]
    if method == AREA:
        # Source pixel j covers [j, j + 1); output pixel o covers [o * scale, (o + 1) * scale).
        taps = math.ceil(scale) + 1
        index = torch.floor(out * scale) + torch.arange(taps, dtype=torch.float64)[None]
        lo, hi = out * scale, (out + 1) * scale
        weight = (torch.minimum(hi, index + 1) - torch.maximum(lo, index)).clamp(min=0.0)
    elif method in KERNELS:
        kernel, support = KERNELS[method]
        stretch = max(scale, 1.0)
        reach = support * stretch
        center = (out + 0.5) * scale                      # in source coordinates, pixel j centred at j + 0.5
        taps = math.ceil(2 * reach) + 2
        index = torch.floor(center - 0.5 - reach) + torch.arange(taps, dtype=torch.float64)[None]
        weight = kernel((index + 0.5 - center) / stretch)
    else:
        raise ValueError(f"resample method must be one of {list(METHODS)}, got {method!r}")
    weight = torch.where((index >= 0) & (index < n_in), weight, torch.zeros_like(weight))
    weight = weight / weight.sum(dim=1, keepdim=True)
    return index.clamp(0, n_in - 1).long(), weight


def _apply_axis(x, dim, n_out, method):
    index, weight = axis_taps(x.shape[dim], n_out, method)
    shape = [1] * x.ndim
    shape[dim] = n_out
    result = None
    for k in range(index.shape[1]):
        w = weight[:, k]
        if not torch.any(w):
            continue
        term = x.index_select(dim, index[:, k].to(x.device)) * w.to(x.device, x.dtype).view(shape)
        result = term if result is None else result.add_(term)
    return result


def resize(image, width, height, method):
    """`image` (NHWC, 0..1) resampled to exactly width x height with `method`. A same-size call
    returns the input unchanged."""
    if int(image.shape[2]) == width and int(image.shape[1]) == height:
        return image
    if method == LANCZOS:
        from . import upscale

        return upscale._lanczos_resize(image, width, height)
    x = image.to(torch.float32)
    if int(x.shape[1]) != height:
        x = _apply_axis(x, 1, height, method)
    if int(x.shape[2]) != width:
        x = _apply_axis(x, 2, width, method)
    return x.clamp_(0.0, 1.0).to(image.dtype)
