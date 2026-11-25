"""Whole-image pixel resampling for the Stepwise Tile Upscale node: every method its upscale_method
and downscale_method widgets offer, in float, on an NHWC IMAGE.

Every method is a separable filter applied here in float32, so a pass never rounds the image to
8 bits. Each output pixel is a normalized weighted sum of source pixels ("taps"), and taps beyond
the image edge are dropped and the rest renormalized, as PIL does. Shrinking widens the kernel by
the scale factor, so every source pixel contributes however large the reduction ("precise"
resampling, as in VirtualDub's precise bicubic), and the result has no aliasing from skipped
pixels.

    area      exact area average: each output pixel is the mean of the source area it covers,
              with fractional weights at the edges (OpenCV INTER_AREA). Non-negative weights, so
              no ringing. Offered for downscaling only (enlarging with it is nearest neighbour
              with blended seams).
    bilinear  triangle kernel. Offered for enlarging only.
    bicubic   the Mitchell-Netravali BC family: (B, C) = (1, 0) B-spline (softest, no ringing),
              (1/3, 1/3) Mitchell, (0, 1/2) Catmull-Rom (PIL's "bicubic").
    lanczos   Lanczos-3, sinc(x) * sinc(x / 3), PIL's (and so core's) lanczos without the
              8-bit round trip core's PIL resize takes.

torch + stdlib at module scope."""
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


def _lanczos3(x):
    return torch.where(x.abs() < 3, torch.sinc(x) * torch.sinc(x / 3), torch.zeros_like(x))


# name -> (kernel, support in source pixels at scale 1)
KERNELS = {
    BILINEAR: (_triangle, 1.0),
    B_SPLINE: (_bc_cubic(1.0, 0.0), 2.0),
    MITCHELL: (_bc_cubic(1.0 / 3.0, 1.0 / 3.0), 2.0),
    CATMULL_ROM: (_bc_cubic(0.0, 0.5), 2.0),
    LANCZOS: (_lanczos3, 3.0),
}
METHODS = (AREA, *KERNELS)


def axis_taps(n_in, n_out, method, scale=None, start=0.0):
    """(index, weight) for one axis, both [n_out, taps]: output pixel o is
    sum_k source[index[o, k]] * weight[o, k], and every row of weights sums to 1. By default
    the output spans the source exactly; `scale` (source px per output px) and `start` (the
    source coordinate of the output's left edge) cut a window out of a fractional resize."""
    scale = n_in / n_out if scale is None else scale
    out = torch.arange(n_out, dtype=torch.float64)[:, None]
    if method == AREA:
        # Source pixel j covers [j, j + 1); output pixel o covers [o * scale, (o + 1) * scale).
        lo, hi = start + out * scale, start + (out + 1) * scale
        taps = math.ceil(scale) + 2
        index = torch.floor(lo) + torch.arange(taps, dtype=torch.float64)[None]
        weight = (torch.minimum(hi, index + 1) - torch.maximum(lo, index)).clamp(min=0.0)
    elif method in KERNELS:
        kernel, support = KERNELS[method]
        stretch = max(scale, 1.0)
        reach = support * stretch
        center = start + (out + 0.5) * scale              # in source coordinates, pixel j centred at j + 0.5
        taps = math.ceil(2 * reach) + 2
        index = torch.floor(center - 0.5 - reach) + torch.arange(taps, dtype=torch.float64)[None]
        weight = kernel((index + 0.5 - center) / stretch)
    else:
        raise ValueError(f"resample method must be one of {list(METHODS)}, got {method!r}")
    weight = torch.where((index >= 0) & (index < n_in), weight, torch.zeros_like(weight))
    weight = weight / weight.sum(dim=1, keepdim=True)
    return index.clamp(0, n_in - 1).long(), weight


def _apply_axis(x, dim, n_out, method, scale=None, start=0.0):
    index, weight = axis_taps(x.shape[dim], n_out, method, scale, start)
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


def resize(image, width, height, method, virtual=None):
    """`image` (NHWC, 0..1) resampled to exactly width x height with `method`. A same-size call
    returns the input unchanged. `virtual` (w, h), floats, resizes to that size instead and keeps
    its centre width x height, so the scale is exact on both axes however the size rounds."""
    in_h, in_w = int(image.shape[1]), int(image.shape[2])
    x = image.to(torch.float32)
    if virtual is None:
        if in_w == width and in_h == height:
            return image
        if in_h != height:
            x = _apply_axis(x, 1, height, method)
        if in_w != width:
            x = _apply_axis(x, 2, width, method)
    else:
        vw, vh = virtual
        sy, sx = in_h / vh, in_w / vw
        x = _apply_axis(x, 1, height, method, sy, (vh - height) / 2 * sy)
        x = _apply_axis(x, 2, width, method, sx, (vw - width) / 2 * sx)
    return x.clamp_(0.0, 1.0).to(image.dtype)
