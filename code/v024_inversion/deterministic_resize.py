"""Deterministic linear resize operators used by the V010 training graph.

PyTorch's CUDA ``upsample_trilinear3d_backward`` is not deterministic.  The
operators below implement the same ``align_corners=False`` linear interpolation
as separable matrix products.  They therefore keep the interpolation rule while
remaining compatible with ``torch.use_deterministic_algorithms(True)``.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


def reflect_pad2d(
    x: torch.Tensor, padding: Sequence[int]
) -> torch.Tensor:
    """Deterministic equivalent of ``F.pad(..., mode='reflect')`` for BCHW.

    CUDA's native reflection-pad backward is nondeterministic.  Slicing,
    flipping and concatenating obey the same no-edge-repeat rule and have a
    deterministic autograd path.
    """
    if x.ndim != 4 or len(padding) != 4:
        raise ValueError(f"expected BCHW and (left,right,top,bottom), got {x.shape}")
    left, right, top, bottom = (int(v) for v in padding)
    height, width = x.shape[-2:]
    if min(left, right, top, bottom) < 0:
        raise ValueError(f"negative reflect padding: {tuple(padding)}")
    if left >= width or right >= width or top >= height or bottom >= height:
        raise ValueError(
            f"reflect padding must be smaller than input: {tuple(padding)} vs {(height, width)}"
        )

    horizontal = []
    if left:
        horizontal.append(torch.flip(x[..., 1:left + 1], dims=(-1,)))
    horizontal.append(x)
    if right:
        horizontal.append(torch.flip(x[..., -(right + 1):-1], dims=(-1,)))
    xh = torch.cat(horizontal, dim=-1)

    vertical = []
    if top:
        vertical.append(torch.flip(xh[..., 1:top + 1, :], dims=(-2,)))
    vertical.append(xh)
    if bottom:
        vertical.append(torch.flip(xh[..., -(bottom + 1):-1, :], dims=(-2,)))
    return torch.cat(vertical, dim=-2)


def _linear_matrix(in_size: int, out_size: int, ref: torch.Tensor) -> torch.Tensor:
    """Return the [out, in] align_corners=False interpolation matrix."""
    if in_size <= 0 or out_size <= 0:
        raise ValueError(f"resize sizes must be positive, got {in_size}->{out_size}")
    if in_size == out_size:
        return torch.eye(in_size, device=ref.device, dtype=ref.dtype)

    # Compute coordinates in float64 so the discrete neighbours exactly match
    # PyTorch's half-pixel convention, then cast weights to the activation type.
    coord = ((torch.arange(out_size, device=ref.device, dtype=torch.float64) + 0.5)
             * (float(in_size) / float(out_size)) - 0.5)
    lo_raw = torch.floor(coord).to(torch.long)
    hi_raw = lo_raw + 1
    w_hi = coord - lo_raw.to(coord.dtype)
    w_lo = 1.0 - w_hi
    lo = lo_raw.clamp(0, in_size - 1)
    hi = hi_raw.clamp(0, in_size - 1)
    matrix = (F.one_hot(lo, num_classes=in_size).to(torch.float64) * w_lo[:, None]
              + F.one_hot(hi, num_classes=in_size).to(torch.float64) * w_hi[:, None])
    return matrix.to(dtype=ref.dtype)


def resize_bilinear2d(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Resize BCHW with bilinear, align_corners=False interpolation."""
    if x.ndim != 4 or len(size) != 2:
        raise ValueError(f"expected BCHW and 2-D size, got {tuple(x.shape)} and {tuple(size)}")
    out_h, out_w = (int(size[0]), int(size[1]))
    if x.shape[-2:] == (out_h, out_w):
        return x
    wh = _linear_matrix(x.shape[-2], out_h, x)
    ww = _linear_matrix(x.shape[-1], out_w, x)
    x = torch.einsum("bchw,oh->bcow", x, wh)
    return torch.einsum("bchw,ow->bcho", x, ww)


def resize_trilinear3d(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Resize BCDHW with trilinear, align_corners=False interpolation."""
    if x.ndim != 5 or len(size) != 3:
        raise ValueError(f"expected BCDHW and 3-D size, got {tuple(x.shape)} and {tuple(size)}")
    out_d, out_h, out_w = (int(size[0]), int(size[1]), int(size[2]))
    if x.shape[-3:] == (out_d, out_h, out_w):
        return x
    wd = _linear_matrix(x.shape[-3], out_d, x)
    wh = _linear_matrix(x.shape[-2], out_h, x)
    ww = _linear_matrix(x.shape[-1], out_w, x)
    x = torch.einsum("bcdhw,od->bcohw", x, wd)
    x = torch.einsum("bcdhw,oh->bcdow", x, wh)
    return torch.einsum("bcdhw,ow->bcdho", x, ww)


def avg_pool3d_2x(x: torch.Tensor) -> torch.Tensor:
    """Deterministic non-overlapping 2x2x2 average pooling for BCDHW."""
    if x.ndim != 5:
        raise ValueError(f"expected BCDHW, got {tuple(x.shape)}")
    batch, channels, depth, height, width = x.shape
    if depth % 2 or height % 2 or width % 2:
        raise ValueError(f"all spatial dimensions must be even, got {(depth, height, width)}")
    grouped = x.reshape(
        batch, channels, depth // 2, 2, height // 2, 2, width // 2, 2
    )
    return grouped.mean(dim=(3, 5, 7))
