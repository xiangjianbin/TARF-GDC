"""V027 signed-component/scale routing and its ordinary convolution control.

The six observation features survive until the K-way mixture.  The resulting
K two-dimensional gates act on channel groups and are broadcast over depth;
they do not introduce an extra depth-dependent inference path.
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .deterministic_resize import reflect_pad2d


@lru_cache(maxsize=128)
def _axis_tables(in_size: int, out_size: int) -> tuple[torch.Tensor, ...]:
    """CPU sparse forward/transpose tables for half-pixel interpolation."""
    forward_indices = [[], []]
    forward_weights = [[], []]
    transpose: list[list[tuple[int, float]]] = [[] for _ in range(in_size)]
    for output in range(out_size):
        coordinate = max(0.0, min(float(in_size - 1),
                                  (output + 0.5) * in_size / out_size - 0.5))
        lower = int(math.floor(coordinate))
        upper = min(lower + 1, in_size - 1)
        fraction = coordinate - lower
        for tap, (source, weight) in enumerate(((lower, 1.0 - fraction),
                                                 (upper, fraction))):
            forward_indices[tap].append(source)
            forward_weights[tap].append(weight)
            if weight:
                transpose[source].append((output, weight))
    contributors = max(len(items) for items in transpose)
    backward_indices, backward_weights = [], []
    for tap in range(contributors):
        entries = [items[tap] if tap < len(items) else (0, 0.0)
                   for items in transpose]
        backward_indices.append([entry[0] for entry in entries])
        backward_weights.append([entry[1] for entry in entries])
    return (
        torch.tensor(forward_indices, dtype=torch.long),
        torch.tensor(forward_weights, dtype=torch.float64),
        torch.tensor(backward_indices, dtype=torch.long),
        torch.tensor(backward_weights, dtype=torch.float64),
    )


@lru_cache(maxsize=256)
def _device_axis_tables(
    in_size: int, out_size: int, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    tables = _axis_tables(in_size, out_size)
    return tuple(value.to(device=device, dtype=dtype if value.is_floating_point()
                          else torch.long) for value in tables)


class _SparseLinearAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, axis: int, out_size: int) -> torch.Tensor:
        axis = axis % value.ndim
        accumulation_dtype = (torch.float32 if value.dtype in
                              (torch.float16, torch.bfloat16) else value.dtype)
        fi, fw, bi, bw = _device_axis_tables(
            value.shape[axis], out_size, value.device, accumulation_dtype)
        ctx.axis, ctx.input_dtype = axis, value.dtype
        ctx.save_for_backward(bi, bw)
        shape = [1] * value.ndim
        shape[axis] = out_size
        working = value.to(accumulation_dtype)
        # No scatter operation appears in either direction.  Each output sum
        # has a fixed order, including CUDA under strict deterministic mode.
        result = working.index_select(axis, fi[0]) * fw[0].reshape(shape)
        result = result + working.index_select(axis, fi[1]) * fw[1].reshape(shape)
        return result.to(value.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        indices, weights = ctx.saved_tensors
        working = grad_output.to(weights.dtype)
        shape = [1] * working.ndim
        shape[ctx.axis] = indices.shape[1]
        result = working.index_select(ctx.axis, indices[0]) * weights[0].reshape(shape)
        for tap in range(1, indices.shape[0]):
            result = result + working.index_select(ctx.axis, indices[tap]) * weights[tap].reshape(shape)
        return result.to(ctx.input_dtype), None, None


def sparse_bilinear2d(value: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Sparse, deterministic bilinear resize with ``align_corners=False``.

    Forward and transpose both gather their fixed neighbours, avoiding dense
    interpolation matrices and nondeterministic scatter-based CUDA backward.
    """
    if value.ndim != 4 or len(size) != 2 or min(size) <= 0:
        raise ValueError(f"expected BCHW and a positive 2-D size: {value.shape}, {size}")
    height, width = int(size[0]), int(size[1])
    if value.shape[-2] != height:
        value = _SparseLinearAxis.apply(value, -2, height)
    if value.shape[-1] != width:
        value = _SparseLinearAxis.apply(value, -1, width)
    return value


def _interpolation_macs(channels: int, source: Sequence[int], target: Sequence[int]) -> int:
    source_h, source_w = (int(item) for item in source)
    target_h, target_w = (int(item) for item in target)
    # A two-term weighted sum is counted as two MACs, including a zero endpoint
    # coefficient.  This is the work performed by the sparse implementation.
    return ((2 * channels * target_h * source_w if source_h != target_h else 0)
            + (2 * channels * target_h * target_w if source_w != target_w else 0))


def smooth_downsample_masked(
    observation: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reflect [1,2,1] outer-product smoothing, normalized by valid weight."""
    if observation.ndim != 4 or mask.shape != (observation.shape[0], 1, *observation.shape[-2:]):
        raise ValueError("observation and one-channel mask shapes must agree")
    weights = observation.new_tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0
    weights = weights.view(1, 1, 3, 3)
    valid = mask.to(observation.dtype)
    denominator = F.conv2d(reflect_pad2d(valid, (1, 1, 1, 1)), weights)
    numerator = F.conv2d(
        reflect_pad2d(observation * valid, (1, 1, 1, 1)),
        weights.expand(observation.shape[1], 1, 3, 3), groups=observation.shape[1])
    normalized = torch.where(denominator > 0, numerator / denominator.clamp_min(1.e-8),
                             torch.zeros_like(numerator))
    return normalized[..., ::2, ::2], denominator[..., ::2, ::2]


def _resize_masked(
    value: torch.Tensor, mask: torch.Tensor, size: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    resized_mask = sparse_bilinear2d(mask.to(value.dtype), size)
    numerator = sparse_bilinear2d(value * mask.to(value.dtype), size)
    result = torch.where(resized_mask > 0,
                         numerator / resized_mask.clamp_min(1.e-8),
                         torch.zeros_like(numerator))
    return result, resized_mask


class _BoundedCorrection(nn.Module):
    def __init__(self, channels: int, feature_groups: int, lambda_g: float) -> None:
        super().__init__()
        if channels % feature_groups or not 0.0 <= lambda_g < 1.0:
            raise ValueError("channels must be divisible by feature_groups and lambda must be in [0,1)")
        self.channels = int(channels)
        self.feature_groups = int(feature_groups)
        self.lambda_g = float(lambda_g)
        self.last_gate_mean = torch.tensor(0.0)
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def _apply_gate(self, feature: torch.Tensor, logits: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        gate = torch.tanh(logits) * (mask > 0).to(logits.dtype)
        gain = self.lambda_g * gate.repeat_interleave(
            self.channels // self.feature_groups, dim=1).unsqueeze(2)
        self.last_gate_mean = gate.detach().float().mean()
        if not self.training:
            self.last_diagnostics = {
                "gain_mean": (1.0 + gain.detach().float()).flatten(1).mean(1),
                "gate_abs_mean": gate.detach().float().abs().flatten(1).mean(1),
            }
        else:
            self.last_diagnostics = {}
        return feature + gain.to(feature.dtype) * feature


class ComponentScaleRouter(_BoundedCorrection):
    """Equations 12--16 of design 05, with independent M and I switches."""
    def __init__(self, channels: int, multiscale: bool, adaptive: bool,
                 branch_width: int = 8, feature_groups: int = 4,
                 hidden_width: int = 16, lambda_g: float = .25) -> None:
        super().__init__(channels, feature_groups, lambda_g)
        self.multiscale, self.adaptive = bool(multiscale), bool(adaptive)
        self.branch_width, self.hidden_width = int(branch_width), int(hidden_width)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(inputs, branch_width, 3, padding=1), nn.GELU())
            for inputs in (1, 2, 2) for _ in range(2)
        ])
        if adaptive:
            self.route = nn.Sequential(
                nn.Conv2d(channels + 6 * branch_width, hidden_width, 1), nn.GELU(),
                nn.Conv2d(hidden_width, feature_groups * 6, 1))
        else:
            self.static_logits = nn.Parameter(torch.zeros(feature_groups, 6))
        self.hidden = nn.Sequential(
            nn.Conv2d(channels + feature_groups * branch_width, hidden_width, 1), nn.GELU())
        self.output = nn.Conv2d(hidden_width, feature_groups, 1)
        self.reset_special_initialization()

    def reset_special_initialization(self) -> None:
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if not self.adaptive:
            nn.init.zeros_(self.static_logits)

    def observation_features(self, observation: torch.Tensor, mask: torch.Tensor,
                             size: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = [(observation * mask, mask)]
        scaled.append(smooth_downsample_masked(observation, mask) if self.multiscale
                      else scaled[0])
        grouped: list[list[torch.Tensor | None]] = [[None, None] for _ in range(3)]
        target_mask = None
        for scale, (values, valid) in enumerate(scaled):
            groups = values.split((1, 2, 2), dim=1)
            features = torch.cat([self.branches[group * 2 + scale](groups[group])
                                  for group in range(3)], dim=1)
            features, resized_valid = _resize_masked(features, valid, size)
            for group, feature in enumerate(features.chunk(3, dim=1)):
                grouped[group][scale] = feature
            target_mask = (resized_valid if target_mask is None
                           else torch.maximum(target_mask, resized_valid))
        return torch.stack([value for group in grouped for value in group], dim=1), target_mask

    def routing_weights(self, pooled: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        batch, _, _, height, width = evidence.shape
        if self.adaptive:
            joined = torch.cat((pooled, evidence.flatten(1, 2)), dim=1)
            logits = self.route(joined).reshape(batch, self.feature_groups, 6, height, width)
        else:
            logits = self.static_logits.view(1, self.feature_groups, 6, 1, 1).expand(
                batch, -1, -1, height, width)
        return logits.float().softmax(dim=2).to(evidence.dtype)

    def forward(self, feature: torch.Tensor, observation: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        pooled = feature.mean(dim=2)
        evidence, target_mask = self.observation_features(observation, mask, pooled.shape[-2:])
        alpha = self.routing_weights(pooled, evidence)
        mixed = torch.einsum("bkshw,bsdhw->bkdhw", alpha, evidence).flatten(1, 2)
        logits = self.output(self.hidden(torch.cat((pooled, mixed), dim=1)))
        return self._apply_gate(feature, logits, target_mask)

    def capacity_report(self, observation_shape: Sequence[int] = (33, 33),
                        target_shape: Sequence[int] = (32, 32),
                        depth: int = 16) -> dict[str, Any]:
        area = math.prod(target_shape)
        native_area = math.prod(observation_shape)
        second = tuple((value + 1) // 2 for value in observation_shape) if self.multiscale else observation_shape
        branch = 9 * 5 * self.branch_width * (native_area + math.prod(second))
        route = ((self.channels + 6 * self.branch_width) * self.hidden_width
                 + self.hidden_width * self.feature_groups * 6) * area if self.adaptive else 0
        head = ((self.channels + self.feature_groups * self.branch_width) * self.hidden_width
                + self.hidden_width * self.feature_groups) * area
        interpolation = sum(_interpolation_macs(3 * self.branch_width + 1, source, target_shape)
                            for source in (observation_shape, second))
        smoothing = (6 * 9 * native_area) if self.multiscale else 0
        mixing = self.feature_groups * 6 * self.branch_width * area
        modulation = self.channels * depth * area
        return {
            "parameters": sum(value.numel() for value in self.parameters()),
            "conv_linear_macs": branch + route + head,
            "fixed_smoothing_macs": smoothing,
            "fixed_interpolation_macs": interpolation,
            "weighted_mixture_macs": mixing,
            "feature_modulation_macs": modulation,
            "total_macs": branch + route + head + interpolation + smoothing + mixing + modulation,
            "mac_convention": "one multiply-accumulate=1; batch=1; includes fixed linear resampling, blur, mixture and residual modulation; excludes bias, normalization, GELU, tanh, softmax, masks/divisions and depth mean",
            "branch_width": self.branch_width, "feature_groups": self.feature_groups,
            "hidden_width": self.hidden_width, "multiscale": self.multiscale,
            "adaptive": self.adaptive,
        }


class ConvolutionControl(_BoundedCorrection):
    """Ordinary convolutions on the same five observations + decoder summary."""
    def __init__(self, channels: int, feature_groups: int = 4,
                 hidden_width: int = 10, lambda_g: float = .25) -> None:
        super().__init__(channels, feature_groups, lambda_g)
        self.hidden_width = int(hidden_width)
        self.hidden = nn.Sequential(
            nn.Conv2d(channels + 5, hidden_width, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_width, hidden_width, 1), nn.GELU())
        self.output = nn.Conv2d(hidden_width, feature_groups, 1)
        self.reset_special_initialization()

    def reset_special_initialization(self) -> None:
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature: torch.Tensor, observation: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        pooled = feature.mean(dim=2)
        evidence, valid = _resize_masked(observation, mask, pooled.shape[-2:])
        logits = self.output(self.hidden(torch.cat((pooled, evidence), dim=1)))
        return self._apply_gate(feature, logits, valid)

    def capacity_report(self, observation_shape: Sequence[int] = (33, 33),
                        target_shape: Sequence[int] = (32, 32),
                        depth: int = 16) -> dict[str, Any]:
        area = math.prod(target_shape)
        convolutions = ((self.channels + 5) * self.hidden_width * 9
                        + self.hidden_width**2 + self.hidden_width * self.feature_groups) * area
        interpolation = _interpolation_macs(6, observation_shape, target_shape)
        modulation = self.channels * depth * area
        return {
            "parameters": sum(value.numel() for value in self.parameters()),
            "conv_linear_macs": convolutions, "fixed_smoothing_macs": 0,
            "fixed_interpolation_macs": interpolation, "weighted_mixture_macs": 0,
            "feature_modulation_macs": modulation,
            "total_macs": convolutions + interpolation + modulation,
            "mac_convention": "one multiply-accumulate=1; batch=1; includes fixed linear resampling and residual modulation; excludes bias, normalization, GELU, tanh, masks/divisions and depth mean",
            "feature_groups": self.feature_groups, "hidden_width": self.hidden_width,
        }
