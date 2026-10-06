"""GSFA with the original product and controlled, identity-start alternatives.

All modes apply F' = F + lambda_g * (2M - 1) * F. ``centered_additive``
uses M = (M_phys + M_att) / 2; both product modes use M = M_phys * M_att.
``centered_product`` names the product control and should be paired with
``identity_init=True``. With that option, each product factor starts at
1 / sqrt(2), while each additive factor starts at 1 / 2.

The default mode and initialization preserve the V025 implementation.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GSFAGate(nn.Module):
    def __init__(
        self,
        channels: int,
        role_scales: tuple[float, ...],
        lambda_g: float = 0.25,
        prompt_width: int = 16,
        channel_reduction: int = 4,
        eps_h_eotvos: float = 1.4,
        eps_e_eotvos: float = 1.4,
        e_ref_eotvos: float = 33.7158,
        fusion_mode: str = "legacy_product",
        identity_init: bool = False,
        prompt_spatial_mode: str = "full",
    ) -> None:
        super().__init__()
        if len(role_scales) != 5:
            raise ValueError(f"role_scales must have five entries, got {role_scales}")
        if not 0.0 <= lambda_g < 1.0:
            raise ValueError(f"lambda_g must lie in [0, 1), got {lambda_g}")
        if fusion_mode not in {"legacy_product", "centered_additive", "centered_product"}:
            raise ValueError(f"unknown GSFA fusion_mode {fusion_mode!r}")
        if prompt_spatial_mode not in {"full", "mean"}:
            raise ValueError(f"unknown GSFA prompt_spatial_mode {prompt_spatial_mode!r}")
        self.lambda_g = float(lambda_g)
        self.fusion_mode = fusion_mode
        self.identity_init = bool(identity_init)
        self.prompt_spatial_mode = prompt_spatial_mode
        self.eps_h_eotvos = float(eps_h_eotvos)
        self.eps_e_eotvos = float(eps_e_eotvos)
        self.e_ref_eotvos = float(e_ref_eotvos)
        self.register_buffer(
            "role_scales",
            torch.tensor(tuple(float(value) for value in role_scales), dtype=torch.float32),
            persistent=False,
        )
        self.prompt_in = nn.Conv2d(3, prompt_width, 3, padding=1)
        self.prompt_act = nn.GELU()
        self.prompt_out = nn.Conv2d(prompt_width, 1, 1)
        reduced = max(channels // int(channel_reduction), 8)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1),
            nn.PReLU(),
            nn.Conv2d(reduced, channels, 1),
            nn.PReLU(),
        )
        self.attention_out = nn.Conv2d(1, 1, 3, padding=1)
        self.last_gate_mean = torch.tensor(0.0)
        self.last_diagnostics: dict[str, torch.Tensor] = {}
        self.reset_special_initialization()

    def reset_special_initialization(self) -> None:
        """Restore the configured identity start after a paired parameter reset."""
        if not self.identity_init:
            return
        probability = 0.5 if self.fusion_mode == "centered_additive" else math.sqrt(0.5)
        bias = math.log(probability / (1.0 - probability))
        for layer in (self.prompt_out, self.attention_out):
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, bias)

    @torch.no_grad()
    def _record_diagnostics(
        self,
        physical: torch.Tensor,
        learned: torch.Tensor,
        alpha: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        """Store detached per-sample values; saturation means p <= .01 or >= .99."""
        gain = (1.0 + self.lambda_g * (2.0 * gate - 1.0)).flatten(1)
        diagnostics = {
            "gain_mean": gain.mean(dim=1),
            "gain_lt_one_fraction": (gain < 1.0).float().mean(dim=1),
        }
        for name, value in (("physical", physical), ("learned", learned)):
            flat = value.detach().float().flatten(1)
            diagnostics[name + "_mean"] = flat.mean(dim=1)
            diagnostics[name + "_spatial_std"] = flat.std(dim=1, unbiased=False)
            diagnostics[name + "_saturated_fraction"] = (
                (flat <= 0.01) | (flat >= 0.99)
            ).float().mean(dim=1)
        probabilities = alpha.detach().float().flatten(1)
        normalizer = math.log(probabilities.shape[1]) if probabilities.shape[1] > 1 else 1.0
        diagnostics["alpha_entropy_normalized"] = -(
            probabilities * probabilities.clamp_min(1.0e-20).log()
        ).sum(dim=1) / normalizer
        diagnostics["alpha_max"], diagnostics["alpha_top_channel"] = probabilities.max(dim=1)
        self.last_diagnostics = diagnostics

    def physical_prompt(self, role_image: torch.Tensor) -> torch.Tensor:
        d5 = role_image.float() * self.role_scales.view(1, 5, 1, 1)
        q, a, b, u, v = d5.unbind(dim=1)
        energy = 1.5 * q.square() + 2.0 * (
            a.square() + b.square() + u.square() + v.square()
        )
        tilt = (2.0 / math.pi) * torch.atan2(
            q, (a.square() + b.square() + self.eps_h_eotvos**2).sqrt()
        )
        amplitude = torch.log1p(energy.sqrt() / self.e_ref_eotvos)
        spin2_ratio = (2.0 * (u.square() + v.square())) / (
            energy + self.eps_e_eotvos**2
        )
        return torch.stack([tilt, amplitude, spin2_ratio], dim=1)

    def forward(self, feature: torch.Tensor, role_image: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 5:
            raise ValueError(f"expected BCZHW feature, got {tuple(feature.shape)}")
        if role_image.ndim != 4 or role_image.shape[1] != 5:
            raise ValueError(
                f"expected Bx5xHxW role image, got {tuple(role_image.shape)}"
            )
        height, width = feature.shape[-2:]

        prompt = self.physical_prompt(role_image)
        if self.prompt_spatial_mode == "mean":
            # Average the three nonlinear descriptors, not the raw observations.
            prompt = prompt.mean(dim=(2, 3), keepdim=True).expand_as(prompt)
        if prompt.shape[-2:] != (height, width):
            prompt = F.adaptive_avg_pool2d(prompt, (height, width))
        physical = torch.sigmoid(
            self.prompt_out(self.prompt_act(self.prompt_in(prompt))).float()
        )

        pooled = feature.mean(dim=2)
        transformed = self.channel_mlp(pooled)
        alpha = torch.softmax(
            transformed.mean(dim=(2, 3), keepdim=True).float(), dim=1
        )
        summary = (alpha * pooled.float()).sum(dim=1, keepdim=True)
        learned = torch.sigmoid(self.attention_out(summary).float())

        gate = (
            0.5 * (physical + learned)
            if self.fusion_mode == "centered_additive"
            else physical * learned
        )
        self.last_gate_mean = gate.detach().float().mean()
        if self.training:
            self.last_diagnostics = {}
        else:
            self._record_diagnostics(physical, learned, alpha, gate)
        modulation = (self.lambda_g * (2.0 * gate.unsqueeze(2) - 1.0)).to(feature.dtype)
        return feature + modulation * feature
