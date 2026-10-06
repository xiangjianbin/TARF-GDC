from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .deterministic_resize import resize_bilinear2d, resize_trilinear3d
from .gsfa import GSFAGate
from .physics import GravityOperator, Normalization
from .utils import state_dict_sha256
from .zhou_router import ComponentScaleRouter, ConvolutionControl
from .complementarity import AnchoredResidualFusion, RelationBranch


DEFAULT_ROLE_SCALES = (33.7158, 34.9304, 34.9304, 13.2083, 13.2083)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class Residual2d(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class Residual3d(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class PlainStem(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(5, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            Residual2d(width, dropout),
        )

    def forward(self, prepared: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.block(prepared["input_image"])


class SpinStem(nn.Module):
    def __init__(self, width: int, dropout: float, gate_bias: float, tarf: bool = False) -> None:
        super().__init__()
        branch_width = max(width // 3, 8)

        def branch(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, branch_width, 3, padding=1, bias=False),
                nn.GroupNorm(_groups(branch_width), branch_width),
                nn.GELU(),
            )

        self.scalar = branch(1)
        self.spin1 = branch(2)
        self.spin2 = branch(2)
        self.role_width = branch_width
        self.uses_tarf = tarf
        self.last_gate_mean = torch.tensor(0.0)
        if tarf:
            self.anchor = nn.Sequential(
                nn.Conv2d(branch_width, width, 3, padding=1, bias=False),
                nn.GroupNorm(_groups(width), width), nn.GELU(), Residual2d(width, dropout),
            )
            self.tarf = AnchoredResidualFusion(width, branch_width)
            return
        joined = branch_width * 3
        self.gate = nn.Conv2d(joined, 3, 1)
        self.gate_bias = float(gate_bias)
        self.reset_special_initialization()
        self.fuse = nn.Sequential(
            nn.Conv2d(joined, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            Residual2d(width, dropout),
        )
        self.last_gate_mean = torch.tensor(0.0)

    def reset_special_initialization(self) -> None:
        if not self.uses_tarf:
            nn.init.constant_(self.gate.bias, self.gate_bias)

    def forward(self, prepared: dict[str, torch.Tensor], return_roles: bool = False):
        roles = (self.scalar(prepared['scalar']), self.spin1(prepared['spin1']),
                 self.spin2(prepared['spin2']))
        if self.uses_tarf:
            fused = self.tarf(self.anchor(roles[0]), roles[1], roles[2])
            self.last_gate_mean = self.tarf.last_gate_mean
        else:
            joined = torch.cat(roles, dim=1)
            gate = torch.sigmoid(self.gate(joined))
            self.last_gate_mean = gate.detach().float().mean()
            fused = self.fuse(joined) * (1.0 + 0.5 * gate.mean(dim=1, keepdim=True))
        return (fused, roles) if return_roles else fused


class EvidenceFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        gate_bias: float,
        mode: str = "legacy",
        lambda_g: float = 0.25,
    ) -> None:
        super().__init__()
        if mode not in {"legacy", "centered"}:
            raise ValueError(f"unknown SAB mode {mode!r}")
        if not 0.0 <= lambda_g < 1.0:
            raise ValueError(f"sab_lambda must lie in [0, 1), got {lambda_g}")
        self.mode = mode
        self.lambda_g = float(lambda_g)
        self.gate_bias = float(gate_bias)
        self.project = nn.Conv2d(5, channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        self.reset_special_initialization()
        self.last_gate_mean = torch.tensor(0.0)

    def reset_special_initialization(self) -> None:
        if self.mode == "centered":
            # Only the output layer starts at zero; its input remains nonzero,
            # so this identity gate can learn on the first optimizer step.
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        else:
            nn.init.constant_(self.gate[-1].bias, self.gate_bias)

    def forward(self, feature: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        pooled = feature.mean(dim=2)
        evidence = resize_bilinear2d(self.project(observation), pooled.shape[-2:])
        gate = torch.sigmoid(self.gate(torch.cat([pooled, evidence], dim=1)))
        self.last_gate_mean = gate.detach().float().mean()
        self.last_gate = gate.detach()
        gain = 1.0 + (
            self.lambda_g * (2.0 * gate - 1.0) if self.mode == "centered" else gate
        )
        self.last_gain = gain.detach()
        return feature * gain.unsqueeze(2)


class VolumeFusion(nn.Module):
    """Identity-start feature update driven by one physical volume."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Conv3d(1, channels, 3, padding=1, bias=False)
        self.update = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
        )
        self.reset_special_initialization()

    def reset_special_initialization(self) -> None:
        nn.init.zeros_(self.update[-1].weight)

    def forward(self, feature: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        if volume.shape[-3:] != feature.shape[-3:]:
            volume = resize_trilinear3d(volume, feature.shape[-3:])
        evidence = self.project(volume)
        return feature + self.update(torch.cat([feature, evidence], dim=1))


class DepthFusion(nn.Module):
    """Pointwise fusion of decoder features with the depth-branch volume."""

    def __init__(self, channels: int, mode: str = "direct", alpha: float = 1.0) -> None:
        super().__init__()
        if mode not in {"direct", "residual"}:
            raise ValueError(f"unknown depth fusion mode {mode!r}")
        self.mode = mode
        if alpha <= 0.0 or alpha > 2.0:
            raise ValueError("depth fusion alpha must be in (0, 2]")
        self.alpha = float(alpha)
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
        )
        if self.mode == "residual":
            self.fuse.append(nn.Conv3d(channels, channels, 1, bias=False))
        self.reset_special_initialization()

    def reset_special_initialization(self) -> None:
        if self.mode == "residual":
            # Do not also zero the first projection or add a zero scalar gate:
            # the final projection must receive nonzero features to start learning.
            nn.init.zeros_(self.fuse[-1].weight)

    def forward(self, feature: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if depth.shape[-3:] != feature.shape[-3:]:
            depth = resize_trilinear3d(depth, feature.shape[-3:])
        fused = self.fuse(torch.cat([feature, depth], dim=1))
        return feature + self.alpha * fused if self.mode == "residual" else self.alpha * fused


class TikBranch(nn.Module):
    """Historical trainable volume path fed by a frozen mode-specific inverse.

    The inverse uses channel-major normalized observations.  Its Tzz version
    has exactly R columns, so the branch cannot consume the other components.
    """

    def __init__(self, width: int, dropout: float, pinv_path: str,
                 input_mode: str = "roles",
                 density_shape: tuple[int, int, int] = (16, 32, 32),
                 receiver_shape: tuple[int, int] = (33, 33)) -> None:
        super().__init__()
        if input_mode not in {"roles", "tzz"}:
            raise ValueError(f"unknown Tikhonov input mode {input_mode!r}")
        path = Path(pinv_path).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Tikhonov pseudoinverse not found: {path}")
        pinv = np.load(path, allow_pickle=False)
        expected = (int(np.prod(density_shape)),
                    (5 if input_mode == "roles" else 1) * int(np.prod(receiver_shape)))
        if pinv.shape != expected:
            raise ValueError(f"{input_mode} pinv shape {pinv.shape} does not match {expected}")
        self.register_buffer("tik_pinv", torch.from_numpy(pinv).float(), persistent=False)
        self.input_mode, self.density_shape = input_mode, tuple(density_shape)
        self.receiver_shape = tuple(receiver_shape)
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(16), 16), nn.GELU(), Residual3d(16, dropout))
        self.down1 = nn.Sequential(
            nn.Conv3d(16, 64, 4, stride=4, bias=False),
            nn.GroupNorm(_groups(64), 64), nn.GELU())
        self.down2 = nn.Conv3d(64, width * 8, 2, stride=2, bias=False)

    def physical_volume(self, prepared: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.input_mode == "tzz":
            observation = prepared["observed_tzz"].transpose(1, 2).reshape(
                -1, 1, *self.receiver_shape)
        else:
            observation = prepared["role_image"]
        observation = observation * prepared["mask_image"]
        flat = observation.float().flatten(1)
        with torch.autocast(device_type=flat.device.type, enabled=False):
            volume = flat @ self.tik_pinv.T
        return volume.reshape(flat.shape[0], 1, *self.density_shape)

    def forward(self, prepared: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.down2(self.down1(self.stem(self.physical_volume(prepared))))


class InversionNet(nn.Module):
    CORE_VARIANTS = {"t0", "b0", "b2r", "gsfa", "dt", "full", "b2r_dt_context_pre"}
    VALID_VARIANTS = CORE_VARIANTS

    def __init__(
        self,
        variant: str,
        width: int = 32,
        dropout: float = 0.05,
        gate_bias: float = 1.0,
        sab_gate_bias: float = 1.0,
        max_density_gcc: float = 0.85,
        module1_config: dict[str, Any] | None = None,
        module2_config: dict[str, Any] | None = None,
        role_scales: tuple[float, ...] | None = None,
        operator: GravityOperator | None = None,
        sab_mode: str = "centered",
        sab_lambda: float = 0.25,
        depth_fusion_mode: str = "residual",
        depth_fusion_alpha: float = 1.0,
        use_tik: bool = False,
        tik_pinv_path: str | None = None,
        input_mode: str = "roles",
        predict_logvar: bool = False,
        zhou_config: dict[str, Any] | None = None,
        architecture: dict[str, Any] | None = None,
        residual_refiner_config: dict[str, Any] | None = None,
        role_noise_scales: list[float] | None = None,
    ) -> None:
        super().__init__()
        if architecture is None and variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        if predict_logvar:
            raise ValueError("V028 fixes predict_logvar=false; no variance head is constructed")
        if variant == "t0" and input_mode != "tzz":
            raise ValueError("T0 must use input_mode=tzz")
        if variant in self.CORE_VARIANTS - {"t0"} and input_mode != "roles":
            raise ValueError(f"{variant} must use input_mode=roles")
        self.input_mode, self.predict_logvar = input_mode, False
        # Retained configuration value is only the historical diagnostic bound.
        # The linear and glin density branches do not bound or clip predictions.
        self.max_density_gcc = float(max_density_gcc)
        self.uses_spin_stem = variant != "plain"
        self.uses_sab = variant not in {"plain", "t0", "b0"}
        self.uses_tik_branch = bool(use_tik)
        self.uses_zhou_correction = variant in {"m0i0", "m1i0", "m0i1", "m1i1", "conv_control"}
        self.uses_backprojection = variant in {"p1", "p1_zero", "p2"}
        self.uses_data_consistency = variant == "p2"
        self.zero_backprojection = variant == "p1_zero"
        self.uses_gsfa = variant in {"gsfa", "full"}
        self.uses_depth_branch = variant in {"dt", "full", "b2r_dt_context_pre"}
        self.uses_depth_context = variant == "b2r_dt_context_pre"
        self.output_mode = 'linear'
        if architecture is not None:
            self.uses_spin_stem = True
            self.uses_sab = bool(architecture['sab'])
            self.uses_tik_branch = bool(architecture['tik'])
            self.uses_depth_branch = architecture['depth'] != 'none'
            self.uses_depth_context = False
            self.uses_gsfa = self.uses_zhou_correction = False
            self.uses_backprojection = self.uses_data_consistency = False
            self.output_mode = architecture['head']
        self.collect_diagnostics = False
        self.last_diagnostics = {}
        self.uses_residual_refiner = (architecture or {}).get('residual_refiner', 'none') == 'transformer'

        self.stem = (
            SpinStem(width, dropout, gate_bias, tarf=bool((architecture or {}).get('tarf', False)))
            if self.uses_spin_stem
            else PlainStem(width, dropout)
        )
        relation_kind = (architecture or {}).get('relation', 'none')
        self.uses_relation = relation_kind != 'none'
        if self.uses_relation:
            self.relation = RelationBranch(self.stem.role_width, width * 4, relation_kind)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(width, width * 2, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(width * 2), width * 2),
            nn.GELU(),
            Residual2d(width * 2, dropout),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(width * 4), width * 4),
            nn.GELU(),
            Residual2d(width * 4, dropout),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(width * 4, width * 8, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(width * 8), width * 8),
            nn.GELU(),
            Residual2d(width * 8, dropout),
        )
        self.bridge = nn.Sequential(
            nn.Conv2d(width * 8, width * 16, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width * 16), width * 16),
            nn.GELU(),
        )
        self.decoder1_up = nn.ConvTranspose3d(
            width * 8, width * 4, kernel_size=(2, 4, 4),
            stride=(2, 2, 2), padding=(0, 1, 1), bias=False,
        )
        self.decoder1 = nn.Sequential(
            nn.GroupNorm(_groups(width * 4), width * 4),
            nn.GELU(),
            Residual3d(width * 4, dropout),
        )
        self.decoder2_up = nn.ConvTranspose3d(
            width * 4, width * 2, kernel_size=(2, 4, 4),
            stride=(2, 2, 2), padding=(0, 1, 1), bias=False,
        )
        self.decoder2 = nn.Sequential(
            nn.GroupNorm(_groups(width * 2), width * 2),
            nn.GELU(),
            Residual3d(width * 2, dropout),
        )
        self.decoder3_up = nn.ConvTranspose3d(
            width * 2, width, kernel_size=(2, 4, 4),
            stride=(2, 2, 2), padding=(0, 1, 1), bias=False,
        )
        self.decoder3 = nn.Sequential(
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            Residual3d(width, dropout),
        )

        if self.uses_tik_branch:
            if not tik_pinv_path:
                raise ValueError("use_tik=true requires operator.tik_pinv_path")
            self.tik_branch = TikBranch(
                width, dropout, tik_pinv_path, input_mode,
                density_shape=operator.density_shape if operator is not None else (16, 32, 32),
                receiver_shape=operator.receiver_shape if operator is not None else (33, 33))

        if self.uses_sab:
            if self.uses_gsfa:
                scales = (
                    tuple(float(value) for value in role_scales)
                    if role_scales is not None
                    else DEFAULT_ROLE_SCALES
                )
                module1 = dict(module1_config or {})
                module1.setdefault("fusion_mode", "centered_additive")
                module1.setdefault("identity_init", True)
                module1.setdefault("lambda_g", .25)
                self.sab1 = GSFAGate(width * 4, scales, **module1)
                self.sab2 = GSFAGate(width * 2, scales, **module1)
                self.sab3 = GSFAGate(width, scales, **module1)
            else:
                self.sab1 = EvidenceFusion(width * 4, sab_gate_bias, sab_mode, sab_lambda)
                self.sab2 = EvidenceFusion(width * 2, sab_gate_bias, sab_mode, sab_lambda)
                self.sab3 = EvidenceFusion(width, sab_gate_bias, sab_mode, sab_lambda)

        if self.uses_zhou_correction:
            zhou = dict(zhou_config or {})
            if variant == "conv_control":
                self.correction = ConvolutionControl(
                    width, feature_groups=int(zhou.get("feature_groups", 4)),
                    hidden_width=int(zhou.get("control_width", 10)),
                    lambda_g=float(zhou.get("lambda_g", .25)))
            else:
                self.correction = ComponentScaleRouter(
                    width, multiscale=variant[1] == "1", adaptive=variant[3] == "1",
                    branch_width=int(zhou.get("branch_width", 8)),
                    feature_groups=int(zhou.get("feature_groups", 4)),
                    hidden_width=int(zhou.get("hidden_width", 16)),
                    lambda_g=float(zhou.get("lambda_g", .25)))

        if self.output_mode in {'dual', 'glin'}:
            self.support_head = nn.Conv3d(width, 1, 3, padding=1)
        self.density_head = nn.Conv3d(width, 1, 3, padding=1)
        if self.uses_residual_refiner:
            from .residual_transformer import PhysicalResidualTransformer
            self.residual_refiner = PhysicalResidualTransformer(
                operator, role_noise_scales, **dict(residual_refiner_config or {}))

        if self.uses_depth_branch:
            if operator is None:
                raise ValueError("dt/full variants require a GravityOperator")
            from .depth_branch import DepthBranch

            module2 = dict(module2_config or {})
            context_mode = module2.pop("context_mode", "pre" if self.uses_depth_context else "none")
            if context_mode not in {"none", "pre", "post"}:
                raise ValueError("unsupported context_mode")
            self.uses_depth_context = context_mode != "none"
            reference_sigma = module2.pop("reference_component_sigma", None)
            if reference_sigma is None:
                raise ValueError(
                    "module2_config requires reference_component_sigma for dt/full"
                )
            module2.pop("width", None)
            self.depth_branch = DepthBranch(
                operator, reference_sigma, width=width, context_mode=context_mode, **module2
            )
            self.depth_fusion = DepthFusion(width, depth_fusion_mode, alpha=depth_fusion_alpha)

        if self.uses_backprojection:
            self.backprojection_fusion = VolumeFusion(width, dropout)
        if self.uses_data_consistency:
            self.data_consistency_fusion = VolumeFusion(width, dropout)

    @staticmethod
    def _resize_skip(skip: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        volume = skip.unsqueeze(2).expand(-1, -1, target.shape[2], -1, -1)
        if volume.shape[-3:] != target.shape[-3:]:
            volume = resize_trilinear3d(volume, target.shape[-3:])
        return volume

    def decode_features(self, prepared: dict[str, torch.Tensor]) -> torch.Tensor:
        stem, roles = self.stem(prepared, return_roles=True)
        encoded1 = self.encoder1(stem)
        encoded2 = self.encoder2(encoded1)
        if self.uses_relation:
            relation_update = self.relation(
                roles, prepared['mask_image'], prepared.get('role_image'))
            if getattr(self, 'collect_relation_diagnostics', False):
                self.last_relation_ratio = (relation_update.detach().float().square().mean().sqrt()
                    / encoded2.detach().float().square().mean().sqrt().clamp_min(1e-12))
            encoded2 = encoded2 + relation_update
        encoded3 = self.encoder3(encoded2)
        bridge = self.bridge(encoded3)
        volume = bridge.reshape(bridge.shape[0], bridge.shape[1] // 2, 2, 4, 4)
        if self.uses_tik_branch:
            volume = volume + self.tik_branch(prepared)

        decoded1 = self.decoder1_up(volume)
        decoded1 = self.decoder1(decoded1 + self._resize_skip(encoded2, decoded1))
        if self.uses_sab:
            decoded1 = self.sab1(decoded1, prepared["role_image"])

        decoded2 = self.decoder2_up(decoded1)
        decoded2 = self.decoder2(decoded2 + self._resize_skip(encoded1, decoded2))
        if self.uses_sab:
            decoded2 = self.sab2(decoded2, prepared["role_image"])

        decoded3 = self.decoder3_up(decoded2)
        decoded3 = self.decoder3(decoded3 + self._resize_skip(stem, decoded3))
        if self.uses_sab:
            decoded3 = self.sab3(decoded3, prepared["role_image"])
        if self.uses_zhou_correction:
            decoded3 = self.correction(decoded3, prepared["role_image"], prepared["mask_image"])
        return decoded3

    def freeze_initial_predictor(self):
        if not self.uses_residual_refiner:
            raise ValueError('A frozen initial predictor requires a refinement stage')
        self.initial_predictor_frozen = True
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith('residual_refiner.'))
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, 'initial_predictor_frozen', False):
            # requires_grad=False alone does not disable dropout or buffer updates.
            for name, module in self.named_children():
                if name != 'residual_refiner':
                    module.eval()
        return self

    def apply_heads(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        """Regress signed density in g/cm³; only tanh/dual bound it, glin gates the raw head."""
        density = self.density_head(feature)
        if self.output_mode in {'tanh', 'dual'}:
            density = self.max_density_gcc * torch.tanh(density)
        if self.output_mode in {'dual', 'glin'}:
            logits = self.support_head(feature)
            support = torch.sigmoid(logits)
            return {'density': support * density, 'signed_density': density,
                    'support': support, 'support_logits': logits}
        return {'density': density}

    def forward(
        self,
        prepared: dict[str, torch.Tensor],
        operator: GravityOperator | None = None,
    ) -> dict[str, torch.Tensor]:
        if getattr(self, 'initial_predictor_frozen', False):
            if operator is None:
                raise ValueError('Frozen refinement requires a forward operator')
            with torch.no_grad():
                initial = self.apply_heads(self.decode_features(prepared))
            return self.residual_refiner(initial, prepared, operator)
        feature = self.decode_features(prepared)
        if self.uses_depth_branch:
            self.depth_branch.collect_diagnostics = self.collect_diagnostics
            depth_observed = prepared["observed_roles"]
            if getattr(self.depth_branch, "residual_evidence", False):
                if operator is None:
                    raise ValueError("residual-evidence DT requires a GravityOperator")
                # Form the DT input from the observation residual left by the
                # backbone before DT is applied.  This makes the candidate a
                # true residual-evidence test rather than a second copy of
                # the original observation.
                base_output = self.apply_heads(feature)
                predicted_roles = operator.density_to_normalized(
                    base_output["density"], mode="roles"
                )
                depth_observed = prepared["observed_roles"] - predicted_roles
            depth = self.depth_branch(
                depth_observed, prepared["receiver_mask"],
                context=feature if self.uses_depth_context else None,
            )
            base = feature
            feature = self.depth_fusion(base, depth)
            if self.collect_diagnostics:
                with torch.no_grad():
                    self.last_diagnostics = dict(self.depth_branch.last_diagnostics)
                    self.last_diagnostics["fusion_residual_to_backbone_rms"] = (
                        (feature.float() - base.float()).square().mean().sqrt()
                        / base.float().square().mean().sqrt().clamp_min(1e-12)
                    ).item()
        backprojection = None
        if self.uses_backprojection:
            if operator is None:
                raise ValueError("physical variants require a GravityOperator")
            backprojection = operator.backproject(
                prepared["observed_roles"], prepared["receiver_mask"], mode="roles"
            )
            fusion_input = torch.zeros_like(backprojection) if self.zero_backprojection else backprojection
            feature = self.backprojection_fusion(feature, fusion_input)

        initial = self.apply_heads(feature)
        if self.uses_residual_refiner:
            if operator is None:
                raise ValueError('Residual refinement requires the forward operator')
            return self.residual_refiner(initial, prepared, operator)
        if not self.uses_data_consistency:
            if backprojection is not None:
                initial["backprojection"] = backprojection
            return initial

        if operator is None:
            raise ValueError("data consistency requires a GravityOperator")
        predicted_roles = operator.density_to_normalized(initial["density"], mode="roles")
        residual = prepared["observed_roles"] - predicted_roles
        correction = operator.backproject(
            residual, prepared["receiver_mask"], mode="roles"
        )
        refined = self.data_consistency_fusion(feature, correction)
        output = self.apply_heads(refined)
        output["initial_density"] = initial["density"]
        output["backprojection"] = backprojection
        output["data_consistency_backprojection"] = correction
        return output

    def common_state_dict(self) -> dict[str, torch.Tensor]:
        """Shared stem/encoder/decoder/heads, excluding variant-specific branches."""
        excluded = (
            "sab1.", "sab2.", "sab3.", "depth_branch.", "depth_fusion.",
            "backprojection_fusion.", "data_consistency_fusion.", "correction.",
            "relation.", "stem.anchor.", "stem.tarf.", "stem.fuse.", "stem.gate.", "support_head.",
            "residual_refiner.",
        )
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith(excluded)
        }

    @torch.no_grad()
    def initialize_paired(self, seed: int) -> None:
        """Reset CPU modules independently of variant construction order.

        Each parameter-owning leaf receives a stable seed based only on the
        training seed and full module name.  Thus common submodules retain
        their initial values even when a sibling is added or a stem differs.
        Built-in resets precede identity initializers; physical buffers stay fixed.
        """
        if any(parameter.device.type != "cpu" for parameter in self.parameters()):
            raise ValueError("paired initialization must precede moving model parameters to a device")
        for name, child in self.named_modules():
            if not any(parameter.requires_grad for parameter in child.parameters(recurse=False)):
                continue
            key = f"v027-paired-init:{int(seed)}:{name}".encode("utf-8")
            module_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)
            with torch.random.fork_rng(devices=[]):
                # Parameters are still on CPU.  Avoid changing CUDA RNG streams.
                torch.default_generator.manual_seed(module_seed)
                reset = getattr(child, "reset_parameters", None)
                if callable(reset):
                    reset()
        for name, module in reversed(list(self.named_modules())):
            special_reset = getattr(module, "reset_special_initialization", None)
            if callable(special_reset):
                # Small nonzero relation projections must also be identical
                # across paired architectures, regardless of construction RNG.
                key = f"v033-special-init:{int(seed)}:{name}".encode('utf-8')
                module_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % (2**63)
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(module_seed)
                    special_reset()

    def extra_capacity_report(self) -> dict[str, Any] | None:
        return self.correction.capacity_report() if self.uses_zhou_correction else None

    def initialization_hashes(self) -> dict[str, str]:
        """Comparable state hashes; call just after construction for an init report."""
        heads = {
            name: value for name, value in self.state_dict().items()
            if name.startswith("density_head.")
        }
        hashes = {
            "backbone": state_dict_sha256(self.common_state_dict()),
            "heads": state_dict_sha256(heads),
        }
        hashes.update({
            name: state_dict_sha256(child.state_dict())
            for name, child in self.named_children()
        })
        return hashes


def build_model(
    config: dict[str, Any],
    normalization: Normalization | None = None,
    operator: GravityOperator | None = None,
) -> InversionNet:
    model_config = config["model"]
    role_scales = None
    if normalization is not None:
        role_scales = (
            float(normalization.scalar_rms),
            float(normalization.spin1_rms),
            float(normalization.spin1_rms),
            float(normalization.spin2_rms),
            float(normalization.spin2_rms),
        )
    module2_config = None
    if "module2" in config:
        module2_config = dict(config["module2"])
        reference_sigma = config.get("noise", {}).get(
            "reference_component_sigma_eotvos"
        )
        if reference_sigma is not None:
            module2_config.setdefault("reference_component_sigma", reference_sigma)
    paired = bool(config.get("initialization", {}).get("paired", False))
    # Paired construction must not leave variant-dependent CPU RNG consumption
    # behind.  Legacy construction deliberately retains its original behavior.
    with torch.random.fork_rng(devices=[]) if paired else nullcontext():
        model = InversionNet(
            variant=str(model_config["variant"]),
            width=int(model_config.get("width", 32)),
            dropout=float(model_config.get("dropout", 0.05)),
            gate_bias=float(model_config.get("gate_bias", 1.0)),
            sab_gate_bias=float(model_config.get("sab_gate_bias", 1.0)),
            max_density_gcc=float(model_config.get("max_density_gcc", 0.85)),
            module1_config=config.get("module1"),
            module2_config=module2_config,
            role_scales=role_scales,
            operator=operator,
            sab_mode=str(model_config.get("sab_mode", "centered")),
            sab_lambda=float(model_config.get("sab_lambda", 0.25)),
            depth_fusion_mode=str(model_config.get("depth_fusion_mode", "residual")),
            depth_fusion_alpha=float(model_config.get("depth_fusion_alpha", 1.0)),
            use_tik=bool(model_config.get("use_tik", False)),
            tik_pinv_path=config.get("operator", {}).get("tik_pinv_path"),
            input_mode=str(config.get("task", {}).get("input_mode", "roles")),
            predict_logvar=bool(model_config.get("predict_logvar", False)),
            zhou_config=config.get("zhou"),
            architecture=model_config.get('architecture'),
            residual_refiner_config=model_config.get('residual_refiner'),
            role_noise_scales=config.get('loss', {}).get('role_noise_scales'),
        )
        if paired:
            model.initialize_paired(int(config["seed"]))
        if config.get('staged_refinement', {}).get('freeze_backbone', False):
            model.freeze_initial_predictor()
        if config.get('control', {}).get('name') == 'observation_only_full':
            # Same capacity as GDC; only the backprojected evidence differs.
            model.residual_refiner.observation_only = True
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
