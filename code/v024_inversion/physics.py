from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .utils import resolve_path


@dataclass(frozen=True)
class Normalization:
    scalar_rms: float
    spin1_rms: float
    spin2_rms: float
    raw6_rms: tuple[float, ...]
    epsilon: float = 1.0e-6

    @classmethod
    def from_json(cls, path: str | Path) -> "Normalization":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))["normalization"]
        return cls(
            scalar_rms=float(payload["scalar_rms"]),
            spin1_rms=float(payload["spin1_rms"]),
            spin2_rms=float(payload["spin2_rms"]),
            raw6_rms=tuple(float(value) for value in payload["raw6_rms"]),
            epsilon=float(payload.get("epsilon", 1.0e-6)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "raw6_rms": list(self.raw6_rms),
            "scalar_rms": self.scalar_rms,
            "spin1_rms": self.spin1_rms,
            "spin2_rms": self.spin2_rms,
        }


def role_matrix(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(
        [
            [-1.0 / 3.0, 0.0, 0.0, -1.0 / 3.0, 0.0, 2.0 / 3.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0, -0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )


def tensor_roles(raw6: torch.Tensor) -> torch.Tensor:
    if raw6.shape[-1] != 6:
        raise ValueError(f"raw6 must end in six components, got {tuple(raw6.shape)}")
    matrix = role_matrix(raw6.dtype).to(raw6.device)
    return raw6 @ matrix.T


class GravityOperator(nn.Module):
    """Frozen six-component forward operator with matched role adjoints."""

    def __init__(
        self,
        g6_path: str | Path,
        normalization: Normalization,
        density_shape: tuple[int, int, int] = (16, 32, 32),
        receiver_shape: tuple[int, int] = (33, 33),
        sensitivity_role_path: str | Path | None = None,
        sensitivity_tzz_path: str | Path | None = None,
        backprojection_scale_role: float = 1.0,
        backprojection_scale_tzz: float = 1.0,
        sensitivity_floor_fraction: float = 1.0e-3,
    ) -> None:
        super().__init__()
        g6 = torch.from_numpy(np.load(Path(g6_path))).float()
        self.density_shape = tuple(int(value) for value in density_shape)
        self.receiver_shape = tuple(int(value) for value in receiver_shape)
        self.n_voxels = int(np.prod(self.density_shape))
        self.n_receivers = int(np.prod(self.receiver_shape))
        expected = (self.n_receivers * 6, self.n_voxels)
        if tuple(g6.shape) != expected:
            raise ValueError(f"g6 shape {tuple(g6.shape)} does not match {expected}")
        self.register_buffer("g6", g6, persistent=False)
        self.register_buffer("role_transform", role_matrix(), persistent=False)
        self.register_buffer(
            "role_scales",
            torch.tensor(
                [
                    normalization.scalar_rms,
                    normalization.spin1_rms,
                    normalization.spin1_rms,
                    normalization.spin2_rms,
                    normalization.spin2_rms,
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "tzz_scale",
            torch.tensor(float(normalization.raw6_rms[5]), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sensitivity_role",
            self._load_optional_vector(sensitivity_role_path),
            persistent=False,
        )
        self.register_buffer(
            "sensitivity_tzz",
            self._load_optional_vector(sensitivity_tzz_path),
            persistent=False,
        )
        self.backprojection_scale_role = float(backprojection_scale_role)
        self.backprojection_scale_tzz = float(backprojection_scale_tzz)
        self.sensitivity_floor_fraction = float(sensitivity_floor_fraction)

    def _load_optional_vector(self, path: str | Path | None) -> torch.Tensor:
        if path is None or not Path(path).is_file():
            return torch.empty(0, dtype=torch.float32)
        value = torch.from_numpy(np.load(Path(path))).float().reshape(-1)
        if value.numel() != self.n_voxels:
            raise ValueError(f"sensitivity has {value.numel()} values, expected {self.n_voxels}")
        return value

    def density_to_d6(self, density: torch.Tensor) -> torch.Tensor:
        if tuple(density.shape[-3:]) != self.density_shape:
            raise ValueError(f"density shape must end in {self.density_shape}")
        prefix = density.shape[:-3]
        flat = density.float().reshape(*prefix, self.n_voxels)
        return (flat @ self.g6.T).reshape(*prefix, self.n_receivers, 6)

    def observed_to_normalized(self, raw6: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "tzz":
            return raw6.float()[..., 5:6] / self.tzz_scale.clamp_min(1.0e-8)
        if mode == "roles":
            return tensor_roles(raw6.float()) / self.role_scales.view(1, 1, 5)
        raise ValueError(f"unknown observation mode {mode!r}")

    def density_to_normalized(self, density: torch.Tensor, mode: str) -> torch.Tensor:
        if density.ndim == 5 and density.shape[1] == 1:
            density = density[:, 0]
        return self.observed_to_normalized(self.density_to_d6(density), mode)

    def prepare_inputs(
        self,
        raw6: torch.Tensor,
        receiver_mask: torch.Tensor,
        input_mode: str,
    ) -> dict[str, torch.Tensor]:
        batch, receivers, components = raw6.shape
        if components != 6 or receivers != self.n_receivers:
            raise ValueError("raw6 has an incompatible shape")
        height, width = self.receiver_shape
        mask = receiver_mask.float().reshape(batch, 1, height, width)
        if input_mode == "tzz":
            tzz = self.observed_to_normalized(raw6, "tzz")
            tzz_image = tzz.reshape(batch, height, width, 1).permute(0, 3, 1, 2) * mask
            input_image = torch.cat(
                [tzz_image, torch.zeros(batch, 4, height, width, device=raw6.device)],
                dim=1,
            )
            # Every observable field exposed to T0 depends only on raw Tzz.
            # The shared SpinStem sees one real scalar and four zero channels.
            role_image = input_image
            roles = torch.zeros(batch, receivers, 5, dtype=raw6.dtype, device=raw6.device)
        elif input_mode == "roles":
            roles = self.observed_to_normalized(raw6, "roles")
            role_image = roles.reshape(batch, height, width, 5).permute(0, 3, 1, 2) * mask
            input_image = role_image
        else:
            raise ValueError(f"unknown input mode {input_mode!r}")
        return {
            "input_image": input_image,
            "role_image": role_image,
            "scalar": role_image[:, 0:1],
            "spin1": role_image[:, 1:3],
            "spin2": role_image[:, 3:5],
            "mask_image": mask,
            "observed_roles": roles,
            "observed_tzz": self.observed_to_normalized(raw6, "tzz"),
            "receiver_mask": receiver_mask,
        }

    def adjoint_normalized(
        self,
        normalized_observation: torch.Tensor,
        receiver_mask: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        values = normalized_observation.float() * receiver_mask.float().unsqueeze(-1)
        if mode == "roles":
            weighted_d6 = (values / self.role_scales.view(1, 1, 5)) @ self.role_transform
        elif mode == "tzz":
            weighted_d6 = torch.zeros(
                values.shape[0], values.shape[1], 6,
                dtype=values.dtype, device=values.device,
            )
            weighted_d6[..., 5] = values[..., 0] / self.tzz_scale
        else:
            raise ValueError(f"unknown adjoint mode {mode!r}")
        return weighted_d6.reshape(values.shape[0], -1) @ self.g6

    @torch.no_grad()
    def compute_sensitivity(self, mode: str) -> torch.Tensor:
        kernels = self.g6.reshape(self.n_receivers, 6, self.n_voxels)
        if mode == "tzz":
            return (kernels[:, 5] / self.tzz_scale).square().sum(dim=0).sqrt()
        if mode != "roles":
            raise ValueError(f"unknown sensitivity mode {mode!r}")
        sensitivity_squared = torch.zeros(
            self.n_voxels, dtype=torch.float32, device=self.g6.device
        )
        for index in range(5):
            coefficients = self.role_transform[index] / self.role_scales[index]
            transformed = torch.einsum("c,ncv->nv", coefficients, kernels)
            sensitivity_squared.add_(transformed.square().sum(dim=0))
        return sensitivity_squared.sqrt()

    def backproject(
        self,
        normalized_observation: torch.Tensor,
        receiver_mask: torch.Tensor,
        mode: str = "roles",
    ) -> torch.Tensor:
        sensitivity = self.sensitivity_role if mode == "roles" else self.sensitivity_tzz
        if sensitivity.numel() != self.n_voxels:
            raise RuntimeError(f"missing {mode} sensitivity artifact; run audit_operator.py")
        floor = sensitivity.amax() * self.sensitivity_floor_fraction
        scale = (
            self.backprojection_scale_role
            if mode == "roles"
            else self.backprojection_scale_tzz
        )
        flat = self.adjoint_normalized(normalized_observation, receiver_mask, mode)
        normalized = flat / sensitivity.clamp_min(floor)
        return (normalized / max(scale, 1.0e-8)).reshape(
            normalized.shape[0], 1, *self.density_shape
        )

    def relative_misfit(
        self,
        density: torch.Tensor,
        raw6: torch.Tensor,
        receiver_mask: torch.Tensor,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted = self.density_to_normalized(density, mode)
        observed = self.observed_to_normalized(raw6, mode)
        mask = receiver_mask.float().unsqueeze(-1)
        valid = mask.sum(dim=1).clamp_min(1.0)
        observed_energy = (observed.square() * mask).sum(dim=1) / valid
        channel_scale = observed_energy.sqrt().clamp_min(1.0e-6)
        residual = (predicted - observed) / channel_scale.unsqueeze(1)
        per_sample = (residual.square() * mask).sum(dim=(1, 2)) / (
            valid[:, 0] * residual.shape[-1]
        )
        return per_sample.mean(), per_sample


def build_operator(
    config: dict[str, Any],
    normalization: Normalization,
    project_root: Path,
) -> GravityOperator:
    operator_config = config["operator"]
    audit_path = resolve_path(operator_config["audit_path"], project_root)
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    return GravityOperator(
        g6_path=resolve_path(operator_config["g6_path"], project_root),
        normalization=normalization,
        density_shape=tuple(operator_config.get("density_shape", (16, 32, 32))),
        receiver_shape=tuple(operator_config.get("receiver_shape", (33, 33))),
        sensitivity_role_path=resolve_path(
            operator_config.get("sensitivity_role_path", "artifacts/sensitivity_role.npy"),
            project_root,
        ),
        sensitivity_tzz_path=resolve_path(
            operator_config.get("sensitivity_tzz_path", "artifacts/sensitivity_tzz.npy"),
            project_root,
        ),
        backprojection_scale_role=float(audit.get("backprojection_scale_role", 1.0)),
        backprojection_scale_tzz=float(audit.get("backprojection_scale_tzz", 1.0)),
        sensitivity_floor_fraction=float(operator_config.get("sensitivity_floor_fraction", 1.0e-3)),
    )
