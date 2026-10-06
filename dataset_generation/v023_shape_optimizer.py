'Optimize smooth, bounded shape directions with Adam and multiple starts. Loss combines negative log complementary/Tzz energy and density/Tzz energy, plus TV. No spin-dominance penalty is used. Directions use tanh, trilinear interpolation and an odd (kernel 3) smoothing kernel. Noise scales match physical Gaussian noise.'
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from v017.ambiguity import reference_component_sigmas_v019


def optimize_shape_directions_multistart(
    volume_fractions: np.ndarray,
    g6: torch.Tensor,
    contract: dict[str, Any],
    seed: int,
    n_starts: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Optimize smooth density redistribution directions with multiple restarts.

    Returns (directions (n0,16,32,32) float32, diagnostics per prototype).
    """
    device = g6.device
    n0 = len(volume_fractions)
    n_starts = int(n_starts)
    fractions = torch.from_numpy(
        np.ascontiguousarray(
            np.repeat(volume_fractions, n_starts, axis=0), dtype=np.float32
        )
    ).to(device)[:, None]
    batch = len(fractions)
    optimization_grid = tuple(
        int(v) for v in contract.get("optimization_grid_zyx", (8, 16, 16))
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    parameters = (
        0.01
        * torch.randn(
            (batch, 1, *optimization_grid), generator=generator, device=device
        )
    ).requires_grad_()
    optimizer = torch.optim.Adam(
        [parameters], lr=float(contract.get("optimizer_learning_rate", 0.08))
    )
    base_density = 0.50 * fractions.reshape(batch, -1)
    fixed = torch.as_tensor(
        reference_component_sigmas_v019(contract), dtype=torch.float32, device=device
    )
    sigma0 = fixed[5].expand(batch)
    sigma1 = fixed[[2, 4]].expand(batch, 2)
    sigma_q1 = torch.sqrt((fixed[0].square() + fixed[3].square()) / 4.0)
    sigma2 = torch.stack((sigma_q1, fixed[1])).expand(batch, 2)
    null_ratio_weight = float(contract.get("null_ratio_objective_weight", 3.0))
    density_null_weight = float(contract.get("density_null_objective_weight", 8.0))
    smoothing_kernel = int(contract.get("direction_smoothing_kernel", 3))
    if smoothing_kernel < 1 or smoothing_kernel % 2 == 0:
        raise ValueError("direction_smoothing_kernel must be a positive odd integer")
    parameterization = str(contract.get("modulation_parameterization", "bounded_tanh"))
    if parameterization != "bounded_tanh":
        raise ValueError("V023 shape optimizer keeps the V022 bounded_tanh parameterization")
    tv_weight = float(contract.get("optimizer_tv_weight", 0.01))

    def _smooth(modulation: torch.Tensor) -> torch.Tensor:
        if smoothing_kernel <= 1:
            return modulation
        return F.avg_pool3d(
            modulation,
            kernel_size=smoothing_kernel,
            stride=1,
            padding=smoothing_kernel // 2,
        )

    def forward_terms():
        smooth = F.interpolate(
            parameters, size=(16, 32, 32), mode="trilinear", align_corners=False
        )
        modulation = _smooth(torch.tanh(smooth))
        direction = fractions.reshape(batch, -1) * modulation.reshape(batch, -1)
        response = (direction @ g6.T).reshape(batch, 1089, 6).permute(0, 2, 1)
        e0 = (response[:, 5] / sigma0[:, None]).square().mean(1).clamp_min(1e-12)
        e1 = (
            (response[:, [2, 4]] / sigma1[:, :, None])
            .square()
            .mean((1, 2))
            .clamp_min(1e-12)
        )
        q = torch.stack(
            ((response[:, 0] - response[:, 3]) / 2.0, response[:, 1]), dim=1
        )
        e2 = (q / sigma2[:, :, None]).square().mean((1, 2)).clamp_min(1e-12)
        density_energy = (
            direction.square().mean(1)
            / base_density.square().mean(1).clamp_min(1e-12)
        ).clamp_min(1e-12)
        return direction, e0, e1, e2, density_energy

    for _ in range(int(contract["local_search_steps"])):
        optimizer.zero_grad(set_to_none=True)
        _, e0, e1, e2, density_energy = forward_terms()
        loss = -null_ratio_weight * torch.log((e1 + e2) / e0)
        if density_null_weight:
            loss -= density_null_weight * torch.log(density_energy / e0)
        tv = (
            (parameters[:, :, 1:] - parameters[:, :, :-1]).square().mean((1, 2, 3, 4))
            + (parameters[:, :, :, 1:] - parameters[:, :, :, :-1])
            .square()
            .mean((1, 2, 3, 4))
            + (parameters[:, :, :, :, 1:] - parameters[:, :, :, :, :-1])
            .square()
            .mean((1, 2, 3, 4))
        )
        objective = (loss + tv_weight * tv).mean()
        objective.backward()
        optimizer.step()

    with torch.no_grad():
        direction, e0, e1, e2, density_energy = forward_terms()
        ratio_density_per_tzz = (density_energy / e0).reshape(n0, n_starts)
        e0r = e0.reshape(n0, n_starts)
        e1r = e1.reshape(n0, n_starts)
        e2r = e2.reshape(n0, n_starts)
        directions = direction.reshape(n0, n_starts, -1)
        best_start = ratio_density_per_tzz.argmax(dim=1)
        chosen = directions[torch.arange(n0, device=device), best_start.to(device)]
        diagnostics = []
        for i in range(n0):
            j = int(best_start[i])
            diagnostics.append(
                {
                    "n_starts": n_starts,
                    "best_start": j,
                    "start_selection": "max_density_energy_over_e0",
                    "density_energy_over_e0_all_starts": [
                        float(v) for v in ratio_density_per_tzz[i].cpu()
                    ],
                    "density_energy_over_e0": float(ratio_density_per_tzz[i, j].cpu()),
                    "linearized_spin1_to_tzz": float(
                        torch.sqrt(e1r[i, j] / e0r[i, j]).cpu()
                    ),
                    "linearized_spin2_to_tzz": float(
                        torch.sqrt(e2r[i, j] / e0r[i, j]).cpu()
                    ),
                    "linearized_spin1_to_spin2": float(
                        torch.sqrt(e1r[i, j] / e2r[i, j]).cpu()
                    ),
                    "optimizer_steps": int(contract["local_search_steps"]),
                    "direction_smoothing_kernel": smoothing_kernel,
                    "density_null_objective_weight": density_null_weight,
                    "modulation_parameterization": parameterization,
                }
            )
        chosen_np = chosen.cpu().numpy().astype(np.float32).reshape(n0, 16, 32, 32)
    return chosen_np, diagnostics
