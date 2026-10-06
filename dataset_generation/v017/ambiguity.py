from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .physics import role_arrays


GLOBAL_DPRIME_METRIC_VERSION = "global_dprime_v018"
ABSOLUTE_COVARIANCE_METRIC_VERSION = "absolute_covariance_global_dprime_v019"


@dataclass(frozen=True)
class PairCandidate:
    first: int
    second: int
    response_group: str
    metrics: dict[str, float]
    scale_second: float


def _centered_cosine(first: np.ndarray, second: np.ndarray) -> float:
    a = first.astype(np.float64) - float(np.mean(first))
    b = second.astype(np.float64) - float(np.mean(second))
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def _symmetric_nrmse(first: np.ndarray, second: np.ndarray) -> float:
    a = first.astype(np.float64)
    b = second.astype(np.float64)
    scale = np.sqrt(0.5 * (np.mean(a * a) + np.mean(b * b)))
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(scale, 1e-30))


def _detectability(first: np.ndarray, second: np.ndarray, snr_db: float) -> float:
    """Diagonal reference covariance using pooled per-channel RMS at fixed SNR."""
    if first.ndim == 1:
        first = first[None]
        second = second[None]
    pooled_rms = np.sqrt(0.5 * (np.mean(first.astype(np.float64) ** 2, axis=1) + np.mean(second.astype(np.float64) ** 2, axis=1)))
    sigma = np.maximum(pooled_rms / (10.0 ** (float(snr_db) / 20.0)), 1e-12)
    whitened = (first.astype(np.float64) - second.astype(np.float64)) / sigma[:, None]
    return float(np.sqrt(np.mean(whitened * whitened)))


def density_nrmse(first: np.ndarray, second: np.ndarray) -> float:
    return _symmetric_nrmse(first.reshape(-1), second.reshape(-1))


def pair_metrics(
    density_a: np.ndarray,
    density_b: np.ndarray,
    response_a: np.ndarray,
    response_b: np.ndarray,
    reference_snr_db: float,
) -> dict[str, float]:
    tzz_a, spin1_a, spin2_a = role_arrays(response_a)
    tzz_b, spin1_b, spin2_b = role_arrays(response_b)
    rms_a = float(np.sqrt(np.mean(tzz_a.astype(np.float64) ** 2)))
    rms_b = float(np.sqrt(np.mean(tzz_b.astype(np.float64) ** 2)))
    d0 = _detectability(tzz_a[None], tzz_b[None], reference_snr_db)
    d1 = _detectability(spin1_a.reshape(2, -1), spin1_b.reshape(2, -1), reference_snr_db)
    d2 = _detectability(spin2_a.reshape(2, -1), spin2_b.reshape(2, -1), reference_snr_db)
    return {
        "tzz_centered_cosine": _centered_cosine(tzz_a, tzz_b),
        "tzz_nrmse": _symmetric_nrmse(tzz_a, tzz_b),
        "tzz_relative_rms_difference": abs(rms_a - rms_b) / max(0.5 * (rms_a + rms_b), 1e-30),
        "D0": d0,
        "D1": d1,
        "D2": d2,
        "density_nrmse": density_nrmse(density_a, density_b),
        "spin1_to_spin2": d1 / max(d2, 1e-12),
        "spin2_to_spin1": d2 / max(d1, 1e-12),
    }


def _pooled_component_sigmas(
    first: np.ndarray,
    second: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Per-component white-noise standard deviations for the V018 contract.

    A single standard deviation is used at every receiver within a component.
    The same pooled value is used for the two endpoints of a pair so that the
    equal-covariance Gaussian detectability calculation is well defined.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] != 6:
        raise ValueError(f"expected matching (6, ny, nx) responses, got {a.shape} and {b.shape}")
    pooled_rms = np.sqrt(0.5 * (
        np.mean(a * a, axis=(1, 2)) + np.mean(b * b, axis=(1, 2))
    ))
    return np.maximum(pooled_rms / (10.0 ** (float(snr_db) / 20.0)), 1e-12)


def pair_metrics_global_dprime(
    density_a: np.ndarray,
    density_b: np.ndarray,
    response_a: np.ndarray,
    response_b: np.ndarray,
    reference_snr_db: float,
) -> dict[str, float]:
    """Whole-grid Gaussian discriminability used by the V018 hard contract.

    Unlike the legacy D0/D1/D2 values, these distances are not divided by the
    square root of the receiver count.  Evidence from all 33x33 receivers is
    therefore accumulated.  Spin-2 uses q1=(Txx-Tyy)/2 and Txy; the q1 noise
    variance is propagated from independent Txx and Tyy noise.
    """
    a = np.asarray(response_a, dtype=np.float64)
    b = np.asarray(response_b, dtype=np.float64)
    sigma = _pooled_component_sigmas(a, b, reference_snr_db)
    delta = a - b
    n_receiver = int(delta.shape[1] * delta.shape[2])

    d0 = float(np.linalg.norm(delta[5] / sigma[5]))
    d1 = float(np.sqrt(
        np.sum((delta[2] / sigma[2]) ** 2)
        + np.sum((delta[4] / sigma[4]) ** 2)
    ))
    delta_q1 = (delta[0] - delta[3]) / 2.0
    sigma_q1 = float(np.sqrt((sigma[0] ** 2 + sigma[3] ** 2) / 4.0))
    d2 = float(np.sqrt(
        np.sum((delta_q1 / sigma_q1) ** 2)
        + np.sum((delta[1] / sigma[1]) ** 2)
    ))

    tzz_a = a[5].reshape(-1)
    tzz_b = b[5].reshape(-1)
    rms_a = float(np.sqrt(np.mean(tzz_a * tzz_a)))
    rms_b = float(np.sqrt(np.mean(tzz_b * tzz_b)))
    supplement = max(d1, d2)
    return {
        "metric_version": GLOBAL_DPRIME_METRIC_VERSION,
        "reference_snr_db": float(reference_snr_db),
        "tzz_centered_cosine": _centered_cosine(tzz_a, tzz_b),
        "tzz_nrmse": _symmetric_nrmse(tzz_a, tzz_b),
        "tzz_relative_rms_difference": abs(rms_a - rms_b) / max(0.5 * (rms_a + rms_b), 1e-30),
        "tzz_global_dprime": d0,
        "spin1_global_dprime": d1,
        "spin2_global_dprime": d2,
        "tzz_whitened_rms_per_receiver": d0 / np.sqrt(float(n_receiver)),
        "spin1_whitened_rms_per_value": d1 / np.sqrt(float(2 * n_receiver)),
        "spin2_whitened_rms_per_value": d2 / np.sqrt(float(2 * n_receiver)),
        "density_nrmse": density_nrmse(density_a, density_b),
        "spin1_to_spin2": d1 / max(d2, 1e-12),
        "spin2_to_spin1": d2 / max(d1, 1e-12),
        "supplement_to_tzz_ratio": supplement / max(d0, 1e-12),
    }


def reference_component_sigmas_v019(contract: dict[str, Any]) -> np.ndarray:
    sigma = np.asarray(contract["reference_component_sigma_eotvos"], dtype=np.float64)
    if sigma.shape != (6,) or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(f"invalid V019 reference component sigma {sigma}")
    return sigma


def pair_metrics_absolute_covariance_v019(
    density_a: np.ndarray,
    density_b: np.ndarray,
    response_a: np.ndarray,
    response_b: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, float]:
    """Whole-grid discriminability under a preregistered absolute noise floor.

    Unlike V018, the covariance is not rescaled from each clean sample. Weak
    and deep sources therefore remain harder than strong, shallow sources.
    """
    a = np.asarray(response_a, dtype=np.float64)
    b = np.asarray(response_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] != 6:
        raise ValueError(f"expected matching (6, ny, nx), got {a.shape}, {b.shape}")
    sigma = reference_component_sigmas_v019(contract)
    delta = a - b
    n_receiver = int(delta.shape[1] * delta.shape[2])
    d0 = float(np.linalg.norm(delta[5] / sigma[5]))
    d1 = float(np.sqrt(
        np.sum((delta[2] / sigma[2]) ** 2)
        + np.sum((delta[4] / sigma[4]) ** 2)
    ))
    delta_q1 = (delta[0] - delta[3]) / 2.0
    sigma_q1 = float(np.sqrt((sigma[0] ** 2 + sigma[3] ** 2) / 4.0))
    d2 = float(np.sqrt(
        np.sum((delta_q1 / sigma_q1) ** 2)
        + np.sum((delta[1] / sigma[1]) ** 2)
    ))
    tzz_a = a[5].reshape(-1)
    tzz_b = b[5].reshape(-1)
    rms_a = float(np.sqrt(np.mean(tzz_a * tzz_a)))
    rms_b = float(np.sqrt(np.mean(tzz_b * tzz_b)))
    supplement = max(d1, d2)
    return {
        "metric_version": ABSOLUTE_COVARIANCE_METRIC_VERSION,
        "reference_component_sigma_eotvos": [float(value) for value in sigma],
        "tzz_centered_cosine": _centered_cosine(tzz_a, tzz_b),
        "tzz_nrmse": _symmetric_nrmse(tzz_a, tzz_b),
        "tzz_relative_rms_difference": abs(rms_a - rms_b) / max(0.5 * (rms_a + rms_b), 1e-30),
        "tzz_global_dprime": d0,
        "spin1_global_dprime": d1,
        "spin2_global_dprime": d2,
        "tzz_whitened_rms_per_receiver": d0 / np.sqrt(float(n_receiver)),
        "spin1_whitened_rms_per_value": d1 / np.sqrt(float(2 * n_receiver)),
        "spin2_whitened_rms_per_value": d2 / np.sqrt(float(2 * n_receiver)),
        "density_nrmse": density_nrmse(density_a, density_b),
        "spin1_to_spin2": d1 / max(d2, 1e-12),
        "spin2_to_spin1": d2 / max(d1, 1e-12),
        "supplement_to_tzz_ratio": supplement / max(d0, 1e-12),
    }


def support_pair_metrics(
    density_a: np.ndarray,
    density_b: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    support_a = np.abs(np.asarray(density_a)) >= float(threshold)
    support_b = np.abs(np.asarray(density_b)) >= float(threshold)
    intersection = int(np.logical_and(support_a, support_b).sum())
    union = int(np.logical_or(support_a, support_b).sum())
    symmetric_difference = int(np.logical_xor(support_a, support_b).sum())
    return {
        "support_intersection_cells": intersection,
        "support_union_cells": union,
        "support_symmetric_difference_cells": symmetric_difference,
        "support_jaccard": float(intersection / max(union, 1)),
        "support_symmetric_difference_fraction": float(
            symmetric_difference / max(union, 1)
        ),
    }


def classify_pair_global_dprime(
    metrics: dict[str, float], contract: dict[str, Any]
) -> str | None:
    """Classify only pairs that satisfy the V018 whole-grid hard gates."""
    d0 = float(metrics["tzz_global_dprime"])
    d1 = float(metrics["spin1_global_dprime"])
    d2 = float(metrics["spin2_global_dprime"])
    supplement_min = float(contract["supplement_global_dprime_min"])
    hard = (
        d0 <= float(contract["tzz_global_dprime_max"])
        and max(d1, d2) >= supplement_min
        and max(d1, d2) / max(d0, 1e-12)
        >= float(contract["supplement_to_tzz_ratio_min"])
        and float(metrics["density_nrmse"]) >= float(contract["density_nrmse_min"])
    )
    if not hard:
        return None
    dominance = float(contract["dominance_ratio_min"])
    if d1 >= supplement_min and d1 / max(d2, 1e-12) >= dominance:
        return "spin1"
    if d2 >= supplement_min and d2 / max(d1, 1e-12) >= dominance:
        return "spin2"
    if min(d1, d2) >= supplement_min:
        return "mixed"
    return None


def classify_pair(metrics: dict[str, float], contract: dict[str, Any]) -> str | None:
    hard = (
        metrics["tzz_centered_cosine"] >= float(contract["tzz_cosine_min"])
        and metrics["tzz_nrmse"] <= float(contract["tzz_nrmse_max"])
        and metrics["tzz_relative_rms_difference"] <= float(contract["tzz_relative_rms_max"])
        and metrics["D0"] <= float(contract["d0_max"])
        and max(metrics["D1"], metrics["D2"]) >= float(contract["supplement_detectability_min"])
        and metrics["density_nrmse"] >= float(contract["density_nrmse_min"])
    )
    if not hard:
        return None
    ratio = float(contract["dominance_ratio_min"])
    threshold = float(contract["supplement_detectability_min"])
    if metrics["D1"] >= threshold and metrics["D1"] / max(metrics["D2"], 1e-12) >= ratio:
        return "spin1"
    if metrics["D2"] >= threshold and metrics["D2"] / max(metrics["D1"], 1e-12) >= ratio:
        return "spin2"
    return "mixed"


def optimal_amplitude_scale(
    tzz_reference: np.ndarray,
    tzz_candidate: np.ndarray,
    contrast_candidate: float,
    contrast_bounds: tuple[float, float],
) -> float:
    a = tzz_reference.astype(np.float64).reshape(-1)
    b = tzz_candidate.astype(np.float64).reshape(-1)
    unconstrained = float(np.dot(a, b) / max(np.dot(b, b), 1e-30))
    lower = float(contrast_bounds[0]) / max(abs(float(contrast_candidate)), 1e-12)
    upper = float(contrast_bounds[1]) / max(abs(float(contrast_candidate)), 1e-12)
    return float(np.clip(unconstrained, lower, upper))


@torch.inference_mode()
def nearest_shortlists(
    tzz: np.ndarray,
    mechanisms: list[str],
    device: str | torch.device,
    shortlist_size: int,
    candidate_pool_per_anchor: int,
    block_size: int = 256,
) -> list[list[int]]:
    """Cosine shortlist on centered Tzz; excludes self and same mechanism."""
    matrix = torch.from_numpy(np.ascontiguousarray(tzz.reshape(len(tzz), -1), dtype=np.float32)).to(device)
    matrix = matrix - matrix.mean(dim=1, keepdim=True)
    matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    mechanism_ids = {name: index for index, name in enumerate(sorted(set(mechanisms)))}
    mids = torch.tensor([mechanism_ids[x] for x in mechanisms], device=device)
    output: list[list[int]] = []
    k = min(max(int(shortlist_size), int(candidate_pool_per_anchor)), len(tzz) - 1)
    for start in range(0, len(tzz), int(block_size)):
        query = matrix[start : start + block_size]
        score = query @ matrix.T
        row_ids = torch.arange(start, start + len(query), device=device)
        score[torch.arange(len(query), device=device), row_ids] = -torch.inf
        score.masked_fill_(mids[None, :] == mids[row_ids][:, None], -torch.inf)
        indices = torch.topk(score, k=k, dim=1, largest=True).indices[:, : int(shortlist_size)]
        output.extend(indices.cpu().tolist())
    return output


def enumerate_pair_candidates(
    densities: np.ndarray,
    responses: np.ndarray,
    records: list[dict[str, Any]],
    contract: dict[str, Any],
    device: str | torch.device,
) -> tuple[list[PairCandidate], dict[str, Any]]:
    shortlists = nearest_shortlists(
        responses[:, 5],
        [str(row["mechanism"]) for row in records],
        device,
        int(contract["shortlist_size"]),
        int(contract["candidate_pool_per_anchor"]),
    )
    contrast_bounds = tuple(float(x) for x in contract["contrast_bounds"])
    accepted: list[PairCandidate] = []
    best_by_group: dict[str, dict[str, float] | None] = {"spin1": None, "spin2": None, "mixed": None}
    evaluated = 0
    unique_keys: set[tuple[int, int]] = set()
    for first, neighbors in enumerate(shortlists):
        for second in neighbors:
            key = (min(first, second), max(first, second))
            if key in unique_keys:
                continue
            unique_keys.add(key)
            scale = optimal_amplitude_scale(
                responses[first, 5], responses[second, 5],
                float(records[second]["contrast_gcc"]), contrast_bounds,
            )
            density_b = densities[second] * scale
            response_b = responses[second] * scale
            metrics = pair_metrics(
                densities[first], density_b, responses[first], response_b,
                float(contract["reference_snr_db"]),
            )
            evaluated += 1
            group = classify_pair(metrics, contract)
            heuristic_group = (
                "spin1" if metrics["D1"] / max(metrics["D2"], 1e-12) >= float(contract["dominance_ratio_min"])
                else "spin2" if metrics["D2"] / max(metrics["D1"], 1e-12) >= float(contract["dominance_ratio_min"])
                else "mixed"
            )
            current = best_by_group[heuristic_group]
            rank = (metrics["tzz_nrmse"], -max(metrics["D1"], metrics["D2"]))
            if current is None or rank < (current["tzz_nrmse"], -max(current["D1"], current["D2"])):
                best_by_group[heuristic_group] = metrics
            if group is not None:
                accepted.append(PairCandidate(first, second, group, metrics, scale))
    accepted.sort(key=lambda row: (
        row.response_group,
        row.metrics["tzz_nrmse"],
        -max(row.metrics["D1"], row.metrics["D2"]),
    ))
    return accepted, {
        "pool_endpoints": len(densities),
        "unique_candidate_pairs_evaluated": evaluated,
        "accepted_candidate_edges": len(accepted),
        "accepted_edges_by_group": {
            group: sum(item.response_group == group for item in accepted)
            for group in ("spin1", "spin2", "mixed")
        },
        "best_rejected_or_accepted_metrics_by_heuristic_group": best_by_group,
    }


def select_disjoint_pairs(
    candidates: Iterable[PairCandidate], quotas: dict[str, int]
) -> tuple[list[PairCandidate], dict[str, int]]:
    selected: list[PairCandidate] = []
    used: set[int] = set()
    counts = {group: 0 for group in quotas}
    # Round-robin by group prevents the mixed class from consuming endpoints needed by a dominant class.
    grouped = {group: [x for x in candidates if x.response_group == group] for group in quotas}
    progress = True
    while progress and any(counts[g] < quotas[g] for g in quotas):
        progress = False
        for group in quotas:
            if counts[group] >= quotas[group]:
                continue
            while grouped[group]:
                item = grouped[group].pop(0)
                if item.first in used or item.second in used:
                    continue
                selected.append(item)
                used.update((item.first, item.second))
                counts[group] += 1
                progress = True
                break
    return selected, counts


def optimize_density_redistribution_batch(
    volume_fractions: np.ndarray,
    target_groups: list[str],
    g6: torch.Tensor,
    contract: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Find smooth density redistributions with near-null Tzz and target spin.

    The optimization variables live on an 8x16x16 grid and are trilinearly
    prolonged, so the learned density modulation cannot create single-voxel
    checkerboards.  Both endpoints retain one density sign and contrasts in
    [0.20, 0.80] g/cc on the same antialiased support.
    """
    device = g6.device
    fractions = torch.from_numpy(
        np.ascontiguousarray(volume_fractions, dtype=np.float32)
    ).to(device)[:, None]
    batch = len(volume_fractions)
    optimization_grid = tuple(int(value) for value in contract.get(
        "optimization_grid_zyx", (8, 16, 16)
    ))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    parameters = (
        0.01 * torch.randn(
            (batch, 1, *optimization_grid), generator=generator, device=device
        )
    ).requires_grad_()
    optimizer = torch.optim.Adam(
        [parameters], lr=float(contract.get("optimizer_learning_rate", 0.08))
    )
    base_density = 0.50 * fractions.reshape(batch, -1)
    with torch.no_grad():
        base_response = (base_density @ g6.T).reshape(batch, 1089, 6).permute(0, 2, 1)
        if str(contract.get("metric_version")) == ABSOLUTE_COVARIANCE_METRIC_VERSION:
            fixed = torch.as_tensor(
                reference_component_sigmas_v019(contract),
                dtype=base_response.dtype,
                device=device,
            )
            sigma0 = fixed[5].expand(batch)
            sigma1 = fixed[[2, 4]].expand(batch, 2)
            sigma_q1 = torch.sqrt((fixed[0].square() + fixed[3].square()) / 4.0)
            sigma2 = torch.stack((sigma_q1, fixed[1])).expand(batch, 2)
        else:
            sigma0 = base_response[:, 5].square().mean(1).sqrt().clamp_min(1e-8) / 10.0
            sigma1 = base_response[:, [2, 4]].square().mean(2).sqrt().clamp_min(1e-8) / 10.0
            base_q = torch.stack(((base_response[:, 0] - base_response[:, 3]) / 2.0, base_response[:, 1]), dim=1)
            sigma2 = base_q.square().mean(2).sqrt().clamp_min(1e-8) / 10.0
    group_code = torch.tensor(
        [{"spin1": 0, "spin2": 1, "mixed": 2}[group] for group in target_groups],
        device=device,
    )
    null_ratio_weight = float(contract.get("null_ratio_objective_weight", 1.0))
    dominance_weight = float(contract.get("dominance_objective_weight", 2.0))
    density_null_weight = float(contract.get("density_null_objective_weight", 0.0))
    last = None
    smoothing_kernel = int(contract.get("direction_smoothing_kernel", 1))
    if smoothing_kernel < 1 or smoothing_kernel % 2 == 0:
        raise ValueError("direction_smoothing_kernel must be a positive odd integer")
    parameterization = str(contract.get("modulation_parameterization", "bounded_tanh"))
    if parameterization not in {"bounded_tanh", "linear_normalized"}:
        raise ValueError(f"unknown modulation_parameterization {parameterization}")
    for _ in range(int(contract["local_search_steps"])):
        optimizer.zero_grad(set_to_none=True)
        smooth = F.interpolate(parameters, size=(16, 32, 32), mode="trilinear", align_corners=False)
        modulation = smooth if parameterization == "linear_normalized" else torch.tanh(smooth)
        if smoothing_kernel > 1:
            modulation = F.avg_pool3d(
                modulation,
                kernel_size=smoothing_kernel,
                stride=1,
                padding=smoothing_kernel // 2,
            )
        if parameterization == "linear_normalized":
            active = (fractions > 0).to(modulation.dtype)
            rms = torch.sqrt(
                (modulation.square() * active).sum((1, 2, 3, 4), keepdim=True)
                / active.sum((1, 2, 3, 4), keepdim=True).clamp_min(1.0)
            ).clamp_min(1e-6)
            modulation = modulation / rms
        direction = fractions.reshape(batch, -1) * modulation.reshape(batch, -1)
        response = (direction @ g6.T).reshape(batch, 1089, 6).permute(0, 2, 1)
        e0 = (response[:, 5] / sigma0[:, None]).square().mean(1).clamp_min(1e-12)
        e1 = (response[:, [2, 4]] / sigma1[:, :, None]).square().mean((1, 2)).clamp_min(1e-12)
        q = torch.stack(((response[:, 0] - response[:, 3]) / 2.0, response[:, 1]), dim=1)
        e2 = (q / sigma2[:, :, None]).square().mean((1, 2)).clamp_min(1e-12)
        density_energy = (
            direction.square().mean(1)
            / base_density.square().mean(1).clamp_min(1e-12)
        ).clamp_min(1e-12)
        loss = torch.zeros(batch, device=device)
        spin1 = group_code == 0
        spin2 = group_code == 1
        mixed = group_code == 2
        if spin1.any():
            loss[spin1] = -null_ratio_weight * torch.log(e1[spin1] / e0[spin1])
            loss[spin1] -= dominance_weight * torch.log(e1[spin1] / e2[spin1])
            loss[spin1] += 12.0 * torch.relu(torch.sqrt(e2[spin1] / e1[spin1]) - 1.0 / 1.55).square()
        if spin2.any():
            loss[spin2] = -null_ratio_weight * torch.log(e2[spin2] / e0[spin2])
            loss[spin2] -= dominance_weight * torch.log(e2[spin2] / e1[spin2])
            loss[spin2] += 12.0 * torch.relu(torch.sqrt(e1[spin2] / e2[spin2]) - 1.0 / 1.55).square()
        if mixed.any():
            ratio = torch.sqrt(e1[mixed] / e2[mixed])
            balance = torch.maximum(ratio, 1.0 / ratio)
            loss[mixed] = -null_ratio_weight * torch.log(
                (e1[mixed] + e2[mixed]) / e0[mixed]
            )
            loss[mixed] += 16.0 * torch.log(ratio).square()
            loss[mixed] += 8.0 * torch.relu(balance - 1.42).square()
        if density_null_weight:
            loss -= density_null_weight * torch.log(density_energy / e0)
        # Low-resolution variables already impose the minimum feature size;
        # a small TV term suppresses residual grid-aligned oscillation.
        tv = (
            (parameters[:, :, 1:] - parameters[:, :, :-1]).square().mean((1, 2, 3, 4))
            + (parameters[:, :, :, 1:] - parameters[:, :, :, :-1]).square().mean((1, 2, 3, 4))
            + (parameters[:, :, :, :, 1:] - parameters[:, :, :, :, :-1]).square().mean((1, 2, 3, 4))
        )
        objective = (loss + float(contract.get("optimizer_tv_weight", 1e-5)) * tv).mean()
        objective.backward()
        optimizer.step()
        last = (e0.detach(), e1.detach(), e2.detach())
    assert last is not None
    with torch.no_grad():
        smooth = F.interpolate(parameters, size=(16, 32, 32), mode="trilinear", align_corners=False)
        modulation = smooth if parameterization == "linear_normalized" else torch.tanh(smooth)
        if smoothing_kernel > 1:
            modulation = F.avg_pool3d(
                modulation,
                kernel_size=smoothing_kernel,
                stride=1,
                padding=smoothing_kernel // 2,
            )
        if parameterization == "linear_normalized":
            active = (fractions > 0).to(modulation.dtype)
            peak = (modulation.abs() * active).amax((1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
            modulation = modulation / peak
        direction = (fractions * modulation).squeeze(1).cpu().numpy().astype(np.float32)
        e0, e1, e2 = last
        diagnostics = [
            {
                "linearized_spin1_to_tzz": float(torch.sqrt(e1[i] / e0[i]).cpu()),
                "linearized_spin2_to_tzz": float(torch.sqrt(e2[i] / e0[i]).cpu()),
                "linearized_spin1_to_spin2": float(torch.sqrt(e1[i] / e2[i]).cpu()),
                "optimizer_steps": int(contract["local_search_steps"]),
                "optimization_grid_zyx": list(optimization_grid),
                "direction_smoothing_kernel": smoothing_kernel,
                "modulation_parameterization": parameterization,
                "null_ratio_objective_weight": null_ratio_weight,
                "dominance_objective_weight": dominance_weight,
                "density_null_objective_weight": density_null_weight,
            }
            for i in range(batch)
        ]
    return direction, diagnostics
