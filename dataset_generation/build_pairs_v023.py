#!/usr/bin/env python3
'Generate separable pairs on identical supports with equal total anomalous mass. Smooth Gaussian directions are volume-fraction demeaned, then scaled. Require Tzz d-prime >=5, max spin d-prime >=5 and symmetric density NRMSE >=0.12. Difficulty-bin targets may be relaxed and are recorded; spin dominance is a label, not a sampling quota.'
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from v017.ambiguity import (
    ABSOLUTE_COVARIANCE_METRIC_VERSION,
    pair_metrics_absolute_covariance_v019,
    support_pair_metrics,
)
from v017.config import (
    load_config,
    resolve_project_path,
    sha256_file,
    stable_hash,
    write_json,
)
from v017.geometry import SobolStream, generate_model, geometry_metrics
from v017.physics import (
    GravityTensorOperator,
    independent_prism_audit,
    response_sha256,
)
from v017.storage import save_split_shards
from v023_noise import V023WhiteNoise


ROOT = Path(__file__).resolve().parent
STRATA = ("small_0p5_2pct", "main_2_8pct", "large_8_18pct")
V022_SEEDS = (
    2026082501, 2026082611, 2026082612, 2026082613,
    2026082691, 2026082692, 2026082693,
    2026082701, 2026082702, 2026082703, 2026082704,
)


# --------------------------------------------------------------------------- #
# Configuration validation and seed registry.
# --------------------------------------------------------------------------- #

def validate_v023_config(config: dict[str, Any]) -> None:
    if str(config.get("schema_version")) != "23.0.0":
        raise ValueError("build_pairs_v023.py requires schema_version=23.0.0")
    amb = config["ambiguity"]
    if str(amb.get("metric_version")) != ABSOLUTE_COVARIANCE_METRIC_VERSION:
        raise ValueError("V023 requires the absolute covariance d-prime metric")
    if str(amb.get("gate_type")) != "reverse_distinguishable_v023":
        raise ValueError("V023 requires gate_type=reverse_distinguishable_v023")
    ref = np.asarray(amb["reference_component_sigma_eotvos"], dtype=np.float64)
    phys = np.asarray(config["noise"]["base_component_sigma_eotvos"], dtype=np.float64)
    if ref.shape != (6,) or not np.array_equal(ref, phys):
        raise ValueError(
            "whitening reference sigma must equal physical noise sigma: "
            f"{ref} vs {phys}"
        )
    if float(config["noise"].get("frozen_amplitude_scale", 1.0)) != 1.0:
        raise ValueError("V023 frozen noise amplitude must be fixed at 1.0")
    bins = [(float(lo), float(hi)) for lo, hi in amb["dprime_bins"]]
    if bins[0][0] != float(amb["tzz_dprime_min"]):
        raise ValueError("dprime_bins[0] must start at tzz_dprime_min")
    if any(bins[i][1] != bins[i + 1][0] for i in range(len(bins) - 1)):
        raise ValueError("dprime_bins must be contiguous")
    if len(bins) != len(amb["dprime_bin_mixture"]):
        raise ValueError("dprime_bins/mixture length mismatch")
    if not np.isclose(sum(float(v) for v in amb["dprime_bin_mixture"]), 1.0):
        raise ValueError("ambiguity.dprime_bin_mixture does not sum to one")
    for name in ("pair_mechanism_mixture", "pair_support_mixture",
                 "pair_geometry_mode_mixture"):
        if not np.isclose(sum(float(v) for v in amb[name].values()), 1.0):
            raise ValueError(f"ambiguity.{name} does not sum to one")
    base_lo, base_hi = (float(v) for v in amb["base_contrast_gcc_range"])
    amp_lo, amp_hi = (float(v) for v in amb["redistribution_amplitude_gcc_range"])
    peak_cap = float(config["density"]["absolute_contrast_gcc_range"][1])
    if base_lo <= 0.0 or base_hi > peak_cap or amp_lo <= 0.0:
        raise ValueError("invalid base_contrast/amplitude ranges")
    # Keep scale_max*amp_hi < base_lo so endpoint signs cannot change.
    if float(amb["scale_search_max"]) * amp_hi >= base_lo:
        raise ValueError(
            "scale_search_max*amplitude_hi must stay below base_contrast_lo "
            "to keep endpoint density single-sign by construction"
        )
    seeds = [int(spec["seed"]) for spec in config["splits"].values()]
    seeds += [int(config["smoke"]["seed"]), int(config["noise"]["training_seed"])]
    seeds += [int(v) for v in config["noise"]["frozen_evaluation_seeds"].values()]
    overlap = sorted(set(seeds) & set(V022_SEEDS))
    if overlap:
        raise ValueError(f"V023 seeds overlap V022 seed registry: {overlap}")


def seed_registry_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": config["dataset_id"],
        "used_seeds": {
            "splits": {k: int(v["seed"]) for k, v in config["splits"].items()},
            "noise": {
                "training": int(config["noise"]["training_seed"]),
                "frozen_evaluation": {
                    k: int(v) for k, v in config["noise"]["frozen_evaluation_seeds"].items()
                },
            },
            "smoke": int(config["smoke"]["seed"]),
        },
        "v022_seeds_do_not_use": list(V022_SEEDS),
    }


# --------------------------------------------------------------------------- #
# Self-contained helpers inherited from earlier generators.
# --------------------------------------------------------------------------- #

def allocate(total: int, mixture: dict[str, float]) -> dict[str, int]:
    raw = {key: int(total) * float(value) for key, value in mixture.items()}
    result = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = int(total) - sum(result.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def exact_labels(counts: dict[str, int], rng: np.random.Generator) -> list[str]:
    values = [key for key, count in counts.items() for _ in range(int(count))]
    rng.shuffle(values)
    return values


def array_hash(value: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(
        np.asarray(value, dtype=dtype, order="C").tobytes()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Separable-pair acceptance gates and d-prime bins.
# --------------------------------------------------------------------------- #

def classify_pair_reverse_v023(
    metrics: dict[str, float], contract: dict[str, Any]
) -> str | None:
    'Require distinguishability in Tzz and at least one complementary spin group.'
    d0 = float(metrics["tzz_global_dprime"])
    d1 = float(metrics["spin1_global_dprime"])
    d2 = float(metrics["spin2_global_dprime"])
    hard = (
        d0 >= float(contract["tzz_dprime_min"])
        and max(d1, d2) >= float(contract["spin_dprime_min"])
        and float(metrics["density_nrmse"]) >= float(contract["density_nrmse_min"])
    )
    return "distinguishable" if hard else None


def dprime_bin_index(contract: dict[str, Any], dprime: float) -> int | None:
    bins = contract["dprime_bins"]
    for i, (lo, hi) in enumerate(bins):
        upper = float(hi) + (1e-12 if i == len(bins) - 1 else 0.0)
        if float(lo) <= float(dprime) < upper:
            return i
    return None


def response_group_label(metrics: dict[str, float], contract: dict[str, Any]) -> str:
    'Label observed spin dominance without imposing spin quotas.'
    d1 = float(metrics["spin1_global_dprime"])
    d2 = float(metrics["spin2_global_dprime"])
    ratio = float(contract.get("dominance_ratio_for_group_label", 1.5))
    if d1 / max(d2, 1e-12) >= ratio:
        return "spin1"
    if d2 / max(d1, 1e-12) >= ratio:
        return "spin2"
    return "mixed"


# --------------------------------------------------------------------------- #
# Target a 10 percent deep quota (support centroid >=600 m).
# --------------------------------------------------------------------------- #

DEEP_INFEASIBLE_MECHANISM = "B_structural"
DEEP_INFEASIBLE_STRATUM = "large_8_18pct"


def deep_target_config(config: dict[str, Any]) -> dict[str, Any]:
    'Copy the geometry configuration and enable deep targeting.'
    deep_cfg = config.get("deep_targeting", {})
    result = dict(config)
    result["geometry"] = {
        "target_min_centroid_depth_km": float(
            deep_cfg.get("target_min_centroid_depth_km", 0.6)
        ),
        "target_max_centroid_depth_km": float(
            deep_cfg.get("target_max_centroid_depth_km", 0.95)
        ),
    }
    return result


def assign_deep_flags(count: int, deep_fraction: float, seed: int) -> list[bool]:
    n_deep = int(round(count * float(deep_fraction)))
    flags = [True] * n_deep + [False] * (count - n_deep)
    rng = np.random.default_rng(int(seed) + 5)
    rng.shuffle(flags)
    return flags


def repair_deep_mechanisms(
    mechanisms: list[str], strata: list[str], deep_flags: list[bool]
) -> tuple[list[str], int]:
    'Swap infeasible deep B_structural/large labels while preserving marginal counts.'
    mechanisms = list(mechanisms)
    swaps = 0
    n = len(mechanisms)
    donor = 0
    for index in range(n):
        if not (
            deep_flags[index]
            and strata[index] == DEEP_INFEASIBLE_STRATUM
            and mechanisms[index] == DEEP_INFEASIBLE_MECHANISM
        ):
            continue
        while True:
            donor += 1
            candidate = (index + donor) % n
            if (
                deep_flags[candidate]
                and mechanisms[candidate] != DEEP_INFEASIBLE_MECHANISM
                and strata[candidate] != DEEP_INFEASIBLE_STRATUM
            ):
                mechanisms[index], mechanisms[candidate] = (
                    mechanisms[candidate], mechanisms[index],
                )
                swaps += 1
                break
            if donor > 2 * n:
                raise RuntimeError("no feasible donor for deep schedule repair")
    return mechanisms, swaps


# --------------------------------------------------------------------------- #
# Task cards preserve mechanism, support-stratum and difficulty marginals.
# --------------------------------------------------------------------------- #

def pair_task_cards(
    config: dict[str, Any], count: int, seed: int, deep_fraction: float = 0.0
) -> list[dict[str, Any]]:
    amb = config["ambiguity"]
    rng = np.random.default_rng(int(seed))
    mechanisms = exact_labels(allocate(count, amb["pair_mechanism_mixture"]), rng)
    strata = exact_labels(allocate(count, amb["pair_support_mixture"]), rng)
    deep_flags = (
        assign_deep_flags(count, deep_fraction, seed) if deep_fraction > 0.0
        else [False] * count
    )
    if deep_fraction > 0.0:
        mechanisms, _ = repair_deep_mechanisms(mechanisms, strata, deep_flags)
    modes = exact_labels(allocate(count, amb["pair_geometry_mode_mixture"]), rng)
    bin_counts = allocate(
        count, {str(i): float(p) for i, p in enumerate(amb["dprime_bin_mixture"])}
    )
    bins = [int(k) for k, c in bin_counts.items() for _ in range(c)]
    rng.shuffle(bins)
    cards = []
    for i in range(count):
        lo, hi = (float(v) for v in amb["dprime_bins"][bins[i]])
        # Draw targets in [lo,2*lo) for the open-ended [40,infinity) bin.
        target_hi = float(lo) * 2.0 if np.isinf(hi) else float(hi)
        cards.append({
            "target_id": f"pair_{i:05d}",
            "mechanism": mechanisms[i],
            "support_stratum": strata[i],
            "pair_geometry_mode": modes[i],
            "deep": bool(deep_flags[i]),
            "target_dprime_bin": int(bins[i]),
            "target_tzz_dprime": float(rng.uniform(lo, target_hi)),
            "base_contrast_gcc": float(
                rng.uniform(*amb["base_contrast_gcc_range"])
            ),
            "redistribution_amplitude_gcc": float(
                rng.uniform(*amb["redistribution_amplitude_gcc_range"])
            ),
            "sign": 1.0 if rng.random() < float(config["density"]["positive_probability"]) else -1.0,
            "attempts": 0,
        })
    rng.shuffle(cards)
    return cards


# --------------------------------------------------------------------------- #
# Smooth random directions replace optimization for separable pairs.
# --------------------------------------------------------------------------- #

def random_smooth_direction(
    fraction: np.ndarray,
    rng: np.random.Generator,
    contract: dict[str, Any],
) -> np.ndarray | None:
    'Upsample coarse Gaussian noise, smooth with kernel 3, remove the volume-fraction weighted mean, and normalize the support peak. The zero-mass direction redistributes density without changing total anomalous mass.'
    coarse = tuple(int(v) for v in contract["direction_coarse_grid_zyx"])
    field = torch.from_numpy(
        rng.standard_normal((1, 1, *coarse)).astype(np.float32)
    )
    up = F.interpolate(field, size=fraction.shape, mode="trilinear", align_corners=False)
    kernel = int(contract["direction_smoothing_kernel"])
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("direction_smoothing_kernel must be a positive odd integer")
    for _ in range(int(contract.get("direction_smooth_passes", 1))):
        up = F.avg_pool3d(up, kernel_size=kernel, stride=1, padding=kernel // 2)
    modulation = up[0, 0].numpy().astype(np.float64)
    mask = fraction > 0.0
    modulation = modulation * mask
    total = float(np.sum(fraction))
    if total <= 0.0:
        return None
    mass = float(np.sum(fraction.astype(np.float64) * modulation))
    modulation = modulation - mass / total
    peak = float(np.max(np.abs(modulation[mask])))
    if peak < 1e-6:
        return None
    modulation = (modulation / peak).astype(np.float32)
    return (fraction * modulation).astype(np.float32)


# --------------------------------------------------------------------------- #
# Vectorized scale screening and endpoint contracts.
# --------------------------------------------------------------------------- #

def _endpoint_contract_ok(
    density: np.ndarray,
    stratum: str,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    active = np.abs(density) >= threshold
    values = density[active]
    if not len(values):
        return False, {"reason": "empty_support"}
    if np.any(values > 0.0) and np.any(values < 0.0):
        return False, {"reason": "mixed_density_sign"}
    stats = geometry_metrics(density, threshold, config)
    low, high = (
        float(value) for value in config["support_fraction_mixture"][stratum]["range"]
    )
    quality = config["geometry_quality"]
    checks = {
        "support_stratum": low <= float(stats["support_fraction"]) <= high,
        "minimum_support_cells": int(stats["support_cells"]) >= int(quality["minimum_support_cells"]),
        "minimum_horizontal_feature": int(stats["minimum_component_horizontal_span_cells"])
        >= int(quality["minimum_horizontal_feature_cells"]),
        "lateral_margin": int(stats["minimum_lateral_margin_cells"])
        >= int(quality["lateral_margin_cells"]),
        "shallow_margin": int(stats["minimum_shallow_margin_cells"])
        >= int(quality["shallow_margin_cells"]),
        "contrast_upper_bound": float(np.max(np.abs(values)))
        <= float(config["density"]["absolute_contrast_gcc_range"][1]) + 1e-6,
    }
    failed = [key for key, passed in checks.items() if not passed]
    return not failed, {"reason": failed[0] if failed else "pass", "geometry": stats}


def evaluate_proposal(
    config: dict[str, Any],
    basis_response: tuple[np.ndarray, np.ndarray],
    basis_density: tuple[np.ndarray, np.ndarray],
    card: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter]:
    'Vectorized screening over scale. Density and response endpoints are common +/- scale times signed direction.'
    amb = config["ambiguity"]
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    mode = str(card["pair_geometry_mode"])
    common, signed_direction = basis_density
    common_response, direction_response = basis_response
    sigma = np.asarray(amb["reference_component_sigma_eotvos"], dtype=np.float64)
    scales = np.linspace(
        float(amb["scale_search_min"]),
        float(amb["scale_search_max"]),
        int(amb["scale_search_steps"]),
    )
    peak_cap = float(config["density"]["absolute_contrast_gcc_range"][1]) + 1e-6
    min_cells = int(config["geometry_quality"]["minimum_support_cells"])
    stratum_low, stratum_high = (
        float(v) for v in config["support_fraction_mixture"][card["support_stratum"]]["range"]
    )

    # Closed-form vectorized d-prime and density NRMSE.
    d = direction_response.astype(np.float64)
    k0 = 2.0 * float(np.linalg.norm(d[5])) / sigma[5]
    k1 = 2.0 * float(np.sqrt(
        np.sum((d[2] / sigma[2]) ** 2) + np.sum((d[4] / sigma[4]) ** 2)
    ))
    dq1 = (d[0] - d[3]) / 2.0
    sigma_q1 = float(np.sqrt((sigma[0] ** 2 + sigma[3] ** 2) / 4.0))
    k2 = 2.0 * float(np.sqrt(
        np.sum((dq1 / sigma_q1) ** 2) + np.sum((d[1] / sigma[1]) ** 2)
    ))
    p_base = float(np.mean(common.astype(np.float64) ** 2))
    q_dir = float(np.mean(signed_direction.astype(np.float64) ** 2))

    d0 = k0 * scales
    d1 = k1 * scales
    d2 = k2 * scales
    nrmse = (
        2.0 * scales * np.sqrt(q_dir) / np.sqrt(p_base + scales**2 * q_dir)
        if q_dir > 0.0 else np.zeros_like(scales)
    )
    gate = (
        (d0 >= float(amb["tzz_dprime_min"]))
        & (np.maximum(d1, d2) >= float(amb["spin_dprime_min"]))
        & (nrmse >= float(amb["density_nrmse_min"]))
    )

    # Vectorized peak, sign, support and Jaccard checks.
    s_col = scales[:, None, None, None].astype(np.float32)
    first_all = common[None] + s_col * signed_direction[None]
    second_all = common[None] - s_col * signed_direction[None]
    active_first = np.abs(first_all) >= threshold
    active_second = np.abs(second_all) >= threshold
    sign_ok = (
        (first_all >= 0.0).all(axis=(1, 2, 3)) | (first_all <= 0.0).all(axis=(1, 2, 3))
    ) & (
        (second_all >= 0.0).all(axis=(1, 2, 3)) | (second_all <= 0.0).all(axis=(1, 2, 3))
    )
    peak_ok = (
        (np.abs(first_all).max(axis=(1, 2, 3)) <= peak_cap)
        & (np.abs(second_all).max(axis=(1, 2, 3)) <= peak_cap)
    )
    cells_first = active_first.sum(axis=(1, 2, 3))
    cells_second = active_second.sum(axis=(1, 2, 3))
    cells_ok = (cells_first >= min_cells) & (cells_second >= min_cells)
    frac_first = cells_first / float(active_first[0].size)
    frac_second = cells_second / float(active_second[0].size)
    stratum_ok = (
        (frac_first >= stratum_low) & (frac_first <= stratum_high)
        & (frac_second >= stratum_low) & (frac_second <= stratum_high)
    )
    jaccard_ok = (active_first ^ active_second).sum(axis=(1, 2, 3)) == 0
    if mode != "shared_support":
        jaccard_ok = np.ones_like(jaccard_ok)  # Variable-support checks are deferred to full candidate validation.

    rejects: Counter[str] = Counter()
    passing: list[dict[str, Any]] = []
    for i, scale in enumerate(scales):
        if not gate[i]:
            if d0[i] < float(amb["tzz_dprime_min"]):
                rejects["tzz_gate"] += 1
            elif np.maximum(d1, d2)[i] < float(amb["spin_dprime_min"]):
                rejects["spin_gate"] += 1
            else:
                rejects["nrmse_gate"] += 1
            continue
        if not sign_ok[i]:
            rejects["single_sign"] += 1
            continue
        if not peak_ok[i]:
            rejects["peak_cap"] += 1
            continue
        if not cells_ok[i] or not stratum_ok[i]:
            rejects["support_stratum"] += 1
            continue
        if not jaccard_ok[i]:
            rejects["jaccard"] += 1
            continue
        passing.append({
            "scale": float(scale),
            "tzz_global_dprime": float(d0[i]),
            "dprime_bin": dprime_bin_index(amb, float(d0[i])),
        })
    return passing, rejects


def select_from_passing_set(
    passing: list[dict[str, Any]],
    card: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    'Prefer the requested d-prime bin and nearest target; record fallback if empty.'
    if not passing:
        return [], ["empty_passing_set"]
    degradation: list[str] = []
    in_bin = [c for c in passing if c["dprime_bin"] == int(card["target_dprime_bin"])]
    pool = in_bin if in_bin else passing
    if not in_bin:
        degradation.append("dprime_bin_relaxed")
    pool = sorted(
        pool,
        key=lambda c: abs(
            float(c["tzz_global_dprime"]) - float(card["target_tzz_dprime"])
        ),
    )
    return pool, degradation


# --------------------------------------------------------------------------- #
# Main pair-generation loop.
# --------------------------------------------------------------------------- #

def build_pairs(
    config: dict[str, Any],
    operator: GravityTensorOperator,
    split: str,
    cards: list[dict[str, Any]],
    seed: int,
    batch_size: int,
    progress_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    amb = config["ambiguity"]
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    jitter = float(amb.get("proposal_jitter_fraction", 0.0))
    rng = np.random.default_rng(int(seed))
    support_sobol = SobolStream(int(seed) + 314159)
    config_deep = deep_target_config(config)
    pending = list(cards)
    for spec in pending:
        spec["failure_history"] = Counter()
        spec["full_passing_seen"] = 0
    accepted: list[dict[str, Any]] = []
    proposals = 0
    batches = 0
    failure_reasons: Counter[str] = Counter()
    scale_reject_reasons: Counter[str] = Counter()
    degradation_log: list[dict[str, Any]] = []
    exhausted_targets: list[dict[str, Any]] = []
    maximum_attempts = int(amb["maximum_proposals_per_target"])
    early_attempts = int(amb.get("early_exhaustion_attempts", maximum_attempts))
    started = time.time()
    while pending:
        batches += 1
        batch_specs = pending[:batch_size]
        fractions: list[np.ndarray] = []
        directions: list[np.ndarray | None] = []
        source_records: list[dict[str, Any]] = []
        for spec in batch_specs:
            model = generate_model(
                rng, support_sobol,
                config_deep if spec.get("deep") else config,
                str(spec["mechanism"]),
                tuple(config["support_fraction_mixture"][spec["support_stratum"]]["range"]),
            )
            fractions.append(model.volume_fraction)
            source_records.append(model.record)
            directions.append(
                random_smooth_direction(model.volume_fraction, rng, amb)
            )
        # Build jittered bases and batch their forward responses.
        bases: list[tuple[np.ndarray, np.ndarray] | None] = []
        actual_params: list[tuple[float, float] | None] = []
        forward_stack: list[np.ndarray] = []
        forward_index: list[int] = []
        for i, spec in enumerate(batch_specs):
            direction = directions[i]
            if direction is None:
                bases.append(None)
                actual_params.append(None)
                continue
            base_lo, base_hi = (float(v) for v in amb["base_contrast_gcc_range"])
            amp_lo, amp_hi = (float(v) for v in amb["redistribution_amplitude_gcc_range"])
            base_contrast = float(np.clip(
                float(spec["base_contrast_gcc"]) * rng.uniform(1.0 - jitter, 1.0 + jitter),
                base_lo, base_hi,
            ))
            amplitude = float(np.clip(
                float(spec["redistribution_amplitude_gcc"]) * rng.uniform(1.0 - jitter, 1.0 + jitter),
                amp_lo, amp_hi,
            ))
            sign = float(spec["sign"])
            common = (sign * base_contrast * fractions[i]).astype(np.float32)
            signed_direction = (sign * amplitude * direction).astype(np.float32)
            bases.append((common, signed_direction))
            actual_params.append((base_contrast, amplitude))
            forward_stack.extend((common, signed_direction))
            forward_index.append(i)
        basis_responses: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(batch_specs)
        if forward_stack:
            responses = operator.forward(
                np.stack(forward_stack), batch_size=max(2 * len(batch_specs), 2)
            )
            for j, i in enumerate(forward_index):
                basis_responses[i] = (responses[2 * j], responses[2 * j + 1])
        completed: set[str] = set()
        for i, spec in enumerate(batch_specs):
            proposals += 1
            spec["attempts"] = int(spec["attempts"]) + 1
            if bases[i] is None or basis_responses[i] is None:
                failure_reasons["direction_degenerate"] += 1
                spec["failure_history"]["direction_degenerate"] += 1
                continue
            passing, rejects = evaluate_proposal(
                config, basis_responses[i], bases[i], spec
            )
            scale_reject_reasons.update(rejects)
            spec["full_passing_seen"] += len(passing)
            pool, degradation = select_from_passing_set(passing, spec)
            chosen: dict[str, Any] | None = None
            chosen_payload: dict[str, Any] | None = None
            for candidate in pool:
                scale = float(candidate["scale"])
                common, signed_direction = bases[i]
                common_response, direction_response = basis_responses[i]
                first = (common + scale * signed_direction).astype(np.float32)
                second = (common - scale * signed_direction).astype(np.float32)
                responses = np.stack((
                    common_response + scale * direction_response,
                    common_response - scale * direction_response,
                )).astype(np.float32)
                metrics = pair_metrics_absolute_covariance_v019(
                    first, second, responses[0], responses[1], amb
                )
                if classify_pair_reverse_v023(metrics, amb) is None:
                    scale_reject_reasons["final_gate_recheck"] += 1
                    continue
                sm = support_pair_metrics(first, second, threshold)
                if str(spec["pair_geometry_mode"]) == "shared_support" and float(
                    sm["support_jaccard"]
                ) != 1.0:
                    scale_reject_reasons["final_jaccard_recheck"] += 1
                    continue
                ok_a, detail_a = _endpoint_contract_ok(
                    first, str(spec["support_stratum"]), config
                )
                ok_b, detail_b = _endpoint_contract_ok(
                    second, str(spec["support_stratum"]), config
                )
                if not (ok_a and ok_b):
                    scale_reject_reasons[
                        f"endpoint_{detail_a['reason'] if not ok_a else detail_b['reason']}"
                    ] += 1
                    continue
                chosen = candidate
                chosen_payload = {
                    "densities": (first, second),
                    "responses": responses,
                    "metrics": metrics,
                    "endpoint_geometry": (detail_a["geometry"], detail_b["geometry"]),
                }
                break
            if chosen is None:
                reason = degradation[0] if degradation else "no_candidate"
                failure_reasons[reason] += 1
                spec["failure_history"][reason] += 1
                early = int(spec["attempts"]) >= early_attempts and int(
                    spec["full_passing_seen"]
                ) == 0
                if early or int(spec["attempts"]) >= maximum_attempts:
                    exhausted_targets.append({
                        "target_id": str(spec["target_id"]),
                        "attempts": int(spec["attempts"]),
                        "exhaustion_mode": "early_infeasible" if early else "max_attempts",
                        "card": {k: v for k, v in spec.items() if k != "attempts"},
                    })
                    completed.add(str(spec["target_id"]))
                continue
            if degradation:
                degradation_log.append({
                    "target_id": str(spec["target_id"]),
                    "degradation": degradation,
                    "target_dprime_bin": int(spec["target_dprime_bin"]),
                    "actual_dprime_bin": chosen["dprime_bin"],
                })
            accepted.append({
                "card": spec,
                "densities": chosen_payload["densities"],
                "responses": chosen_payload["responses"],
                "volume_fraction": fractions[i],
                "metrics": chosen_payload["metrics"],
                "scale": float(chosen["scale"]),
                "dprime_bin": chosen["dprime_bin"],
                "selection_degradation": degradation,
                "source_record": source_records[i],
                "base_contrast_gcc_actual": float(actual_params[i][0]),
                "redistribution_amplitude_gcc_actual": float(actual_params[i][1]),
                "generation_attempts_for_target": int(spec["attempts"]),
            })
            completed.add(str(spec["target_id"]))
        pending = [s for s in pending if str(s["target_id"]) not in completed]
        if pending:
            rotation = min(len(batch_specs), len(pending))
            pending = pending[rotation:] + pending[:rotation]
        if progress_path is not None:
            write_json(progress_path, {
                "split": split,
                "batch": batches,
                "accepted": len(accepted),
                "target_pairs": len(cards),
                "proposals": proposals,
                "elapsed_seconds": time.time() - started,
                "acceptance_rate_so_far": len(accepted) / max(proposals, 1),
                "failure_reasons": dict(failure_reasons),
            })
        if batches == 1 or batches % 5 == 0 or not pending:
            print(
                f"[{time.strftime('%F %T')}] V023 {split}: accepted "
                f"{len(accepted)}/{len(cards)}, proposals={proposals}, "
                f"exhausted={len(exhausted_targets)}, "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    elapsed = time.time() - started
    funnel = {
        "split": split,
        "target_pairs": len(cards),
        "accepted_pairs": len(accepted),
        "shortfall": len(cards) - len(accepted),
        "proposals": proposals,
        "acceptance_rate": len(accepted) / max(proposals, 1),
        "proposals_per_accepted_pair": proposals / max(len(accepted), 1),
        "seconds_per_accepted_pair": elapsed / max(len(accepted), 1),
        "proposals_per_second": proposals / max(elapsed, 1e-9),
        "failure_reasons": dict(failure_reasons),
        "scale_level_reject_reasons": dict(scale_reject_reasons),
        "selection_degradation_count": len(degradation_log),
        "selection_degradation_log": degradation_log[:500],
        "exhausted_targets": exhausted_targets,
        "attempts_distribution": {
            "max": max((int(s["attempts"]) for s in cards), default=0),
            "mean": float(np.mean([int(s["attempts"]) for s in cards])) if cards else 0.0,
        },
        "metric_version": ABSOLUTE_COVARIANCE_METRIC_VERSION,
        "elapsed_seconds": elapsed,
    }
    return accepted, funnel


# --------------------------------------------------------------------------- #
# Assemble canonical endpoint metadata.
# --------------------------------------------------------------------------- #

def _refresh_record(
    source: dict[str, Any],
    density: np.ndarray,
    volume_fraction: np.ndarray,
    response: np.ndarray,
    config: dict[str, Any],
    split: str,
    sample_id: str,
    support_stratum: str,
) -> dict[str, Any]:
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    result = dict(source)
    generation_contrast = result.pop("contrast_gcc", None)
    fresh_geometry = geometry_metrics(density, threshold, config)
    for key in fresh_geometry:
        result.pop(key, None)
    result.update(fresh_geometry)
    active = np.abs(density) >= threshold
    peak = float(np.max(np.abs(density[active])))
    result.update({
        "sample_id": sample_id,
        "split": split,
        "support_stratum": support_stratum,
        "generation_contrast_gcc": (
            None if generation_contrast is None else float(generation_contrast)
        ),
        "contrast_gcc": peak,
        "peak_abs_density_gcc": peak,
        "density_sign": 1 if bool(np.all(density[active] > 0.0)) else -1,
        "density_sha256": array_hash(density, "<f4"),
        "support_sha256": array_hash(active, "|u1"),
        "volume_fraction_sha256": array_hash(volume_fraction, "<f2"),
        "clean_d6_sha256": response_sha256(response),
    })
    result["prototype_id"] = result["volume_fraction_sha256"][:24]
    if not result.get("base_prototype_id"):
        result["base_prototype_id"] = result["prototype_id"]
    return result


def assemble_pairs_split(
    config: dict[str, Any],
    split: str,
    seed: int,
    pair_items: list[dict[str, Any]],
    output: Path,
    save_frozen: bool,
    id_infix: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.time()
    amb = config["ambiguity"]
    endpoints: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    pair_counter: Counter[tuple[str, str]] = Counter()
    for item in pair_items:
        card = item["card"]
        stratum = str(card["support_stratum"])
        group = response_group_label(item["metrics"], amb)
        pair_number = pair_counter[(stratum, group)]
        pair_counter[(stratum, group)] += 1
        pair_id = (
            f"{split}_{stratum}_{group}_{pair_number:05d}_"
            f"{stable_hash([seed, stratum, group, pair_number], 10)}"
        )
        base_prototype_id = array_hash(item["volume_fraction"], "<f2")[:24]
        for endpoint_index, role in enumerate(("A", "B")):
            source = dict(item["source_record"])
            source.update({
                "mechanism": str(card["mechanism"]),
                "support_mechanism": str(card["mechanism"]),
                "support_subtype": str(item["source_record"]["subtype"]),
                "subtype": str(amb.get("pair_subtype", "density_redistribution")),
                "density_pattern": "random_smooth_redistribution",
                "ambiguity_group_id": pair_id,
                "ambiguity_pair_role": role,
                "response_group": group,
                "pair_geometry_mode": str(card["pair_geometry_mode"]),
                "pair_metrics": item["metrics"],
                "density_redistribution_scale": float(item["scale"]),
                "target_tzz_global_dprime": float(card["target_tzz_dprime"]),
                "tzz_target_absolute_error": abs(
                    float(item["metrics"]["tzz_global_dprime"])
                    - float(card["target_tzz_dprime"])
                ),
                "candidate_selection": "v023_passing_set_bin_targeted",
                "base_prototype_id": base_prototype_id,
                "base_contrast_gcc": float(item["base_contrast_gcc_actual"]),
                "redistribution_amplitude_gcc": float(
                    item["redistribution_amplitude_gcc_actual"]
                ),
                "target_dprime_bin": int(card["target_dprime_bin"]),
                "actual_dprime_bin": item["dprime_bin"],
                "selection_degradation": list(item["selection_degradation"]),
                "generation_attempts_for_target": int(item["generation_attempts_for_target"]),
                "direction_construction": str(amb["direction_construction"]),
                "direction_smoothing_kernel": int(amb["direction_smoothing_kernel"]),
            })
            endpoints.append({
                "density": item["densities"][endpoint_index],
                "volume_fraction": item["volume_fraction"],
                "response": item["responses"][endpoint_index],
                "record": source,
                "support_stratum": stratum,
            })
        pair_rows.append({
            "split": split,
            "ambiguity_group_id": pair_id,
            "response_group": group,
            "gate": "distinguishable",
            "mechanism": str(card["mechanism"]),
            "support_subtype": str(item["source_record"]["subtype"]),
            "support_stratum": stratum,
            "pair_geometry_mode": str(card["pair_geometry_mode"]),
            "density_redistribution_scale": float(item["scale"]),
            "base_contrast_gcc": float(item["base_contrast_gcc_actual"]),
            "redistribution_amplitude_gcc": float(item["redistribution_amplitude_gcc_actual"]),
            "direction_smoothing_kernel": int(amb["direction_smoothing_kernel"]),
            "target_tzz_global_dprime": float(card["target_tzz_dprime"]),
            "tzz_target_absolute_error": abs(
                float(item["metrics"]["tzz_global_dprime"])
                - float(card["target_tzz_dprime"])
            ),
            "target_dprime_bin": int(card["target_dprime_bin"]),
            "actual_dprime_bin": item["dprime_bin"],
            "selection_degradation": list(item["selection_degradation"]),
            "generation_attempts_for_target": int(item["generation_attempts_for_target"]),
            **item["metrics"],
        })

    rng = np.random.default_rng(seed + 90000)
    endpoints = [endpoints[int(i)] for i in rng.permutation(len(endpoints))]
    records = []
    prefix = str(config.get("sample_id_prefix", "v023"))
    for index, endpoint in enumerate(endpoints):
        sample_id = (
            f"{prefix}_{split}_{id_infix}{index:05d}_"
            f"{stable_hash([seed, index, endpoint['record']['subtype']], 10)}"
        )
        records.append(_refresh_record(
            endpoint["record"], endpoint["density"], endpoint["volume_fraction"],
            endpoint["response"], config, split, sample_id,
            endpoint["support_stratum"],
        ))

    arrays: dict[str, Any] = {
        "density_zyx": np.stack([v["density"] for v in endpoints]).astype(
            config["storage"]["density_dtype"]
        ),
        "volume_fraction_zyx": np.stack([v["volume_fraction"] for v in endpoints]).astype(np.float16),
        "clean_d6": np.stack([v["response"] for v in endpoints]).astype(
            config["storage"]["response_dtype"]
        ),
        "sample_id": np.asarray([r["sample_id"] for r in records], dtype="U64"),
        "mechanism": np.asarray([r["mechanism"] for r in records], dtype="U24"),
        "subtype": np.asarray([r["subtype"] for r in records], dtype="U32"),
        "support_subtype": np.asarray([r["support_subtype"] for r in records], dtype="U32"),
        "density_pattern": np.asarray([r["density_pattern"] for r in records], dtype="U40"),
        "ambiguity_group_id": np.asarray([r["ambiguity_group_id"] for r in records], dtype="U96"),
        "ambiguity_pair_role": np.asarray([r["ambiguity_pair_role"] for r in records], dtype="U1"),
        "response_group": np.asarray([r["response_group"] for r in records], dtype="U12"),
        "support_stratum": np.asarray([r["support_stratum"] for r in records], dtype="U24"),
        "pair_geometry_mode": np.asarray([r["pair_geometry_mode"] for r in records], dtype="U20"),
    }
    noise_check: dict[str, Any] | None = None
    if save_frozen:
        noise = V023WhiteNoise(config)
        n_real = int(config["noise"]["frozen_realizations"])
        frozen_obs = np.empty((len(records), n_real, 6, 33, 33), dtype=np.float32)
        frozen_mode = np.empty((len(records), n_real), dtype="U24")
        frozen_scale = np.empty((len(records), n_real), dtype=np.float32)
        frozen_seed = np.empty((len(records), n_real), dtype=np.uint64)
        for index, record in enumerate(records):
            clean = arrays["clean_d6"][index]
            real_meta = []
            for r in range(n_real):
                view, draw_seed = noise.frozen_realization(
                    clean, record["sample_id"], split, r
                )
                frozen_obs[index, r] = view.observed_d6.numpy()
                frozen_mode[index, r] = view.mode
                frozen_scale[index, r] = view.amplitude_scale
                frozen_seed[index, r] = draw_seed
                real_meta.append({
                    "realization": r,
                    "mode": view.mode,
                    "amplitude_scale": float(view.amplitude_scale),
                    "seed": int(draw_seed),
                })
            record["frozen_noise_realizations"] = real_meta
        arrays["frozen_observed_d6"] = frozen_obs
        arrays["frozen_noise_mode"] = frozen_mode
        arrays["frozen_noise_scale"] = frozen_scale
        arrays["frozen_noise_seed"] = frozen_seed
        residual = frozen_obs.astype(np.float64) - np.repeat(
            arrays["clean_d6"][:, None].astype(np.float64), n_real, axis=1
        )
        sigma = np.asarray(
            config["noise"]["base_component_sigma_eotvos"], dtype=np.float64
        )
        rms = np.sqrt(np.mean(residual**2, axis=(0, 1, 3, 4)))
        rel = rms / sigma - 1.0
        noise_check = {
            "components": list(config["operator"]["component_order"]),
            "configured_sigma_eotvos": [float(v) for v in sigma],
            "observed_residual_rms_eotvos": [float(v) for v in rms],
            "relative_error": [float(v) for v in rel],
            "within_tolerance": bool(
                np.all(np.abs(rel) <= float(config["audit"]["noise_rms_relative_tolerance"]))
            ),
            "tolerance": float(config["audit"]["noise_rms_relative_tolerance"]),
        }
    index_payload = save_split_shards(
        output, split, arrays, records, int(config["storage"]["shard_size"])
    )
    with (output / split / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in pair_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return index_payload, pair_rows, records, noise_check


# --------------------------------------------------------------------------- #
# Smoke diagnostics and summary reports.
# --------------------------------------------------------------------------- #

def _quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        str(q): float(np.quantile(arr, q))
        for q in (0.0, 0.25, 0.5, 0.75, 1.0)
    }


def build_summary(
    config: dict[str, Any],
    split: str,
    funnel: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    noise_check: dict[str, Any] | None,
    index_payload: dict[str, Any],
) -> dict[str, Any]:
    amb = config["ambiguity"]
    tzz = [float(r["tzz_global_dprime"]) for r in pair_rows]
    spin1 = [float(r["spin1_global_dprime"]) for r in pair_rows]
    spin2 = [float(r["spin2_global_dprime"]) for r in pair_rows]
    maxspin = [max(a, b) for a, b in zip(spin1, spin2)]
    nrmse = [float(r["density_nrmse"]) for r in pair_rows]
    scales = [float(r["density_redistribution_scale"]) for r in pair_rows]
    amps = [float(r["redistribution_amplitude_gcc"]) for r in pair_rows]
    bases = [float(r["base_contrast_gcc"]) for r in pair_rows]
    peaks = [float(r["peak_abs_density_gcc"]) for r in records]
    cells = [int(r["support_cells"]) for r in records]
    signs = {int(r["density_sign"]) for r in records}
    single_sign_ok = all(
        r["density_sign"] in (1, -1) for r in records
    )
    bins = amb["dprime_bins"]
    bin_counts = Counter(int(r["actual_dprime_bin"]) for r in pair_rows)
    n_pairs = max(len(pair_rows), 1)
    relaxed = [d for d in funnel["selection_degradation_log"]
               if "dprime_bin_relaxed" in d["degradation"]]
    relaxed_by_target_bin = Counter(
        int(d["target_dprime_bin"]) for d in relaxed
    )
    accept_by_stratum = Counter(str(r["support_stratum"]) for r in pair_rows)
    cards_by_stratum = Counter()
    summary = {
        "split": split,
        "pairs": len(pair_rows),
        "endpoints": len(records),
        "funnel": {
            "proposals": funnel["proposals"],
            "accepted_pairs": funnel["accepted_pairs"],
            "target_pairs": funnel["target_pairs"],
            "acceptance_rate": funnel["acceptance_rate"],
            "proposals_per_accepted_pair": funnel["proposals_per_accepted_pair"],
            "seconds_per_accepted_pair": funnel["seconds_per_accepted_pair"],
            "elapsed_seconds": funnel["elapsed_seconds"],
        },
        "tzz_dprime": {
            **_quantiles(tzz),
            "min_ok": bool(min(tzz, default=0.0) >= float(amb["tzz_dprime_min"])),
            "bin_edges": bins,
            "bin_counts": {str(i): bin_counts.get(i, 0) for i in range(len(bins))},
            "bin_fractions": {
                str(i): bin_counts.get(i, 0) / n_pairs for i in range(len(bins))
            },
        },
        "spin_dprime": {
            "spin1": _quantiles(spin1),
            "spin2": _quantiles(spin2),
            "max_spin": _quantiles(maxspin),
            "min_ok": bool(min(maxspin, default=0.0) >= float(amb["spin_dprime_min"])),
        },
        "density_nrmse": {
            **_quantiles(nrmse),
            "min_ok": bool(min(nrmse, default=0.0) >= float(amb["density_nrmse_min"])),
        },
        "noise_check": noise_check,
        "geometry_check": {
            "peak_abs_density_max": max(peaks, default=0.0),
            "peak_cap_gcc": float(config["density"]["absolute_contrast_gcc_range"][1]),
            "peak_ok": bool(
                max(peaks, default=0.0)
                <= float(config["density"]["absolute_contrast_gcc_range"][1]) + 1e-6
            ),
            "density_signs_present": sorted(signs),
            "single_sign_ok": bool(single_sign_ok),
            "support_cells": {
                "min": min(cells, default=0),
                "median": float(np.median(cells)) if cells else 0.0,
                "max": max(cells, default=0),
                "min_ok": bool(
                    min(cells, default=0)
                    >= int(config["geometry_quality"]["minimum_support_cells"])
                ),
            },
        },
        "scale": _quantiles(scales),
        "redistribution_amplitude_gcc": _quantiles(amps),
        "base_contrast_gcc": _quantiles(bases),
        "response_group_counts_pairs": dict(Counter(
            str(r["response_group"]) for r in pair_rows
        )),
        "mechanism_counts_pairs": dict(Counter(str(r["mechanism"]) for r in pair_rows)),
        "support_stratum_counts_pairs": dict(accept_by_stratum),
        "dprime_bin_relaxed_count": len(relaxed),
        "dprime_bin_relaxed_by_target_bin": {
            str(k): v for k, v in relaxed_by_target_bin.items()
        },
        "failure_reasons": funnel["failure_reasons"],
        "scale_level_reject_reasons": funnel["scale_level_reject_reasons"],
        "exhausted_targets": funnel["exhausted_targets"],
        "index": index_payload,
    }
    return summary


def format_report(summary: dict[str, Any]) -> str:
    f = summary["funnel"]
    lines = [
        f"V023 smoke report (split={summary['split']})",
        "=" * 60,
        f"1) proposals {f['proposals']} / accepted {f['accepted_pairs']} / target {f['target_pairs']}"
        f" | acceptance rate {f['acceptance_rate']:.3f}"
        f" | proposals per accepted pair {f['proposals_per_accepted_pair']:.2f}"
        f" | seconds per accepted pair {f['seconds_per_accepted_pair']:.2f}s"
        f" | total elapsed {f['elapsed_seconds']:.1f}s",
    ]
    t = summary["tzz_dprime"]
    lines.append(
        f"2) Tzz d′: min {t['0.0']:.2f} / median {t['0.5']:.2f} / max {t['1.0']:.2f}"
        f" | four-bin fractions {json.dumps(t['bin_fractions'], sort_keys=True)}"
        f" | all>={t['bin_edges'][0][0]}: {t['min_ok']}"
    )
    s = summary["spin_dprime"]
    lines.append(
        f"3) spin d′(max(spin1,spin2)): min {s['max_spin']['0.0']:.2f}"
        f" / median {s['max_spin']['0.5']:.2f} / max {s['max_spin']['1.0']:.2f}"
        f" | all>=5: {s['min_ok']}"
        f" | spin1 min {s['spin1']['0.0']:.2f} | spin2 min {s['spin2']['0.0']:.2f}"
    )
    n = summary["density_nrmse"]
    lines.append(
        f"4) density NRMSE: min {n['0.0']:.3f} / median {n['0.5']:.3f} / max {n['1.0']:.3f}"
        f" | all>=0.12: {n['min_ok']}"
    )
    nc = summary.get("noise_check")
    if nc:
        rel = ", ".join(f"{c}:{r:+.3%}" for c, r in zip(nc["components"], nc["relative_error"]))
        lines.append(
            f"5) noise self-check (frozen implementation observed-clean RMS vs sigma): {rel}"
            f" | within ±5%: {nc['within_tolerance']}"
        )
    else:
        lines.append("5) noise self-check: frozen implementation not written")
    g = summary["geometry_check"]
    lines.append(
        f"6) geometry contract: peak max {g['peak_abs_density_max']:.4f} (cap 0.80): {g['peak_ok']}"
        f" | single sign: {g['single_sign_ok']} (sign set {g['density_signs_present']})"
        f" | support voxels min/median/max {g['support_cells']['min']}"
        f"/{g['support_cells']['median']:.0f}/{g['support_cells']['max']}"
        f" | >=82: {g['support_cells']['min_ok']}"
    )
    sc, am = summary["scale"], summary["redistribution_amplitude_gcc"]
    lines.append(
        f"7) scale: min {sc['0.0']:.3f} / median {sc['0.5']:.3f} / max {sc['1.0']:.3f}"
        f" | redistribution amplitude: min {am['0.0']:.3f} / median {am['0.5']:.3f} / max {am['1.0']:.3f}"
    )
    lines.append(
        f"8) gate conflicts: d′ bin relaxed {summary['dprime_bin_relaxed_count']} times"
        f" (by target bin {json.dumps(summary['dprime_bin_relaxed_by_target_bin'], sort_keys=True)})"
        f" | proposal-level failures {json.dumps(summary['failure_reasons'], sort_keys=True)}"
        f" | exhausted {len(summary['exhausted_targets'])}"
    )
    lines.append(
        f"   response_group actual distribution (no quota): "
        f"{json.dumps(summary['response_group_counts_pairs'], sort_keys=True)}"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command-line workflow.
# --------------------------------------------------------------------------- #

def build(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    validate_v023_config(config)
    if args.seed is not None:
        cli_seed = int(args.seed)
        if cli_seed in V022_SEEDS:
            raise ValueError(f"--seed {cli_seed} collides with V022 registry")
    deep_fraction = (
        float(args.deep_fraction) if args.deep_fraction is not None else 0.0
    )
    if args.mode == "smoke":
        config["ambiguity"]["maximum_proposals_per_target"] = int(
            config["smoke"]["maximum_proposals_per_target"]
        )
    n_pairs = int(args.n_pairs) if args.n_pairs else (
        int(config["smoke"]["default_pairs"]) if args.mode == "smoke"
        else sum(int(s["ambiguity_pairs"]) for s in config["splits"].values())
    )
    if args.mode == "smoke":
        seed = int(args.seed) if args.seed is not None else int(config["smoke"]["seed"])
        split = "smoke"
        requested = (
            Path(args.output).resolve()
            if args.output is not None
            else (ROOT / ".." / "smoke").resolve()
        )
        if requested.exists():
            raise FileExistsError('Choose a new smoke-output directory: ' + str(requested))
    else:
        split = str(args.split) if args.split else "train"
        if split not in config["splits"]:
            raise ValueError(f"unknown split {split}")
        seed = (
            int(args.seed) if args.seed is not None
            else int(config["splits"][split]["seed"])
        )
        requested = (
            Path(args.output).resolve()
            if args.output is not None
            else resolve_project_path(config["output_directory"])
        )
        if (requested / split).exists():
            raise FileExistsError(f"refusing to overwrite existing V023 output: {requested / split}")
    building = requested
    building.mkdir(parents=True, exist_ok=True)
    write_json(
        building / "config_snapshot.json",
        {k: v for k, v in config.items() if not k.startswith("_")},
    )
    write_json(building / "seed_registry.json", seed_registry_payload(config))
    operator = GravityTensorOperator(config, args.device)
    try:
        physics_audit = independent_prism_audit(operator, config)
        write_json(building / "independent_physics_audit.json", physics_audit)
        if float(physics_audit["maximum_component_relative_l2"]) > float(
            config["audit"]["independent_prism_relative_l2_max"]
        ):
            raise RuntimeError(f"independent prism audit failed: {physics_audit}")

        cards = pair_task_cards(config, n_pairs, seed + 71, deep_fraction=deep_fraction)
        batch_size = int(args.batch_size) if args.batch_size else 32
        print(
            f"[{time.strftime('%F %T')}] V023 {args.mode} {split}: building {n_pairs} pairs "
            f"on {args.device} (batch_size={batch_size}, seed={seed}, "
            f"deep_fraction={deep_fraction})",
            flush=True,
        )
        pair_items, funnel = build_pairs(
            config, operator, split, cards, seed + 10000,
            batch_size, progress_path=building / f"build_progress_{split}.json",
        )
        write_json(building / f"acceptance_funnel_{split}.json", funnel)
        save_frozen = (
            args.mode == "smoke"
            and bool(config["storage"].get("smoke_save_frozen_observation"))
        ) or split in set(config["storage"]["save_frozen_observation_for"])
        index_payload, pair_rows, records, noise_check = assemble_pairs_split(
            config, split, seed, pair_items, building, save_frozen,
            id_infix=str(args.id_infix or ""),
        )
        summary = build_summary(
            config, split, funnel, pair_rows, records, noise_check, index_payload
        )
        write_json(building / f"summary_{split}.json", summary)
        report = format_report(summary)
        (building / "smoke_report.txt").write_text(report + "\n", encoding="utf-8")
        print(report, flush=True)

        manifest = {
            "dataset_id": config["dataset_id"] + ("_SMOKE" if args.mode == "smoke" else ""),
            "schema_version": config["schema_version"],
            "build_mode": args.mode,
            "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "operator_path": str(operator.operator_path),
            "operator_sha256": operator.operator_sha256,
            "config_sha256": sha256_file(args.config),
            "split": split,
            "pairs": len(pair_rows),
            "seed_registry": seed_registry_payload(config),
            "gate_contract": {
                "gate_type": config["ambiguity"]["gate_type"],
                "tzz_dprime_min": float(config["ambiguity"]["tzz_dprime_min"]),
                "spin_dprime_min": float(config["ambiguity"]["spin_dprime_min"]),
                "density_nrmse_min": float(config["ambiguity"]["density_nrmse_min"]),
                "dprime_bins": config["ambiguity"]["dprime_bins"],
                "metric_version": ABSOLUTE_COVARIANCE_METRIC_VERSION,
            },
            "noise_contract": config["noise"],
            "noise_storage_policy": (
                "clean responses stored; train uses online white noise "
                "(amplitude U(0.75,1.5), never stored); frozen evaluation "
                "realizations use fixed amplitude 1.0, 5 per endpoint, "
                "pair endpoints independent (v023_noise.V023WhiteNoise)"
            ),
            "independent_physics_audit": physics_audit,
        }
        write_json(building / "manifest.json", manifest)
        return building
    except Exception as exc:
        write_json(building / "BUILD_FAILED.json", {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the V023 reverse-gate distinguishable-pair dataset"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config_v023.json")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Proposal batch size (geometry/forward batching)")
    parser.add_argument("--n-pairs", type=int, default=None,
                        help="Override the number of generated pairs (smoke defaults to config smoke.default_pairs)")
    parser.add_argument("--split", default=None,
                        help="Target split in full mode (train/validation_iid/test_locked)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the generation seed (for top-ups/workers; must not collide with the V022 registry)")
    parser.add_argument("--deep-fraction", type=float, default=None,
                        help="Deep-prototype quota (e.g. 0.1; targeted generation with support centroid >=600 m)")
    parser.add_argument("--id-infix", default=None,
                        help="Extra infix appended to sample_id (to distinguish top-up batches, e.g. t_)")
    return parser.parse_args()


if __name__ == "__main__":
    destination = build(parse_args())
    print(destination)
