#!/usr/bin/env python3
'Generate shape-ambiguity pairs from a shared core and signed peripheral allocation. Optimize four starts for low Tzz response and high complementary response. Final tier 1 requires Tzz d-prime <=1, max spin d-prime >=25, density NRMSE >=0.12 and Jaccard <=0.7. Historical tier-2 fallbacks must be excluded when assembling the final dataset.'
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage

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
from v017.geometry import SobolStream, generate_model
from v017.physics import GravityTensorOperator, independent_prism_audit
from v017.storage import save_split_shards
from build_pairs_v023 import (
    V022_SEEDS,
    _endpoint_contract_ok,
    _quantiles,
    _refresh_record,
    allocate,
    exact_labels,
)
from v023_shape_optimizer import optimize_shape_directions_multistart


ROOT = Path(__file__).resolve().parent
STRATA = ("small_0p5_2pct", "main_2_8pct", "large_8_18pct")
V023_SEPARABLE_SEEDS = (
    2026091201, 2026091202, 2026091203,
    2026091211, 2026091212, 2026091213, 2026091214, 2026091299,
)


# --------------------------------------------------------------------------- #
# Configuration validation and seed registry.
# --------------------------------------------------------------------------- #

def validate_shape_config(config: dict[str, Any]) -> None:
    if str(config.get("schema_version")) != "23.1.0":
        raise ValueError("build_shape_pairs_v023.py requires schema_version=23.1.0")
    amb = config["ambiguity"]
    if str(amb.get("metric_version")) != ABSOLUTE_COVARIANCE_METRIC_VERSION:
        raise ValueError("V023 shape requires the absolute covariance d-prime metric")
    if str(amb.get("gate_type")) != "shape_tzz_equivalent_v023":
        raise ValueError("V023 shape requires gate_type=shape_tzz_equivalent_v023")
    ref = np.asarray(amb["reference_component_sigma_eotvos"], dtype=np.float64)
    phys = np.asarray(config["noise"]["base_component_sigma_eotvos"], dtype=np.float64)
    if ref.shape != (6,) or not np.array_equal(ref, phys):
        raise ValueError(f"whitening sigma must equal physical noise sigma: {ref} vs {phys}")
    if not (0.0 < float(amb["tzz_dprime_max"]) <= 1.0):
        raise ValueError("tzz_dprime_max must be in (0, 1]")
    tier2 = amb["shape_gate_tier2"]
    if not (
        float(amb["shape_jaccard_max"]) < float(tier2["shape_jaccard_max"])
        and float(amb["shape_symmetric_difference_min"])
        > float(tier2["shape_symmetric_difference_min"])
    ):
        raise ValueError("tier2 shape gate must be strictly looser than tier1")
    seeds = [int(spec["seed"]) for spec in config["splits"].values()]
    seeds.append(int(config["smoke"]["seed"]))
    overlap = sorted(set(seeds) & (set(V022_SEEDS) | set(V023_SEPARABLE_SEEDS)))
    if overlap:
        raise ValueError(f"shape seeds overlap earlier registries: {overlap}")


def seed_registry_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": config["dataset_id"],
        "used_seeds": {
            "splits": {k: int(v["seed"]) for k, v in config["splits"].items()},
            "smoke": int(config["smoke"]["seed"]),
        },
        "earlier_seeds_do_not_use": {
            "v022": list(V022_SEEDS),
            "v023_separable": list(V023_SEPARABLE_SEEDS),
        },
    }


# --------------------------------------------------------------------------- #
# Acceptance gates.
# --------------------------------------------------------------------------- #

def classify_shape_pair_v023(
    metrics: dict[str, float], contract: dict[str, Any]
) -> str | None:
    'Require Tzz similarity, complementary-response separation and density difference.'
    d0 = float(metrics["tzz_global_dprime"])
    d1 = float(metrics["spin1_global_dprime"])
    d2 = float(metrics["spin2_global_dprime"])
    hard = (
        d0 <= float(contract["tzz_dprime_max"])
        and max(d1, d2) >= float(contract["spin_dprime_min"])
        and float(metrics["density_nrmse"]) >= float(contract["density_nrmse_min"])
    )
    return "shape_ambiguous" if hard else None


def shape_gate(
    jaccard: float,
    symmetric_difference_fraction: float,
    jaccard_max: float,
    symmetric_difference_min: float,
) -> bool:
    return bool(
        float(jaccard) <= float(jaccard_max)
        and float(symmetric_difference_fraction) >= float(symmetric_difference_min)
    )


def response_group_label_shape(metrics: dict[str, float], contract: dict[str, Any]) -> str:
    d1 = float(metrics["spin1_global_dprime"])
    d2 = float(metrics["spin2_global_dprime"])
    ratio = float(contract.get("dominance_ratio_for_group_label", 1.5))
    if d1 / max(d2, 1e-12) >= ratio:
        return "spin1"
    if d2 / max(d1, 1e-12) >= ratio:
        return "spin2"
    return "mixed"


# --------------------------------------------------------------------------- #
# Task cards preserve mechanism, support-stratum and difficulty marginals.
# --------------------------------------------------------------------------- #

def shape_task_cards(
    config: dict[str, Any], count: int, seed: int
) -> list[dict[str, Any]]:
    amb = config["ambiguity"]
    rng = np.random.default_rng(int(seed))
    mechanisms = exact_labels(allocate(count, amb["pair_mechanism_mixture"]), rng)
    strata = exact_labels(allocate(count, amb["pair_support_mixture"]), rng)
    cards = []
    for i in range(count):
        cards.append({
            "target_id": f"shape_{i:05d}",
            "mechanism": mechanisms[i],
            "support_stratum": strata[i],
            "base_contrast_gcc": float(rng.uniform(*amb["base_contrast_gcc_range"])),
            "redistribution_amplitude_gcc": float(
                rng.uniform(*amb["redistribution_amplitude_gcc_range"])
            ),
            "sign": 1.0 if rng.random() < float(config["density"]["positive_probability"]) else -1.0,
            "attempts": 0,
        })
    rng.shuffle(cards)
    return cards


def prototype_support_range(
    config: dict[str, Any], stratum: str
) -> tuple[float, float]:
    'Use the upper 80 percent of the support stratum to leave room after splitting. Geometry failures reject proposals rather than terminating the build.'
    low, high = (
        float(v) for v in config["support_fraction_mixture"][stratum]["range"]
    )
    bias = float(config["ambiguity"].get("prototype_support_range_upper_fraction", 0.2))
    return (low + bias * (high - low), high)


# --------------------------------------------------------------------------- #
# Vectorized scale screening and endpoint contracts.
# --------------------------------------------------------------------------- #

def evaluate_shape_proposal(
    config: dict[str, Any],
    basis_response: tuple[np.ndarray, np.ndarray, np.ndarray],
    basis_density: tuple[np.ndarray, np.ndarray, np.ndarray],
    card: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    'Return tier-1 candidates, tier-2-only candidates and scale-level rejection reasons. Endpoints are (common + scale*absolute_direction) +/- scale*signed_direction.'
    amb = config["ambiguity"]
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    common, absolute, signed = basis_density
    common_response, absolute_response, signed_response = basis_response
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

    d = signed_response.astype(np.float64)
    k0 = 2.0 * float(np.linalg.norm(d[5])) / sigma[5]
    k1 = 2.0 * float(np.sqrt(
        np.sum((d[2] / sigma[2]) ** 2) + np.sum((d[4] / sigma[4]) ** 2)
    ))
    dq1 = (d[0] - d[3]) / 2.0
    sigma_q1 = float(np.sqrt((sigma[0] ** 2 + sigma[3] ** 2) / 4.0))
    k2 = 2.0 * float(np.sqrt(
        np.sum((dq1 / sigma_q1) ** 2) + np.sum((d[1] / sigma[1]) ** 2)
    ))
    d0 = k0 * scales
    d1 = k1 * scales
    d2 = k2 * scales
    metric_gate = (
        (d0 <= float(amb["tzz_dprime_max"]))
        & (np.maximum(d1, d2) >= float(amb["spin_dprime_min"]))
    )

    s_col = scales[:, None, None, None].astype(np.float32)
    base_all = common[None] + s_col * absolute[None]
    first_all = base_all + s_col * signed[None]
    second_all = base_all - s_col * signed[None]
    q_dir = float(np.mean(signed.astype(np.float64) ** 2))
    p_base = np.mean(base_all.astype(np.float64) ** 2, axis=(1, 2, 3))
    nrmse = (
        2.0 * scales * np.sqrt(q_dir) / np.sqrt(p_base + scales**2 * q_dir)
        if q_dir > 0.0 else np.zeros_like(scales)
    )
    nrmse_gate = nrmse >= float(amb["density_nrmse_min"])

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
    inter = (active_first & active_second).sum(axis=(1, 2, 3)).astype(np.float64)
    union = (active_first | active_second).sum(axis=(1, 2, 3)).astype(np.float64)
    jaccard = inter / np.maximum(union, 1.0)
    symdiff = 1.0 - jaccard

    tier2_cfg = amb["shape_gate_tier2"]
    rejects: Counter[str] = Counter()
    passing1: list[dict[str, Any]] = []
    passing2: list[dict[str, Any]] = []
    for i, scale in enumerate(scales):
        if not metric_gate[i]:
            if d0[i] > float(amb["tzz_dprime_max"]):
                rejects["tzz_gate"] += 1
            else:
                rejects["spin_gate"] += 1
            continue
        if not nrmse_gate[i]:
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
        row = {
            "scale": float(scale),
            "tzz_global_dprime": float(d0[i]),
            "support_jaccard": float(jaccard[i]),
            "support_symmetric_difference_fraction": float(symdiff[i]),
        }
        if shape_gate(
            jaccard[i], symdiff[i],
            float(amb["shape_jaccard_max"]),
            float(amb["shape_symmetric_difference_min"]),
        ):
            passing1.append(row)
        elif shape_gate(
            jaccard[i], symdiff[i],
            float(tier2_cfg["shape_jaccard_max"]),
            float(tier2_cfg["shape_symmetric_difference_min"]),
        ):
            passing2.append(row)
        else:
            rejects["shape_gate"] += 1
    return passing1, passing2, rejects


def select_shape_candidate(
    passing: list[dict[str, Any]], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    'Prefer Tzz d-prime below the ceiling, then descending scale for larger signals.'
    if not passing:
        return []
    ceiling = float(contract.get("tzz_dprime_selection_ceiling", 0.85))
    under = [c for c in passing if float(c["tzz_global_dprime"]) <= ceiling]
    pool = under if under else passing
    return sorted(pool, key=lambda c: -float(c["scale"]))


# --------------------------------------------------------------------------- #
# Main pair-generation loop.
# --------------------------------------------------------------------------- #

def build_shape_pairs(
    config: dict[str, Any],
    operator: GravityTensorOperator,
    split: str,
    cards: list[dict[str, Any]],
    seed: int,
    batch_size: int,
    progress_path: Path | None = None,
    max_seconds: float | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    amb = config["ambiguity"]
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    jitter = float(amb.get("proposal_jitter_fraction", 0.0))
    tier2_patience = int(amb["shape_gate_tier2"]["tier1_patience_attempts"])
    n_starts = int(amb["optimizer_multistart"])
    rng = np.random.default_rng(int(seed))
    support_sobol = SobolStream(int(seed) + 314159)
    pending = list(cards)
    for spec in pending:
        spec["failure_history"] = Counter()
        spec["full_passing_seen"] = 0
    accepted: list[dict[str, Any]] = []
    accepted_since_spill: list[dict[str, Any]] = []
    proposals = 0
    batches = 0
    failure_reasons: Counter[str] = Counter()
    scale_reject_reasons: Counter[str] = Counter()
    exhausted_targets: list[dict[str, Any]] = []
    card_outcomes: list[dict[str, Any]] = []
    maximum_attempts = int(amb["maximum_proposals_per_target"])
    early_attempts = int(amb.get("early_exhaustion_attempts", maximum_attempts))
    started = time.time()
    wall_clock_stop = False
    while pending:
        if max_seconds is not None and time.time() - started > max_seconds:
            wall_clock_stop = True
            for spec in pending:
                card_outcomes.append(_card_outcome(spec, config, accepted=False,
                                                   stop="wall_clock_stop"))
            break
        batches += 1
        batch_specs = pending[:batch_size]
        fractions: list[np.ndarray] = []
        source_records: list[dict[str, Any]] = []
        active_specs: list[dict[str, Any]] = []
        skipped_specs: list[dict[str, Any]] = []
        for spec in batch_specs:
            try:
                model = generate_model(
                    rng, support_sobol, config, str(spec["mechanism"]),
                    prototype_support_range(config, str(spec["support_stratum"])),
                )
            except RuntimeError:
                # Reject a geometry proposal after 120 unsuccessful attempts; do not abort.
                skipped_specs.append(spec)
                continue
            fractions.append(model.volume_fraction)
            source_records.append(model.record)
            active_specs.append(spec)
            # Storage is deep-to-shallow; physical depth reverses the geometry z coordinate.
            spec["last_prototype_depth_km"] = float(
                1.0 - float(model.record["centroid_zyx_km"][0])
            )
        batch_specs = active_specs
        completed: set[str] = set()
        for spec in skipped_specs:
            proposals += 1
            spec["attempts"] = int(spec["attempts"]) + 1
            failure_reasons["geometry_infeasible"] += 1
            spec["failure_history"]["geometry_infeasible"] += 1
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
                card_outcomes.append(_card_outcome(spec, config, accepted=False,
                                                   stop="exhausted"))
                completed.add(str(spec["target_id"]))
        if not batch_specs:
            pending = [s for s in pending if str(s["target_id"]) not in completed]
            continue
        directions, diagnostics = optimize_shape_directions_multistart(
            np.stack(fractions).astype(np.float32),
            operator.g6, amb,
            int(seed) + 500000 + batches * 7919,
            n_starts,
        )
        # Build jittered bases and batch their forward responses.
        bases: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        actual_params: list[tuple[float, float]] = []
        forward_stack: list[np.ndarray] = []
        for i, spec in enumerate(batch_specs):
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
            mask = fractions[i] > 0.0
            core = ndimage.binary_erosion(
                mask,
                structure=ndimage.generate_binary_structure(3, 1),
                iterations=int(amb["support_varying_common_core_erosion_iterations"]),
                border_value=0,
            )
            direction = directions[i]
            common = (sign * base_contrast * fractions[i] * core.astype(np.float32)).astype(np.float32)
            absolute = (sign * amplitude * np.abs(direction)).astype(np.float32)
            signed = (sign * amplitude * direction).astype(np.float32)
            bases.append((common, absolute, signed))
            actual_params.append((base_contrast, amplitude))
            forward_stack.extend((common, absolute, signed))
        responses_all = operator.forward(
            np.stack(forward_stack), batch_size=max(3 * len(batch_specs), 3)
        )
        for i, spec in enumerate(batch_specs):
            proposals += 1
            spec["attempts"] = int(spec["attempts"]) + 1
            basis_response = (
                responses_all[3 * i], responses_all[3 * i + 1], responses_all[3 * i + 2]
            )
            passing1, passing2, rejects = evaluate_shape_proposal(
                config, basis_response, bases[i], spec
            )
            scale_reject_reasons.update(rejects)
            spec["full_passing_seen"] += len(passing1)
            tier = None
            pool: list[dict[str, Any]] = []
            if passing1:
                tier = 1
                pool = select_shape_candidate(passing1, amb)
            elif passing2 and int(spec["attempts"]) >= tier2_patience:
                tier = 2
                pool = select_shape_candidate(passing2, amb)
            chosen: dict[str, Any] | None = None
            chosen_payload: dict[str, Any] | None = None
            for candidate in pool:
                scale = float(candidate["scale"])
                common, absolute, signed = bases[i]
                common_response, absolute_response, signed_response = basis_response
                base = common + scale * absolute
                first = (base + scale * signed).astype(np.float32)
                second = (base - scale * signed).astype(np.float32)
                response_base = common_response + scale * absolute_response
                responses = np.stack((
                    response_base + scale * signed_response,
                    response_base - scale * signed_response,
                )).astype(np.float32)
                metrics = pair_metrics_absolute_covariance_v019(
                    first, second, responses[0], responses[1], amb
                )
                if classify_shape_pair_v023(metrics, amb) is None:
                    scale_reject_reasons["final_gate_recheck"] += 1
                    continue
                sm = support_pair_metrics(first, second, threshold)
                gate = amb if tier == 1 else amb["shape_gate_tier2"]
                if not shape_gate(
                    sm["support_jaccard"],
                    sm["support_symmetric_difference_fraction"],
                    float(gate["shape_jaccard_max"]),
                    float(gate["shape_symmetric_difference_min"]),
                ):
                    scale_reject_reasons["final_shape_recheck"] += 1
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
                    "support_metrics": sm,
                }
                break
            if chosen is None:
                reason = "empty_passing_set" if not (passing1 or passing2) else (
                    "tier1_only_passing_set" if passing2 else "recheck_failed"
                )
                if passing2 and int(spec["attempts"]) < tier2_patience:
                    reason = "tier2_waiting_patience"
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
                    card_outcomes.append(_card_outcome(spec, config, accepted=False,
                                                       stop="exhausted"))
                    completed.add(str(spec["target_id"]))
                continue
            accepted.append({
                "card": spec,
                "densities": chosen_payload["densities"],
                "responses": chosen_payload["responses"],
                "volume_fraction": fractions[i],
                "metrics": chosen_payload["metrics"],
                "support_metrics": chosen_payload["support_metrics"],
                "scale": float(chosen["scale"]),
                "shape_gate_tier": int(tier),
                "source_record": source_records[i],
                "optimizer": diagnostics[i],
                "base_contrast_gcc_actual": float(actual_params[i][0]),
                "redistribution_amplitude_gcc_actual": float(actual_params[i][1]),
                "generation_attempts_for_target": int(spec["attempts"]),
            })
            card_outcomes.append(_card_outcome(spec, config, accepted=True, stop=""))
            accepted_since_spill.append(accepted[-1])
            completed.add(str(spec["target_id"]))
        pending = [s for s in pending if str(s["target_id"]) not in completed]
        if pending:
            rotation = min(len(batch_specs), len(pending))
            pending = pending[rotation:] + pending[:rotation]
        if checkpoint_dir is not None and checkpoint_every and (
            batches % int(checkpoint_every) == 0 or not pending
        ):
            _spill_checkpoint(checkpoint_dir, batches, accepted_since_spill)
            accepted_since_spill = []
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
        if batches == 1 or batches % 2 == 0 or not pending:
            print(
                f"[{time.strftime('%F %T')}] V023-shape {split}: accepted "
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
        "failure_reasons": dict(failure_reasons),
        "scale_level_reject_reasons": dict(scale_reject_reasons),
        "exhausted_targets": exhausted_targets,
        "card_outcomes": card_outcomes,
        "wall_clock_stop": wall_clock_stop,
        "shape_gate_tier_counts": dict(Counter(
            str(item["shape_gate_tier"]) for item in accepted
        )),
        "metric_version": ABSOLUTE_COVARIANCE_METRIC_VERSION,
        "elapsed_seconds": elapsed,
    }
    return accepted, funnel


def _spill_checkpoint(
    checkpoint_dir: Path, batch: int, items: list[dict[str, Any]]
) -> None:
    'Persist accepted pairs and diagnostics incrementally. This protects accepted work but does not resume the optimizer/random stream.'
    if not items:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    densities = np.stack([np.stack(it["densities"]) for it in items]).astype(np.float32)
    responses = np.stack([np.stack(it["responses"]) for it in items]).astype(np.float32)
    fractions = np.stack([it["volume_fraction"] for it in items]).astype(np.float32)
    np.savez(
        checkpoint_dir / f"accepted_batch_{batch:05d}.npz",
        densities=densities, responses=responses, volume_fraction=fractions,
    )
    with (checkpoint_dir / "accepted_meta.jsonl").open("a", encoding="utf-8") as handle:
        for it in items:
            card = {
                k: (dict(v) if isinstance(v, Counter) else v)
                for k, v in it["card"].items()
            }
            meta = {
                "spilled_at_batch": int(batch),
                "card": card,
                "metrics": it["metrics"],
                "support_metrics": it["support_metrics"],
                "scale": float(it["scale"]),
                "shape_gate_tier": int(it["shape_gate_tier"]),
                "source_record": it["source_record"],
                "optimizer": it["optimizer"],
                "base_contrast_gcc_actual": float(it["base_contrast_gcc_actual"]),
                "redistribution_amplitude_gcc_actual": float(
                    it["redistribution_amplitude_gcc_actual"]
                ),
                "generation_attempts_for_target": int(it["generation_attempts_for_target"]),
            }
            handle.write(json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n")


def _card_outcome(
    spec: dict[str, Any], config: dict[str, Any], accepted: bool, stop: str
) -> dict[str, Any]:
    return {
        "target_id": str(spec["target_id"]),
        "mechanism": str(spec["mechanism"]),
        "support_stratum": str(spec["support_stratum"]),
        "last_prototype_depth_km": spec.get("last_prototype_depth_km"),
        "attempts": int(spec["attempts"]),
        "accepted": bool(accepted),
        "stop": stop,
        "failure_history": dict(spec["failure_history"]),
    }


# --------------------------------------------------------------------------- #
# Assemble canonical endpoint metadata.
# --------------------------------------------------------------------------- #

def assemble_shape_split(
    config: dict[str, Any],
    split: str,
    seed: int,
    pair_items: list[dict[str, Any]],
    output: Path,
    worker_tag: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    amb = config["ambiguity"]
    endpoints: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    pair_counter: Counter[tuple[str, str]] = Counter()
    for item in pair_items:
        card = item["card"]
        stratum = str(card["support_stratum"])
        group = response_group_label_shape(item["metrics"], amb)
        pair_number = pair_counter[(stratum, group)]
        pair_counter[(stratum, group)] += 1
        pair_id = (
            f"{split}_{stratum}_{group}_"
            f"{worker_tag + '_' if worker_tag else ''}shape{pair_number:05d}_"
            f"{stable_hash([seed, stratum, group, pair_number], 10)}"
        )
        prototype_id = item["source_record"].get("prototype_id") or stable_hash(
            [seed, str(item["source_record"].get("subtype"))], 16
        )
        for endpoint_index, role in enumerate(("A", "B")):
            source = dict(item["source_record"])
            source.update({
                "mechanism": str(card["mechanism"]),
                "support_mechanism": str(card["mechanism"]),
                "support_subtype": str(item["source_record"]["subtype"]),
                "subtype": str(amb["pair_subtype"]),
                "density_pattern": "optimized_smooth_redistribution_shape",
                "ambiguity_group_id": pair_id,
                "ambiguity_pair_role": role,
                "response_group": group,
                "pair_geometry_mode": "support_varying",
                "pair_metrics": item["metrics"],
                "support_jaccard": float(item["support_metrics"]["support_jaccard"]),
                "support_symmetric_difference_fraction": float(
                    item["support_metrics"]["support_symmetric_difference_fraction"]
                ),
                "shape_gate_tier": int(item["shape_gate_tier"]),
                "density_redistribution_scale": float(item["scale"]),
                "candidate_selection": "v023_shape_passing_set_max_scale",
                "base_prototype_id": prototype_id,
                "base_contrast_gcc": float(item["base_contrast_gcc_actual"]),
                "redistribution_amplitude_gcc": float(
                    item["redistribution_amplitude_gcc_actual"]
                ),
                "generation_attempts_for_target": int(item["generation_attempts_for_target"]),
                **item["optimizer"],
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
            "gate": "shape_ambiguous",
            "shape_gate_tier": int(item["shape_gate_tier"]),
            "mechanism": str(card["mechanism"]),
            "support_subtype": str(item["source_record"]["subtype"]),
            "support_stratum": stratum,
            "pair_geometry_mode": "support_varying",
            "density_redistribution_scale": float(item["scale"]),
            "base_contrast_gcc": float(item["base_contrast_gcc_actual"]),
            "redistribution_amplitude_gcc": float(item["redistribution_amplitude_gcc_actual"]),
            "direction_smoothing_kernel": int(amb["direction_smoothing_kernel"]),
            "support_jaccard": float(item["support_metrics"]["support_jaccard"]),
            "support_symmetric_difference_fraction": float(
                item["support_metrics"]["support_symmetric_difference_fraction"]
            ),
            "support_intersection_cells": int(item["support_metrics"]["support_intersection_cells"]),
            "support_union_cells": int(item["support_metrics"]["support_union_cells"]),
            "optimizer_n_starts": int(item["optimizer"]["n_starts"]),
            "optimizer_best_start": int(item["optimizer"]["best_start"]),
            "optimizer_density_energy_over_e0": float(
                item["optimizer"]["density_energy_over_e0"]
            ),
            "linearized_spin1_to_tzz": float(item["optimizer"]["linearized_spin1_to_tzz"]),
            "linearized_spin2_to_tzz": float(item["optimizer"]["linearized_spin2_to_tzz"]),
            "generation_attempts_for_target": int(item["generation_attempts_for_target"]),
            **item["metrics"],
        })

    rng = np.random.default_rng(seed + 90000)
    endpoints = [endpoints[int(i)] for i in rng.permutation(len(endpoints))]
    records = []
    prefix = str(config.get("sample_id_prefix", "v023s"))
    worker_infix = f"{worker_tag}_" if worker_tag else ""
    for index, endpoint in enumerate(endpoints):
        sample_id = (
            f"{prefix}_{split}_{worker_infix}{index:05d}_"
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
        "subtype": np.asarray([r["subtype"] for r in records], dtype="U40"),
        "support_subtype": np.asarray([r["support_subtype"] for r in records], dtype="U32"),
        "density_pattern": np.asarray([r["density_pattern"] for r in records], dtype="U48"),
        "ambiguity_group_id": np.asarray([r["ambiguity_group_id"] for r in records], dtype="U96"),
        "ambiguity_pair_role": np.asarray([r["ambiguity_pair_role"] for r in records], dtype="U1"),
        "response_group": np.asarray([r["response_group"] for r in records], dtype="U12"),
        "support_stratum": np.asarray([r["support_stratum"] for r in records], dtype="U24"),
        "pair_geometry_mode": np.asarray([r["pair_geometry_mode"] for r in records], dtype="U20"),
    }
    index_payload = save_split_shards(
        output, split, arrays, records, int(config["storage"]["shard_size"])
    )
    with (output / split / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in pair_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return index_payload, pair_rows, records


# --------------------------------------------------------------------------- #
# Aggregate diagnostics.
# --------------------------------------------------------------------------- #

def build_shape_summary(
    config: dict[str, Any],
    split: str,
    funnel: dict[str, Any],
    pair_items: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    index_payload: dict[str, Any],
) -> dict[str, Any]:
    amb = config["ambiguity"]
    tzz = [float(r["tzz_global_dprime"]) for r in pair_rows]
    spin1 = [float(r["spin1_global_dprime"]) for r in pair_rows]
    spin2 = [float(r["spin2_global_dprime"]) for r in pair_rows]
    maxspin = [max(a, b) for a, b in zip(spin1, spin2)]
    nrmse = [float(r["density_nrmse"]) for r in pair_rows]
    jaccard = [float(r["support_jaccard"]) for r in pair_rows]
    symdiff = [float(r["support_symmetric_difference_fraction"]) for r in pair_rows]
    scales = [float(r["density_redistribution_scale"]) for r in pair_rows]
    peaks = [float(r["peak_abs_density_gcc"]) for r in records]
    cells = [int(r["support_cells"]) for r in records]
    # Stratify acceptance by mechanism, support size and depth third.
    outcomes = funnel["card_outcomes"]
    depth_of = {}
    for item in pair_items:
        z = float(item["source_record"]["centroid_zyx_km"][0])
        depth_of[str(item["card"]["target_id"])] = 1.0 - z  # Storage is deep-to-shallow; physical depth reverses the geometry z coordinate.
    def _rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        acc = sum(1 for r in rows if r["accepted"])
        return {"cards": n, "accepted": acc, "acceptance_rate": acc / max(n, 1)}
    by_mechanism: dict[str, list[dict[str, Any]]] = {}
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    by_depth: dict[str, list[dict[str, Any]]] = {}
    for row in outcomes:
        by_mechanism.setdefault(row["mechanism"], []).append(row)
        by_stratum.setdefault(row["support_stratum"], []).append(row)
        depth = row.get("last_prototype_depth_km")
        if depth is None:
            depth = depth_of.get(row["target_id"])
        third = ("unknown" if depth is None
                 else "shallow_0_333m" if depth < 1.0 / 3.0
                 else "mid_333_667m" if depth < 2.0 / 3.0
                 else "deep_667_1000m")
        by_depth.setdefault(third, []).append(row)
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
            "wall_clock_stop": funnel["wall_clock_stop"],
        },
        "shape_difference": {
            "support_jaccard": _quantiles(jaccard),
            "support_symmetric_difference_fraction": _quantiles(symdiff),
            "tier1_gate": {
                "jaccard_max": float(amb["shape_jaccard_max"]),
                "symdiff_min": float(amb["shape_symmetric_difference_min"]),
            },
            "tier_counts": funnel["shape_gate_tier_counts"],
            "jaccard_max_ok": bool(
                max(jaccard, default=1.0) <= float(amb["shape_jaccard_max"]) + 1e-12
                or funnel["shape_gate_tier_counts"].get("2")
            ),
        },
        "tzz_dprime": {
            **_quantiles(tzz),
            "max_ok": bool(max(tzz, default=0.0) <= float(amb["tzz_dprime_max"]) + 1e-12),
            "near_ceiling_fraction_ge_0p95": float(
                np.mean(np.asarray(tzz) >= 0.95)
            ) if tzz else None,
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
        "acceptance_by_mechanism": {k: _rate(v) for k, v in sorted(by_mechanism.items())},
        "acceptance_by_stratum": {k: _rate(v) for k, v in sorted(by_stratum.items())},
        "acceptance_by_depth_third": {k: _rate(v) for k, v in sorted(by_depth.items())},
        "geometry_check": {
            "peak_abs_density_max": max(peaks, default=0.0),
            "peak_ok": bool(max(peaks, default=0.0) <= 0.80 + 1e-6),
            "support_cells": {
                "min": min(cells, default=0),
                "median": float(np.median(cells)) if cells else 0.0,
                "max": max(cells, default=0),
                "min_ok": bool(min(cells, default=0) >= 82),
            },
        },
        "scale": _quantiles(scales),
        "failure_reasons": funnel["failure_reasons"],
        "scale_level_reject_reasons": funnel["scale_level_reject_reasons"],
        "exhausted_targets_count": len(funnel["exhausted_targets"]),
        "response_group_counts_pairs": dict(Counter(
            str(r["response_group"]) for r in pair_rows
        )),
        "extrapolation_full_2087": {
            "seconds_per_pair": funnel["seconds_per_accepted_pair"],
            "estimated_hours": funnel["seconds_per_accepted_pair"] * 2087 / 3600.0,
            "proposals_per_pair": funnel["proposals_per_accepted_pair"],
        },
        "index": index_payload,
    }
    return summary


def format_shape_report(summary: dict[str, Any]) -> str:
    f = summary["funnel"]
    sd = summary["shape_difference"]
    t = summary["tzz_dprime"]
    s = summary["spin_dprime"]
    n = summary["density_nrmse"]
    g = summary["geometry_check"]
    lines = [
        f"V023 shape-ambiguity pair smoke report (split={summary['split']})",
        "=" * 64,
        f"1) proposals {f['proposals']} / accepted {f['accepted_pairs']} / target {f['target_pairs']}"
        f" | acceptance rate {f['acceptance_rate']:.3f}"
        f" | proposals per pair {f['proposals_per_accepted_pair']:.2f}"
        f" | {f['seconds_per_accepted_pair']:.1f}s/pair"
        f" | total elapsed {f['elapsed_seconds']:.0f}s"
        f" | wall_clock_stop={f['wall_clock_stop']}",
        f"2) reachable shape difference: Jaccard min {sd['support_jaccard']['0.0']:.3f}"
        f" / median {sd['support_jaccard']['0.5']:.3f}"
        f" / max {sd['support_jaccard']['1.0']:.3f}"
        f" | symmetric difference min {sd['support_symmetric_difference_fraction']['0.0']:.3f}"
        f" / median {sd['support_symmetric_difference_fraction']['0.5']:.3f}"
        f" / max {sd['support_symmetric_difference_fraction']['1.0']:.3f}"
        f" | tier counts {json.dumps(sd['tier_counts'], sort_keys=True)}",
        f"3) Tzz d′: min {t['0.0']:.3f} / median {t['0.5']:.3f} / max {t['1.0']:.3f}"
        f" | all<=1: {t['max_ok']} | >=0.95 near-ceiling fraction {t['near_ceiling_fraction_ge_0p95']:.3f}"
        f" || max spin d′: min {s['max_spin']['0.0']:.1f} / median {s['max_spin']['0.5']:.1f}"
        f" / max {s['max_spin']['1.0']:.1f} | all>=25: {s['min_ok']}"
        f" || NRMSE: min {n['0.0']:.3f} / median {n['0.5']:.3f} / max {n['1.0']:.3f}"
        f" | all>=0.12: {n['min_ok']}",
        "4) stratified acceptance rates:",
    ]
    for name, block in (("mechanism", summary["acceptance_by_mechanism"]),
                        ("support tier", summary["acceptance_by_stratum"]),
                        ("depth", summary["acceptance_by_depth_third"])):
        for key, row in block.items():
            lines.append(
                f"   {name} {key}: {row['accepted']}/{row['cards']}"
                f" = {row['acceptance_rate']:.3f}"
            )
    lines.append(
        f"5) proposal-level failure causes {json.dumps(summary['failure_reasons'], sort_keys=True)}"
        f" | top scale-level failure causes: {json.dumps(summary['scale_level_reject_reasons'], sort_keys=True)}"
    )
    ex = summary["extrapolation_full_2087"]
    lines.append(
        f"6) full 2087-pair runtime extrapolation: {ex['seconds_per_pair']:.1f}s/pair"
        f" -> about {ex['estimated_hours']:.2f} hours"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command-line workflow.
# --------------------------------------------------------------------------- #

def build(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    validate_shape_config(config)
    if args.seed is not None:
        cli_seed = int(args.seed)
        forbidden = set(V022_SEEDS) | set(V023_SEPARABLE_SEEDS)
        if cli_seed in forbidden:
            raise ValueError(f"--seed {cli_seed} collides with earlier registries")
    worker_tag = f"w{int(args.worker_id)}" if args.worker_id is not None else None
    if args.tier1_patience is not None:
        # Large patience disables tier-2 fallback; the final dataset uses tier 1 only.
        config["ambiguity"]["shape_gate_tier2"]["tier1_patience_attempts"] = int(
            args.tier1_patience
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
            else (ROOT / ".." / "smoke_shape").resolve()
        )
        if requested.exists():
            raise FileExistsError('Choose a new smoke-output directory: ' + str(requested))
    else:
        split = str(args.split) if args.split else "train"
        if split not in ("train", "validation_iid", "test_locked"):
            raise ValueError(f"unknown split {split}")
        seed = (
            int(args.seed) if args.seed is not None
            else int(config["splits"]["train"]["seed"])
        )
        requested = (
            Path(args.output).resolve()
            if args.output is not None
            else resolve_project_path(config["output_directory"])
        )
        if requested.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {requested}")
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

        cards = shape_task_cards(config, n_pairs, seed + 71)
        batch_size = int(args.batch_size) if args.batch_size else 16
        max_seconds = float(args.max_minutes) * 60.0 if args.max_minutes else None
        print(
            f"[{time.strftime('%F %T')}] V023-shape {args.mode}: building {n_pairs} pairs "
            f"on {args.device} (batch_size={batch_size}, seed={seed}, "
            f"worker={worker_tag}, "
            f"optimizer {config['ambiguity']['local_search_steps']} steps x "
            f"{config['ambiguity']['optimizer_multistart']} starts)",
            flush=True,
        )
        pair_items, funnel = build_shape_pairs(
            config, operator, split, cards, seed + 10000,
            batch_size, progress_path=building / f"build_progress_{split}.json",
            max_seconds=max_seconds,
            checkpoint_dir=(
                building / "checkpoint"
                if int(args.checkpoint_every) > 0 else None
            ),
            checkpoint_every=int(args.checkpoint_every),
        )
        write_json(building / f"acceptance_funnel_{split}.json", funnel)
        index_payload, pair_rows, records = assemble_shape_split(
            config, split, seed, pair_items, building, worker_tag=worker_tag,
        )
        summary = build_shape_summary(
            config, split, funnel, pair_items, pair_rows, records, index_payload
        )
        write_json(building / f"summary_{split}.json", summary)
        report = format_shape_report(summary)
        (building / "smoke_report.txt").write_text(report + "\n", encoding="utf-8")
        print(report, flush=True)

        manifest = {
            "dataset_id": config["dataset_id"] + ("_SMOKE" if args.mode == "smoke" else ""),
            "schema_version": config["schema_version"],
            "build_mode": args.mode,
            "worker_tag": worker_tag,
            "generation_seed": int(seed),
            "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "operator_path": str(operator.operator_path),
            "operator_sha256": operator.operator_sha256,
            "config_sha256": sha256_file(args.config),
            "split": split,
            "pairs": len(pair_rows),
            "seed_registry": seed_registry_payload(config),
            "gate_contract": {
                "gate_type": config["ambiguity"]["gate_type"],
                "tzz_dprime_max": float(config["ambiguity"]["tzz_dprime_max"]),
                "spin_dprime_min": float(config["ambiguity"]["spin_dprime_min"]),
                "density_nrmse_min": float(config["ambiguity"]["density_nrmse_min"]),
                "shape_jaccard_max": float(config["ambiguity"]["shape_jaccard_max"]),
                "shape_symmetric_difference_min": float(
                    config["ambiguity"]["shape_symmetric_difference_min"]
                ),
                "shape_gate_tier2": config["ambiguity"]["shape_gate_tier2"],
                "metric_version": ABSOLUTE_COVARIANCE_METRIC_VERSION,
            },
            "noise_contract": "v023_noise.V023WhiteNoise(halved sigma, online only)",
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
        description="Build the V023 shape-ambiguity pairs"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config_shape_v023.json")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Proposal batch size (each proposal is jointly optimized from 4 starting points)")
    parser.add_argument("--n-pairs", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Wall-clock limit (for smoke runs; on timeout the run wraps up with the pairs accepted so far)")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Flush accepted pairs incrementally to <output>/checkpoint/ every N batches (0=disabled)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the generation seed (for parallel workers; must not collide with the historical registry)")
    parser.add_argument("--split", default=None,
                        help="Target split in full mode (train/validation_iid/test_locked)")
    parser.add_argument("--worker-id", type=int, default=None,
                        help="Worker index: sample_id/pair_id get a w<N> tag")
    parser.add_argument("--tier1-patience", type=int, default=None,
                        help="Override tier-1 patience (a large value disables the tier-2 fallback; the final dataset is all tier 1)")
    return parser.parse_args()


if __name__ == "__main__":
    destination = build(parse_args())
    print(destination)
