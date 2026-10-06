#!/usr/bin/env python3
'Merge worker pairs, globally shuffle with seed 2026091471, and recheck gates on stored arrays. Enforce endpoint/group uniqueness and optionally remove tier-2 pairs. Write merged arrays, metadata, acceptance funnels and seed provenance.'
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from v017.ambiguity import (
    pair_metrics_absolute_covariance_v019,
    support_pair_metrics,
)
from v017.config import load_config, sha256_file, write_json
from v017.physics import GravityTensorOperator
from v017.storage import load_split, save_split_shards
from build_pairs_v023 import _endpoint_contract_ok, _quantiles
from build_shape_pairs_v023 import (
    V022_SEEDS,
    V023_SEPARABLE_SEEDS,
    classify_shape_pair_v023,
    seed_registry_payload,
    shape_gate,
    validate_shape_config,
)


ROOT = Path(__file__).resolve().parent
MERGE_SHUFFLE_SEED = 2026091471
WORKER_SEEDS = [2026091401 + 1000 * w for w in range(6)]
EXPECTED_COUNTS = [348, 348, 348, 348, 348, 347]
REFERENCE_DATASETS = (
    str(ROOT.parent / "generation_work/pairs_separable"),
    str(ROOT.parent / "generation_work/smoke"),
    str(ROOT.parent / "generation_work/smoke_shape"),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def merge(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    validate_shape_config(config)
    amb = config["ambiguity"]
    root = Path(args.root).resolve()
    workers_root = root / "workers"
    n_workers = int(args.workers)
    split = str(args.split)
    worker_seeds = (
        [int(args.seed_base) + int(args.seed_step) * w for w in range(n_workers)]
        if args.seed_base is not None else WORKER_SEEDS[:n_workers]
    )
    expected_counts = (
        [int(v) for v in str(args.counts).split(",")]
        if args.counts else EXPECTED_COUNTS[:n_workers]
    )
    if len(expected_counts) != n_workers:
        raise ValueError("--counts length must equal --workers")

    # Collect completed worker outputs.
    per_worker: list[dict[str, Any]] = []
    for w in range(n_workers):
        wdir = workers_root / f"w{w}"
        arrays, records = load_split(wdir, split)
        pairs = _read_jsonl(wdir / split / "pairs.jsonl")
        funnel = json.loads((wdir / f"acceptance_funnel_{split}.json").read_text(encoding="utf-8"))
        manifest = json.loads((wdir / "manifest.json").read_text(encoding="utf-8"))
        per_worker.append({
            "worker": w, "dir": str(wdir), "arrays": arrays, "records": records,
            "pairs": pairs, "funnel": funnel, "manifest": manifest,
        })
        print(f"worker w{w}: pairs={len(pairs)}, endpoints={len(records)}, "
              f"proposals={funnel['proposals']}, elapsed={funnel['elapsed_seconds']:.0f}s",
              flush=True)
    total_pairs = sum(len(w["pairs"]) for w in per_worker)
    expected_total = sum(expected_counts)
    assert total_pairs == expected_total, f"pairs total {total_pairs} != {expected_total}"

    # Remove tier-2 pairs for the final release.
    excluded_rows: list[dict[str, Any]] = []
    if args.tier1_only:
        keep_gid = {
            str(p["ambiguity_group_id"]) for p in
            [row for w in per_worker for row in w["pairs"]]
            if int(p.get("shape_gate_tier", 1)) == 1
        }
        for w in per_worker:
            for row in w["pairs"]:
                if str(row["ambiguity_group_id"]) not in keep_gid:
                    excluded_rows.append(row)
            keep_idx = [
                i for i, r in enumerate(w["records"])
                if str(r["ambiguity_group_id"]) in keep_gid
            ]
            w["arrays"] = {
                key: value[keep_idx] for key, value in w["arrays"].items()
            }
            w["records"] = [w["records"][i] for i in keep_idx]
            w["pairs"] = [
                row for row in w["pairs"]
                if str(row["ambiguity_group_id"]) in keep_gid
            ]
        (root).mkdir(parents=True, exist_ok=True)
        with (root / f"{split}_tier2_excluded.jsonl").open("w", encoding="utf-8") as handle:
            for row in excluded_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"tier1-only: excluded {len(excluded_rows)} tier2 pairs", flush=True)
    total_pairs = sum(len(w["pairs"]) for w in per_worker)

    # Merge and globally shuffle.
    array_keys = list(per_worker[0]["arrays"].keys())
    merged_arrays = {
        key: np.concatenate([w["arrays"][key] for w in per_worker]) for key in array_keys
    }
    merged_records = [r for w in per_worker for r in w["records"]]
    merged_pairs = [p for w in per_worker for p in w["pairs"]]
    rng = np.random.default_rng(MERGE_SHUFFLE_SEED)
    order = rng.permutation(len(merged_records))
    merged_arrays = {key: value[order] for key, value in merged_arrays.items()}
    merged_records = [merged_records[int(i)] for i in order]

    # Recompute all acceptance gates from stored arrays.
    started_verify = time.time()
    ids = [r["sample_id"] for r in merged_records]
    assert len(set(ids)) == len(ids), "sample_id not unique"
    for reference in REFERENCE_DATASETS:
        ref = Path(reference)
        if not ref.exists():
            continue
        for split_dir in sorted(p for p in ref.iterdir() if p.is_dir()):
            _, ref_records = load_split(ref, split_dir.name)
            overlap = set(ids) & {r["sample_id"] for r in ref_records}
            assert not overlap, f"sample_id overlap with {ref}: {len(overlap)}"
    subtypes = {str(r["subtype"]) for r in merged_records}
    assert subtypes == {str(amb["pair_subtype"])}, subtypes

    by_gid: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(merged_records):
        by_gid[str(r["ambiguity_group_id"])].append(i)
    assert len(by_gid) == len(merged_pairs)
    assert all(len(v) == 2 for v in by_gid.values())
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    prow = {p["ambiguity_group_id"]: p for p in merged_pairs}
    max_rel = 0.0
    max_jac_err = 0.0
    for gid, (ia, ib) in by_gid.items():
        m = pair_metrics_absolute_covariance_v019(
            merged_arrays["density_zyx"][ia], merged_arrays["density_zyx"][ib],
            merged_arrays["clean_d6"][ia], merged_arrays["clean_d6"][ib], amb,
        )
        assert classify_shape_pair_v023(m, amb) is not None, f"gate failed: {gid}"
        sm = support_pair_metrics(
            merged_arrays["density_zyx"][ia], merged_arrays["density_zyx"][ib], threshold
        )
        p = prow[gid]
        gate_cfg = amb if int(p.get("shape_gate_tier", 1)) == 1 else amb["shape_gate_tier2"]
        assert shape_gate(
            sm["support_jaccard"], sm["support_symmetric_difference_fraction"],
            float(gate_cfg["shape_jaccard_max"]),
            float(gate_cfg["shape_symmetric_difference_min"]),
        ), f"shape gate failed: {gid}"
        p = prow[gid]
        for key in ("tzz_global_dprime", "spin1_global_dprime",
                    "spin2_global_dprime", "density_nrmse"):
            max_rel = max(max_rel, abs(m[key] - p[key]) / max(abs(p[key]), 1e-12))
        max_jac_err = max(max_jac_err, abs(sm["support_jaccard"] - p["support_jaccard"]))
    for i in range(len(merged_records)):
        ok, detail = _endpoint_contract_ok(
            merged_arrays["density_zyx"][i],
            str(merged_records[i]["support_stratum"]), config,
        )
        assert ok, f"endpoint contract failed {merged_records[i]['sample_id']}: {detail}"
    verification = {
        "pairs_recomputed": len(merged_pairs),
        "tier2_excluded_pairs": len(excluded_rows),
        "tier1_only": bool(args.tier1_only),
        "metric_max_rel_error_vs_records": float(max_rel),
        "jaccard_max_abs_error_vs_records": float(max_jac_err),
        "gates_recomputed_all_pass": True,
        "endpoints_contract_all_pass": len(merged_records),
        "sample_id_unique": True,
        "zero_overlap_with_reference_datasets": True,
        "subtype": str(amb["pair_subtype"]),
        "elapsed_seconds": time.time() - started_verify,
    }
    print(f"verification: {json.dumps(verification, ensure_ascii=False)}", flush=True)

    # Write canonical shards.
    index_payload = save_split_shards(
        root, split, merged_arrays, merged_records, int(config["storage"]["shard_size"])
    )
    with (root / split / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in merged_pairs:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # Merge acceptance funnels.
    failure_reasons: Counter[str] = Counter()
    scale_rejects: Counter[str] = Counter()
    card_outcomes: list[dict[str, Any]] = []
    exhausted: list[dict[str, Any]] = []
    tier_counts: Counter[str] = Counter()
    proposals = 0
    for w in per_worker:
        f = w["funnel"]
        proposals += int(f["proposals"])
        failure_reasons.update(f["failure_reasons"])
        scale_rejects.update(f["scale_level_reject_reasons"])
        card_outcomes.extend(f["card_outcomes"])
        exhausted.extend(f["exhausted_targets"])
        tier_counts.update(f["shape_gate_tier_counts"])
    worker_elapsed = [float(w["funnel"]["elapsed_seconds"]) for w in per_worker]
    funnel = {
        "split": split,
        "target_pairs": expected_total,
        "accepted_pairs": total_pairs,
        "shortfall": expected_total - total_pairs,
        "proposals": proposals,
        "acceptance_rate": total_pairs / max(proposals, 1),
        "proposals_per_accepted_pair": proposals / max(total_pairs, 1),
        "failure_reasons": dict(failure_reasons),
        "scale_level_reject_reasons": dict(scale_rejects),
        "exhausted_targets": exhausted,
        "card_outcomes": card_outcomes,
        "shape_gate_tier_counts": dict(tier_counts),
        "parallel_workers": n_workers,
        "per_worker": [
            {
                "worker": w["worker"],
                "seed": worker_seeds[w["worker"]],
                "pairs": len(w["pairs"]),
                "proposals": int(w["funnel"]["proposals"]),
                "acceptance_rate": float(w["funnel"]["acceptance_rate"]),
                "elapsed_seconds": float(w["funnel"]["elapsed_seconds"]),
            }
            for w in per_worker
        ],
        "elapsed_seconds_per_worker": worker_elapsed,
        "wall_clock_seconds_estimate": max(worker_elapsed),
    }
    write_json(root / f"acceptance_funnel_{split}.json", funnel)

    # Write aggregate summaries.
    tzz = [float(r["tzz_global_dprime"]) for r in merged_pairs]
    spin1 = [float(r["spin1_global_dprime"]) for r in merged_pairs]
    spin2 = [float(r["spin2_global_dprime"]) for r in merged_pairs]
    maxspin = [max(a, b) for a, b in zip(spin1, spin2)]
    nrmse = [float(r["density_nrmse"]) for r in merged_pairs]
    jaccard = [float(r["support_jaccard"]) for r in merged_pairs]
    symdiff = [float(r["support_symmetric_difference_fraction"]) for r in merged_pairs]
    scales = [float(r["density_redistribution_scale"]) for r in merged_pairs]
    peaks = [float(r["peak_abs_density_gcc"]) for r in merged_records]
    cells = [int(r["support_cells"]) for r in merged_records]

    def _rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        acc = sum(1 for r in rows if r["accepted"])
        return {"cards": n, "accepted": acc, "acceptance_rate": acc / max(n, 1)}

    by_mechanism: dict[str, list[dict[str, Any]]] = {}
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    by_depth: dict[str, list[dict[str, Any]]] = {}
    for row in card_outcomes:
        by_mechanism.setdefault(str(row["mechanism"]), []).append(row)
        by_stratum.setdefault(str(row["support_stratum"]), []).append(row)
        depth = row.get("last_prototype_depth_km")
        third = ("unknown" if depth is None
                 else "shallow_0_333m" if depth < 1.0 / 3.0
                 else "mid_333_667m" if depth < 2.0 / 3.0
                 else "deep_667_1000m")
        by_depth.setdefault(third, []).append(row)
    summary = {
        "split": split,
        "pairs": total_pairs,
        "endpoints": len(merged_records),
        "parallel_workers": n_workers,
        "worker_seeds": worker_seeds,
        "merge_shuffle_seed": MERGE_SHUFFLE_SEED,
        "funnel": {
            "proposals": proposals,
            "accepted_pairs": total_pairs,
            "target_pairs": expected_total,
            "acceptance_rate": funnel["acceptance_rate"],
            "proposals_per_accepted_pair": funnel["proposals_per_accepted_pair"],
            "per_worker": funnel["per_worker"],
            "wall_clock_seconds_estimate": funnel["wall_clock_seconds_estimate"],
            "single_process_seconds_per_pair_smoke": 18.1,
            "speedup_vs_single_process_estimate": (
                18.1 * expected_total / max(max(worker_elapsed), 1e-9)
            ),
        },
        "shape_difference": {
            "support_jaccard": _quantiles(jaccard),
            "support_symmetric_difference_fraction": _quantiles(symdiff),
            "jaccard_le_0p5_fraction": float(np.mean(np.asarray(jaccard) <= 0.5)),
            "tier_counts": dict(tier_counts),
        },
        "tzz_dprime": {
            **_quantiles(tzz),
            "max_ok": bool(max(tzz) <= float(amb["tzz_dprime_max"]) + 1e-12),
            "near_ceiling_fraction_ge_0p95": float(np.mean(np.asarray(tzz) >= 0.95)),
        },
        "spin_dprime": {
            "spin1": _quantiles(spin1),
            "spin2": _quantiles(spin2),
            "max_spin": _quantiles(maxspin),
            "min_ok": bool(min(maxspin) >= float(amb["spin_dprime_min"])),
        },
        "density_nrmse": {
            **_quantiles(nrmse),
            "min_ok": bool(min(nrmse) >= float(amb["density_nrmse_min"])),
        },
        "acceptance_by_mechanism": {k: _rate(v) for k, v in sorted(by_mechanism.items())},
        "acceptance_by_stratum": {k: _rate(v) for k, v in sorted(by_stratum.items())},
        "acceptance_by_depth_third": {k: _rate(v) for k, v in sorted(by_depth.items())},
        "geometry_check": {
            "peak_abs_density_max": max(peaks),
            "peak_ok": bool(max(peaks) <= 0.80 + 1e-6),
            "support_cells": {
                "min": min(cells),
                "median": float(np.median(cells)),
                "max": max(cells),
                "min_ok": bool(min(cells) >= 82),
            },
        },
        "scale": _quantiles(scales),
        "response_group_counts_pairs": dict(Counter(
            str(r["response_group"]) for r in merged_pairs
        )),
        "mechanism_counts_pairs": dict(Counter(str(r["mechanism"]) for r in merged_pairs)),
        "support_stratum_counts_pairs": dict(Counter(
            str(r["support_stratum"]) for r in merged_pairs
        )),
        "verification": verification,
        "index": index_payload,
    }
    write_json(root / f"summary_{split}.json", summary)

    manifest = {
        "dataset_id": config["dataset_id"],
        "schema_version": config["schema_version"],
        "build_mode": "full_parallel",
        "parallel_workers": n_workers,
        "worker_seeds": worker_seeds,
        "worker_pairs": expected_counts,
        "merge_shuffle_seed": MERGE_SHUFFLE_SEED,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operator_sha256": per_worker[0]["manifest"].get(
            "operator_sha256", config["operator"]["g6_sha256"]
        ),
        "config_sha256": sha256_file(args.config),
        "split": split,
        "pairs": total_pairs,
        "gate_contract": {
            "gate_type": config["ambiguity"]["gate_type"],
            "tzz_dprime_max": float(amb["tzz_dprime_max"]),
            "spin_dprime_min": float(amb["spin_dprime_min"]),
            "density_nrmse_min": float(amb["density_nrmse_min"]),
            "shape_jaccard_max": float(amb["shape_jaccard_max"]),
            "shape_symmetric_difference_min": float(amb["shape_symmetric_difference_min"]),
        },
        "noise_contract": "v023_noise.V023WhiteNoise(halved sigma, online only)",
        "seed_registry": seed_registry_payload(config),
    }
    write_json(root / "manifest.json", manifest)
    registry = seed_registry_payload(config)
    registry["used_seeds"]["shape_workers"] = {
        f"w{w}": worker_seeds[w] for w in range(n_workers)
    }
    registry["used_seeds"]["merge_shuffle"] = MERGE_SHUFFLE_SEED
    write_json(root / "seed_registry.json", registry)
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge V023 shape-pair workers")
    parser.add_argument("--config", type=Path, default=ROOT / "config_shape_v023.json")
    parser.add_argument("--root", type=Path, default=ROOT / ".." / "pairs_shape")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--split", default="train")
    parser.add_argument("--counts", default=None,
                        help="Target pair count per worker, comma separated (default: batch B1's 348x5+347)")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="Worker seed base (default: batch B1's 2026091401)")
    parser.add_argument("--seed-step", type=int, default=1000)
    parser.add_argument("--tier1-only", action="store_true",
                        help="Exclude tier-2 pairs (writes <root>/<split>_tier2_excluded.jsonl); the final layout contains only tier 1")
    return parser.parse_args()


if __name__ == "__main__":
    print(merge(parse_args()))
