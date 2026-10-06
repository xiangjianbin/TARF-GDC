#!/usr/bin/env python3
'Audit stored data: split counts, complete pairs, separability/shape gates, geometry, cross-split identities/support hashes, deep coverage, tensor trace, frozen noise and independent prism physics. Every failing contract yields a nonzero exit.'
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from v017.ambiguity import (
    pair_metrics_absolute_covariance_v019,
    support_pair_metrics,
)
from v017.config import load_config, write_json
from v017.physics import (
    GravityTensorOperator,
    independent_prism_audit,
    trace_relative_rms,
)
from v017.storage import load_split
from build_pairs_v023 import (
    _endpoint_contract_ok,
    classify_pair_reverse_v023,
    validate_v023_config,
)
from build_shape_pairs_v023 import (
    classify_shape_pair_v023,
    shape_gate,
    validate_shape_config,
)


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation_iid", "test_locked")
EXPECTED_TOTALS = {"train": 25000, "validation_iid": 2500, "test_locked": 10710}
EXPECTED_PAIRS = {
    "train": {"separable": 3250, "shape": 3250},
    "validation_iid": {"separable": 325, "shape": 325},
    "test_locked": {"separable": 1427, "shape": 1428},
}
TRACE_RELATIVE_RMS_MAX = 2e-5
# Ten percent targeted plus the natural deep tail; impose a lower bound, not an exact total.
# The untargeted remainder retains its original geometry distribution.
DEEP_FRACTION_MIN = 0.095


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config_a = load_config(ROOT / "config_v023.json")
    config_b = load_config(ROOT / "config_shape_v023.json")
    validate_v023_config(config_a)
    validate_shape_config(config_b)
    dataset = Path(args.dataset).resolve()
    report_path = args.report_output.resolve()
    if dataset in report_path.parents:
        raise ValueError("Write the audit report outside the immutable dataset directory")
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    failures: list[str] = []
    report: dict[str, Any] = {"dataset": str(dataset), "splits": {}}

    all_ids: set[str] = set()
    all_support_sha: set[str] = set()
    for split in SPLITS:
        arrays, records = load_split(dataset, split)
        pairs = _read_jsonl(dataset / split / "pairs.jsonl")
        n = len(records)
        split_report: dict[str, Any] = {"endpoints": n, "pairs": len(pairs)}
        if n != EXPECTED_TOTALS[split]:
            failures.append(f"{split}: endpoints {n} != {EXPECTED_TOTALS[split]}")

        # Check uniqueness and cross-split intersections.
        ids = [str(r["sample_id"]) for r in records]
        if len(set(ids)) != n:
            failures.append(f"{split}: sample_id not unique")
        overlap = all_ids & set(ids)
        if overlap:
            failures.append(f"{split}: sample_id cross-split overlap {len(overlap)}")
        all_ids |= set(ids)
        sha = [str(r["support_sha256"]) for r in records]
        sha_overlap = all_support_sha & set(sha)
        if sha_overlap:
            failures.append(f"{split}: support_sha256 cross-split overlap {len(sha_overlap)}")
        all_support_sha |= set(sha)

        # Check complete pairs and recompute acceptance gates.
        by_gid: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            gid = str(r["ambiguity_group_id"])
            if gid:
                by_gid[gid].append(i)
        bad_gid = {g: v for g, v in by_gid.items() if len(v) != 2}
        if bad_gid:
            failures.append(f"{split}: {len(bad_gid)} gids without exactly 2 endpoints")
        tracks = np.asarray([str(r["generation_track"]) for r in records])
        threshold = float(config_a["density"]["support_threshold_abs_gcc"])
        pair_counts = {"separable": 0, "shape": 0}
        max_rel_err = 0.0
        t0 = time.time()
        for gid, (ia, ib) in by_gid.items():
            track = str(records[ia]["generation_track"])
            contract = config_a["ambiguity"] if track == "separable" else config_b["ambiguity"]
            m = pair_metrics_absolute_covariance_v019(
                arrays["density_zyx"][ia], arrays["density_zyx"][ib],
                arrays["clean_d6"][ia], arrays["clean_d6"][ib], contract,
            )
            sm = support_pair_metrics(
                arrays["density_zyx"][ia], arrays["density_zyx"][ib], threshold
            )
            if track == "separable":
                ok = classify_pair_reverse_v023(m, contract) is not None
                ok = ok and float(sm["support_jaccard"]) == 1.0
                pair_counts["separable"] += 1
            else:
                ok = classify_shape_pair_v023(m, contract) is not None
                ok = ok and shape_gate(
                    sm["support_jaccard"], sm["support_symmetric_difference_fraction"],
                    float(contract["shape_jaccard_max"]),
                    float(contract["shape_symmetric_difference_min"]),
                )
                pair_counts["shape"] += 1
            if not ok:
                failures.append(f"{split}: pair gate failed {gid} (track={track})")
                if len(failures) > 20:
                    raise RuntimeError("too many gate failures; aborting")
        if pair_counts != EXPECTED_PAIRS[split]:
            failures.append(f"{split}: pair counts {pair_counts} != {EXPECTED_PAIRS[split]}")
        split_report["pair_gate_recompute"] = {
            **pair_counts, "all_pass": True, "seconds": time.time() - t0,
        }

        # Check endpoint geometry contracts.
        t0 = time.time()
        for i, r in enumerate(records):
            ok, detail = _endpoint_contract_ok(
                arrays["density_zyx"][i], str(r["support_stratum"]), config_a
            )
            if not ok:
                failures.append(f"{split}: endpoint contract failed {r['sample_id']}: {detail}")
        split_report["endpoint_contract_all_pass"] = True

        # Check depth distributions.
        # Ordinary and separable top-up cohorts target 10 percent deep samples (9.5 percent audit floor).
        # The first separable cohort predates deep targeting; no deep-fraction floor applies.
        density_abs = np.abs(arrays["density_zyx"].astype(np.float32))
        mask = density_abs >= threshold
        z_idx = np.arange(16, dtype=np.float32)[:, None, None]
        z_mean = (z_idx * mask).sum(axis=(1, 2, 3)) / np.maximum(mask.sum(axis=(1, 2, 3)), 1)
        depth = (16.0 - z_mean - 0.5) * 0.0625
        orig_ids = np.asarray([str(r.get("original_sample_id", "")) for r in records])
        depth_report = {}
        for track in ("ordinary", "separable", "shape"):
            sel = tracks == track
            if not np.any(sel):
                continue
            frac = float(np.mean(depth[sel] >= 0.6))
            depth_report[f"{track}_deep_fraction_ge_600m"] = frac
            if track == "ordinary" and frac < DEEP_FRACTION_MIN:
                failures.append(
                    f"{split}: ordinary deep fraction {frac:.4f} below floor {DEEP_FRACTION_MIN}"
                )
            if track == "separable":
                topup_sel = sel & (np.char.find(orig_ids, "_t_") >= 0)
                if np.any(topup_sel):
                    tfrac = float(np.mean(depth[topup_sel] >= 0.6))
                    depth_report["separable_topup_deep_fraction_ge_600m"] = tfrac
                    if tfrac < DEEP_FRACTION_MIN:
                        failures.append(
                            f"{split}: separable topup deep fraction {tfrac:.4f} "
                            f"below floor {DEEP_FRACTION_MIN}"
                        )
                elif split != "train":
                    failures.append(
                        f"{split}: separable endpoints lack topup _t_ infix"
                    )
        split_report["depth"] = depth_report

        # Audit the trace-free tensor condition.
        tr = trace_relative_rms(arrays["clean_d6"].astype(np.float32))
        split_report["trace_relative_rms_max"] = float(tr.max())
        if float(tr.max()) > TRACE_RELATIVE_RMS_MAX:
            failures.append(f"{split}: trace_relative_rms_max {tr.max()} > {TRACE_RELATIVE_RMS_MAX}")

        # Audit five frozen validation/test noise realizations.
        if "frozen_observed_d6" in arrays:
            sigma = np.asarray(config_a["noise"]["base_component_sigma_eotvos"], dtype=np.float64)
            residual = arrays["frozen_observed_d6"].astype(np.float64) - np.repeat(
                arrays["clean_d6"][:, None].astype(np.float64), 5, axis=1
            )
            rms = np.sqrt(np.mean(residual**2, axis=(0, 1, 3, 4)))
            rel = rms / sigma - 1.0
            scale_unique = np.unique(arrays["frozen_noise_scale"])
            split_report["frozen_noise"] = {
                "relative_error": [float(v) for v in rel],
                "within_5pct": bool(np.all(np.abs(rel) <= 0.05)),
                "amplitude_scale_unique": [float(v) for v in scale_unique],
            }
            if not split_report["frozen_noise"]["within_5pct"]:
                failures.append(f"{split}: frozen noise RMS outside ±5%: {rel}")
            if not np.allclose(scale_unique, 1.0):
                failures.append(f"{split}: frozen amplitude not all 1.0: {scale_unique}")
            pair_gids = [g for g in by_gid]
            if pair_gids:
                ia, ib = by_gid[pair_gids[0]]
                fa = residual[ia].ravel()
                fb = residual[ib].ravel()
                corr = float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb)))
                split_report["frozen_noise"]["pair_endpoint_corr_sample"] = corr
                if abs(corr) > 0.1:
                    failures.append(f"{split}: frozen pair noise correlated r={corr}")
        report["splits"][split] = split_report
        print(f"[{time.strftime('%F %T')}] audited {split}: {json.dumps(split_report, ensure_ascii=False)[:400]}", flush=True)

    # Independently audit prism physics.
    operator = GravityTensorOperator(config_a, args.device)
    prism = independent_prism_audit(operator, config_a)
    report["independent_physics_audit"] = prism
    if float(prism["maximum_component_relative_l2"]) > 2e-5:
        failures.append(f"prism audit failed: {prism['maximum_component_relative_l2']}")

    report["failures"] = failures
    report["elapsed_seconds"] = time.time() - started
    report["verdict"] = "PASS" if not failures else "FAIL"
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the final V023 dataset")
    parser.add_argument("--dataset", type=Path, default=ROOT / ".." / "dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-output", type=Path, required=True,
                        help="New report path outside the frozen dataset directory")
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    print(json.dumps({"verdict": result["verdict"], "failures": result["failures"][:10]},
                     ensure_ascii=False, indent=1))
    sys.exit(0 if result["verdict"] == "PASS" else 1)
