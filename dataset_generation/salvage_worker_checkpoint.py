#!/usr/bin/env python3
'Recover accepted pairs from interrupted worker checkpoints. Rebuild canonical shards, records, pair metadata and summaries for worker merging. Proposal counts are approximate when taken from the last progress record; this is recovery, not exact process resumption.'
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from v017.config import load_config, write_json
from build_shape_pairs_v023 import assemble_shape_split


ROOT = Path(__file__).resolve().parent


def salvage(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    wdir = Path(args.worker_dir).resolve()
    split = str(args.split)
    if (wdir / split).exists() or (wdir / "freeze_manifest.json").exists():
        raise FileExistsError("Recover into a copy of an unfinished worker, not completed/frozen data")
    ckpt = wdir / "checkpoint"
    meta_path = ckpt / "accepted_meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"no checkpoint meta: {meta_path}")
    metas = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines() if l]
    densities_parts = []
    responses_parts = []
    fractions_parts = []
    for npz_path in sorted(ckpt.glob("accepted_batch_*.npz")):
        with np.load(npz_path) as payload:
            densities_parts.append(payload["densities"])
            responses_parts.append(payload["responses"])
            fractions_parts.append(payload["volume_fraction"])
    densities = np.concatenate(densities_parts)
    responses = np.concatenate(responses_parts)
    fractions = np.concatenate(fractions_parts)
    assert len(metas) == len(densities), (len(metas), len(densities))
    pair_items = []
    for i, meta in enumerate(metas):
        card = dict(meta["card"])
        card["failure_history"] = Counter(card.get("failure_history", {}))
        pair_items.append({
            "card": card,
            "densities": (densities[i, 0], densities[i, 1]),
            "responses": (responses[i, 0], responses[i, 1]),
            "volume_fraction": fractions[i],
            "metrics": meta["metrics"],
            "support_metrics": meta["support_metrics"],
            "scale": float(meta["scale"]),
            "shape_gate_tier": int(meta["shape_gate_tier"]),
            "source_record": meta["source_record"],
            "optimizer": meta["optimizer"],
            "base_contrast_gcc_actual": float(meta["base_contrast_gcc_actual"]),
            "redistribution_amplitude_gcc_actual": float(
                meta["redistribution_amplitude_gcc_actual"]
            ),
            "generation_attempts_for_target": int(meta["generation_attempts_for_target"]),
        })
    seed = int(args.seed)
    worker_tag = f"w{int(args.worker_id)}"
    index_payload, pair_rows, records = assemble_shape_split(
        config, split, seed, pair_items, wdir, worker_tag=worker_tag,
    )
    progress = {}
    progress_path = wdir / f"build_progress_{split}.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    funnel = {
        "split": split,
        "target_pairs": int(progress.get("target_pairs", len(pair_rows))),
        "accepted_pairs": len(pair_rows),
        "shortfall": int(progress.get("target_pairs", len(pair_rows))) - len(pair_rows),
        "proposals": int(progress.get("proposals", 0)),
        "acceptance_rate": len(pair_rows) / max(int(progress.get("proposals", 0)), 1),
        "proposals_per_accepted_pair": int(progress.get("proposals", 0)) / max(len(pair_rows), 1),
        "seconds_per_accepted_pair": 0.0,
        "failure_reasons": progress.get("failure_reasons", {}),
        "scale_level_reject_reasons": {},
        "exhausted_targets": [],
        "card_outcomes": [],
        "wall_clock_stop": True,
        "shape_gate_tier_counts": dict(Counter(str(r["shape_gate_tier"]) for r in pair_rows)),
        "metric_version": "absolute_covariance_global_dprime_v019",
        "elapsed_seconds": float(progress.get("elapsed_seconds", 0.0)),
        "salvaged_from_checkpoint": True,
    }
    write_json(wdir / f"acceptance_funnel_{split}.json", funnel)
    manifest = json.loads((wdir / "manifest.json").read_text(encoding="utf-8")) if (
        wdir / "manifest.json"
    ).exists() else {}
    manifest.update({
        "salvaged_from_checkpoint": True,
        "pairs": len(pair_rows),
        "salvaged_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    write_json(wdir / "manifest.json", manifest)
    print(f"salvaged {len(pair_rows)} pairs from {len(metas)} checkpoint items")
    return wdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Salvage a killed worker checkpoint")
    parser.add_argument("--config", type=Path, default=ROOT / "config_shape_v023.json")
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(salvage(parse_args()))
