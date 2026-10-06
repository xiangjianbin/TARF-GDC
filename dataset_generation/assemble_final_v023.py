#!/usr/bin/env python3
'Assemble ordinary, separable and shape stages into the frozen standalone dataset. Deterministically trim pair quotas by group hash, shuffle, reassign sample IDs, and generate five independent frozen-noise realizations for validation/test. Preserve original sample IDs; training noise remains online.'
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from v017.config import load_config, sha256_file, stable_hash, write_json
from v017.storage import load_split, save_split_shards
from build_pairs_v023 import _quantiles
from v023_noise import V023WhiteNoise


ROOT = Path(__file__).resolve().parent
V023_ROOT = ROOT / ".."
FINAL_SEEDS = {"train": 2026091601, "validation_iid": 2026091602, "test_locked": 2026091603}
SHUFFLE_SEED_BASE = 2026091571
EXPECTED = {
    "train": {"ordinary": 12000, "separable_pairs": 3250, "shape_pairs": 3250, "total": 25000},
    "validation_iid": {"ordinary": 1200, "separable_pairs": 325, "shape_pairs": 325, "total": 2500},
    "test_locked": {"ordinary": 5000, "separable_pairs": 1427, "shape_pairs": 1428, "total": 10710},
}
TRACK_OF_SUBTYPE = {
    "ordinary": "ordinary",
    "ambiguity_redistribution_separable": "separable",
    "ambiguity_shape_tzz_equivalent": "shape",
}
FROZEN_SPLITS = ("validation_iid", "test_locked")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def _load_source(root: Path, split: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    if not (root / split).exists():
        return {}, [], []
    arrays, records = load_split(root, split)
    pairs = _read_jsonl(root / split / "pairs.jsonl")
    return arrays, records, pairs


def assemble_split(
    config: dict[str, Any],
    split: str,
    sources: dict[str, tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]],
    output: Path,
) -> dict[str, Any]:
    started = time.time()
    expected = EXPECTED[split]
    final_seed = FINAL_SEEDS[split]
    array_keys = ("density_zyx", "volume_fraction_zyx", "clean_d6")
    merged_arrays: dict[str, list[np.ndarray]] = {key: [] for key in array_keys}
    merged_records: list[dict[str, Any]] = []
    merged_pairs: list[dict[str, Any]] = []
    track_counts: Counter[str] = Counter()
    trimmed_rows: dict[str, list[dict[str, Any]]] = {"separable": [], "shape": []}
    for group_name, source_names, pair_quota in (
        ("ordinary", ("ordinary",), None),
        ("separable", ("separable", "separable_topup"), expected["separable_pairs"]),
        ("shape", ("shape", "shape_topup"), expected["shape_pairs"]),
    ):
        group_arrays: dict[str, list[np.ndarray]] = {key: [] for key in array_keys}
        group_records: list[dict[str, Any]] = []
        group_pairs: list[dict[str, Any]] = []
        for name in source_names:
            arrays, records, pairs = sources.get(name, ({}, [], []))
            if not records:
                continue
            for key in array_keys:
                group_arrays[key].append(arrays[key])
            group_records.extend(records)
            group_pairs.extend(pairs)
        if not group_records:
            if group_name != "ordinary" and pair_quota:
                raise AssertionError(f"{split}: no endpoints for track {group_name}")
            continue
        group_cat = {key: np.concatenate(value) for key, value in group_arrays.items()}
        if pair_quota is not None:
            if len(group_pairs) < pair_quota:
                raise AssertionError(
                    f"{split}: {group_name} pairs {len(group_pairs)} < quota {pair_quota}"
                )
            if len(group_pairs) > pair_quota:
                # Keep the first quota pairs ordered by stable group hash; retain rejected metadata.
                ranked = sorted(
                    group_pairs,
                    key=lambda p: stable_hash(str(p["ambiguity_group_id"]), 16),
                )
                keep_pairs = ranked[:pair_quota]
                trimmed = ranked[pair_quota:]
                keep_gids = {str(p["ambiguity_group_id"]) for p in keep_pairs}
                keep_idx = np.asarray([
                    i for i, r in enumerate(group_records)
                    if str(r["ambiguity_group_id"]) in keep_gids
                ])
                group_records = [group_records[int(i)] for i in keep_idx]
                group_cat = {key: value[keep_idx] for key, value in group_cat.items()}
                trimmed_rows[group_name].extend(trimmed)
                group_pairs = keep_pairs
        for key in array_keys:
            merged_arrays[key].append(group_cat[key])
        for row in group_records:
            row["generation_track"] = group_name
            merged_records.append(row)
            track_counts[group_name] += 1
        for row in group_pairs:
            row = dict(row)
            row["track"] = group_name
            merged_pairs.append(row)
    count = len(merged_records)
    sep_pairs = sum(1 for p in merged_pairs if p["track"] == "separable")
    shape_pairs = sum(1 for p in merged_pairs if p["track"] == "shape")
    assert sep_pairs == expected["separable_pairs"], (split, sep_pairs)
    assert shape_pairs == expected["shape_pairs"], (split, shape_pairs)
    assert track_counts["ordinary"] == expected["ordinary"], (split, track_counts)
    assert count == expected["total"], (
        f"{split}: endpoints {count} != expected {expected['total']}"
    )
    for track, rows in trimmed_rows.items():
        if rows:
            with (output / f"{split}_{track}_trimmed.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # Globally shuffle and reassign sample IDs.
    rng = np.random.default_rng(SHUFFLE_SEED_BASE + list(EXPECTED).index(split))
    order = rng.permutation(count)
    gid_of_index: list[str] = [""] * count
    new_records: list[dict[str, Any]] = []
    for new_index, old_index in enumerate(order):
        record = dict(merged_records[int(old_index)])
        old_id = str(record["sample_id"])
        sample_id = (
            f"v023_{split}_{new_index:06d}_"
            f"{stable_hash([final_seed, new_index, record['subtype']], 10)}"
        )
        record["original_sample_id"] = old_id
        record["sample_id"] = sample_id
        record["split"] = split
        new_records.append(record)
        gid_of_index[new_index] = str(record.get("ambiguity_group_id", ""))
    arrays_out: dict[str, Any] = {
        key: np.concatenate(merged_arrays[key])[order].astype(
            config["storage"]["density_dtype"] if key == "density_zyx"
            else config["storage"]["response_dtype"] if key == "clean_d6"
            else np.float16
        )
        for key in array_keys
    }
    arrays_out.update({
        "sample_id": np.asarray([r["sample_id"] for r in new_records], dtype="U64"),
        "mechanism": np.asarray([r["mechanism"] for r in new_records], dtype="U24"),
        "subtype": np.asarray([r["subtype"] for r in new_records], dtype="U40"),
        "support_subtype": np.asarray([r["support_subtype"] for r in new_records], dtype="U32"),
        "density_pattern": np.asarray([r["density_pattern"] for r in new_records], dtype="U48"),
        "ambiguity_group_id": np.asarray([r["ambiguity_group_id"] for r in new_records], dtype="U96"),
        "ambiguity_pair_role": np.asarray([r["ambiguity_pair_role"] for r in new_records], dtype="U1"),
        "response_group": np.asarray([r["response_group"] for r in new_records], dtype="U12"),
        "support_stratum": np.asarray([r["support_stratum"] for r in new_records], dtype="U24"),
        "pair_geometry_mode": np.asarray([r["pair_geometry_mode"] for r in new_records], dtype="U20"),
        "generation_track": np.asarray([r["generation_track"] for r in new_records], dtype="U12"),
        "original_sample_id": np.asarray([r["original_sample_id"] for r in new_records], dtype="U64"),
    })

    noise_check = None
    if split in FROZEN_SPLITS:
        noise = V023WhiteNoise(config)
        n_real = int(config["noise"]["frozen_realizations"])
        frozen_obs = np.empty((count, n_real, 6, 33, 33), dtype=np.float32)
        frozen_mode = np.empty((count, n_real), dtype="U24")
        frozen_scale = np.empty((count, n_real), dtype=np.float32)
        frozen_seed = np.empty((count, n_real), dtype=np.uint64)
        for index, record in enumerate(new_records):
            clean = arrays_out["clean_d6"][index]
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
                    "realization": r, "mode": view.mode,
                    "amplitude_scale": float(view.amplitude_scale),
                    "seed": int(draw_seed),
                })
            record["frozen_noise_realizations"] = real_meta
        arrays_out["frozen_observed_d6"] = frozen_obs
        arrays_out["frozen_noise_mode"] = frozen_mode
        arrays_out["frozen_noise_scale"] = frozen_scale
        arrays_out["frozen_noise_seed"] = frozen_seed
        residual = frozen_obs.astype(np.float64) - np.repeat(
            arrays_out["clean_d6"][:, None].astype(np.float64), n_real, axis=1
        )
        sigma = np.asarray(config["noise"]["base_component_sigma_eotvos"], dtype=np.float64)
        rms = np.sqrt(np.mean(residual**2, axis=(0, 1, 3, 4)))
        rel = rms / sigma - 1.0
        noise_check = {
            "components": list(config["operator"]["component_order"]),
            "configured_sigma_eotvos": [float(v) for v in sigma],
            "observed_residual_rms_eotvos": [float(v) for v in rms],
            "relative_error": [float(v) for v in rel],
            "within_tolerance": bool(np.all(np.abs(rel) <= 0.05)),
            "tolerance": 0.05,
            "realizations_per_endpoint": n_real,
            "amplitude_scale_unique": sorted({float(v) for v in frozen_scale.ravel()}),
        }

    index_payload = save_split_shards(
        output, split, arrays_out, new_records, int(config["storage"]["shard_size"])
    )
    with (output / split / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in merged_pairs:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # Storage is deep-to-shallow; physical depth reverses the geometry z coordinate.
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    density_abs = np.abs(arrays_out["density_zyx"].astype(np.float32))
    mask = density_abs >= threshold
    z_idx = np.arange(16, dtype=np.float32)[:, None, None]
    z_mean = (z_idx * mask).sum(axis=(1, 2, 3)) / np.maximum(
        mask.sum(axis=(1, 2, 3)), 1
    )
    depth_arr = (16.0 - z_mean - 0.5) * 0.0625
    tracks = np.asarray([r["generation_track"] for r in new_records])
    ordinary_depth = depth_arr[tracks == "ordinary"]
    pair_endpoint_depth = depth_arr[tracks != "ordinary"]
    # Separable endpoints share the prototype support/depth; shape pairs occupy shallow/middle depths.
    summary = {
        "split": split,
        "total_endpoints": count,
        "ordinary_endpoints": int(track_counts["ordinary"]),
        "separable_pairs": sep_pairs,
        "shape_pairs": shape_pairs,
        "final_seed": final_seed,
        "track_counts_endpoints": dict(track_counts),
        "trimmed_pairs": {k: len(v) for k, v in trimmed_rows.items()},
        "depth": {
            "ordinary_deep_fraction_ge_600m": float(np.mean(ordinary_depth >= 0.6)),
            "separable_endpoint_deep_fraction_ge_600m": float(
                np.mean(pair_endpoint_depth[tracks[tracks != "ordinary"] == "separable"] >= 0.6)
            ) if np.any(tracks == "separable") else None,
            "shape_endpoint_deep_fraction_ge_600m": float(
                np.mean(pair_endpoint_depth[tracks[tracks != "ordinary"] == "shape"] >= 0.6)
            ) if np.any(tracks == "shape") else None,
            "all_depth_km_quantiles": _quantiles([float(v) for v in depth_arr]),
        },
        "response_group_counts": dict(Counter(
            str(r["response_group"]) for r in new_records
        )),
        "frozen_noise_check": noise_check,
        "elapsed_seconds": time.time() - started,
        "index": index_payload,
    }
    write_json(output / f"summary_{split}.json", summary)
    return summary


def assemble(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    v023_root = Path(args.v023_root).resolve()
    output = v023_root / "dataset"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    started = time.time()
    summaries = {}
    for split in ("train", "validation_iid", "test_locked"):
        staging = {
            "ordinary": v023_root / "ordinary_stage",
            "separable": v023_root / "pairs_separable",
            "separable_topup": v023_root / "pairs_separable_topup",
            "shape": v023_root / "pairs_shape",
            "shape_topup": v023_root / "pairs_shape_topup" / split,
        }
        sources = {name: _load_source(root, split) for name, root in staging.items()}
        print(f"[{time.strftime('%F %T')}] assembling {split}", flush=True)
        summaries[split] = assemble_split(config, split, sources, output)
        print(
            f"[{time.strftime('%F %T')}] {split}: {summaries[split]['total_endpoints']} endpoints, "
            f"elapsed={summaries[split]['elapsed_seconds']:.0f}s",
            flush=True,
        )
    manifest = {
        "dataset_id": "V023_STANDALONE_FULL",
        "schema_version": "23.2.0",
        "build_mode": "final_assembly",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "splits": {k: {"total_endpoints": v["total_endpoints"],
                       "ordinary_endpoints": v["ordinary_endpoints"],
                       "separable_pairs": v["separable_pairs"],
                       "shape_pairs": v["shape_pairs"],
                       "final_seed": v["final_seed"]}
                   for k, v in summaries.items()},
        "expected_table": EXPECTED,
        "shuffle_seed_base": SHUFFLE_SEED_BASE,
        "frozen_noise": {
            "type": "white_gaussian_v023",
            "realizations_per_endpoint": int(config["noise"]["frozen_realizations"]),
            "amplitude_scale": 1.0,
            "splits": list(FROZEN_SPLITS),
            "seeds": {k: int(v) for k, v in config["noise"]["frozen_evaluation_seeds"].items()},
        },
        "train_noise_policy": "online white noise U(0.75,1.5), never stored",
        "staging_sources": {
            "ordinary": "ordinary_stage/<split>",
            "separable": "pairs_separable/train",
            "separable_topup": "pairs_separable_topup/<split>",
            "shape": "pairs_shape/train",
            "shape_topup": "pairs_shape_topup/<split>/<split>",
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "manifest.json", manifest)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the final V023 dataset")
    parser.add_argument("--config", type=Path, default=ROOT / "config_v023.json")
    parser.add_argument("--v023-root", type=Path, default=ROOT / "..")
    return parser.parse_args()


if __name__ == "__main__":
    print(assemble(parse_args()))
