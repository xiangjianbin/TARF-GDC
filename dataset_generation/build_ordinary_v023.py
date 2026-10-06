#!/usr/bin/env python3
'Generate ordinary V023 endpoints with Sobol geometry and signed density. Ten percent of designated samples target support centroids at least 600 m deep. Swap infeasible deep mechanism labels without changing marginal counts. Preserve shape names in support_subtype and density_pattern.'
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v017.config import load_config, sha256_file, stable_hash, write_json
from v017.geometry import SobolStream, generate_model
from v017.physics import GravityTensorOperator, independent_prism_audit
from v017.storage import save_split_shards
from build_pairs_v023 import (
    V022_SEEDS,
    _refresh_record,
    _quantiles,
    allocate,
    assign_deep_flags,
    deep_target_config,
    exact_labels,
    repair_deep_mechanisms,
)


ROOT = Path(__file__).resolve().parent
V023_USED_SEEDS = (
    2026091201, 2026091202, 2026091203,
    2026091211, 2026091212, 2026091213, 2026091214, 2026091299,
    2026091301, 2026091311,
    2026091401, 2026092401, 2026093401, 2026094401, 2026095401, 2026096401,
    2026091471,
)
STRATA = ("small_0p5_2pct", "main_2_8pct", "large_8_18pct")
DEEP_INFEASIBLE_MECHANISM = "B_structural"
DEEP_INFEASIBLE_STRATUM = "large_8_18pct"
CELL_Z_KM = 0.0625
EXTENT_Z_KM = 1.0


def support_centroid_depth_km(density: np.ndarray, threshold: float) -> float:
    'Unweighted support-centroid depth; array index zero is the deepest layer.'
    mask = np.abs(density) >= threshold
    z_index = np.nonzero(mask)[0]
    return float((density.shape[0] - float(z_index.mean()) - 0.5) * CELL_Z_KM)


def ordinary_schedule(
    config: dict[str, Any], count: int, seed: int
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(int(seed))
    mechanisms = exact_labels(
        allocate(count, config["ordinary_mechanism_mixture"]), rng
    )
    strata = exact_labels(
        allocate(
            count,
            {k: float(config["support_fraction_mixture"][k]["probability"]) for k in STRATA},
        ),
        rng,
    )
    return mechanisms, strata


def build_ordinary(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    split = str(args.split)
    if split not in ("train", "validation_iid", "test_locked"):
        raise ValueError(split)
    seed = int(args.seed)
    forbidden = set(V022_SEEDS) | set(V023_USED_SEEDS)
    if seed in forbidden:
        raise ValueError(f"--seed {seed} collides with earlier registries")
    count = int(args.n)
    deep_cfg = config.get("deep_targeting", {})
    deep_fraction = (
        float(args.deep_fraction) if args.deep_fraction is not None
        else float(deep_cfg.get("fraction", 0.0))
    )
    output = Path(args.output).resolve()
    if (output / split).exists():
        raise FileExistsError(f"refusing to overwrite: {output / split}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(output / "config_snapshot.json",
               {k: v for k, v in config.items() if not k.startswith("_")})
    operator = GravityTensorOperator(config, args.device)
    physics_audit = independent_prism_audit(operator, config)
    write_json(output / "independent_physics_audit.json", physics_audit)
    if float(physics_audit["maximum_component_relative_l2"]) > float(
        config["audit"]["independent_prism_relative_l2_max"]
    ):
        raise RuntimeError(f"independent prism audit failed: {physics_audit}")

    mechanisms, strata = ordinary_schedule(config, count, seed)
    deep_flags = assign_deep_flags(count, deep_fraction, seed)
    mechanisms, swaps = repair_deep_mechanisms(mechanisms, strata, deep_flags)
    config_deep = deep_target_config(config)
    rng = np.random.default_rng(seed + 1000)
    sobol = SobolStream(seed + 1000)
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    densities: list[np.ndarray] = []
    fractions: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    depths: list[float] = []
    for index, (mechanism, stratum, deep) in enumerate(zip(mechanisms, strata, deep_flags)):
        model = generate_model(
            rng, sobol, config_deep if deep else config, mechanism,
            tuple(config["support_fraction_mixture"][stratum]["range"]),
        )
        depth = support_centroid_depth_km(model.density, threshold)
        if deep and depth < float(config_deep["geometry"]["target_min_centroid_depth_km"]):
            raise RuntimeError(f"deep targeting failed for sample {index}: {depth}")
        record = dict(model.record)
        record.update({
            "support_mechanism": record["mechanism"],
            "support_subtype": record["subtype"],
            "subtype": "ordinary",
            "density_pattern": model.record["subtype"],
            "ambiguity_group_id": "",
            "ambiguity_pair_role": "",
            "response_group": "ordinary",
            "pair_geometry_mode": "",
            "candidate_selection": (
                "ordinary_sobol_geometry_deep_targeted" if deep
                else "ordinary_sobol_geometry"
            ),
            "deep_targeted": bool(deep),
            "support_centroid_depth_km_v025": depth,
        })
        densities.append(model.density)
        fractions.append(model.volume_fraction)
        records.append(record)
        depths.append(depth)
        if (index + 1) % 500 == 0:
            write_json(output / f"build_progress_{split}.json", {
                "split": split, "generated": index + 1, "target": count,
                "elapsed_seconds": time.time() - started,
            })
            print(
                f"[{time.strftime('%F %T')}] V023 ordinary {split}: "
                f"{index + 1}/{count}, elapsed={time.time() - started:.0f}s",
                flush=True,
            )
    density = np.stack(densities).astype(np.float32)
    response = operator.forward(density, batch_size=128)

    rng_shuffle = np.random.default_rng(seed + 90000)
    order = rng_shuffle.permutation(count)
    endpoints = [{
        "density": density[int(i)],
        "volume_fraction": fractions[int(i)],
        "response": response[int(i)],
        "record": records[int(i)],
        "support_stratum": strata[int(i)],
    } for i in order]
    final_records = []
    for index, endpoint in enumerate(endpoints):
        sample_id = (
            f"v023o_{split}_{index:05d}_"
            f"{stable_hash([seed, index, 'ordinary'], 10)}"
        )
        final_records.append(_refresh_record(
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
        "sample_id": np.asarray([r["sample_id"] for r in final_records], dtype="U64"),
        "mechanism": np.asarray([r["mechanism"] for r in final_records], dtype="U24"),
        "subtype": np.asarray([r["subtype"] for r in final_records], dtype="U40"),
        "support_subtype": np.asarray([r["support_subtype"] for r in final_records], dtype="U32"),
        "density_pattern": np.asarray([r["density_pattern"] for r in final_records], dtype="U40"),
        "ambiguity_group_id": np.asarray([r["ambiguity_group_id"] for r in final_records], dtype="U96"),
        "ambiguity_pair_role": np.asarray([r["ambiguity_pair_role"] for r in final_records], dtype="U1"),
        "response_group": np.asarray([r["response_group"] for r in final_records], dtype="U12"),
        "support_stratum": np.asarray([r["support_stratum"] for r in final_records], dtype="U24"),
        "pair_geometry_mode": np.asarray([r["pair_geometry_mode"] for r in final_records], dtype="U20"),
    }
    index_payload = save_split_shards(
        output, split, arrays, final_records, int(config["storage"]["shard_size"])
    )
    depth_arr = np.asarray(depths)
    summary = {
        "split": split,
        "count": count,
        "seed": seed,
        "deep_fraction_target": deep_fraction,
        "deep_targeted_count": int(sum(deep_flags)),
        "deep_actual_fraction_ge_600m": float(np.mean(depth_arr >= 0.6)),
        "depth_km_quantiles": _quantiles([float(v) for v in depth_arr]),
        "schedule_swaps_deep_infeasible": int(swaps),
        "mechanism_counts": dict(Counter(mechanisms)),
        "support_stratum_counts": dict(Counter(strata)),
        "elapsed_seconds": time.time() - started,
        "index": index_payload,
    }
    write_json(output / f"summary_{split}.json", summary)
    manifest = {
        "dataset_id": f"V023_ORDINARY_{split.upper()}",
        "build_mode": "ordinary",
        "split": split,
        "seed": seed,
        "count": count,
        "deep_fraction": deep_fraction,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operator_sha256": operator.operator_sha256,
        "config_sha256": sha256_file(args.config),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k: summary[k] for k in (
        "count", "deep_targeted_count", "deep_actual_fraction_ge_600m",
        "schedule_swaps_deep_infeasible", "elapsed_seconds")}, ensure_ascii=False), flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V023 ordinary endpoints")
    parser.add_argument("--config", type=Path, default=ROOT / "config_v023.json")
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deep-fraction", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(build_ordinary(parse_args()))
