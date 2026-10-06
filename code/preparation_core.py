"""Audit frozen V023 and prepare training-only normalization and linear inputs.

Default preparation only hashes/reads metadata and array headers of test_locked.
Its array cache requires the explicit final comparison freeze issued by the
pipeline. All derivative files live in V027/artifacts, never in the dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import time
import zipfile

import numpy as np
import scipy.linalg
import torch
from threadpoolctl import threadpool_limits

from v024_inversion.data import (DATASET_ID, SCHEMA_VERSION, SPLIT_COUNTS,
                                  SUPPORT_THRESHOLD, read_json, read_split_metadata)
from v024_inversion.noise import V023_NOISE_CONFIG, V023WhiteNoise
from v024_inversion.physics import GravityOperator, Normalization, role_matrix
from v024_inversion.utils import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
DATASET = None  # The public CLI always supplies the asset directory.
EXPECTED_SUMS = "75f2d230b211897ecf154ec3f6bc00592425835a6e26a505914eae19afc2c26e"
EXPECTED_G6 = "544d1ceeb63f7da3e83be606e613fb5c3ff6cbe11b7585e1f85c281d71346b0c"
G6_NAME = "g6_vinton_4x4x1km_16x32x32_rx33x33_v0003.npy"
SHAPES = {"density_zyx": (16, 32, 32), "clean_d6": (6, 33, 33),
          "frozen_observed_d6": (5, 6, 33, 33)}


def note(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def audit_frozen(dataset: Path, artifacts: Path) -> dict:
    started = time.perf_counter()
    freeze = read_json(dataset / "freeze_manifest.json")
    digest = sha256_file(dataset / "SHA256SUMS")
    if digest != EXPECTED_SUMS or digest != freeze["sha256sums_sha256"]:
        raise RuntimeError("V023 SHA256SUMS does not match its declared freeze")
    entries = {}
    for line in (dataset / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (dataset / name).resolve()
        if dataset not in path.parents or name in entries:
            raise RuntimeError("invalid frozen file entry")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen source changed: {name}")
        entries[name] = actual
    if len(entries) != freeze["file_count"]:
        raise RuntimeError("frozen file count mismatch")
    manifest = read_json(dataset / "manifest.json")
    if (manifest["dataset_id"], manifest["schema_version"]) != (DATASET_ID, SCHEMA_VERSION):
        raise RuntimeError("unexpected dataset identity")
    previous: dict[str, set] = {}
    result = {"status": "PASS", "source_sha256sums": digest, "frozen_files": len(entries),
              "frozen_bytes": sum((dataset / name).stat().st_size for name in entries), "splits": {}}
    for split, count in SPLIT_COUNTS.items():
        records, pairs = read_split_metadata(dataset, split)
        intersections = {}
        for field in ("sample_id", "original_sample_id", "ambiguity_group_id", "support_sha256", "density_sha256", "base_prototype_id"):
            values = {str(r.get(field, "")) for r in records} - {""}
            overlap = previous.get(field, set()) & values
            intersections[field] = len(overlap)
            if overlap:
                raise RuntimeError(f"cross-split {field} overlap in {split}: {len(overlap)}")
            previous.setdefault(field, set()).update(values)
        index = read_json(dataset / split / "index.json")
        expected_start = 0
        for shard in index["shards"]:
            n = int(shard["count"])
            if shard["start"] != expected_start or shard["stop"] != expected_start + n:
                raise RuntimeError(f"noncontiguous shard index in {split}")
            expected_start += n
            # Read NPZ directory + NPY headers only; no test numeric arrays here.
            with zipfile.ZipFile(dataset / split / shard["file"]) as archive:
                names = set(archive.namelist())
                keys = ("density_zyx", "clean_d6") + (("frozen_observed_d6",) if split != "train" else ())
                for key in keys:
                    with archive.open(key + ".npy") as handle:
                        version = np.lib.format.read_magic(handle)
                        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                        shape, _, dtype = reader(handle)
                    if shape != (n, *SHAPES[key]) or dtype != np.dtype("float32"):
                        raise RuntimeError(f"{split}/{shard['file']}/{key}: shape/dtype mismatch")
                if split == "train" and "frozen_observed_d6.npy" in names:
                    raise RuntimeError("train unexpectedly contains frozen observations")
        if expected_start != count or index["count"] != count:
            raise RuntimeError("split coverage mismatch")
        result["splits"][split] = {"endpoints": count, "pairs": len(pairs),
            "track_endpoints": dict(Counter(r["generation_track"] for r in records)),
            "cross_split_overlap": intersections, "shard_headers_checked": len(index["shards"])}
        note(f"metadata/header audit {split}: {count} endpoints, {len(pairs)} groups")
    source_config_path = ROOT / "dataset_generation/config_v023.json"
    source_config = read_json(source_config_path)
    for key, value in V023_NOISE_CONFIG.items():
        if source_config["noise"].get(key) != value:
            raise RuntimeError(f"local V023 noise contract differs from production: {key}")
    if source_config["density"]["support_threshold_abs_gcc"] != SUPPORT_THRESHOLD:
        raise RuntimeError("support threshold differs from production")
    g6_path = dataset.parent / "operator/g6.npy"
    if sha256_file(g6_path) != EXPECTED_G6:
        raise RuntimeError("frozen forward operator hash differs")
    original_audit = read_json(dataset / "audit_report.json")
    if original_audit.get("verdict") != "PASS" or original_audit.get("failures"):
        raise RuntimeError("frozen production audit is not PASS")
    result.update(source_noise_config_sha256=sha256_file(source_config_path),
                  source_noise_code_sha256=sha256_file(ROOT / "dataset_generation/v023_noise.py"),
                  g6_path=str(g6_path), g6_sha256=EXPECTED_G6,
                  production_audit_sha256=entries["audit_report.json"],
                  test_access="hashes, metadata, and array headers only; no test predictions/arrays",
                  elapsed_seconds=round(time.perf_counter() - started, 2))
    write_json(artifacts / "data_contract_audit.json", result)
    write_json(artifacts / "noise_config.json", V023_NOISE_CONFIG)
    note(f"frozen source audit PASS in {result['elapsed_seconds']}s")
    return result


def require_source_audit(dataset: Path, artifacts: Path) -> dict:
    audit = read_json(artifacts / "data_contract_audit.json")
    if audit.get("status") != "PASS" or audit.get("source_sha256sums") != sha256_file(dataset / "SHA256SUMS"):
        raise RuntimeError("run the frozen source audit before preparation")
    return audit


def build_cache(dataset: Path, artifacts: Path, split: str) -> dict:
    audit = require_source_audit(dataset, artifacts)
    cache_dir = artifacts / "dataset_cache" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing["source_sha256sums"] != audit["source_sha256sums"]:
            raise RuntimeError("existing derivative cache belongs to another source")
        if all(sha256_file(cache_dir / value["file"]) == value["sha256"] for value in existing["arrays"].values()):
            note(f"reuse verified memmap cache: {split}")
            return existing
        raise RuntimeError("existing derivative cache changed; preserve it and investigate")
    started = time.perf_counter()
    records, _ = read_split_metadata(dataset, split)
    index = read_json(dataset / split / "index.json")
    keys = ("density_zyx", "clean_d6") + (("frozen_observed_d6",) if split != "train" else ())
    arrays = {key: np.lib.format.open_memmap(cache_dir / (key + ".tmp.npy"), mode="w+", dtype=np.float32,
                                         shape=(len(records), *SHAPES[key])) for key in keys}
    source_hashes = {name: digest for digest, name in (
        line.split(maxsplit=1) for line in (dataset / "SHA256SUMS").read_text().splitlines())}
    for shard in index["shards"]:
        start, stop = shard["start"], shard["stop"]
        source = dataset / split / shard["file"]
        if sha256_file(source) != source_hashes.get(str(source.relative_to(dataset))):
            raise RuntimeError("frozen source shard changed before cache construction: " + str(source))
        with np.load(source, allow_pickle=False) as archive:
            if archive["sample_id"].astype(str).tolist() != [r["sample_id"] for r in records[start:stop]]:
                raise RuntimeError("records/shard row identity differs")
            for key in keys:
                value = archive[key]
                if not np.isfinite(value).all():
                    raise RuntimeError(f"nonfinite array {split}/{shard['file']}/{key}")
                arrays[key][start:stop] = value
            if split != "train":
                if not np.all(archive["frozen_noise_scale"] == 1.0) or not np.all(archive["frozen_noise_mode"] == "independent_white"):
                    raise RuntimeError("frozen noise mode/amplitude differs")
        if stop % 5000 == 0:
            note(f"cache {split}: {stop}/{len(records)}")
    report = {"source_sha256sums": audit["source_sha256sums"], "count": len(records), "arrays": {}}
    for key, array in arrays.items():
        array.flush()
        path = cache_dir / (key + ".npy")
        (cache_dir / (key + ".tmp.npy")).replace(path)
        report["arrays"][key] = {"file": path.name, "shape": list(array.shape), "dtype": "float32", "sha256": sha256_file(path)}
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    write_json(manifest_path, report)
    note(f"cache {split} complete in {report['elapsed_seconds']}s")
    return report


def build_normalization(dataset: Path, artifacts: Path) -> dict:
    started = time.perf_counter()
    audit = require_source_audit(dataset, artifacts)
    records, _ = read_split_metadata(dataset, "train")
    clean = np.load(artifacts / "dataset_cache/train/clean_d6.npy", mmap_mode="r")
    noise = V023WhiteNoise(V023_NOISE_CONFIG)
    raw_sums, role_sums = np.zeros(6), np.zeros(5)
    transform = role_matrix().numpy().astype(np.float64)
    for i, record in enumerate(records):
        observed = noise(clean[i], sample_id=record["sample_id"], epoch=0).observed_d6.numpy().astype(np.float64).reshape(6, -1)
        raw_sums += np.square(observed).sum(axis=1)
        role_sums += np.square(transform @ observed).sum(axis=1)
    values_per_channel = len(records) * 1089
    raw_rms, role_rms = np.sqrt(raw_sums / values_per_channel), np.sqrt(role_sums / values_per_channel)
    normalization = Normalization(float(role_rms[0]), float(np.sqrt(role_sums[1:3].sum() / (2 * values_per_channel))),
                                  float(np.sqrt(role_sums[3:5].sum() / (2 * values_per_channel))), tuple(raw_rms))
    payload = {"normalization": normalization.to_dict(), "source_sha256sums": audit["source_sha256sums"],
               "split": "train", "sample_count": len(records), "noise_epoch": 0,
               "method": "global RMS of all train endpoints with deterministic V023 online epoch-0 noise; paired spin components share RMS",
               "elapsed_seconds": round(time.perf_counter() - started, 2)}
    write_json(artifacts / "normalization.json", payload)
    note(f"train-only RMS complete in {payload['elapsed_seconds']}s")
    return payload


def normalized_matrix(g6: np.ndarray, normalization: Normalization, mode: str) -> np.ndarray:
    """A maps density to channel-major normalized observations (C, receiver)."""
    kernels = g6.reshape(1089, 6, 16384)
    if mode == "tzz":
        return np.array(kernels[:, 5, :] / normalization.raw6_rms[5], dtype=np.float64)
    if mode != "roles":
        raise ValueError(mode)
    scales = np.array([normalization.scalar_rms, normalization.spin1_rms, normalization.spin1_rms,
                       normalization.spin2_rms, normalization.spin2_rms])
    matrix = np.empty((5, 1089, 16384), dtype=np.float64)
    coefficients = role_matrix().numpy().astype(np.float64) / scales[:, None]
    for channel in range(5):
        matrix[channel] = np.einsum("c,rcv->rv", coefficients[channel], kernels, dtype=np.float64)
    return matrix.reshape(5445, 16384)


def tikhonov_dual(a: np.ndarray, relative_lambda: float = 0.01) -> tuple[np.ndarray, float]:
    """Exactly the legacy ridge objective, solved in its smaller dual system."""
    absolute_lambda = relative_lambda * float(np.square(a).sum() / a.shape[1])
    gram = a @ a.T
    gram.flat[::gram.shape[0] + 1] += absolute_lambda
    factor = scipy.linalg.cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
    solved = scipy.linalg.cho_solve(factor, a, overwrite_b=False, check_finite=False)
    return solved.T, absolute_lambda


def prepare_operators(dataset: Path, artifacts: Path) -> dict:
    started = time.perf_counter()
    audit = require_source_audit(dataset, artifacts)
    normalization_path = artifacts / "normalization.json"
    normalization = Normalization.from_json(normalization_path)
    g6 = np.load(audit["g6_path"], mmap_mode="r")
    records, _ = read_split_metadata(dataset, "train")
    clean = np.load(artifacts / "dataset_cache/train/clean_d6.npy", mmap_mode="r")
    density = np.load(artifacts / "dataset_cache/train/density_zyx.npy", mmap_mode="r")
    noise = V023WhiteNoise(V023_NOISE_CONFIG)
    indices = np.random.default_rng(1107 + 20260905).choice(len(records), size=256, replace=False)
    observed = np.stack([noise(clean[i], sample_id=records[i]["sample_id"], epoch=0).observed_d6.numpy() for i in indices])
    report = {"status": "PASS", "g6_path": audit["g6_path"], "g6_sha256": audit["g6_sha256"],
              "normalization_sha256": sha256_file(normalization_path), "backprojection_quantile": 0.995,
              "backprojection_sample_count": 256, "backprojection_sample_seed": 1107 + 20260905,
              "sensitivity_floor_fraction": 0.001, "density_shape": [16, 32, 32], "receiver_shape": [33, 33],
              "z_storage_order": "deep_to_shallow; depth_m=(15.5-index)*62.5", "tik": {}}
    transform = role_matrix().numpy().astype(np.float64)
    for mode in ("tzz", "roles"):
        mode_started = time.perf_counter()
        note(f"building {mode} normalized operator and sensitivity")
        a = normalized_matrix(g6, normalization, mode)
        sensitivity = np.sqrt(np.square(a).sum(axis=0))
        singular_mode = "role" if mode == "roles" else "tzz"
        sensitivity_path = artifacts / f"sensitivity_{singular_mode}.npy"
        np.save(sensitivity_path, sensitivity.astype(np.float32))
        if mode == "tzz":
            observation = observed[:, 5].reshape(256, -1).astype(np.float64) / normalization.raw6_rms[5]
        else:
            scales = np.array([normalization.scalar_rms, normalization.spin1_rms, normalization.spin1_rms,
                               normalization.spin2_rms, normalization.spin2_rms])
            observation = np.einsum("oc,ncr->nor", transform, observed.reshape(256, 6, -1).astype(np.float64))
            observation = (observation / scales[None, :, None]).reshape(256, -1)
        backprojection = (observation @ a) / np.maximum(sensitivity, sensitivity.max() * 0.001)
        report[f"backprojection_scale_{singular_mode}"] = float(np.quantile(np.abs(backprojection), 0.995))
        report[f"sensitivity_{singular_mode}_sha256"] = sha256_file(sensitivity_path)
        del backprojection
        # Check the channel-major layout against the public physical operator.
        probe_density = np.asarray(density[int(indices[0])]).reshape(-1).astype(np.float64)
        raw_probe = (g6 @ probe_density).reshape(1089, 6)
        expected = raw_probe[:, 5] / normalization.raw6_rms[5] if mode == "tzz" else ((raw_probe @ transform.T) / scales).T.reshape(-1)
        layout_error = float(np.linalg.norm(a @ probe_density - expected) / max(np.linalg.norm(expected), 1e-12))
        if layout_error > 2e-6:
            raise RuntimeError(f"Tikhonov input layout mismatch: {mode}, {layout_error}")
        note(f"{mode}: solving dual ridge system, lambda_rel=0.01")
        pinv, absolute_lambda = tikhonov_dual(a)
        # A random observation checks the original primal normal equations
        # without materializing the 16384 x 16384 matrix.
        probe = np.random.default_rng(1107).normal(size=a.shape[0])
        solution = pinv @ probe
        rhs = a.T @ probe
        relative_residual = float(np.linalg.norm(a.T @ (a @ solution) + absolute_lambda * solution - rhs) / np.linalg.norm(rhs))
        if relative_residual > 1e-8 or not np.isfinite(pinv).all():
            raise RuntimeError(f"Tikhonov solve failed: {mode}, {relative_residual}")
        path = artifacts / f"tik_pinv_{mode}_lambda0.01.npy"
        np.save(path, pinv.astype(np.float32))
        report["tik"][mode] = {"path": str(path), "sha256": sha256_file(path), "shape": list(pinv.shape),
            "input_layout": "channel_major: normalized (C,33,33).flatten(); C=1 raw Tzz or C=5 q,Txz,Tyz,u,Txy",
            "relative_lambda": 0.01, "absolute_lambda": absolute_lambda,
            "regularization_definition": "lambda_abs=0.01*mean(diag(A.T@A)); P=A.T@(A@A.T+lambda_abs*I)^-1",
            "normalization_sha256": report["normalization_sha256"], "layout_relative_error": layout_error,
            "primal_normal_equation_relative_residual_float64": relative_residual,
            "elapsed_seconds": round(time.perf_counter() - mode_started, 2)}
        note(f"{mode} tik complete in {report['tik'][mode]['elapsed_seconds']}s, residual {relative_residual:.3g}")
        del a, pinv, observation
    operator = GravityOperator(audit["g6_path"], normalization)
    rng = torch.Generator().manual_seed(1107)
    probe_density = torch.randn(2, 16, 32, 32, generator=rng)
    mask = torch.rand(2, 1089, generator=rng) > 0.1
    for mode, channels in (("roles", 5), ("tzz", 1)):
        probe = torch.randn(2, 1089, channels, generator=rng)
        left = (operator.density_to_normalized(probe_density, mode) * probe * mask.unsqueeze(-1)).sum()
        right = (probe_density.reshape(2, -1) * operator.adjoint_normalized(probe, mask, mode)).sum()
        error = float((left - right).abs() / torch.maximum(left.abs(), right.abs()).clamp_min(1e-12))
        if error > 2e-5:
            raise RuntimeError(f"adjoint audit failed: {mode}, {error}")
        report[f"adjoint_relative_error_{'role' if mode == 'roles' else 'tzz'}"] = error
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    write_json(artifacts / "operator_audit.json", report)
    note(f"operator and tik preparation PASS in {report['elapsed_seconds']}s")
    return report


