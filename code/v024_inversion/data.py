"""Single-endpoint reader for frozen V023. Pair identity is evaluation metadata."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .noise import V023_NOISE_CONFIG, V023WhiteNoise

DATASET_ID = "V023_STANDALONE_FULL"
SCHEMA_VERSION = "23.2.0"
SPLIT_COUNTS = {"train": 25000, "validation_iid": 2500, "test_locked": 10710}
EXPECTED_TRACKS = {
    "train": {"ordinary": 12000, "separable": 6500, "shape": 6500},
    "validation_iid": {"ordinary": 1200, "separable": 650, "shape": 650},
    "test_locked": {"ordinary": 5000, "separable": 2854, "shape": 2856},
}
SUPPORT_THRESHOLD = 0.005
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "artifacts/dataset_cache"


class DatasetContractError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dprime_bin(track: str, value: float) -> str:
    if track == "ordinary":
        return "ordinary"
    if track == "shape":
        return "shape_tzz_le1"
    for edge, label in ((10, "[5,10)"), (20, "[10,20)"), (40, "[20,40)")):
        if value < edge:
            return label
    return "[40,inf)"


def depth_group(depth_m: float) -> str:
    return "0_333" if depth_m < 1000 / 3 else "333_667" if depth_m < 2000 / 3 else "667_1000"


def support_depth_m(density: np.ndarray) -> float:
    counts = (np.abs(density) >= SUPPORT_THRESHOLD).sum(axis=(1, 2))
    total = int(counts.sum())
    if total == 0:
        raise DatasetContractError("V023 endpoint has an empty support")
    return float(np.dot((15.5 - np.arange(16)) * 62.5, counts) / total)


def read_split_metadata(root: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate endpoint/group relationships using metadata only, including test."""
    if split not in SPLIT_COUNTS:
        raise DatasetContractError(f"unsupported V023 split {split!r}")
    records = read_jsonl(root / split / "records.jsonl")
    pair_rows = read_jsonl(root / split / "pairs.jsonl")
    if len(records) != SPLIT_COUNTS[split]:
        raise DatasetContractError(f"unexpected {split} endpoint count {len(records)}")
    ids = [str(row["sample_id"]) for row in records]
    if len(set(ids)) != len(ids):
        raise DatasetContractError(f"duplicate {split} sample_id")
    tracks = Counter(row["generation_track"] for row in records)
    if dict(tracks) != EXPECTED_TRACKS[split]:
        raise DatasetContractError(f"unexpected {split} track counts: {tracks}")
    pairs = {str(row["ambiguity_group_id"]): row for row in pair_rows}
    if len(pairs) != len(pair_rows) or "" in pairs:
        raise DatasetContractError(f"invalid or duplicate {split} pair identifiers")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["split"] != split:
            raise DatasetContractError(f"record belongs to a different split: {row['sample_id']}")
        group = str(row["ambiguity_group_id"])
        if row["generation_track"] == "ordinary":
            if group or row.get("ambiguity_pair_role", ""):
                raise DatasetContractError("ordinary endpoint has a pair identity")
        else:
            if not group:
                raise DatasetContractError("paired endpoint has no group identity")
            groups[group].append(row)
    if groups.keys() != pairs.keys():
        raise DatasetContractError(f"{split} pairs.jsonl and records.jsonl group sets differ")
    for group, endpoints in groups.items():
        pair = pairs[group]
        if len(endpoints) != 2 or {r["ambiguity_pair_role"] for r in endpoints} != {"A", "B"}:
            raise DatasetContractError(f"pair must contain exactly A and B: {group}")
        if {r["generation_track"] for r in endpoints} != {pair["track"]} or pair["split"] != split:
            raise DatasetContractError(f"pair track/split mismatch: {group}")
        if not np.allclose(pair["reference_component_sigma_eotvos"], V023_NOISE_CONFIG["base_component_sigma_eotvos"], rtol=0, atol=0):
            raise DatasetContractError(f"pair whitening sigma differs from V023 noise: {group}")
        for key in ("tzz_global_dprime", "spin1_global_dprime", "spin2_global_dprime"):
            value = float(pair[key])
            if not np.isfinite(value) or value < 0:
                raise DatasetContractError(f"invalid pair metric {group}/{key}")
            for record in endpoints:
                if key in record.get("pair_metrics", {}) and not np.isclose(value, float(record["pair_metrics"][key]), rtol=1e-8, atol=1e-10):
                    raise DatasetContractError(f"pair metric mismatch {group}/{key}")
    return records, pairs


def compact_record(record: dict[str, Any], pairs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = ("sample_id", "original_sample_id", "mechanism", "subtype", "support_stratum",
             "generation_track", "ambiguity_group_id", "ambiguity_pair_role", "response_group",
             "pair_geometry_mode", "prototype_id", "base_prototype_id", "support_sha256", "density_sha256")
    result = {name: str(record.get(name, "")) for name in names}
    track = result["generation_track"]
    pair = pairs.get(result["ambiguity_group_id"], {})
    result.update(track=track, is_pair=bool(pair), spin_dominance=result["response_group"])
    for key in ("tzz_global_dprime", "spin1_global_dprime", "spin2_global_dprime"):
        result[key] = float(pair.get(key, 0.0))
    result["dprime_bin"] = dprime_bin(track, result["tzz_global_dprime"])
    return result


class SingleSampleDataset(Dataset):
    """Train with deterministic online white noise; evaluate stored r0--r4.

    Optional read-only memmaps avoid repeated NPZ decompression across workers.
    The cache manifest is tied to the source SHA256SUMS and frozen in the run.
    """

    def __init__(self, root: str | Path, split: str, observation_mode: str,
                 noise_realization: int = 0, cache_shards: int = 2,
                 limit: int | None = None, cache_directory: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.split = str(split)
        self.observation_mode = str(observation_mode)
        self.noise_realization = int(noise_realization)
        self.cache_shards = max(1, int(cache_shards))
        self.cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._epoch = mp.Value("q", 0, lock=True)
        manifest = read_json(self.root / "manifest.json")
        if manifest.get("dataset_id") != DATASET_ID or manifest.get("schema_version") != SCHEMA_VERSION:
            raise DatasetContractError("expected frozen standalone V023 schema 23.2.0")
        if self.split not in SPLIT_COUNTS:
            raise DatasetContractError(f"unsupported split {self.split!r}")
        if self.observation_mode not in {"online", "frozen", "clean"}:
            raise ValueError("observation_mode must be online, frozen, or clean")
        if self.split != "train" and self.observation_mode == "online":
            raise DatasetContractError("online noise is restricted to training")
        if self.split == "train" and self.observation_mode == "frozen":
            raise DatasetContractError("V023 train has no frozen observations")
        if not 0 <= self.noise_realization < 5:
            raise DatasetContractError("V023 only provides frozen r0--r4")
        frozen = manifest.get("frozen_noise", {})
        if frozen.get("type") != V023WhiteNoise.TYPE or frozen.get("realizations_per_endpoint") != 5 or frozen.get("amplitude_scale") != 1.0:
            raise DatasetContractError("frozen noise manifest differs from V023")
        self.index = read_json(self.root / self.split / "index.json")
        self.shards = list(self.index["shards"])
        self.starts = [int(item["start"]) for item in self.shards]
        self.stops = [int(item["stop"]) for item in self.shards]
        total = int(self.index["count"])
        if self.index.get("split") != self.split or total != SPLIT_COUNTS[self.split]:
            raise DatasetContractError("split index count/identity mismatch")
        if not self.starts or self.starts[0] != 0 or self.stops[-1] != total or self.starts[1:] != self.stops[:-1]:
            raise DatasetContractError("invalid contiguous shard coverage")
        if any(int(s["count"]) != b - a for s, a, b in zip(self.shards, self.starts, self.stops)):
            raise DatasetContractError("invalid shard count")
        self.length = min(total, int(limit)) if limit is not None else total
        if self.length <= 0:
            raise ValueError("dataset limit must be positive")
        records, self.pairs = read_split_metadata(self.root, self.split)
        self.records = [compact_record(row, self.pairs) for row in records[:self.length]]
        self.records_by_id = {row["sample_id"]: row for row in self.records}
        self.support_threshold = SUPPORT_THRESHOLD
        self.noise = V023WhiteNoise(V023_NOISE_CONFIG) if self.observation_mode == "online" else None
        self._memmaps: dict[str, np.ndarray] = {}
        self.cache_directory = Path(cache_directory) if cache_directory is not None else DEFAULT_CACHE
        cache_manifest = self.cache_directory / self.split / "manifest.json"
        if cache_manifest.is_file():
            payload = read_json(cache_manifest)
            source_digest = hashlib.sha256((self.root / "SHA256SUMS").read_bytes()).hexdigest()
            if payload.get("source_sha256sums") != source_digest or payload.get("count") != total:
                raise DatasetContractError("derived memmap cache does not match the frozen source")
            self._cache_fields = payload["arrays"]
        else:
            self._cache_fields = {}

    def __len__(self) -> int:
        return self.length

    @property
    def epoch(self) -> int:
        with self._epoch.get_lock():
            return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _locate(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        shard_index = bisect_right(self.starts, index) - 1
        return shard_index, index - self.starts[shard_index]

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if shard_index in self.cache:
            payload = self.cache.pop(shard_index)
            self.cache[shard_index] = payload
            return payload
        path = self.root / self.split / self.shards[shard_index]["file"]
        with np.load(path, allow_pickle=False) as archive:
            count = self.stops[shard_index] - self.starts[shard_index]
            source_ids = archive["sample_id"].astype(str).tolist()
            expected_ids = [r["sample_id"] for r in self.records[self.starts[shard_index]:self.stops[shard_index]]]
            if len(source_ids) != count or source_ids[:len(expected_ids)] != expected_ids:
                raise DatasetContractError(f"shard/records row order differs: {path.name}")
            payload = {name: archive[name] for name in ("density_zyx", "clean_d6")}
            if payload["density_zyx"].shape != (count, 16, 32, 32) or payload["clean_d6"].shape != (count, 6, 33, 33):
                raise DatasetContractError("V023 shard array shape differs from physical grid")
            if self.observation_mode == "frozen":
                frozen = archive["frozen_observed_d6"]
                if frozen.shape != (count, 5, 6, 33, 33):
                    raise DatasetContractError("V023 frozen observation shape is invalid")
                payload["observed"] = frozen[:, self.noise_realization].copy()
        self.cache[shard_index] = payload
        while len(self.cache) > self.cache_shards:
            self.cache.popitem(last=False)
        return payload

    def _cached(self, name: str, index: int) -> np.ndarray:
        if name not in self._memmaps:
            self._memmaps[name] = np.load(self.cache_directory / self.split / self._cache_fields[name]["file"], mmap_mode="r", allow_pickle=False)
        return self._memmaps[name][index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        shard, local = self._locate(index)
        record = self.records[index]
        if self._cache_fields:
            clean = self._cached("clean_d6", index)
            density_array = self._cached("density_zyx", index)
            frozen = self._cached("frozen_observed_d6", index)[self.noise_realization] if self.observation_mode == "frozen" else None
        else:
            payload = self._load_shard(shard)
            clean, density_array = payload["clean_d6"][local], payload["density_zyx"][local]
            frozen = payload["observed"][local] if self.observation_mode == "frozen" else None
        if self.observation_mode == "online":
            view = self.noise(clean, sample_id=record["sample_id"], epoch=self.epoch)
            observed, noise_mode = view.observed_d6, view.mode
        else:
            observed = torch.from_numpy(np.array(frozen if frozen is not None else clean, dtype=np.float32, copy=True))
            noise_mode = "independent_white" if frozen is not None else "clean"
        density = torch.from_numpy(np.array(density_array, dtype=np.float32, copy=True))
        raw6 = observed.permute(1, 2, 0).contiguous().reshape(-1, 6)
        # Supervision only: never passed into operator.prepare_inputs/model.
        clean_raw6 = torch.from_numpy(np.array(clean, dtype=np.float32, copy=True)).permute(1, 2, 0).contiguous().reshape(-1, 6)
        physical_depth = support_depth_m(density_array)
        return {**record, "raw6": raw6, "clean_raw6": clean_raw6,
                "receiver_mask": torch.ones(raw6.shape[0], dtype=torch.bool),
                "density": density, "support": density.abs().ge(self.support_threshold),
                "noise_mode": noise_mode, "physical_depth_m": physical_depth,
                "depth_group": depth_group(physical_depth)}
