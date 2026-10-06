from __future__ import annotations

import json
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .noise import V017OnlineNoise, V018SimpleGaussianNoise, V019AbsoluteHybridNoise


ARRAY_KEYS = (
    "density_zyx", "volume_fraction_zyx", "clean_d6", "sample_id",
    "mechanism", "subtype", "ambiguity_group_id", "ambiguity_pair_role",
    "response_group", "support_stratum",
)


def save_split_shards(
    output: Path,
    split: str,
    arrays: dict[str, Any],
    records: list[dict[str, Any]],
    shard_size: int,
) -> dict[str, Any]:
    split_dir = output / split
    split_dir.mkdir(parents=True, exist_ok=True)
    count = len(records)
    shards = []
    for shard_index, start in enumerate(range(0, count, int(shard_size))):
        stop = min(start + int(shard_size), count)
        name = f"shard-{shard_index:04d}.npz"
        payload = {key: np.asarray(value[start:stop]) for key, value in arrays.items()}
        np.savez(split_dir / name, **payload)
        shards.append({"file": name, "start": start, "stop": stop, "count": stop - start})
    with (split_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    index = {"split": split, "count": count, "shards": shards, "array_keys": list(arrays)}
    (split_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def load_split(dataset: str | Path, split: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    split_dir = Path(dataset) / split
    index = json.loads((split_dir / "index.json").read_text(encoding="utf-8"))
    parts: dict[str, list[np.ndarray]] = {}
    for shard in index["shards"]:
        with np.load(split_dir / shard["file"], allow_pickle=False) as payload:
            for key in payload.files:
                parts.setdefault(key, []).append(payload[key])
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    records = [json.loads(line) for line in (split_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return arrays, records


class ShardedV017Dataset(Dataset):
    """Lazy shard reader; noise is resampled every item access."""

    def __init__(
        self,
        dataset: str | Path,
        split: str = "train",
        noise: V017OnlineNoise | V018SimpleGaussianNoise | None = None,
        cache_shards: int = 2,
        subset_index: str | Path | None = None,
    ) -> None:
        self.split_dir = Path(dataset) / split
        self.index = json.loads((self.split_dir / "index.json").read_text(encoding="utf-8"))
        self.noise = noise
        self.cache_shards = int(cache_shards)
        self.cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.subset_indices: list[int] | None = None
        if subset_index is not None:
            subset = json.loads(Path(subset_index).read_text(encoding="utf-8"))
            self.subset_indices = [int(value) for value in subset["indices"]]

    def __len__(self) -> int:
        return len(self.subset_indices) if self.subset_indices is not None else int(self.index["count"])

    def _locate(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.subset_indices is not None:
            index = self.subset_indices[index]
        shard_size = int(self.index["shards"][0]["count"])
        shard_index = min(index // shard_size, len(self.index["shards"]) - 1)
        shard = self.index["shards"][shard_index]
        return shard_index, index - int(shard["start"])

    def _shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if shard_index in self.cache:
            value = self.cache.pop(shard_index)
            self.cache[shard_index] = value
            return value
        path = self.split_dir / self.index["shards"][shard_index]["file"]
        with np.load(path, allow_pickle=False) as payload:
            value = {key: payload[key] for key in payload.files}
        self.cache[shard_index] = value
        while len(self.cache) > self.cache_shards:
            self.cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, local = self._locate(index)
        row = self._shard(shard_index)
        clean = row["clean_d6"][local].astype(np.float32)
        noise_view = self.noise(clean) if self.noise is not None else None
        return {
            "density": torch.from_numpy(row["density_zyx"][local].astype(np.float32)),
            "clean_d6": torch.from_numpy(clean),
            "observed_d6": noise_view.observed_d6 if noise_view else torch.from_numpy(clean.copy()),
            "sample_id": str(row["sample_id"][local]),
            "mechanism": str(row["mechanism"][local]),
            "subtype": str(row["subtype"][local]),
            "ambiguity_group_id": str(row["ambiguity_group_id"][local]),
            "ambiguity_pair_role": str(row["ambiguity_pair_role"][local]),
            "response_group": str(row["response_group"][local]),
            "noise_mode": noise_view.mode if noise_view else "clean",
            "snr_db": noise_view.snr_db if noise_view else None,
        }

    def metadata_column(self, key: str) -> list[str]:
        values: list[str] = []
        for shard_index in range(len(self.index["shards"])):
            values.extend(str(value) for value in self._shard(shard_index)[key])
        if self.subset_indices is not None:
            values = [values[index] for index in self.subset_indices]
        return values


class PairAwareBatchSampler(Sampler):
    """Keep every ambiguity pair intact while shuffling atomic units."""

    def __init__(
        self, ambiguity_group_ids: Sequence[str], batch_size: int,
        seed: int = 0, shuffle: bool = True, drop_last: bool = False,
    ) -> None:
        if int(batch_size) < 2:
            raise ValueError("pair-aware batch_size must be >= 2")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        groups: dict[str, list[int]] = defaultdict(list)
        singles = []
        for index, value in enumerate(ambiguity_group_ids):
            gid = str(value) if value else ""
            (groups[gid] if gid else singles).append(index)
        invalid = {key: value for key, value in groups.items() if len(value) != 2}
        if invalid:
            raise ValueError(f"ambiguity groups must contain exactly two endpoints: {invalid}")
        self.units = [tuple(value) for _, value in sorted(groups.items())]
        self.units.extend((index,) for index in singles)
        self.sample_count = len(ambiguity_group_ids)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.units)) if self.shuffle else np.arange(len(self.units))
        batch: list[int] = []
        for unit_index in order:
            unit = self.units[int(unit_index)]
            if batch and len(batch) + len(unit) > self.batch_size:
                if not self.drop_last or len(batch) == self.batch_size:
                    yield batch
                batch = []
            batch.extend(unit)
        if batch and (not self.drop_last or len(batch) == self.batch_size):
            yield batch

    def __len__(self) -> int:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.units)) if self.shuffle else np.arange(len(self.units))
        count = 0
        batch_size = 0
        for unit_index in order:
            unit_size = len(self.units[int(unit_index)])
            if batch_size and batch_size + unit_size > self.batch_size:
                if not self.drop_last or batch_size == self.batch_size:
                    count += 1
                batch_size = 0
            batch_size += unit_size
        if batch_size and (not self.drop_last or batch_size == self.batch_size):
            count += 1
        return count


class ShardedV019Dataset(ShardedV017Dataset):
    """V019 reader with deterministic online train noise and frozen eval views."""

    def __init__(
        self,
        dataset: str | Path,
        split: str = "train",
        noise: V019AbsoluteHybridNoise | None = None,
        use_frozen_observation: bool = True,
        cache_shards: int = 2,
        subset_index: str | Path | None = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            split=split,
            noise=None,
            cache_shards=cache_shards,
            subset_index=subset_index,
        )
        self.v019_noise = noise
        self.use_frozen_observation = bool(use_frozen_observation)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, local = self._locate(index)
        row = self._shard(shard_index)
        clean = row["clean_d6"][local].astype(np.float32)
        sample_id = str(row["sample_id"][local])
        noise_mode = "clean"
        noise_scale = float("nan")
        if self.v019_noise is not None:
            view = self.v019_noise(clean, sample_id=sample_id, epoch=self.epoch)
            observed = view.observed_d6
            noise_mode = view.mode
            noise_scale = view.amplitude_scale
        elif self.use_frozen_observation and "frozen_observed_d6" in row:
            observed = torch.from_numpy(
                row["frozen_observed_d6"][local].astype(np.float32)
            )
            noise_mode = str(row["frozen_noise_mode"][local])
            noise_scale = float(row["frozen_noise_scale"][local])
        else:
            observed = torch.from_numpy(clean.copy())
        result: dict[str, Any] = {
            "density": torch.from_numpy(row["density_zyx"][local].astype(np.float32)),
            "volume_fraction": torch.from_numpy(
                row["volume_fraction_zyx"][local].astype(np.float32)
            ),
            "clean_d6": torch.from_numpy(clean),
            "observed_d6": observed,
            "sample_id": sample_id,
            "mechanism": str(row["mechanism"][local]),
            "subtype": str(row["subtype"][local]),
            "support_subtype": str(row["support_subtype"][local]),
            "density_pattern": str(row["density_pattern"][local]),
            "ambiguity_group_id": str(row["ambiguity_group_id"][local]),
            "ambiguity_pair_role": str(row["ambiguity_pair_role"][local]),
            "response_group": str(row["response_group"][local]),
            "support_stratum": str(row["support_stratum"][local]),
            "pair_geometry_mode": str(row["pair_geometry_mode"][local]),
            "noise_mode": noise_mode,
            "noise_scale": noise_scale,
        }
        return result


class PairNoiseCollator:
    """Apply one absolute noise field to both endpoints of every pair."""

    def __init__(self, noise: V017OnlineNoise) -> None:
        self.noise = noise

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(items):
            gid = str(item["ambiguity_group_id"])
            if gid:
                groups[gid].append(index)
        invalid = {key: value for key, value in groups.items() if len(value) != 2}
        if invalid:
            raise ValueError(f"batch split an ambiguity pair: {invalid}")
        observed: list[torch.Tensor | None] = [None] * len(items)
        modes: list[str | None] = [None] * len(items)
        snrs: list[float | None] = [None] * len(items)
        paired_indices = {index for values in groups.values() for index in values}
        for indices in groups.values():
            first, second = indices
            view_a, view_b = self.noise.shared_pair(items[first]["clean_d6"], items[second]["clean_d6"])
            for index, view in ((first, view_a), (second, view_b)):
                observed[index], modes[index], snrs[index] = view.observed_d6, view.mode, view.snr_db
        for index, item in enumerate(items):
            if index in paired_indices:
                continue
            view = self.noise(item["clean_d6"])
            observed[index], modes[index], snrs[index] = view.observed_d6, view.mode, view.snr_db
        tensor_keys = ("density", "clean_d6")
        result: dict[str, Any] = {
            key: torch.stack([item[key] for item in items]) for key in tensor_keys
        }
        result["observed_d6"] = torch.stack([value for value in observed if value is not None])
        for key in ("sample_id", "mechanism", "subtype", "ambiguity_group_id", "ambiguity_pair_role", "response_group"):
            result[key] = [item[key] for item in items]
        result["noise_mode"] = modes
        result["snr_db"] = torch.tensor([float("nan") if value is None else float(value) for value in snrs])
        return result
