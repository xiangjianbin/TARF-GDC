'Independent V023 Gaussian noise. Sigma in Eotvos is [0.425,0.31,0.70,0.425,0.70,0.74]; evaluation amplitude is 1. Training seeds depend on sample ID and epoch; frozen seeds include split and realization. Pair endpoints have independent noise.'
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class V023NoiseView:
    """One deterministic white-noise observation."""

    observed_d6: torch.Tensor
    mode: str
    amplitude_scale: float
    component_sigma_eotvos: tuple[float, ...]


class V023WhiteNoise:
    """I.i.d. Gaussian white noise with fixed per-component sigma."""

    TYPE = "white_gaussian_v023"
    MODE = "independent_white"

    def __init__(self, config: dict[str, Any], seed: int | None = None) -> None:
        self.config = config["noise"] if "noise" in config else config
        if str(self.config.get("type")) != self.TYPE:
            raise ValueError(f"V023WhiteNoise requires type={self.TYPE}")
        self.base_sigma = np.asarray(
            self.config["base_component_sigma_eotvos"], dtype=np.float64
        )
        if self.base_sigma.shape != (6,) or np.any(self.base_sigma <= 0.0):
            raise ValueError(f"invalid V023 component sigma {self.base_sigma}")
        low, high = (float(v) for v in self.config["amplitude_scale_range"])
        if not (0.0 < low <= high):
            raise ValueError("invalid amplitude_scale_range")
        self.amplitude_range = (low, high)
        self.frozen_amplitude = float(self.config.get("frozen_amplitude_scale", 1.0))
        self.seed = int(self.config["training_seed"] if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _stable_seed(*values: Any) -> int:
        raw = "|".join(str(value) for value in values).encode("utf-8")
        return int.from_bytes(
            hashlib.sha256(raw).digest()[:8], "little", signed=False
        )

    def _draw(
        self,
        clean_d6: torch.Tensor | np.ndarray,
        rng: np.random.Generator,
        amplitude_scale: float | None = None,
    ) -> V023NoiseView:
        clean = np.asarray(clean_d6, dtype=np.float32)
        if clean.ndim != 3 or clean.shape[0] != 6:
            raise ValueError(f"expected clean response (6, ny, nx), got {clean.shape}")
        if amplitude_scale is None:
            amplitude_scale = float(rng.uniform(*self.amplitude_range))
        amplitude_scale = float(amplitude_scale)
        sigma = self.base_sigma * amplitude_scale
        field = rng.normal(size=clean.shape).astype(np.float32)
        observed = clean + field * sigma[:, None, None].astype(np.float32)
        return V023NoiseView(
            observed_d6=torch.from_numpy(observed.astype(np.float32)),
            mode=self.MODE,
            amplitude_scale=amplitude_scale,
            component_sigma_eotvos=tuple(float(value) for value in sigma),
        )

    def __call__(
        self,
        clean_d6: torch.Tensor | np.ndarray,
        sample_id: str | None = None,
        epoch: int = 0,
    ) -> V023NoiseView:
        """Online training noise: amplitude ~ U(amplitude_scale_range)."""
        if sample_id is None:
            return self._draw(clean_d6, self.rng)
        seed = self._stable_seed(self.seed, sample_id, int(epoch), "train")
        return self._draw(clean_d6, np.random.default_rng(seed))

    def frozen_realization(
        self,
        clean_d6: torch.Tensor | np.ndarray,
        sample_id: str,
        split: str,
        realization: int,
    ) -> tuple[V023NoiseView, int]:
        """One frozen evaluation realization; amplitude fixed at frozen_amplitude_scale."""
        seeds = self.config["frozen_evaluation_seeds"]
        if split not in seeds:
            raise ValueError(f"no frozen evaluation seed for split {split}")
        seed = self._stable_seed(
            int(seeds[split]), sample_id, split, "frozen", int(realization)
        )
        view = self._draw(
            clean_d6,
            np.random.default_rng(seed),
            amplitude_scale=self.frozen_amplitude,
        )
        return view, seed

    def independent_pair(
        self,
        clean_a: torch.Tensor | np.ndarray,
        clean_b: torch.Tensor | np.ndarray,
        sample_id_a: str,
        sample_id_b: str,
        epoch: int = 0,
    ) -> tuple[V023NoiseView, V023NoiseView]:
        return (
            self(clean_a, sample_id=sample_id_a, epoch=epoch),
            self(clean_b, sample_id=sample_id_b, epoch=epoch),
        )
