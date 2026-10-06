from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import torch
from scipy import ndimage


@dataclass(frozen=True)
class NoiseView:
    observed_d6: torch.Tensor
    mode: str
    snr_db: float | None


@dataclass(frozen=True)
class V019NoiseView:
    observed_d6: torch.Tensor
    mode: str
    amplitude_scale: float
    component_sigma_eotvos: tuple[float, ...]


class V017OnlineNoise:
    """Epoch-resampled noise contract, including strict shared-noise pair views."""

    def __init__(self, config: dict[str, Any], seed: int = 0) -> None:
        self.config = config["noise_online"] if "noise_online" in config else config
        self.rng = np.random.default_rng(int(seed))

    def _draw_mode(self) -> tuple[str, float | None]:
        p0 = float(self.config["clean_or_40db_probability"])
        p1 = float(self.config["independent_gaussian_probability"])
        draw = float(self.rng.random())
        if draw < p0:
            if self.rng.random() < float(self.config["clean_probability_within_clean_or_40db"]):
                return "clean", None
            return "near_clean_gaussian", 40.0
        if draw < p0 + p1:
            low, high = self.config["gaussian_snr_db_range"]
            return "independent_gaussian", float(self.rng.uniform(low, high))
        low, high = self.config["smooth_or_stripe_snr_db_range"]
        return "smooth_correlated" if self.rng.random() < 0.5 else "stripe", float(self.rng.uniform(low, high))

    def _unit_field(self, mode: str, shape: tuple[int, ...]) -> np.ndarray:
        field = self.rng.normal(size=shape).astype(np.float32)
        if mode == "smooth_correlated":
            for c in range(shape[0]):
                field[c] = ndimage.gaussian_filter(field[c], sigma=float(self.rng.uniform(0.8, 2.0)), mode="reflect")
        elif mode == "stripe":
            for c in range(shape[0]):
                if self.rng.random() < 0.5:
                    stripe = self.rng.normal(size=(shape[1], 1)).astype(np.float32)
                else:
                    stripe = self.rng.normal(size=(1, shape[2])).astype(np.float32)
                field[c] = 0.35 * field[c] + stripe
        rms = np.sqrt(np.mean(field.astype(np.float64) ** 2, axis=(1, 2), keepdims=True))
        return field / np.maximum(rms, 1e-12)

    @staticmethod
    def _sigma(reference: np.ndarray, snr_db: float) -> np.ndarray:
        rms = np.sqrt(np.mean(reference.astype(np.float64) ** 2, axis=(1, 2), keepdims=True))
        return (rms / (10.0 ** (float(snr_db) / 20.0))).astype(np.float32)

    def __call__(self, clean_d6: torch.Tensor | np.ndarray) -> NoiseView:
        clean = np.asarray(clean_d6, dtype=np.float32)
        mode, snr = self._draw_mode()
        if mode == "clean":
            return NoiseView(torch.from_numpy(clean.copy()), mode, snr)
        field = self._unit_field(mode, tuple(clean.shape))
        observed = clean + field * self._sigma(clean, float(snr))
        return NoiseView(torch.from_numpy(observed), mode, snr)

    def shared_pair(
        self,
        clean_a: torch.Tensor | np.ndarray,
        clean_b: torch.Tensor | np.ndarray,
    ) -> tuple[NoiseView, NoiseView]:
        first = np.asarray(clean_a, dtype=np.float32)
        second = np.asarray(clean_b, dtype=np.float32)
        mode, snr = self._draw_mode()
        if mode == "clean":
            return (
                NoiseView(torch.from_numpy(first.copy()), mode, snr),
                NoiseView(torch.from_numpy(second.copy()), mode, snr),
            )
        field = self._unit_field(mode, tuple(first.shape))
        pooled = np.sqrt(0.5 * (first.astype(np.float64) ** 2 + second.astype(np.float64) ** 2))
        sigma = self._sigma(pooled, float(snr))
        absolute_noise = field * sigma
        return (
            NoiseView(torch.from_numpy(first + absolute_noise), mode, snr),
            NoiseView(torch.from_numpy(second + absolute_noise), mode, snr),
        )


class V018SimpleGaussianNoise:
    """One fixed-SNR, independent Gaussian white-noise model.

    For component ``c``, every receiver gets independent N(0, sigma_c^2)
    noise with ``sigma_c = RMS(clean_c) / 10**(SNR/20)``.  There are no clean,
    stripe, smoothed, or random-SNR branches.  Pair endpoints are sampled
    independently; ``shared_pair`` is retained only as a loader-compatible
    name and deliberately does not share the realized noise field.
    """

    def __init__(self, config: dict[str, Any], seed: int = 0) -> None:
        self.config = config["noise_online"] if "noise_online" in config else config
        if str(self.config.get("type")) != "independent_gaussian_white":
            raise ValueError("V018SimpleGaussianNoise requires type=independent_gaussian_white")
        self.snr_db = float(self.config["snr_db"])
        self.seed = int(seed)
        self._rng_key: tuple[int, int] | None = None
        self.rng = np.random.default_rng(self.seed)

    def _worker_rng(self) -> np.random.Generator:
        info = torch.utils.data.get_worker_info()
        key = (-1, self.seed) if info is None else (int(info.id), int(info.seed))
        if key != self._rng_key:
            sequence = np.random.SeedSequence((self.seed, key[0] + 1, key[1]))
            self.rng = np.random.default_rng(sequence)
            self._rng_key = key
        return self.rng

    @staticmethod
    def _sigma(reference: np.ndarray, snr_db: float) -> np.ndarray:
        rms = np.sqrt(np.mean(reference.astype(np.float64) ** 2, axis=(1, 2), keepdims=True))
        return np.maximum(rms / (10.0 ** (float(snr_db) / 20.0)), 1e-12).astype(np.float32)

    def __call__(self, clean_d6: torch.Tensor | np.ndarray) -> NoiseView:
        clean = np.asarray(clean_d6, dtype=np.float32)
        field = self._worker_rng().normal(size=clean.shape).astype(np.float32)
        observed = clean + field * self._sigma(clean, self.snr_db)
        return NoiseView(torch.from_numpy(observed), "independent_gaussian_white", self.snr_db)

    def shared_pair(
        self,
        clean_a: torch.Tensor | np.ndarray,
        clean_b: torch.Tensor | np.ndarray,
    ) -> tuple[NoiseView, NoiseView]:
        # V018 intentionally avoids identical pair noise, because identical
        # noise would cancel under endpoint subtraction and exaggerate the
        # apparent separability of the supplemental components.
        return self(clean_a), self(clean_b)


class V019AbsoluteHybridNoise:
    """Absolute, heterogeneous noise with order-independent seeded views.

    The base component standard deviations are fixed in physical response
    units. They are never scaled from the clean sample, so weak sources remain
    weak. Spatially correlated and stripe modes are included in addition to
    white noise. A small common spatial field introduces component covariance.
    """

    def __init__(self, config: dict[str, Any], seed: int | None = None) -> None:
        self.config = config["noise"] if "noise" in config else config
        if str(self.config.get("type")) != "absolute_hybrid_v019":
            raise ValueError("V019AbsoluteHybridNoise requires type=absolute_hybrid_v019")
        self.base_sigma = np.asarray(
            self.config["base_component_sigma_eotvos"], dtype=np.float64
        )
        if self.base_sigma.shape != (6,) or np.any(self.base_sigma <= 0.0):
            raise ValueError(f"invalid V019 component sigma {self.base_sigma}")
        self.seed = int(self.config["training_seed"] if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)
        probabilities = self.config["mode_probabilities"]
        self.modes = tuple(probabilities)
        self.mode_probabilities = np.asarray(
            [float(probabilities[key]) for key in self.modes], dtype=np.float64
        )
        self.mode_probabilities /= self.mode_probabilities.sum()

    @staticmethod
    def _stable_seed(*values: Any) -> int:
        raw = "|".join(str(value) for value in values).encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little", signed=False)

    def _draw(
        self,
        clean_d6: torch.Tensor | np.ndarray,
        rng: np.random.Generator,
    ) -> V019NoiseView:
        clean = np.asarray(clean_d6, dtype=np.float32)
        if clean.shape[0] != 6 or clean.ndim != 3:
            raise ValueError(f"expected clean response (6, ny, nx), got {clean.shape}")
        mode = str(rng.choice(self.modes, p=self.mode_probabilities))
        field = rng.normal(size=clean.shape).astype(np.float32)
        if mode == "smooth_correlated":
            low, high = self.config["smooth_sigma_cells_range"]
            for component in range(6):
                field[component] = ndimage.gaussian_filter(
                    field[component], sigma=float(rng.uniform(low, high)), mode="reflect"
                )
        elif mode == "stripe_plus_white":
            for component in range(6):
                if rng.random() < 0.5:
                    stripe = rng.normal(size=(clean.shape[1], 1)).astype(np.float32)
                else:
                    stripe = rng.normal(size=(1, clean.shape[2])).astype(np.float32)
                field[component] = 0.55 * field[component] + stripe
        elif mode != "independent_white":
            raise ValueError(mode)

        common_fraction = float(self.config["component_common_mode_fraction"])
        if common_fraction:
            common = rng.normal(size=clean.shape[1:]).astype(np.float32)
            if mode == "smooth_correlated":
                low, high = self.config["smooth_sigma_cells_range"]
                common = ndimage.gaussian_filter(
                    common, sigma=float(rng.uniform(low, high)), mode="reflect"
                )
            common /= max(float(np.sqrt(np.mean(common.astype(np.float64) ** 2))), 1e-12)
            field = (
                np.sqrt(max(1.0 - common_fraction**2, 0.0)) * field
                + common_fraction * common[None]
            )
        rms = np.sqrt(np.mean(field.astype(np.float64) ** 2, axis=(1, 2), keepdims=True))
        field = field / np.maximum(rms, 1e-12)
        low_scale, high_scale = (
            float(value) for value in self.config["amplitude_scale_range"]
        )
        amplitude_scale = float(rng.uniform(low_scale, high_scale))
        sigma = self.base_sigma * amplitude_scale
        observed = clean + field * sigma[:, None, None].astype(np.float32)
        return V019NoiseView(
            observed_d6=torch.from_numpy(observed.astype(np.float32)),
            mode=mode,
            amplitude_scale=amplitude_scale,
            component_sigma_eotvos=tuple(float(value) for value in sigma),
        )

    def __call__(
        self,
        clean_d6: torch.Tensor | np.ndarray,
        sample_id: str | None = None,
        epoch: int = 0,
    ) -> V019NoiseView:
        if sample_id is None:
            return self._draw(clean_d6, self.rng)
        seed = self._stable_seed(self.seed, sample_id, int(epoch), "train")
        return self._draw(clean_d6, np.random.default_rng(seed))

    def frozen(
        self,
        clean_d6: torch.Tensor | np.ndarray,
        sample_id: str,
        split: str,
    ) -> V019NoiseView:
        seeds = self.config["frozen_evaluation_seeds"]
        if split not in seeds:
            raise ValueError(f"no frozen evaluation seed for split {split}")
        seed = self._stable_seed(int(seeds[split]), sample_id, split, "frozen")
        return self._draw(clean_d6, np.random.default_rng(seed))

    def independent_pair(
        self,
        clean_a: torch.Tensor | np.ndarray,
        clean_b: torch.Tensor | np.ndarray,
        sample_id_a: str,
        sample_id_b: str,
        epoch: int = 0,
    ) -> tuple[V019NoiseView, V019NoiseView]:
        return (
            self(clean_a, sample_id=sample_id_a, epoch=epoch),
            self(clean_b, sample_id=sample_id_b, epoch=epoch),
        )
