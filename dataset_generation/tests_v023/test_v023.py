'Test separable-pair gates, Gaussian noise, geometry and covariance-normalized d-prime.'
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v017.ambiguity import pair_metrics_absolute_covariance_v019  # noqa: E402
from v017.config import load_config  # noqa: E402
from v017.geometry import SobolStream, generate_model  # noqa: E402
from v017.physics import GravityTensorOperator  # noqa: E402
from build_pairs_v023 import (  # noqa: E402
    _endpoint_contract_ok,
    classify_pair_reverse_v023,
    dprime_bin_index,
    evaluate_proposal,
    pair_task_cards,
    random_smooth_direction,
    response_group_label,
    validate_v023_config,
)
from v023_noise import V023WhiteNoise  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_v023.json"
SIGMA = np.array([0.425, 0.31, 0.70, 0.425, 0.70, 0.74])


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def contract(config: dict) -> dict:
    return config["ambiguity"]


# --------------------------------------------------------------------------- #
# Test separable-pair classifier boundaries.
# --------------------------------------------------------------------------- #

def _metrics(d0: float, d1: float, d2: float, nrmse: float) -> dict[str, float]:
    return {
        "tzz_global_dprime": d0,
        "spin1_global_dprime": d1,
        "spin2_global_dprime": d2,
        "density_nrmse": nrmse,
    }


def test_reverse_gate_boundaries(contract: dict) -> None:
    # Accept exact threshold values.
    assert classify_pair_reverse_v023(_metrics(5.0, 5.0, 0.0, 0.12), contract) == "distinguishable"
    assert classify_pair_reverse_v023(_metrics(5.0, 0.0, 5.0, 0.12), contract) == "distinguishable"
    assert classify_pair_reverse_v023(_metrics(100.0, 100.0, 100.0, 2.0), contract) == "distinguishable"
    # Reject a pair that narrowly fails any required gate.
    assert classify_pair_reverse_v023(_metrics(4.9999, 50.0, 50.0, 0.5), contract) is None
    assert classify_pair_reverse_v023(_metrics(50.0, 4.9999, 4.9999, 0.5), contract) is None
    assert classify_pair_reverse_v023(_metrics(50.0, 50.0, 50.0, 0.1199), contract) is None
    # Use the maximum spin d-prime: either complementary group may pass.
    assert classify_pair_reverse_v023(_metrics(5.0, 5.0, 0.1, 0.12), contract) == "distinguishable"


def test_dprime_bin_index(contract: dict) -> None:
    assert dprime_bin_index(contract, 5.0) == 0
    assert dprime_bin_index(contract, 9.9999) == 0
    assert dprime_bin_index(contract, 10.0) == 1
    assert dprime_bin_index(contract, 20.0) == 2
    assert dprime_bin_index(contract, 40.0) == 3
    assert dprime_bin_index(contract, 123456.0) == 3
    assert dprime_bin_index(contract, 4.9999) is None


def test_response_group_label(contract: dict) -> None:
    assert response_group_label(_metrics(10.0, 30.0, 10.0, 0.2), contract) == "spin1"
    assert response_group_label(_metrics(10.0, 10.0, 30.0, 0.2), contract) == "spin2"
    assert response_group_label(_metrics(10.0, 20.0, 25.0, 0.2), contract) == "mixed"


def test_config_validation(config: dict) -> None:
    validate_v023_config(config)
    broken = {k: v for k, v in config.items()}
    broken["ambiguity"] = dict(config["ambiguity"])
    broken["ambiguity"]["reference_component_sigma_eotvos"] = [1.0] * 6
    with pytest.raises(ValueError, match="whitening reference sigma"):
        validate_v023_config(broken)


# --------------------------------------------------------------------------- #
# Independent white-noise tests.
# --------------------------------------------------------------------------- #

def test_noise_rms_matches_sigma(config: dict) -> None:
    noise = V023WhiteNoise(config)
    clean = np.zeros((6, 33, 33), dtype=np.float32)
    residuals = []
    for i in range(40):
        view, _ = noise.frozen_realization(clean, f"sample_{i:04d}", "smoke", 0)
        residuals.append(view.observed_d6.numpy())
    residual = np.stack(residuals)
    rms = np.sqrt(np.mean(residual.astype(np.float64) ** 2, axis=(0, 2, 3)))
    rel = rms / SIGMA - 1.0
    assert np.all(np.abs(rel) <= 0.05), f"per-component RMS off by {rel}"
    # Frozen evaluation amplitude is exactly 1.
    assert all(
        v.amplitude_scale == 1.0
        for v, _ in (noise.frozen_realization(clean, "x", "smoke", r) for r in range(5))
    )


def test_noise_deterministic(config: dict) -> None:
    noise = V023WhiteNoise(config)
    clean = np.random.default_rng(0).normal(size=(6, 33, 33)).astype(np.float32)
    a1 = noise(clean, sample_id="s1", epoch=3).observed_d6.numpy()
    a2 = noise(clean, sample_id="s1", epoch=3).observed_d6.numpy()
    assert np.array_equal(a1, a2), "same (seed, sample_id, epoch) must be bitwise identical"
    b = noise(clean, sample_id="s1", epoch=4).observed_d6.numpy()
    assert not np.array_equal(a1, b), "different epoch must differ"
    f1, seed1 = noise.frozen_realization(clean, "s1", "smoke", 2)
    f2, seed2 = noise.frozen_realization(clean, "s1", "smoke", 2)
    assert seed1 == seed2 and np.array_equal(f1.observed_d6.numpy(), f2.observed_d6.numpy())
    f3, _ = noise.frozen_realization(clean, "s1", "smoke", 3)
    assert not np.array_equal(f1.observed_d6.numpy(), f3.observed_d6.numpy())
    # Training amplitude lies in [0.75,1.5].
    scales = [noise(clean, sample_id=f"s{i}", epoch=0).amplitude_scale for i in range(30)]
    assert all(0.75 <= s <= 1.5 for s in scales)


def test_pair_noise_independent(config: dict) -> None:
    noise = V023WhiteNoise(config)
    clean = np.zeros((6, 33, 33), dtype=np.float32)
    va, vb = noise.independent_pair(clean, clean, "endpoint_a", "endpoint_b", epoch=0)
    fa = va.observed_d6.numpy().reshape(-1).astype(np.float64)
    fb = vb.observed_d6.numpy().reshape(-1).astype(np.float64)
    corr = float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb)))
    assert abs(corr) < 0.05, f"pair endpoint noise should be independent, got r={corr}"
    assert not np.array_equal(fa, fb)


# --------------------------------------------------------------------------- #
# Target a 10 percent deep quota (support centroid >=600 m).
# --------------------------------------------------------------------------- #

def test_deep_targeting_and_repair(config: dict) -> None:
    from build_pairs_v023 import (
        assign_deep_flags, deep_target_config, repair_deep_mechanisms,
    )
    from build_ordinary_v023 import support_centroid_depth_km

    # Repair infeasible deep mechanisms without changing marginal counts.
    mechanisms = ["B_structural", "A_analytic", "C_irregular", "B_structural"]
    strata = ["large_8_18pct", "main_2_8pct", "small_0p5_2pct", "main_2_8pct"]
    flags = [True, True, True, False]
    repaired, swaps = repair_deep_mechanisms(mechanisms, strata, flags)
    assert swaps == 1
    assert repaired[0] == "A_analytic" and repaired[1] == "B_structural"
    assert not any(
        f and s == "large_8_18pct" and m == "B_structural"
        for m, s, f in zip(repaired, strata, flags)
    )
    # Quota assignment is deterministic.
    f1 = assign_deep_flags(100, 0.1, 7)
    assert sum(f1) == 10 and f1 == assign_deep_flags(100, 0.1, 7)
    # Targeted support-centroid depth must be at least 0.6 km.
    config_deep = deep_target_config(config)
    assert config_deep["geometry"]["target_min_centroid_depth_km"] == 0.6
    rng = np.random.default_rng(5150)
    sobol = SobolStream(5151)
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    for mechanism in ("A_analytic", "C_irregular"):
        model = generate_model(
            rng, sobol, config_deep, mechanism, (0.02, 0.05)
        )
        depth = support_centroid_depth_km(model.density, threshold)
        assert depth >= 0.6, f"{mechanism} deep targeting failed: {depth}"


# --------------------------------------------------------------------------- #
# Test covariance-normalized d-prime on artificial pairs.
# --------------------------------------------------------------------------- #

def test_dprime_synthetic_offsets(contract: dict) -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(size=(6, 33, 33)).astype(np.float32) * 5.0
    density = np.ones((16, 32, 32), dtype=np.float32) * 0.4

    # Constant Tzz offset: d0=33*delta/sigma5; other groups are unchanged.
    delta = 0.7
    shifted = base.copy()
    shifted[5] += delta
    m = pair_metrics_absolute_covariance_v019(density, density * 1.1, base, shifted, contract)
    assert m["tzz_global_dprime"] == pytest.approx(33.0 * delta / SIGMA[5], rel=1e-6)
    assert m["spin1_global_dprime"] == pytest.approx(0.0, abs=1e-9)
    assert m["spin2_global_dprime"] == pytest.approx(0.0, abs=1e-9)

    # Txz offset: d1=33*delta/sigma2; spin-2 is unchanged.
    shifted = base.copy()
    shifted[2] += delta
    m = pair_metrics_absolute_covariance_v019(density, density * 1.1, base, shifted, contract)
    assert m["spin1_global_dprime"] == pytest.approx(33.0 * delta / SIGMA[2], rel=1e-6)
    assert m["tzz_global_dprime"] == pytest.approx(0.0, abs=1e-9)

    # Propagate Txx through (Txx-Tyy)/2 with sigma=sqrt((s0^2+s3^2)/4).
    shifted = base.copy()
    shifted[0] += delta
    sigma_q1 = float(np.sqrt((SIGMA[0] ** 2 + SIGMA[3] ** 2) / 4.0))
    m = pair_metrics_absolute_covariance_v019(density, density * 1.1, base, shifted, contract)
    assert m["spin2_global_dprime"] == pytest.approx(33.0 * (delta / 2.0) / sigma_q1, rel=1e-6)

    # Use symmetric density NRMSE.
    a = np.full((16, 32, 32), 0.5, dtype=np.float32)
    b = np.full((16, 32, 32), 0.4, dtype=np.float32)
    expected = abs(0.5 - 0.4) / np.sqrt(0.5 * (0.5**2 + 0.4**2))
    m = pair_metrics_absolute_covariance_v019(a, b, base, base, contract)
    assert m["density_nrmse"] == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- #
# Integration: prototype, smooth direction, scale search and valid endpoints.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def operator(config: dict) -> GravityTensorOperator:
    if not Path(config['operator']['g6_path']).is_file():
        pytest.skip('Set TARF_GDC_ASSETS to run the full-operator integration test')
    return GravityTensorOperator(config, "cpu")


def test_geometry_contract_on_prototypes(config: dict) -> None:
    rng = np.random.default_rng(123)
    sobol = SobolStream(456)
    for mechanism in ("A_analytic", "B_structural", "C_irregular", "D_composite"):
        model = generate_model(
            rng, sobol, config, mechanism,
            tuple(config["support_fraction_mixture"]["main_2_8pct"]["range"]),
        )
        threshold = float(config["density"]["support_threshold_abs_gcc"])
        active = np.abs(model.density) >= threshold
        assert int(active.sum()) >= 82
        assert float(np.max(np.abs(model.density))) <= 0.80 + 1e-6
        values = model.density[active]
        assert np.all(values > 0.0) or np.all(values < 0.0)


def test_random_smooth_direction_properties(config: dict) -> None:
    rng = np.random.default_rng(7)
    sobol = SobolStream(8)
    model = generate_model(
        rng, sobol, config, "C_irregular",
        tuple(config["support_fraction_mixture"]["main_2_8pct"]["range"]),
    )
    direction = random_smooth_direction(model.volume_fraction, rng, config["ambiguity"])
    assert direction is not None
    assert direction.shape == model.volume_fraction.shape
    mask = model.volume_fraction > 0.0
    assert np.all(direction[~mask] == 0.0), "direction must live on the support"
    assert float(np.abs(direction[mask]).max()) <= float(model.volume_fraction.max()) + 1e-6
    # Redistribute mass only within the existing support.
    mass = float(np.sum(direction.astype(np.float64)))
    assert abs(mass) <= 1e-5 * float(np.sum(model.volume_fraction))


def test_scale_sweep_matches_canonical_metrics(
    config: dict, operator: GravityTensorOperator
) -> None:
    'Check vectorized metrics against canonical pair metrics and enforce peak, sign, support-size and Jaccard contracts.'
    rng = np.random.default_rng(2026)
    sobol = SobolStream(912)
    amb = config["ambiguity"]
    cards = pair_task_cards(config, 4, 777)
    hits = 0
    for card in cards:
        model = generate_model(
            rng, sobol, config, str(card["mechanism"]),
            tuple(config["support_fraction_mixture"][card["support_stratum"]]["range"]),
        )
        direction = random_smooth_direction(model.volume_fraction, rng, amb)
        assert direction is not None
        sign = float(card["sign"])
        common = (sign * float(card["base_contrast_gcc"]) * model.volume_fraction).astype(np.float32)
        signed_direction = (sign * float(card["redistribution_amplitude_gcc"]) * direction).astype(np.float32)
        responses = operator.forward(np.stack([common, signed_direction]), batch_size=2)
        passing, _ = evaluate_proposal(
            config, (responses[0], responses[1]), (common, signed_direction), card
        )
        if not passing:
            continue
        hits += 1
        cand = passing[len(passing) // 2]
        s = float(cand["scale"])
        first = common + s * signed_direction
        second = common - s * signed_direction
        resp_a = responses[0] + s * responses[1]
        resp_b = responses[0] - s * responses[1]
        m = pair_metrics_absolute_covariance_v019(first, second, resp_a, resp_b, amb)
        assert m["tzz_global_dprime"] == pytest.approx(cand["tzz_global_dprime"], rel=1e-6)
        assert classify_pair_reverse_v023(m, amb) == "distinguishable"
        assert dprime_bin_index(amb, m["tzz_global_dprime"]) == cand["dprime_bin"]
        for endpoint in (first, second):
            ok, detail = _endpoint_contract_ok(endpoint, str(card["support_stratum"]), config)
            assert ok, detail
            active = np.abs(endpoint) >= float(config["density"]["support_threshold_abs_gcc"])
            assert int(active.sum()) >= 82
            assert float(np.abs(endpoint).max()) <= 0.80 + 1e-6
            values = endpoint[active]
            assert np.all(values > 0.0) or np.all(values < 0.0)
        # Identical supports have Jaccard=1.
        threshold = float(config["density"]["support_threshold_abs_gcc"])
        assert np.array_equal(
            np.abs(first) >= threshold, np.abs(second) >= threshold
        )
    assert hits >= 2, "most random proposals should yield a non-empty passing set"
