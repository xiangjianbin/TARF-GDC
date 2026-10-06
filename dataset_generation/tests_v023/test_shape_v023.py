'Test shape gates, classifier boundaries, optimizer determinism and endpoint contracts.'
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
from build_pairs_v023 import _endpoint_contract_ok, random_smooth_direction  # noqa: E402
from build_shape_pairs_v023 import (  # noqa: E402
    classify_shape_pair_v023,
    evaluate_shape_proposal,
    prototype_support_range,
    select_shape_candidate,
    shape_gate,
    shape_task_cards,
    validate_shape_config,
)
from v023_shape_optimizer import optimize_shape_directions_multistart  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_shape_v023.json"
SIGMA = np.array([0.425, 0.31, 0.70, 0.425, 0.70, 0.74])


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def contract(config: dict) -> dict:
    return config["ambiguity"]


# --------------------------------------------------------------------------- #
# Test shape gates and classifier boundaries.
# --------------------------------------------------------------------------- #

def test_shape_gate_boundaries() -> None:
    # Tier 1: Jaccard <=0.7 and symmetric difference >=0.20.
    assert shape_gate(0.7, 0.3, 0.7, 0.2)
    assert shape_gate(0.0, 1.0, 0.7, 0.2)
    assert shape_gate(0.5, 0.2, 0.7, 0.2)
    assert not shape_gate(0.7000001, 0.3, 0.7, 0.2)
    assert not shape_gate(0.5, 0.1999, 0.7, 0.2)
    # tier2: <=0.8 / >=0.15
    assert shape_gate(0.8, 0.2, 0.8, 0.15)
    assert not shape_gate(0.8000001, 0.2, 0.8, 0.15)
    assert not shape_gate(0.75, 0.149, 0.8, 0.15)


def test_classify_shape_pair_boundaries(contract: dict) -> None:
    def m(d0, d1, d2, nrmse):
        return {
            "tzz_global_dprime": d0,
            "spin1_global_dprime": d1,
            "spin2_global_dprime": d2,
            "density_nrmse": nrmse,
        }
    assert classify_shape_pair_v023(m(1.0, 25.0, 0.0, 0.12), contract) == "shape_ambiguous"
    assert classify_shape_pair_v023(m(0.0, 0.0, 25.0, 0.12), contract) == "shape_ambiguous"
    assert classify_shape_pair_v023(m(1.0001, 100.0, 100.0, 0.5), contract) is None
    assert classify_shape_pair_v023(m(0.5, 24.9999, 24.0, 0.5), contract) is None
    assert classify_shape_pair_v023(m(0.5, 50.0, 50.0, 0.1199), contract) is None


def test_validate_shape_config(config: dict) -> None:
    validate_shape_config(config)


# --------------------------------------------------------------------------- #
# Test shape gates on artificial supports.
# --------------------------------------------------------------------------- #

def test_support_metrics_synthetic() -> None:
    from v017.ambiguity import support_pair_metrics

    a = np.zeros((16, 32, 32), dtype=np.float32)
    b = np.zeros((16, 32, 32), dtype=np.float32)
    a[4:8, 4:12, 4:12] = 0.5   # 4x8x8 = 256 cells
    b[4:8, 8:16, 4:12] = 0.5   # Adjacent 4x8x8 blocks intersect at y=8:12.
    inter = 4 * 4 * 8  # y 8..11
    union = 256 + 256 - inter
    sm = support_pair_metrics(a, b, 0.005)
    assert sm["support_intersection_cells"] == inter
    assert sm["support_union_cells"] == union
    assert sm["support_jaccard"] == pytest.approx(inter / union)
    assert sm["support_symmetric_difference_fraction"] == pytest.approx(
        (union - inter) / union
    )
    # Jaccard is 128/384 = 1/3, satisfying tier 1.
    assert shape_gate(sm["support_jaccard"], sm["support_symmetric_difference_fraction"], 0.7, 0.2)


# --------------------------------------------------------------------------- #
# Check deterministic CPU optimization with a small step budget.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def operator(config: dict) -> GravityTensorOperator:
    if not Path(config['operator']['g6_path']).is_file():
        pytest.skip('Set TARF_GDC_ASSETS to run the full-operator integration test')
    return GravityTensorOperator(config, "cpu")


def test_optimizer_deterministic_cpu(config: dict, operator: GravityTensorOperator) -> None:
    rng = np.random.default_rng(11)
    sobol = SobolStream(22)
    models = [
        generate_model(rng, sobol, config, "A_analytic", (0.03, 0.06)),
        generate_model(rng, sobol, config, "C_irregular", (0.03, 0.06)),
    ]
    fractions = np.stack([m.volume_fraction for m in models]).astype(np.float32)
    contract = dict(config["ambiguity"])
    contract["local_search_steps"] = 30
    d1, diag1 = optimize_shape_directions_multistart(
        fractions, operator.g6, contract, seed=12345, n_starts=2
    )
    d2, diag2 = optimize_shape_directions_multistart(
        fractions, operator.g6, contract, seed=12345, n_starts=2
    )
    assert np.array_equal(d1, d2), "same seed must reproduce directions bitwise (CPU)"
    assert diag1[0]["density_energy_over_e0"] == diag2[0]["density_energy_over_e0"]
    d3, _ = optimize_shape_directions_multistart(
        fractions, operator.g6, contract, seed=54321, n_starts=2
    )
    assert not np.array_equal(d1, d3), "different seed must differ"
    for diag in diag1:
        assert diag["n_starts"] == 2
        assert np.isfinite(diag["density_energy_over_e0"])
        assert np.isfinite(diag["linearized_spin1_to_tzz"])


# --------------------------------------------------------------------------- #
# Test endpoint contracts and scale screening with synthetic responses, without G6.
# --------------------------------------------------------------------------- #

def _synthetic_basis(config: dict, fraction: np.ndarray, rng: np.random.Generator):
    'Synthetic variable-support basis with weak Tzz and strong Txz responses.'
    from scipy import ndimage

    sign = 1.0
    base_contrast, amplitude = 0.42, 0.30
    mask = fraction > 0.0
    core = ndimage.binary_erosion(
        mask, structure=ndimage.generate_binary_structure(3, 1), iterations=1
    )
    field = rng.normal(size=fraction.shape).astype(np.float32)
    field = ndimage.gaussian_filter(field, sigma=1.0)
    field *= mask
    field /= max(float(np.abs(field[mask]).max()), 1e-6)
    direction = (fraction * field).astype(np.float32)
    common = (sign * base_contrast * fraction * core.astype(np.float32)).astype(np.float32)
    absolute = (sign * amplitude * np.abs(direction)).astype(np.float32)
    signed = (sign * amplitude * direction).astype(np.float32)
    resp_common = rng.normal(size=(6, 33, 33)).astype(np.float32) * 3.0
    resp_abs = rng.normal(size=(6, 33, 33)).astype(np.float32) * 0.5
    resp_signed = rng.normal(size=(6, 33, 33)).astype(np.float32)
    resp_signed[5] *= 1e-3   # Near-zero Tzz makes scalar d-prime small.
    resp_signed[2] *= 20.0   # Large Txz makes spin-1 d-prime large.
    return (common, absolute, signed), (resp_common, resp_abs, resp_signed)


def test_shape_scale_sweep_matches_canonical(config: dict) -> None:
    rng = np.random.default_rng(2027)
    sobol = SobolStream(33)
    model = generate_model(rng, sobol, config, "B_structural", (0.05, 0.08))
    basis_density, basis_response = _synthetic_basis(config, model.volume_fraction, rng)
    card = {"support_stratum": "main_2_8pct"}
    passing1, passing2, rejects = evaluate_shape_proposal(
        config, basis_response, basis_density, card
    )
    assert passing1 or passing2, f"synthetic proposal should pass some scale: {rejects}"
    passing = passing1 if passing1 else passing2
    tier = 1 if passing1 else 2
    pool = select_shape_candidate(passing, config["ambiguity"])
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    checked = 0
    for cand in pool[:3]:
        s = float(cand["scale"])
        common, absolute, signed = basis_density
        rc, ra, rs = basis_response
        base = common + s * absolute
        first = base + s * signed
        second = base - s * signed
        resp_a = rc + s * ra + s * rs
        resp_b = rc + s * ra - s * rs
        m = pair_metrics_absolute_covariance_v019(
            first, second, resp_a, resp_b, config["ambiguity"]
        )
        # Float32 endpoint arithmetic can round cancellation-sensitive d-prime values.
        # The scale sweep uses float64; the canonical endpoint check remains authoritative.
        # Use relative tolerance 1e-4 here; production rejects candidates failing canonical checks.
        assert m["tzz_global_dprime"] == pytest.approx(cand["tzz_global_dprime"], rel=1e-4)
        from v017.ambiguity import support_pair_metrics
        sm = support_pair_metrics(first, second, threshold)
        # Different float32 conversion order can move threshold-boundary voxels.
        # Jaccard may differ by about 1e-3 at voxel threshold boundaries.
        # Production advances to the next candidate after a failed canonical check.
        assert sm["support_jaccard"] == pytest.approx(cand["support_jaccard"], abs=2e-3)
        gate = config["ambiguity"] if tier == 1 else config["ambiguity"]["shape_gate_tier2"]
        assert shape_gate(
            sm["support_jaccard"], sm["support_symmetric_difference_fraction"],
            float(gate["shape_jaccard_max"]), float(gate["shape_symmetric_difference_min"]),
        )
        for endpoint in (first, second):
            ok, detail = _endpoint_contract_ok(endpoint, "main_2_8pct", config)
            assert ok, detail
            active = np.abs(endpoint) >= threshold
            assert int(active.sum()) >= 82
            assert float(np.abs(endpoint).max()) <= 0.80 + 1e-6
            values = endpoint[active]
            assert np.all(values > 0.0) or np.all(values < 0.0)
        checked += 1
    assert checked >= 1


def test_prototype_support_range_bias(config: dict) -> None:
    low, high = prototype_support_range(config, "main_2_8pct")
    full_low, full_high = config["support_fraction_mixture"]["main_2_8pct"]["range"]
    assert full_low < low < high == full_high
    cards = shape_task_cards(config, 8, 99)
    assert len(cards) == 8 and {c["support_stratum"] for c in cards}


def test_assemble_worker_tag(config: dict, tmp_path) -> None:
    'Worker-specific sample and pair IDs must include distinct w<N> tags.'
    from build_shape_pairs_v023 import assemble_shape_split

    rng = np.random.default_rng(2027)
    sobol = SobolStream(33)
    model = generate_model(rng, sobol, config, "B_structural", (0.05, 0.08))
    basis_density, basis_response = _synthetic_basis(config, model.volume_fraction, rng)
    card = {
        "target_id": "shape_t", "mechanism": "B_structural",
        "support_stratum": "main_2_8pct", "sign": 1.0,
        "base_contrast_gcc": 0.42, "redistribution_amplitude_gcc": 0.30,
        "attempts": 1,
    }
    passing1, passing2, _ = evaluate_shape_proposal(
        config, basis_response, basis_density, card
    )
    passing = passing1 if passing1 else passing2
    cand = select_shape_candidate(passing, config["ambiguity"])[0]
    s = float(cand["scale"])
    common, absolute, signed = basis_density
    rc, ra, rs = basis_response
    base = common + s * absolute
    from v017.ambiguity import pair_metrics_absolute_covariance_v019, support_pair_metrics
    item = {
        "card": card,
        "densities": (base + s * signed, base - s * signed),
        "responses": np.stack((rc + s * ra + s * rs, rc + s * ra - s * rs)),
        "volume_fraction": model.volume_fraction,
        "metrics": pair_metrics_absolute_covariance_v019(
            base + s * signed, base - s * signed,
            rc + s * ra + s * rs, rc + s * ra - s * rs, config["ambiguity"],
        ),
        "support_metrics": support_pair_metrics(
            base + s * signed, base - s * signed,
            float(config["density"]["support_threshold_abs_gcc"]),
        ),
        "scale": s,
        "shape_gate_tier": 1 if passing1 else 2,
        "source_record": model.record,
        "optimizer": {"n_starts": 4, "best_start": 0, "density_energy_over_e0": 1.0,
                      "linearized_spin1_to_tzz": 1.0, "linearized_spin2_to_tzz": 1.0},
        "base_contrast_gcc_actual": 0.42,
        "redistribution_amplitude_gcc_actual": 0.30,
        "generation_attempts_for_target": 1,
    }
    _, pair_rows, records = assemble_shape_split(
        config, "train", 2026091401, [item], tmp_path, worker_tag="w3"
    )
    assert len(records) == 2 and len(pair_rows) == 1
    assert all(r["sample_id"].startswith("v023s_train_w3_") for r in records)
    assert "_w3_shape" in pair_rows[0]["ambiguity_group_id"]
    _, pair_rows_w4, records_w4 = assemble_shape_split(
        config, "train", 2026094401, [item], tmp_path / "w4", worker_tag="w4"
    )
    assert not ({r["sample_id"] for r in records} & {r["sample_id"] for r in records_w4})
    assert pair_rows[0]["ambiguity_group_id"] != pair_rows_w4[0]["ambiguity_group_id"]
