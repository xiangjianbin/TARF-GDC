"""V031 statistics: registered training seeds, five noise views, geological resampling.

The primary estimator is an equal-endpoint mean. Pair clusters are sampling
units, never equally weighted substitutes for their two endpoint observations.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable

import numpy as np


from .experiment import ALL as CORE, comparison_plan as registered_comparisons

CORE_VARIANTS = CORE
TRACKS = ("ordinary", "separable", "shape")
NOISE_IDS = (0, 1, 2, 3, 4)
GEOMETRY_METRICS = ("density_centroid_error_m", "density_top_depth_error_m")
ENDPOINT_METRICS = (
    "density_sae", "density_sse", "density_mse", "foreground_nrmse",
    "density_support_dice", "density_support_precision", "density_support_recall",
    "density_support_false_positive_rate", "density_support_macro_unit_dice",
    "density_support_volume_relative_error", "density_support_voxel_count",
    "physics_nrms", "empty_prediction_rate", "empty_truth_rate",
    "density_geometry_valid_rate", "density_min", "density_max", "density_out_of_range_rate",
    *GEOMETRY_METRICS,
)
PAIR_METRICS = (
    "pdacc_direction_correct", "pdacc_cos", "difference_magnitude_ratio",
    "difference_nrmse",
)
METADATA_COLUMNS = (
    "sample_id", "ambiguity_group_id", "ambiguity_pair_role", "generation_track",
    "track", "response_group", "spin_dominance", "dprime_bin", "depth_group",
    "mechanism", "subtype", "support_stratum", "is_pair", "split", "checkpoint",
    "epoch", "sample_id_A", "sample_id_B", "tzz_global_dprime",
    "spin1_global_dprime", "spin2_global_dprime", "physical_depth_m",
    "truth_support_voxel_count", "truth_top_depth_m", "truth_bottom_depth_m",
)


def comparison_plan(include_optional: bool = False) -> list[dict[str, Any]]:
    """The labels and coefficients are frozen before any locked-test inference."""
    if not include_optional:
        return registered_comparisons()
    if CORE_VARIANTS == ("b2r_dt_context_pre",):
        if include_optional:
            raise ValueError("no optional training candidate is registered")
        # Cross-project contrasts are frozen in configs/joint_registry.json.
        return []
    if include_optional:
        raise ValueError("V028 contains only the six requested models; no optional extensions")
    pairs = [
        ("b0_vs_t0", "b0", "t0", "exploratory"),
        ("b2r_vs_b0", "b2r", "b0", "exploratory"),
        ("gsfa_vs_b2r", "gsfa", "b2r", "exploratory"),
        ("dt_vs_b2r", "dt", "b2r", "exploratory"),
        ("full_vs_b2r", "full", "b2r", "exploratory"),
        ("full_vs_gsfa", "full", "gsfa", "exploratory"),
        ("full_vs_dt", "full", "dt", "exploratory"),
    ]
    result = [
        {"id": name, "metric": "density_sae", "role": role,
         "coefficients": {candidate: 1.0, reference: -1.0}}
        for name, candidate, reference, role in pairs
    ]
    result.append({
        "id": "gsfa_dt_interaction", "metric": "density_sae",
        "role": "exploratory", "coefficients": {
            "full": 1.0, "gsfa": -1.0, "dt": -1.0, "b2r": 1.0,
        },
    })
    return result


def numeric_mean(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("means require nonempty finite values")
    return float(array.mean())


def metric_mean(name: str, values: Iterable[Any]) -> float:
    """Only undefined geometry may be omitted; every numerical failure is fatal."""
    array = np.asarray(list(values), dtype=np.float64)
    if name in {"density_min", "density_max"}:
        numeric_mean(array)  # Validate the complete set before taking its range.
        return float(array.min() if name == "density_min" else array.max())
    if name not in GEOMETRY_METRICS:
        return numeric_mean(array)
    if array.size == 0 or np.isinf(array).any():
        raise ValueError(f"invalid geometry values for {name}")
    valid = np.isfinite(array)
    return float(array[valid].mean()) if valid.any() else float("nan")


def summarize_metric_rows(rows: list[dict[str, Any]], names: Iterable[str]) -> dict[str, Any]:
    """Summarize means and explicit geometry denominators without reweighting endpoints."""
    result: dict[str, Any] = {}
    for name in names:
        result[name] = metric_mean(name, (row[name] for row in rows))
        if name in GEOMETRY_METRICS:
            result[name + "_valid_count"] = sum(math.isfinite(float(row[name])) for row in rows)
            result[name + "_total_count"] = len(rows)
            if rows and name + "_valid_count" in rows[0]:
                result[name + "_valid_noise_count"] = sum(int(row[name + "_valid_count"]) for row in rows)
                result[name + "_total_noise_count"] = sum(int(row[name + "_total_count"]) for row in rows)
    return result


def collapse_realizations(
    rows: list[dict[str, Any]],
    key: str = "sample_id",
    required_ids: Iterable[int] = NOISE_IDS,
    metric_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Average metrics, not predictions, within each endpoint/pair over noise."""
    required = set(int(value) for value in required_ids)
    if not required or not rows:
        raise ValueError("noise aggregation requires nonempty rows and realizations")
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        identifier = str(row[key])
        realization = int(row["noise_realization"])
        if realization not in required or realization in grouped[identifier]:
            raise ValueError(f"unexpected or duplicate noise view for {identifier}")
        grouped[identifier][realization] = row
    result = []
    for identifier in sorted(grouped):
        views = grouped[identifier]
        if set(views) != required:
            raise ValueError(f"incomplete noise views for {identifier}: {sorted(views)}")
        ordered = [views[index] for index in sorted(required)]
        first = ordered[0]
        names = list(metric_names) if metric_names is not None else [
            name for name in first if name not in METADATA_COLUMNS
            and name not in {"noise_realization", "noise_mode", "realization_count"}
            and not name.endswith(".alpha_top_channel")
        ]
        merged = {name: first[name] for name in METADATA_COLUMNS if name in first}
        for row in ordered:
            for name, value in merged.items():
                if row.get(name) != value:
                    raise ValueError(f"metadata {name} changed across noise for {identifier}")
            if any(name not in row for name in names):
                raise ValueError(f"metric columns changed across noise for {identifier}")
        for name in names:
            merged[name] = metric_mean(name, (row[name] for row in ordered))
            if name in GEOMETRY_METRICS:
                merged[name + "_valid_count"] = sum(math.isfinite(float(row[name])) for row in ordered)
                merged[name + "_total_count"] = len(ordered)
        merged["realization_count"] = len(required)
        result.append(merged)
    return result


def track_of(row: dict[str, Any]) -> str:
    track = str(row.get("track", row.get("generation_track", "")))
    if track not in TRACKS:
        raise ValueError(f"invalid generation track {track!r}")
    return track


def endpoint_units(rows: list[dict[str, Any]]) -> dict[str, list[list[int]]]:
    """Validate complete A/B pairs and construct the stratified sampling units."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if not rows or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("endpoint inference requires distinct nonempty geological IDs")
    group_tracks: dict[str, str] = {}
    for index, row in enumerate(rows):
        track = track_of(row)
        group = str(row.get("ambiguity_group_id", ""))
        if track == "ordinary":
            if group or row.get("ambiguity_pair_role", ""):
                raise ValueError("ordinary observations cannot have ambiguity metadata")
            unit = str(row["sample_id"])
        else:
            if not group or str(row.get("ambiguity_pair_role", "")) not in {"A", "B"}:
                raise ValueError("paired tracks require a group and a legal endpoint role")
            if group in group_tracks and group_tracks[group] != track:
                raise ValueError("an ambiguity group spans generation tracks")
            group_tracks[group] = track
            unit = group
        groups[(track, unit)].append(index)
    strata: dict[str, list[list[int]]] = defaultdict(list)
    for (track, _), indexes in sorted(groups.items()):
        if track != "ordinary" and (
            len(indexes) != 2
            or {str(rows[index]["ambiguity_pair_role"]) for index in indexes} != {"A", "B"}
        ):
            raise ValueError("each ambiguity group must contain exactly its A/B endpoints")
        strata[track].append(indexes)
    return dict(strata)


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    pairs = {str(row.get("ambiguity_group_id", "")) for row in rows}
    pairs.discard("")
    ordinary = sum(track_of(row) == "ordinary" for row in rows)
    return {"n_endpoints": len(rows), "n_ordinary_models": ordinary,
            "n_ambiguity_groups": len(pairs), "n_sampling_units": ordinary + len(pairs)}


def stratified_cluster_bootstrap(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    replicates: int = 2000,
    seed: int = 1107,
    selected: np.ndarray | None = None,
) -> dict[str, Any]:
    """Paired bootstrap of any endpoint contrast, with endpoint count weights.

    A multidimensional values array uses identical draws for every contrast.
    For subgroup estimates, whole geological groups are still sampled; only
    qualifying endpoint contributions enter that subgroup's numerator/denominator.
    """
    strata = endpoint_units(rows)
    array = np.asarray(values, dtype=np.float64)
    scalar = array.ndim == 1
    if scalar:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != len(rows) or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be finite and aligned with endpoint rows")
    mask = np.ones(len(rows), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    if mask.shape != (len(rows),) or not mask.any() or replicates < 100:
        raise ValueError("bootstrap requires a nonempty selection and at least 100 replicates")
    sums_and_counts = []
    for track in sorted(strata):
        unit_values, unit_counts = [], []
        for indexes in strata[track]:
            qualifying = [index for index in indexes if mask[index]]
            if qualifying:
                unit_values.append(array[qualifying].sum(axis=0))
                unit_counts.append(len(qualifying))
        if unit_counts:
            sums_and_counts.append((np.asarray(unit_values), np.asarray(unit_counts)))
    random = np.random.default_rng(seed)
    draws = np.empty((replicates, array.shape[1]), dtype=np.float64)
    # Bounded intermediate storage, including the larger locked-test population.
    for start in range(0, replicates, 32):
        stop = min(start + 32, replicates)
        numerator = np.zeros((stop - start, array.shape[1]), dtype=np.float64)
        denominator = np.zeros(stop - start, dtype=np.float64)
        for unit_values, unit_counts in sums_and_counts:
            indexes = random.integers(0, len(unit_counts), size=(stop - start, len(unit_counts)))
            numerator += unit_values[indexes].sum(axis=1)
            denominator += unit_counts[indexes].sum(axis=1)
        draws[start:stop] = numerator / denominator[:, None]
    estimate = array[mask].mean(axis=0)
    lower, upper = np.percentile(draws, [2.5, 97.5], axis=0)
    convert = (lambda value: float(value[0])) if scalar else (lambda value: value.tolist())
    return {"estimate": convert(estimate), "ci95_low": convert(lower),
            "ci95_high": convert(upper), "replicates": replicates, "bootstrap_seed": seed,
            **counts([row for index, row in enumerate(rows) if mask[index]])}


def subgroup_masks(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    result = {"all": np.ones(len(rows), dtype=bool)}
    for track in TRACKS:
        track_mask = np.asarray([track_of(row) == track for row in rows])
        if not track_mask.any():
            continue
        result[f"track/{track}"] = track_mask
        for field in ("dprime_bin", "spin_dominance", "depth_group"):
            categories = sorted({str(row.get(field, "missing")) for index, row in enumerate(rows)
                                 if track_mask[index]})
            for category in categories:
                if category in {"", "missing", "ordinary"} and field != "depth_group":
                    continue
                result[f"track/{track}/{field}/{category}"] = track_mask & np.asarray(
                    [str(row.get(field, "missing")) == category for row in rows]
                )
    for category in sorted({str(row.get("depth_group", "missing")) for row in rows}):
        result[f"depth_group/{category}"] = np.asarray(
            [str(row.get("depth_group", "missing")) == category for row in rows]
        )
    return result


def summarize_endpoints(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for group, mask in subgroup_masks(rows).items():
        subset = [row for index, row in enumerate(rows) if mask[index]]
        result[group] = {**counts(subset), **summarize_metric_rows(subset, ENDPOINT_METRICS)}
    return result


def summarize_pairs(rows: list[dict[str, Any]], replicates: int | None = None) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    result = {}
    endpoints = [dict(row, sample_id=row[f"sample_id_{role}"], ambiguity_pair_role=role)
                 for row in rows for role in ("A", "B")]
    values = np.asarray([[float(row[name]) for name in PAIR_METRICS] for row in endpoints])
    # Pair depth groups use the mean physical support depth of their two endpoints.
    for group, mask in subgroup_masks(rows).items():
        subset = [row for index, row in enumerate(rows) if mask[index]]
        result[group] = {"n_pairs": len(subset), **{
            name: numeric_mean(row[name] for row in subset) for name in PAIR_METRICS
        }}
        if replicates is not None:
            interval = stratified_cluster_bootstrap(
                endpoints, values, replicates=replicates, selected=np.repeat(mask, 2), seed=1107,
            )
            for index, name in enumerate(PAIR_METRICS):
                result[group][name + "_ci95_low"] = interval["ci95_low"][index]
                result[group][name + "_ci95_high"] = interval["ci95_high"][index]
            result[group]["bootstrap_replicates"] = replicates
            result[group]["bootstrap_seed"] = 1107
    return result


def interpretation(estimate: float, low: float, high: float, primary: bool) -> str:
    if not all(math.isfinite(value) for value in (estimate, low, high)):
        raise ValueError("nonfinite comparison result")
    if high < 0.0:
        return ("Improvement observed under the registered seeds and this data setting" if primary
                else "Exploratory comparison: negative Δ observed, interval below zero")
    if low > 0.0:
        return "The candidate metric is higher under the registered seeds and this data setting"
    return "No clear difference detected; no equivalence bound was registered this round, so no equivalence claim is made"
