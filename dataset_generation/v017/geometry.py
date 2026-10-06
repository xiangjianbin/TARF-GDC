from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import ndimage
from scipy.stats import qmc


DENSITY_SHAPE = (16, 32, 32)
CELL_SIZE_ZYX_KM = np.array((0.0625, 0.125, 0.125), dtype=np.float64)
EXTENT_ZYX_KM = np.array((1.0, 4.0, 4.0), dtype=np.float64)
MECHANISMS = ("A_analytic", "B_structural", "C_irregular", "D_composite")
SUBTYPES = {
    "A_analytic": ("ellipsoid", "superellipsoid", "box", "cylinder", "lens", "frustum"),
    "B_structural": ("dipping_slab", "dike", "wedge", "faulted_body", "curved_layer"),
    "C_irregular": ("grf_levelset", "harmonic_levelset", "blob_union"),
    "D_composite": ("body_flank", "notched_body", "weak_appendage", "density_gradient", "zoned_density"),
}


@dataclass
class GeneratedModel:
    density: np.ndarray
    volume_fraction: np.ndarray
    record: dict[str, Any]


class SobolStream:
    """Deterministic low-discrepancy numbers with a separate stream per split."""

    def __init__(self, seed: int, dimension: int = 32) -> None:
        self.engine = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))

    def next(self) -> np.ndarray:
        return self.engine.random(1)[0]


def _fine_axes(supersampling: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = np.asarray(DENSITY_SHAPE) * int(supersampling)
    axes = []
    for n, extent in zip(shape, EXTENT_ZYX_KM):
        axes.append((np.arange(n, dtype=np.float32) + 0.5) * (extent / n))
    return tuple(axes)  # type: ignore[return-value]


def _local_coordinates(
    supersampling: int,
    center_zyx: np.ndarray,
    strike_rad: float,
    dip_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis, y_axis, x_axis = _fine_axes(supersampling)
    z = z_axis[:, None, None] - center_zyx[0]
    y = y_axis[None, :, None] - center_zyx[1]
    x = x_axis[None, None, :] - center_zyx[2]
    # strike rotation in x-y followed by dip about the local y axis.
    along = x * math.cos(strike_rad) + y * math.sin(strike_rad)
    across0 = -x * math.sin(strike_rad) + y * math.cos(strike_rad)
    vertical = z * math.cos(dip_rad) - across0 * math.sin(dip_rad)
    across = z * math.sin(dip_rad) + across0 * math.cos(dip_rad)
    return vertical, across, along


def _remove_tiny_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(minimum)
    keep[0] = False
    return keep[labels]


def _downsample(mask: np.ndarray, supersampling: int) -> np.ndarray:
    s = int(supersampling)
    return mask.reshape(16, s, 32, s, 32, s).mean(axis=(1, 3, 5)).astype(np.float32)


def _shape_field(
    subtype: str,
    vertical: np.ndarray,
    across: np.ndarray,
    along: np.ndarray,
    radii: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    rz, ry, rx = np.maximum(radii, 1e-4)
    u, v, w = vertical / rz, across / ry, along / rx
    parameters: dict[str, Any] = {}
    if subtype == "ellipsoid":
        return 1.0 - (u * u + v * v + w * w), parameters
    if subtype == "superellipsoid":
        exponent = float(rng.uniform(2.5, 5.5))
        parameters["superellipsoid_exponent"] = exponent
        return 1.0 - (np.abs(u) ** exponent + np.abs(v) ** exponent + np.abs(w) ** exponent), parameters
    if subtype == "box":
        return 1.0 - np.maximum.reduce(np.broadcast_arrays(np.abs(u), np.abs(v), np.abs(w))), parameters
    if subtype == "cylinder":
        return np.minimum(1.0 - (u * u + v * v), 1.0 - np.abs(w)), parameters
    if subtype == "lens":
        taper = np.maximum(0.15, 1.0 - w * w)
        return taper - (u * u + v * v), parameters
    if subtype == "frustum":
        taper = np.clip(1.0 - 0.45 * w, 0.35, 1.65)
        return np.minimum(taper - np.sqrt(u * u + v * v), 1.0 - np.abs(w)), parameters
    if subtype == "dipping_slab":
        return 1.0 - np.maximum.reduce(np.broadcast_arrays(np.abs(u), np.abs(v), np.abs(w))), parameters
    if subtype == "dike":
        waviness = float(rng.uniform(0.05, 0.18))
        parameters["waviness"] = waviness
        return 1.0 - np.maximum.reduce(np.broadcast_arrays(np.abs(u), np.abs(v - waviness * np.sin(2.5 * w)), np.abs(w))), parameters
    if subtype == "wedge":
        thickness = np.clip(1.0 - 0.70 * (w + 1.0) / 2.0, 0.25, 1.0)
        return np.minimum.reduce(np.broadcast_arrays(thickness - np.abs(u), 1.0 - np.abs(v), 1.0 - np.abs(w))), parameters
    if subtype == "faulted_body":
        throw = float(rng.uniform(0.25, 0.60))
        shifted = u - np.where(w >= 0.0, throw, -throw)
        gap = np.abs(w) > float(rng.uniform(0.08, 0.20))
        parameters["normalized_throw"] = throw
        return np.where(gap, 1.0 - (shifted * shifted + v * v + w * w), -1.0), parameters
    if subtype == "curved_layer":
        curvature = float(rng.uniform(0.15, 0.45))
        centerline = curvature * (w * w - 0.4) + 0.15 * np.sin(2.5 * w)
        parameters["curvature"] = curvature
        return 1.0 - np.maximum.reduce(np.broadcast_arrays(np.abs((u - centerline)), np.abs(v), np.abs(w))), parameters
    if subtype in {"grf_levelset", "harmonic_levelset"}:
        envelope = 1.0 - (u * u + v * v + w * w)
        field = np.zeros(np.broadcast_shapes(u.shape, v.shape, w.shape), dtype=np.float32)
        n_terms = 10 if subtype == "grf_levelset" else 6
        for _ in range(n_terms):
            kz, ky, kx = rng.uniform(0.7, 3.2, size=3)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amplitude = rng.normal() / (kz * kz + ky * ky + kx * kx)
            field += amplitude * np.sin(kz * u + ky * v + kx * w + phase)
        field /= max(float(field.std()), 1e-6)
        strength = float(rng.uniform(0.18, 0.42))
        parameters["random_field_strength"] = strength
        return envelope + strength * field, parameters
    if subtype == "blob_union":
        result = 1.0 - (u * u + v * v + w * w)
        offsets = []
        for _ in range(int(rng.integers(2, 5))):
            oz, oy, ox = rng.uniform(-0.65, 0.65, size=3)
            scale = float(rng.uniform(0.35, 0.65))
            blob = 1.0 - (((u - oz) / scale) ** 2 + ((v - oy) / scale) ** 2 + ((w - ox) / scale) ** 2)
            result = np.maximum(result, blob)
            offsets.append([float(oz), float(oy), float(ox), scale])
        parameters["blob_offsets"] = offsets
        return result, parameters
    # D support starts from an ellipsoid and then applies a structural modifier.
    base = 1.0 - (u * u + v * v + w * w)
    if subtype == "body_flank":
        flank = 1.0 - (((u - 0.10) / 0.65) ** 2 + ((v + 0.65) / 0.55) ** 2 + ((w - 0.70) / 0.80) ** 2)
        return np.maximum(base, flank), parameters
    if subtype == "notched_body":
        notch = 1.0 - (((u + 0.15) / 0.45) ** 2 + ((v - 0.45) / 0.40) ** 2 + ((w + 0.15) / 0.55) ** 2)
        return np.where(notch > 0.0, -1.0, base), parameters
    if subtype == "weak_appendage":
        appendage = 1.0 - (((u - 0.20) / 0.55) ** 2 + ((v + 0.75) / 0.45) ** 2 + ((w - 0.65) / 0.70) ** 2)
        return np.maximum(base, appendage), parameters
    if subtype in {"density_gradient", "zoned_density"}:
        return base, parameters
    raise ValueError(f"unknown subtype {subtype}")


def _density_from_fraction(
    fraction: np.ndarray,
    subtype: str,
    contrast: float,
    sign: float,
    local_coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    density = fraction * (contrast * sign)
    if subtype == "weak_appendage":
        # Fraction already represents the union; retain one-sign support and a weak side.
        coarse_axis = np.linspace(-1.0, 1.0, 32, dtype=np.float32)[None, None, :]
        density *= np.where(coarse_axis > 0.25, 0.55, 1.0)
    elif subtype == "density_gradient":
        coarse_axis = np.linspace(-1.0, 1.0, 32, dtype=np.float32)[None, None, :]
        density *= 0.65 + 0.35 * (coarse_axis + 1.0) / 2.0
    elif subtype == "zoned_density":
        z = np.linspace(-1.0, 1.0, 16, dtype=np.float32)[:, None, None]
        density *= np.where(z < 0.0, 0.65, 1.0)
    return density.astype(np.float32)


def geometry_metrics(
    density: np.ndarray,
    threshold: float = 0.005,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None:
        density_shape = DENSITY_SHAPE
        cell_size_zyx_km = CELL_SIZE_ZYX_KM
    else:
        density_shape = tuple(int(value) for value in config["grid"]["density_shape_zyx"])
        dx, dy, dz = (float(value) for value in config["grid"]["cell_size_xyz_m"])
        cell_size_zyx_km = np.asarray((dz, dy, dx), dtype=np.float64) / 1000.0
    if tuple(density.shape) != tuple(density_shape):
        raise ValueError(f"density shape {density.shape} does not match configured {density_shape}")
    mask = np.abs(density) >= threshold
    coords = np.argwhere(mask)
    if not len(coords):
        raise ValueError("empty model")
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel())[1:]
    weights = np.abs(density[mask]).astype(np.float64)
    physical = (coords + 0.5) * cell_size_zyx_km
    centroid = np.average(physical, axis=0, weights=weights)
    centered = physical - centroid
    covariance = (centered * weights[:, None]).T @ centered / max(float(weights.sum()), 1e-12)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    dip = math.degrees(math.atan2(abs(float(major[0])), max(math.hypot(float(major[1]), float(major[2])), 1e-12)))
    horizontal = centered[:, [2, 1]]
    hcov = (horizontal * weights[:, None]).T @ horizontal / max(float(weights.sum()), 1e-12)
    hvalues, hvectors = np.linalg.eigh(hcov)
    horder = np.argsort(hvalues)[::-1]
    hmajor = hvectors[:, horder[0]]
    azimuth = math.degrees(math.atan2(float(hmajor[1]), float(hmajor[0]))) % 180.0
    aspect = math.sqrt(max(float(hvalues[horder[0]]), 1e-12) / max(float(hvalues[horder[-1]]), 1e-12))
    component_horizontal_spans = []
    for label_id in range(1, count + 1):
        component_coords = np.argwhere(labels == label_id)
        y_span = int(component_coords[:, 1].max() - component_coords[:, 1].min() + 1)
        x_span = int(component_coords[:, 2].max() - component_coords[:, 2].min() + 1)
        component_horizontal_spans.append(max(y_span, x_span))
    minimum_gap_cells: float | None = None
    if count > 1:
        gaps = []
        for label_id in range(1, count + 1):
            distance = ndimage.distance_transform_edt(labels != label_id)
            other = (labels != 0) & (labels != label_id)
            if np.any(other):
                gaps.append(max(float(distance[other].min()) - 1.0, 0.0))
        minimum_gap_cells = min(gaps) if gaps else None
    z_max, y_max, x_max = (value - 1 for value in density_shape)
    lateral_margin = int(min(
        coords[:, 1].min(), y_max - coords[:, 1].max(),
        coords[:, 2].min(), x_max - coords[:, 2].max(),
    ))
    shallow_margin = int(coords[:, 0].min())
    return {
        "support_cells": int(mask.sum()),
        "support_fraction": float(mask.mean()),
        "component_count": int(count),
        "smallest_component_cells": int(sizes.min()) if len(sizes) else 0,
        "minimum_gap_cells": minimum_gap_cells,
        "minimum_component_horizontal_span_cells": int(min(component_horizontal_spans)),
        "minimum_lateral_margin_cells": lateral_margin,
        "minimum_shallow_margin_cells": shallow_margin,
        "centroid_zyx_km": [float(x) for x in centroid],
        "top_depth_km": float((coords[:, 0].min() + 0.5) * cell_size_zyx_km[0]),
        "bottom_depth_km": float((coords[:, 0].max() + 0.5) * cell_size_zyx_km[0]),
        "horizontal_azimuth_deg": float(azimuth),
        "principal_axis_dip_deg": float(dip),
        "horizontal_aspect_ratio": float(aspect),
        "density_min_gcc": float(density[mask].min()),
        "density_max_gcc": float(density[mask].max()),
        "touches_lateral_boundary": bool(
            np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == y_max)
            or np.any(coords[:, 2] == 0) or np.any(coords[:, 2] == x_max)
        ),
        "touches_shallow_boundary": bool(np.any(coords[:, 0] == 0)),
    }


def generate_model(
    rng: np.random.Generator,
    sobol: SobolStream,
    config: dict[str, Any],
    mechanism: str,
    support_range: tuple[float, float],
    forced_subtype: str | None = None,
) -> GeneratedModel:
    if mechanism not in MECHANISMS:
        raise ValueError(mechanism)
    grid = config["grid"]
    configured_shape = tuple(int(value) for value in grid["density_shape_zyx"])
    configured_extent = tuple(float(value) / 1000.0 for value in (
        grid["physical_extent_xyz_m"][2],
        grid["physical_extent_xyz_m"][1],
        grid["physical_extent_xyz_m"][0],
    ))
    if configured_shape != DENSITY_SHAPE or not np.allclose(configured_extent, EXTENT_ZYX_KM):
        raise ValueError(
            "the current antialiased geometry backend supports only "
            f"shape={DENSITY_SHAPE}, extent_zyx_km={tuple(EXTENT_ZYX_KM)}; "
            f"got {configured_shape}, {configured_extent}"
        )
    s = int(grid["subvoxel_supersampling"])
    threshold = float(config["density"]["support_threshold_abs_gcc"])
    quality = config["geometry_quality"]
    # Opt-in deep targeting (config["geometry"]["target_min_centroid_depth_km"]).
    # When absent the default path below is byte-identical to the V022 production
    # generator.  Depth follows the V025 diagnostic convention: the stored z axis
    # is deep-first, depth_km = extent_z_km - z_geometry, and the gate uses the
    # unweighted support centroid (|density| >= threshold), not the density-weighted
    # geometry_metrics centroid.
    deep_cfg = config.get("geometry") or {}
    deep_min_km = deep_cfg.get("target_min_centroid_depth_km")
    deep_mode = deep_min_km is not None
    if deep_mode:
        deep_min_km = float(deep_min_km)
        deep_max_km = float(deep_cfg.get("target_max_centroid_depth_km", 0.95))
        cell_z_km = float(CELL_SIZE_ZYX_KM[0])
        extent_z_km = float(EXTENT_ZYX_KM[0])
        deep_z_hi = extent_z_km - deep_min_km
        deep_z_lo = max(extent_z_km - deep_max_km, 1.5 * cell_z_km)
    for attempt in range(int(quality["maximum_generation_attempts"])):
        u = sobol.next()
        subtype = forced_subtype or SUBTYPES[mechanism][int(u[0] * len(SUBTYPES[mechanism])) % len(SUBTYPES[mechanism])]
        target = float(support_range[0] + (support_range[1] - support_range[0]) * u[1])
        strike = float(np.pi * u[5])
        if mechanism == "B_structural":
            dip = float(np.deg2rad(8.0 + 62.0 * u[6]))
        else:
            dip = float(np.deg2rad(-20.0 + 40.0 * u[6]))
        aspect_y = float(0.65 + 1.70 * u[7])
        aspect_x = float(0.65 + 2.35 * u[8])
        if subtype in {"dipping_slab", "dike", "wedge", "curved_layer"}:
            aspect_x = float(2.0 + 2.0 * u[8])
            aspect_y = float(0.35 + 0.45 * u[7])
        lateral_margin = float(np.clip(0.45 + 1.65 * target ** (1.0 / 3.0), 0.65, 1.35))
        shallow_margin = float(np.clip(0.10 + 0.75 * target ** (1.0 / 3.0), 0.18, 0.48))
        if deep_mode:
            # Bias the u[2] mapping so support centroids land deep (geometry z
            # near index 0).  Radii are needed before the center, so they are
            # computed here with the same formulas as the default path, then the
            # vertical radius is capped so the body fits above the bottom layer.
            scale = float(np.clip((target / 0.04) ** (1.0 / 3.0), 0.45, 1.75))
            radii = np.minimum(
                np.array((0.30, 0.48 * aspect_y, 0.48 * aspect_x)) * scale,
                np.array((0.72, 1.55, 1.55)),
            )
            rz_cap = max(deep_z_hi - 2.0 * cell_z_km, 2.0 * cell_z_km)
            radii[0] = min(float(radii[0]), rz_cap)
            c_lo = max(deep_z_lo, float(radii[0]) + 1.5 * cell_z_km)
            c_hi = max(deep_z_hi, c_lo)
            center = np.array((
                c_lo + (c_hi - c_lo) * u[2],
                lateral_margin + (4.0 - 2.0 * lateral_margin) * u[3],
                lateral_margin + (4.0 - 2.0 * lateral_margin) * u[4],
            ))
        else:
            center = np.array((
                shallow_margin + (0.88 - shallow_margin) * u[2],
                lateral_margin + (4.0 - 2.0 * lateral_margin) * u[3],
                lateral_margin + (4.0 - 2.0 * lateral_margin) * u[4],
            ))
        coords = _local_coordinates(s, center, strike, dip)
        contrast = float(config["density"]["absolute_contrast_gcc_range"][0] + (
            config["density"]["absolute_contrast_gcc_range"][1]
            - config["density"]["absolute_contrast_gcc_range"][0]
        ) * u[9])
        sign = 1.0 if u[10] < float(config["density"]["positive_probability"]) else -1.0

        # One 4x field evaluation followed by a level-set quantile is equivalent
        # to scaling an implicit body, while avoiding 16 repeated million-point
        # evaluations per sample.  The threshold is selected on the fine grid,
        # then occupancy is measured only after volume averaging.
        if not deep_mode:
            scale = float(np.clip((target / 0.04) ** (1.0 / 3.0), 0.45, 1.75))
            radii = np.minimum(
                np.array((0.30, 0.48 * aspect_y, 0.48 * aspect_x)) * scale,
                np.array((0.72, 1.55, 1.55)),
            )
        field, extra = _shape_field(subtype, *coords, radii, rng)
        level = float(np.quantile(field, 1.0 - target))
        fine_mask = field >= level
        fraction = _downsample(fine_mask, s)
        # Clear tiny coarse fragments, retaining antialiasing on the accepted support.
        support = fraction >= (threshold / contrast)
        support = _remove_tiny_components(support, int(quality["remove_isolated_components_smaller_than_cells"]))
        fraction = np.where(support, fraction, 0.0).astype(np.float32)
        density = _density_from_fraction(fraction, subtype, contrast, sign, coords)
        stats = geometry_metrics(density, threshold, config)
        if not (support_range[0] <= stats["support_fraction"] <= support_range[1]):
            continue
        if stats["support_cells"] < int(quality["minimum_support_cells"]):
            continue
        if stats["minimum_component_horizontal_span_cells"] < int(
            quality["minimum_horizontal_feature_cells"]
        ):
            continue
        if stats["minimum_lateral_margin_cells"] < int(quality["lateral_margin_cells"]):
            continue
        if stats["minimum_shallow_margin_cells"] < int(quality["shallow_margin_cells"]):
            continue
        support_centroid_depth_km: float | None = None
        if deep_mode:
            # Storage is deep-to-shallow; physical depth reverses the geometry z coordinate.
            deep_mask = np.abs(density) >= threshold
            z_index = np.nonzero(deep_mask)[0]
            support_centroid_depth_km = float(
                (density.shape[0] - float(z_index.mean()) - 0.5) * cell_z_km
            )
            if support_centroid_depth_km < deep_min_km:
                continue
        record = {
            "mechanism": mechanism,
            "subtype": subtype,
            "target_support_fraction": target,
            "contrast_gcc": contrast,
            "density_sign": int(sign),
            "center_zyx_km": [float(x) for x in center],
            "strike_deg": float(np.rad2deg(strike)),
            "dip_deg": float(np.rad2deg(dip)),
            "radii_zyx_km": [float(x) for x in radii],
            "supersampling": s,
            "generation_attempt": attempt + 1,
            **extra,
            **stats,
        }
        if deep_mode:
            record["deep_targeting"] = {
                "target_min_centroid_depth_km": deep_min_km,
                "target_max_centroid_depth_km": deep_max_km,
                "support_centroid_depth_km_v025": support_centroid_depth_km,
            }
        return GeneratedModel(density=density, volume_fraction=fraction, record=record)
    raise RuntimeError(f"failed to generate {mechanism} in support stratum {support_range}")
