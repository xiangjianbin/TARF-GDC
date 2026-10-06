#!/usr/bin/env python3
"""Run the frozen Full model on the Vinton six-component field grid.

This evaluator deliberately has no density ground truth claim: it records the
forward response fit and the signed 3-D density-difference volume for later
figures.  The same script is used for all three formal Full checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label


COMPONENTS = ("Txx", "Txy", "Txz", "Tyy", "Tyz", "Tzz")
RAW_SIZE = 41
MODEL_SIZE = 33
WIDTH_M = 4000.0
DEPTH_M = 1000.0


def load_field_data(directory: Path) -> np.ndarray:
    grids = []
    for component in COMPONENTS:
        table = np.loadtxt(directory / f"{component}.txt", delimiter=",")
        if table.shape != (RAW_SIZE * RAW_SIZE, 5):
            raise ValueError(f"unexpected {component} table shape {table.shape}")
        grid = np.full((RAW_SIZE, RAW_SIZE), np.nan, dtype=np.float64)
        x_index = np.rint(table[:, 0] / 100.0).astype(int)
        y_index = np.rint(table[:, 1] / 100.0).astype(int)
        if np.any((x_index < 0) | (x_index >= RAW_SIZE) | (y_index < 0) | (y_index >= RAW_SIZE)):
            raise ValueError(f"out-of-range coordinates for {component}")
        grid[y_index, x_index] = table[:, 3] * 1.0e9
        if np.isnan(grid).any():
            raise ValueError(f"incomplete Vinton grid for {component}")
        grids.append(grid)
    return np.stack(grids)


def resample_to_model_grid(grids: np.ndarray) -> np.ndarray:
    source = np.linspace(0.0, WIDTH_M, RAW_SIZE)
    target = np.linspace(0.0, WIDTH_M, MODEL_SIZE)
    target_y, target_x = np.meshgrid(target, target, indexing="ij")
    points = np.stack((target_y.ravel(), target_x.ravel()), axis=-1)
    return np.stack([
        RegularGridInterpolator((source, source), grid)(points).reshape(MODEL_SIZE, MODEL_SIZE)
        for grid in grids
    ])


def to_z_up(grids: np.ndarray) -> np.ndarray:
    converted = grids.copy()
    converted[2] *= -1.0
    converted[4] *= -1.0
    return converted


def fit_metrics(observed: np.ndarray, predicted: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for index, component in enumerate(COMPONENTS):
        actual = observed[index].ravel()
        estimate = predicted[index].ravel()
        residual = actual - estimate
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        observed_rms = float(np.sqrt(np.mean(actual ** 2)))
        residual_rms = float(np.sqrt(np.mean(residual ** 2)))
        rows.append({
            "component": component,
            "r2": 1.0 - float(np.sum(residual ** 2)) / max(denominator, 1.0e-12),
            "relative_residual": residual_rms / max(observed_rms, 1.0e-12),
            "rms_ratio": float(np.sqrt(np.mean(estimate ** 2))) / max(observed_rms, 1.0e-12),
            "observed_rms_E": observed_rms,
            "predicted_rms_E": float(np.sqrt(np.mean(estimate ** 2))),
            "residual_rms_E": residual_rms,
            "residual_sae_E": float(np.abs(residual).sum()),
            "residual_sse_E2": float(np.square(residual).sum()),
        })
    return rows


def structure_at_threshold(density: np.ndarray, threshold: float) -> dict[str, float | int]:
    components, count = label(density >= threshold, structure=np.ones((3, 3, 3), dtype=int))
    peak = np.unravel_index(np.argmax(density), density.shape)
    component_index = int(components[peak])
    if count == 0 or component_index == 0:
        return {"threshold_gcc": threshold, "voxel_count": 0}
    selected = components == component_index
    z_index, y_index, x_index = np.where(selected)
    dz = DEPTH_M / density.shape[0]
    horizontal = WIDTH_M / density.shape[1]
    return {
        "threshold_gcc": threshold,
        "voxel_count": int(selected.sum()),
        "volume_km3": float(selected.sum() * dz * horizontal * horizontal / 1.0e9),
        "top_depth_m": float((density.shape[0] - z_index.max() - 0.5) * dz),
        "bottom_depth_m": float((density.shape[0] - z_index.min() - 0.5) * dz),
        "x_extent_m": float((x_index.max() - x_index.min() + 1) * horizontal),
        "y_extent_m": float((y_index.max() - y_index.min() + 1) * horizontal),
    }


