from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import resolve_project_path, sha256_file


COMPONENTS = ("Txx", "Txy", "Txz", "Tyy", "Tyz", "Tzz")


class GravityTensorOperator:
    """Frozen six-component linear forward operator with batched GPU execution."""

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu") -> None:
        op = config["operator"]
        grid = config["grid"]
        density_shape = tuple(int(value) for value in grid["density_shape_zyx"])
        receiver_shape = tuple(int(value) for value in grid["receiver_shape_yx"])
        component_order = tuple(str(value) for value in op["component_order"])
        if component_order != COMPONENTS:
            raise ValueError(f"unsupported component order {component_order}; expected {COMPONENTS}")
        path = resolve_project_path(op["g6_path"])
        actual = sha256_file(path)
        if actual != op["g6_sha256"]:
            raise ValueError(f"G6 hash mismatch: expected {op['g6_sha256']}, got {actual}")
        raw = np.load(path, mmap_mode="r")
        expected_shape = (
            int(np.prod(receiver_shape)) * len(component_order),
            int(np.prod(density_shape)),
        )
        if tuple(raw.shape) != expected_shape:
            raise ValueError(f"unexpected G6 shape {raw.shape}; expected {expected_shape}")

        manifest_path = resolve_project_path(op["manifest_path"])
        if op.get("manifest_sha256"):
            manifest_hash = sha256_file(manifest_path)
            if manifest_hash != op["manifest_sha256"]:
                raise ValueError(
                    f"operator manifest hash mismatch: expected {op['manifest_sha256']}, got {manifest_hash}"
                )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest = {
            "density_shape_zyx": list(density_shape),
            "receiver_shape_yx": list(receiver_shape),
            "cell_size_xyz_m": [float(value) for value in grid["cell_size_xyz_m"]],
            "origin_xyz_m": [float(value) for value in grid["origin_xyz_m"]],
            "physical_extent_xyz_m": [float(value) for value in grid["physical_extent_xyz_m"]],
            "receiver_height_m": float(grid["receiver_height_m"]),
            "g6_shape": list(expected_shape),
            "g6_sha256": actual,
        }
        mismatches = {
            key: {"config": expected, "manifest": manifest.get(key)}
            for key, expected in expected_manifest.items()
            if manifest.get(key) != expected
        }
        normalized_manifest_components = tuple(
            "T" + str(value)[1:] if str(value).startswith("g") else str(value)
            for value in manifest.get("components", [])
        )
        if normalized_manifest_components != component_order:
            mismatches["components"] = {
                "config": list(component_order),
                "manifest": manifest.get("components"),
            }
        if mismatches:
            raise ValueError(f"operator manifest/config mismatch: {mismatches}")

        self.device = torch.device(device)
        self.g6 = torch.from_numpy(np.asarray(raw, dtype=np.float32).copy()).to(self.device)
        self.density_shape = density_shape
        self.receiver_shape = receiver_shape
        self.receiver_count = int(np.prod(receiver_shape))
        self.component_order = component_order
        # File layout is receiver-major: [n_receiver, 6, n_density].
        self.g6_receiver_component = self.g6.reshape(
            self.receiver_count, len(component_order), int(np.prod(density_shape))
        )
        self.gzz = self.g6_receiver_component[:, 5, :]
        self.operator_path = path
        self.operator_sha256 = actual
        self.manifest_path = manifest_path
        self.manifest = manifest

    @torch.inference_mode()
    def forward(self, density: np.ndarray, batch_size: int = 128) -> np.ndarray:
        if density.ndim == 3:
            density = density[None]
        outputs: list[np.ndarray] = []
        for start in range(0, len(density), int(batch_size)):
            batch = torch.from_numpy(
                np.ascontiguousarray(density[start : start + batch_size], dtype=np.float32)
            ).to(self.device)
            response = batch.reshape(len(batch), -1) @ self.g6.T
            response = response.reshape(
                len(batch), self.receiver_count, len(self.component_order)
            ).permute(0, 2, 1)
            outputs.append(
                response.reshape(len(batch), len(self.component_order), *self.receiver_shape)
                .cpu().numpy()
            )
        return np.concatenate(outputs).astype(np.float32)

    @torch.inference_mode()
    def tzz(self, density: np.ndarray, batch_size: int = 256) -> np.ndarray:
        if density.ndim == 3:
            density = density[None]
        outputs: list[np.ndarray] = []
        for start in range(0, len(density), int(batch_size)):
            batch = torch.from_numpy(
                np.ascontiguousarray(density[start : start + batch_size], dtype=np.float32)
            ).to(self.device)
            result = batch.reshape(len(batch), -1) @ self.gzz.T
            outputs.append(result.reshape(len(batch), *self.receiver_shape).cpu().numpy())
        return np.concatenate(outputs).astype(np.float32)


GRAVITATIONAL_CONSTANT = 6.6743e-11


def independent_prism_tensor(
    receiver_xyz: np.ndarray,
    bounds_xyz: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    density_gcc: float,
) -> np.ndarray:
    """Independent Nagy-style rectangular-prism tensor in Eotvos.

    This implementation deliberately does not call SimPEG or Geoana and is
    used only as an external physics check on the frozen G6 artifact.
    """
    receiver = np.asarray(receiver_xyz, dtype=np.float64)
    output = np.zeros((len(receiver), 6), dtype=np.float64)
    for ix, x_corner in enumerate(bounds_xyz[0]):
        for iy, y_corner in enumerate(bounds_xyz[1]):
            for iz, z_corner in enumerate(bounds_xyz[2]):
                x = receiver[:, 0] - float(x_corner)
                y = receiver[:, 1] - float(y_corner)
                z = receiver[:, 2] - float(z_corner)
                radius = np.sqrt(x * x + y * y + z * z)
                sign = (-1.0) ** (ix + iy + iz)
                output += sign * np.column_stack((
                    -np.arctan2(y * z, x * radius),
                    np.log(z + radius),
                    np.log(y + radius),
                    -np.arctan2(x * z, y * radius),
                    np.log(x + radius),
                    -np.arctan2(x * y, z * radius),
                ))
    return output * GRAVITATIONAL_CONSTANT * (float(density_gcc) * 1000.0) * 1e9


def independent_prism_audit(
    operator: GravityTensorOperator,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate G6 against three independently evaluated uniform prisms."""
    grid = config["grid"]
    nz, ny, nx = (int(value) for value in grid["density_shape_zyx"])
    dx, dy, dz = (float(value) for value in grid["cell_size_xyz_m"])
    ox, oy, oz = (float(value) for value in grid["origin_xyz_m"])
    extent_x, extent_y, extent_z = (
        float(value) for value in grid["physical_extent_xyz_m"]
    )
    ry, rx = (int(value) for value in grid["receiver_shape_yx"])
    x_receivers = np.linspace(ox, ox + extent_x, rx)
    y_receivers = np.linspace(oy, oy + extent_y, ry)
    yy, xx = np.meshgrid(y_receivers, x_receivers, indexing="ij")
    receiver_xyz = np.column_stack((
        xx.reshape(-1),
        yy.reshape(-1),
        np.full(xx.size, oz + extent_z + float(grid["receiver_height_m"])),
    ))
    definitions = (
        (slice(4, 10), slice(12, 20), slice(10, 18), 0.50),
        (slice(1, 5), slice(3, 11), slice(20, 27), -0.35),
        (slice(10, 15), slice(21, 29), slice(4, 12), 0.65),
    )
    rows = []
    for index, (zs, ys, xs, contrast) in enumerate(definitions):
        density = np.zeros((nz, ny, nx), dtype=np.float32)
        density[zs, ys, xs] = float(contrast)
        predicted = operator.forward(density, batch_size=1)[0].reshape(6, -1).T
        bounds = (
            (ox + xs.start * dx, ox + xs.stop * dx),
            (oy + ys.start * dy, oy + ys.stop * dy),
            (oz + zs.start * dz, oz + zs.stop * dz),
        )
        analytic = independent_prism_tensor(receiver_xyz, bounds, contrast)
        component_relative_l2 = np.linalg.norm(predicted - analytic, axis=0) / np.maximum(
            np.linalg.norm(analytic, axis=0), 1e-30
        )
        rows.append({
            "prism_index": index,
            "bounds_xyz_m": [list(value) for value in bounds],
            "density_gcc": float(contrast),
            "component_relative_l2": [float(value) for value in component_relative_l2],
            "maximum_component_relative_l2": float(component_relative_l2.max()),
        })
    return {
        "method": "independent_nagy_rectangular_prism_corner_sum",
        "prism_count": len(rows),
        "maximum_component_relative_l2": max(
            row["maximum_component_relative_l2"] for row in rows
        ),
        "rows": rows,
    }


def trace_relative_rms(response_d6: np.ndarray) -> np.ndarray:
    trace = response_d6[:, 0] + response_d6[:, 3] + response_d6[:, 5]
    scale = np.sqrt(np.mean(response_d6 * response_d6, axis=(1, 2, 3)))
    return np.sqrt(np.mean(trace * trace, axis=(1, 2))) / np.maximum(scale, 1e-12)


def response_sha256(response: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(response, dtype="<f4", order="C").tobytes()).hexdigest()


def role_arrays(response: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Tzz, spin-1 (Txz,Tyz), and spin-2 ((Txx-Tyy)/2,Txy)."""
    tzz = response[5].reshape(-1)
    spin1 = response[[2, 4]].reshape(-1)
    spin2 = np.stack(((response[0] - response[3]) / 2.0, response[1])).reshape(-1)
    return tzz, spin1, spin2
