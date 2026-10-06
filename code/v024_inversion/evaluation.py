"""Full validation/final evaluation; training selection stays in trainer.py."""
from __future__ import annotations

from collections import Counter
import time
from typing import Any

import numpy as np
import torch

from .metrics import metrics_per_sample
from .statistics import ENDPOINT_METRICS, GEOMETRY_METRICS, METADATA_COLUMNS, summarize_metric_rows, track_of


def pair_difference_metrics(
    prediction_a: np.ndarray, prediction_b: np.ndarray,
    truth_a: np.ndarray, truth_b: np.ndarray,
) -> dict[str, float]:
    prediction = np.asarray(prediction_a, dtype=np.float64).ravel() - np.asarray(
        prediction_b, dtype=np.float64
    ).ravel()
    truth = np.asarray(truth_a, dtype=np.float64).ravel() - np.asarray(
        truth_b, dtype=np.float64
    ).ravel()
    if prediction.shape != truth.shape or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("pair predictions and truth must be finite, matching complete grids")
    truth_norm = float(np.linalg.norm(truth))
    prediction_norm = float(np.linalg.norm(prediction))
    if truth_norm <= 1.0e-12:
        raise ValueError("a legal ambiguity pair requires distinct density endpoints")
    cosine = float(np.dot(prediction, truth) / max(prediction_norm * truth_norm, 1.0e-30))
    return {
        "pdacc_cos": float(np.clip(cosine, -1.0, 1.0)),
        "pdacc_direction_correct": float(cosine > 0.0),
        "difference_magnitude_ratio": prediction_norm / truth_norm,
        "difference_nrmse": float(np.linalg.norm(prediction - truth)) / truth_norm,
        "predicted_difference_l2": prediction_norm,
        "truth_difference_l2": truth_norm,
    }


def _item(batch: dict[str, Any], name: str, index: int, fallback: Any = "") -> Any:
    if name not in batch:
        return fallback
    value = batch[name][index]
    return value.detach().cpu().item() if torch.is_tensor(value) else value


def _depth_group(depth: float) -> str:
    return "0_333" if depth < 1000.0 / 3.0 else ("333_667" if depth < 2000.0 / 3.0 else "667_1000")


@torch.inference_mode()
def evaluate_with_pairs(
    model: torch.nn.Module, operator: torch.nn.Module, loader: Any,
    device: torch.device, amp_dtype: torch.dtype, input_mode: str, physics_mode: str,
    allow_partial: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if input_mode != getattr(model, 'input_mode', None):
        raise ValueError(
            f"Evaluation input_mode={input_mode!r} disagrees with "
            f"model.input_mode={getattr(model, 'input_mode', None)!r}")
    model.eval()
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    finished: set[str] = set()
    diagnostic_categories: dict[str, Counter] = {}
    forward_seconds = 0.0
    started = time.perf_counter()
    for cpu_batch in loader:
        batch = {name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                 for name, value in cpu_batch.items()}
        prepared = operator.prepare_inputs(batch["raw6"], batch["receiver_mask"], input_mode)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda" and amp_dtype != torch.float32):
            output = model(prepared, operator)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - forward_started
        metrics = metrics_per_sample(output, batch, operator, physics_mode)
        for layer_name in ("sab1", "sab2", "sab3"):
            layer = getattr(model, layer_name, None)
            for name, value in getattr(layer, "last_diagnostics", {}).items():
                if not torch.is_tensor(value) or value.shape != (len(batch["sample_id"]),):
                    raise ValueError(f"{layer_name}.{name} must contain one diagnostic per endpoint")
                column = f"{layer_name}.{name}"
                if name == "alpha_top_channel":
                    diagnostic_categories.setdefault(column, Counter()).update(
                        str(int(item)) for item in value.detach().cpu().tolist()
                    )
                else:
                    metrics[column] = value.float()
        values = {name: tensor.detach().float().cpu().numpy() for name, tensor in metrics.items()}
        for name, value in values.items():
            if np.isinf(value).any() or (name not in GEOMETRY_METRICS and np.isnan(value).any()):
                raise FloatingPointError(f"nonfinite held-out metric: {name}")
        prediction = output["density"][:, 0].detach().float().cpu().numpy()
        truth = cpu_batch["density"].float().numpy()
        for index, sample_id in enumerate(cpu_batch["sample_id"]):
            row = {name: _item(cpu_batch, name, index) for name in METADATA_COLUMNS
                   if name in cpu_batch}
            row["sample_id"] = str(sample_id)
            row["track"] = str(row.get("track", row.get("generation_track", "")))
            track = track_of(row)
            row["generation_track"] = track
            row["noise_mode"] = str(_item(cpu_batch, "noise_mode", index))
            row.update({name: float(value[index]) for name, value in values.items()})
            row["depth_group"] = _depth_group(row["physical_depth_m"])
            rows.append(row)
            group = str(row.get("ambiguity_group_id", ""))
            if not group:
                if track != "ordinary":
                    raise ValueError("paired track is missing its ambiguity group ID")
                continue
            role = str(row.get("ambiguity_pair_role", ""))
            if track == "ordinary" or role not in {"A", "B"} or group in finished:
                raise ValueError("invalid or repeated ambiguity endpoint")
            slot = pending.setdefault(group, {})
            if role in slot:
                raise ValueError(f"duplicate role {role} in ambiguity group {group}")
            # Copies release batch-sized backing storage while waiting for the mate.
            slot[role] = (prediction[index].copy(), truth[index].copy(), row)
            if set(slot) == {"A", "B"}:
                pred_a, true_a, meta_a = slot["A"]
                pred_b, true_b, meta_b = slot["B"]
                pair = {name: meta_a[name] for name in (
                    "ambiguity_group_id", "generation_track", "track", "response_group",
                    "spin_dominance", "dprime_bin", "tzz_global_dprime", "spin1_global_dprime",
                    "spin2_global_dprime",
                ) if name in meta_a}
                if any(meta_b.get(name) != value for name, value in pair.items()):
                    raise ValueError(f"pair metadata disagree between endpoints in {group}")
                pair["sample_id_A"], pair["sample_id_B"] = meta_a["sample_id"], meta_b["sample_id"]
                pair["physical_depth_m"] = (meta_a["physical_depth_m"] + meta_b["physical_depth_m"]) / 2.0
                pair["depth_group"] = _depth_group(pair["physical_depth_m"])
                pair.update(pair_difference_metrics(pred_a, pred_b, true_a, true_b))
                pair_rows.append(pair)
                del pending[group]
                finished.add(group)
    identifiers = [row["sample_id"] for row in rows]
    if not rows or len(set(identifiers)) != len(identifiers):
        raise ValueError("evaluation requires distinct, nonempty endpoint IDs")
    if pending and not allow_partial:
        raise ValueError(f"evaluation is missing partners for {len(pending)} ambiguity groups")
    names = [name for name in rows[0] if name in values]
    aggregate = summarize_metric_rows(rows, names)
    if not all(name in aggregate for name in ENDPOINT_METRICS):
        raise ValueError("required endpoint metrics are missing")
    summary = {
        "aggregate": aggregate, "sample_count": len(rows), "pair_count": len(pair_rows),
        "incomplete_pair_count": len(pending), "elapsed_seconds": time.perf_counter() - started,
        "forward_seconds": forward_seconds, "forward_seconds_per_endpoint": forward_seconds / len(rows),
        "categorical_diagnostics": {name: dict(counter) for name, counter in diagnostic_categories.items()},
    }
    return summary, rows, sorted(pair_rows, key=lambda row: row["ambiguity_group_id"])
