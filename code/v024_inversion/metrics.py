from __future__ import annotations

from typing import Any

import torch

from .physics import GravityOperator


PREDICTION_SUPPORT_THRESHOLD_GCC = 0.005
HISTORICAL_DENSITY_BOUND_GCC = 0.85
NULLABLE_GEOMETRY_METRICS = (
    "density_centroid_error_m",
    "density_top_depth_error_m",
)


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1.0e-6)


@torch.no_grad()
def metrics_per_sample(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    operator: GravityOperator,
    physics_mode: str,
) -> dict[str, torch.Tensor]:
    prediction = output["density"].float()[:, 0]
    truth = batch["density"].float()
    # Thresholding is for geometry only. Density losses and the forward model
    # continue to consume the complete, unclipped linear density prediction.
    predicted_support = prediction.abs() >= PREDICTION_SUPPORT_THRESHOLD_GCC
    truth_support = batch["support"].bool()
    dimensions = (1, 2, 3)

    true_positive = (predicted_support & truth_support).sum(dim=dimensions).float()
    false_positive = (predicted_support & ~truth_support).sum(dim=dimensions).float()
    false_negative = (~predicted_support & truth_support).sum(dim=dimensions).float()
    true_negative = (~predicted_support & ~truth_support).sum(dim=dimensions).float()
    predicted_count = predicted_support.sum(dim=dimensions).float()
    truth_count = truth_support.sum(dim=dimensions).float()
    empty_prediction = predicted_count == 0
    empty_truth = truth_count == 0
    geometry_valid = ~empty_prediction & ~empty_truth
    support_dice = _ratio(2.0 * true_positive + 1.0, predicted_count + truth_count + 1.0)
    precision = _ratio(true_positive + 1.0, predicted_count + 1.0)
    recall = _ratio(true_positive + 1.0, truth_count + 1.0)
    false_positive_rate = _ratio(false_positive, true_negative + false_positive)

    error = prediction - truth
    # Per-model totals over the complete grid: no voxel-count or amplitude
    # normalization. aggregate_metrics subsequently averages model totals.
    density_sae = error.abs().sum(dim=dimensions)
    density_sse = error.square().sum(dim=dimensions)
    density_mae = error.abs().mean(dim=dimensions)
    density_mse = error.square().mean(dim=dimensions)
    density_rmse = density_mse.sqrt()
    foreground_error = torch.where(truth_support, error, torch.zeros_like(error))
    foreground_truth = torch.where(truth_support, truth, torch.zeros_like(truth))
    foreground_nrmse = _ratio(
        _ratio(foreground_error.square().sum(dim=dimensions), truth_count).sqrt(),
        _ratio(foreground_truth.square().sum(dim=dimensions), truth_count).sqrt(),
    )

    sign_mask = truth_support & truth.abs().ge(0.05)
    sign_count = sign_mask.sum(dim=dimensions).float()
    sign_correct = (((prediction >= 0.0) == (truth >= 0.0)) & sign_mask).sum(
        dim=dimensions
    ).float()
    sign_accuracy = _ratio(sign_correct, sign_count)

    predicted_class = torch.zeros_like(prediction, dtype=torch.long)
    predicted_class[predicted_support & (prediction >= 0.0)] = 1
    predicted_class[predicted_support & (prediction < 0.0)] = 2
    truth_class = torch.zeros_like(truth, dtype=torch.long)
    truth_class[truth_support & (truth >= 0.0)] = 1
    truth_class[truth_support & (truth < 0.0)] = 2
    class_dice = []
    absent_sign_false_positive = torch.zeros_like(truth_support)
    for class_index in range(3):
        predicted = predicted_class == class_index
        expected = truth_class == class_index
        overlap = (predicted & expected).sum(dim=dimensions).float()
        class_dice.append(
            _ratio(
                2.0 * overlap + 1.0,
                predicted.sum(dim=dimensions) + expected.sum(dim=dimensions) + 1.0,
            )
        )
        if class_index in (1, 2):
            missing_class = ~expected.any(dim=dimensions)
            absent_sign_false_positive |= predicted & missing_class[:, None, None, None]
    macro_unit_dice = torch.stack(class_dice, dim=1).mean(dim=1)

    depth_centers = (
        truth.shape[1]
        - torch.arange(truth.shape[1], device=truth.device).float()
        - 0.5
    ) * 62.5
    depth_volume = depth_centers.view(1, -1, 1, 1).expand_as(truth)
    missing_depth = torch.full_like(depth_volume, 1000.0)
    predicted_top = torch.where(predicted_support, depth_volume, missing_depth).amin(
        dim=dimensions
    )
    truth_top = torch.where(truth_support, depth_volume, missing_depth).amin(dim=dimensions)
    top_depth_error_m = torch.where(
        geometry_valid, (predicted_top - truth_top).abs(), torch.nan
    )

    z = depth_centers
    y = (torch.arange(truth.shape[2], device=truth.device).float() + 0.5) * 125.0
    x = (torch.arange(truth.shape[3], device=truth.device).float() + 0.5) * 125.0
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    def centroid(weights: torch.Tensor) -> torch.Tensor:
        denominator = weights.sum(dim=dimensions).clamp_min(1.0e-6)
        return torch.stack(
            [
                (weights * zz).sum(dim=dimensions) / denominator,
                (weights * yy).sum(dim=dimensions) / denominator,
                (weights * xx).sum(dim=dimensions) / denominator,
            ],
            dim=1,
        )

    centroid_error_m = torch.where(
        geometry_valid,
        (centroid(predicted_support.float()) - centroid(truth_support.float()))
        .square().sum(dim=1).sqrt(),
        torch.nan,
    )
    volume_relative_error = _ratio((predicted_count - truth_count).abs(), truth_count)
    _, physics_squared = operator.relative_misfit(
        output["density"].float(),
        batch["raw6"].float(),
        batch["receiver_mask"],
        physics_mode,
    )
    predicted_raw6 = operator.density_to_d6(prediction)
    observation_error = torch.where(
        batch["receiver_mask"].bool().unsqueeze(-1),
        predicted_raw6 - batch["raw6"].float(),
        torch.zeros_like(predicted_raw6),
    )

    result = {
        "absent_sign_false_positive_count": absent_sign_false_positive.sum(dim=dimensions).float(),
        "absent_sign_false_positive_density_sae": torch.where(
            absent_sign_false_positive, error.abs(), torch.zeros_like(error)
        ).sum(dim=dimensions),
        "density_centroid_error_m": centroid_error_m,
        "density_mae": density_mae,
        "density_mse": density_mse,
        "density_rmse": density_rmse,
        "density_sae": density_sae,
        "density_sse": density_sse,
        "density_min": prediction.amin(dim=dimensions),
        "density_max": prediction.amax(dim=dimensions),
        "density_out_of_range_rate": (
            prediction.abs() > HISTORICAL_DENSITY_BOUND_GCC
        ).float().mean(dim=dimensions),
        "density_support_false_positive_rate": false_positive_rate,
        "foreground_nrmse": foreground_nrmse,
        "density_nrmse": _ratio(error.square().sum(dim=dimensions).sqrt(), truth.square().sum(dim=dimensions).sqrt()),
        "density_support_macro_unit_dice": macro_unit_dice,
        "observation_sae": observation_error.abs().sum(dim=(1, 2)),
        "observation_sse": observation_error.square().sum(dim=(1, 2)),
        "physics_nrms": physics_squared.sqrt(),
        "density_support_precision": precision,
        "density_support_recall": recall,
        "sign_accuracy": sign_accuracy,
        "density_support_dice": support_dice,
        "truth_support_voxel_count": truth_count,
        "density_support_voxel_count": predicted_count,
        "empty_prediction_rate": empty_prediction.float(),
        "empty_truth_rate": empty_truth.float(),
        "density_geometry_valid_rate": geometry_valid.float(),
        "physical_depth_m": _ratio(
            (truth_support.float() * depth_volume).sum(dim=dimensions), truth_count
        ),
        "truth_top_depth_m": truth_top,
        "truth_bottom_depth_m": torch.where(
            truth_support, depth_volume, torch.zeros_like(depth_volume)
        ).amax(dim=dimensions),
        "density_top_depth_error_m": top_depth_error_m,
        "density_support_volume_relative_error": volume_relative_error,
    }
    for index, component in enumerate(('Txx', 'Txy', 'Txz', 'Tyy', 'Tyz', 'Tzz')):
        result[f'observation_sae_{component}'] = observation_error[..., index].abs().sum(dim=1)
        result[f'observation_nrms_{component}'] = _ratio(
            observation_error[..., index].square().sum(dim=1).sqrt(),
            (batch['raw6'][..., index].float().square() * batch['receiver_mask']).sum(dim=1).sqrt())
    depth_sae = error.abs().sum(dim=(2, 3))
    result.update({f"density_sae_z{index:02d}": depth_sae[:, index]
                   for index in range(depth_sae.shape[1])})
    depth_support = truth_support.sum(dim=(2, 3)).float()
    result.update({f"truth_support_voxel_count_z{index:02d}": depth_support[:, index]
                   for index in range(depth_support.shape[1])})
    if "initial_density" in output:
        initial_error = output["initial_density"].float()[:,0] - batch["density"].float()
        result["initial_density_sae"] = initial_error.abs().sum(dim=(1,2,3))
        result["initial_density_mse"] = initial_error.square().mean(dim=(1,2,3))
        result["refinement_sae_reduction"] = result["initial_density_sae"] - result["density_sae"]
        initial_support = output['initial_density'].float()[:,0].abs() >= PREDICTION_SUPPORT_THRESHOLD_GCC
        initial_tp = (initial_support & truth_support).sum(dim=dimensions).float()
        result['initial_density_support_dice'] = _ratio(
            2*initial_tp+1, initial_support.sum(dim=dimensions)+truth_count+1)
        result['density_background_sae'] = (error.abs() * ~truth_support).sum(dim=dimensions)
        result['initial_density_background_sae'] = (initial_error.abs() * ~truth_support).sum(dim=dimensions)
        result['density_foreground_sae'] = (error.abs() * truth_support).sum(dim=dimensions)
        result['initial_density_foreground_sae'] = (initial_error.abs() * truth_support).sum(dim=dimensions)
        initial_output = dict(output)
        initial_output["density"] = output["initial_density"]
        _, initial_squared = operator.relative_misfit(
            initial_output["density"].float(),
            batch["raw6"].float(),
            batch["receiver_mask"],
            physics_mode,
        )
        result["initial_physics_nrms"] = initial_squared.sqrt()
        result["dc_density_update_rms"] = (
            output["density"].float() - output["initial_density"].float()
        ).square().mean(dim=(1, 2, 3, 4)).sqrt()
    return result


def aggregate_metrics(batches: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    if not batches:
        return {}
    names = sorted(set.intersection(*(set(batch) for batch in batches)))
    result = {}
    for name in names:
        values = torch.cat([batch[name].detach().cpu() for batch in batches])
        if name in NULLABLE_GEOMETRY_METRICS:
            # Empty masks have no location. Always expose their denominator;
            # failures still contribute to Dice/recall and empty_prediction_rate.
            result[name] = float(values.nanmean())
            result[f"{name}_valid_count"] = int((~values.isnan()).sum())
            result[f"{name}_total_count"] = values.numel()
        elif name == "density_min":
            result[name] = float(values.min())
        elif name == "density_max":
            result[name] = float(values.max())
        else:
            result[name] = float(values.mean())
    return result


def objective_score(metrics: dict[str, float]) -> float:
    return (
        metrics.get("density_support_dice", 0.0)
        + 0.25 * metrics.get("density_support_macro_unit_dice", 0.0)
        - 0.25 * metrics.get("density_support_false_positive_rate", 0.0)
    )


def v032_objective_score(metrics: dict[str, float]) -> float:
    """Validation-only score aligned with the registered Stage-2 targets."""
    dice = metrics.get("density_support_dice", 0.0)
    physics = metrics.get("physics_nrms", 1.0)
    centroid = metrics.get("density_centroid_error_m", 1000.0)
    top = metrics.get("density_top_depth_error_m", 1000.0)
    # Dice is dimensionless; geometry errors are normalized to the physical
    # 1-km depth extent.  SAE remains a safety report, not a hidden selector.
    return dice - 0.25 * physics - 0.10 * centroid / 1000.0 - 0.10 * top / 1000.0
