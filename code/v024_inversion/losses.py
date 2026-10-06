"""Density supervision plus differentiable forward-physics supervision."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .physics import tensor_roles


def _role_physics_loss(prediction, batch, operator, loss_config):
    """Compare predicted response with the explicitly selected 2D label.

    Clean-forward supervision is the default; observed-data consistency is
    supported but must be explicitly configured, with no silent fallback.
    """
    density = prediction[:, 0] if prediction.ndim == 5 and prediction.shape[1] == 1 else prediction
    predicted_roles = tensor_roles(operator.density_to_d6(density).float())
    target_kind = loss_config.get('physics_target')
    if target_kind not in ('clean', 'observed'):
        raise ValueError('physics_target must be clean or observed')
    target_key = 'clean_raw6' if target_kind == 'clean' else 'raw6'
    observed_roles = tensor_roles(batch[target_key].float()).detach()
    if observed_roles.shape != predicted_roles.shape:
        raise ValueError('Forward response and label shapes differ')
    scales = torch.as_tensor(
        loss_config['role_noise_scales'], dtype=predicted_roles.dtype,
        device=predicted_roles.device).reshape(1, 1, 5).clamp_min(1e-8)
    residual = (predicted_roles - observed_roles) / scales
    mask = batch['receiver_mask'].float().reshape(residual.shape[0], residual.shape[1], 1)
    denominator = mask.sum(dim=(1, 2)).clamp_min(1.0) * residual.shape[-1]
    return ((residual.square() * mask).sum(dim=(1, 2)) / denominator).mean()


def compute_loss(output, batch, operator, loss_config, physics_mode):
    if physics_mode != 'roles':
        raise ValueError('The registered physical loss uses role-space observations')
    prediction = output['density'].float()
    target = batch['density'].float().unsqueeze(1)
    if prediction.shape != target.shape:
        raise ValueError('Expected identical [B,1,Z,Y,X] shapes, no broadcasting')
    objective = loss_config.get('objective')
    if objective not in ('mse', 'mse_physics'):
        raise ValueError(f'Unknown loss objective: {objective!r}')
    if objective == 'mse' and loss_config != {'objective': 'mse'}:
        raise ValueError('Unexpected density-only loss fields')
    density_loss = F.mse_loss(prediction, target)
    if objective == 'mse':
        total = density_loss
        physics_loss = prediction.new_zeros(())
    else:
        physics_loss = _role_physics_loss(prediction, batch, operator, loss_config)
        total = (float(loss_config.get('density_weight', 1.0)) * density_loss
                 + float(loss_config['physics_weight']) * physics_loss)
    values = {
        'total': float(total.detach()),
        'density_mse': float(density_loss.detach()),
        'physics_loss': float(physics_loss.detach()),
        'physics_weighted': float((float(loss_config.get('physics_weight', 0.0)) * physics_loss).detach()),
    }
    return total, values
