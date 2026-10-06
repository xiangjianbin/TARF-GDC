"""Independent projected Newton-CG warm starts for convex box quadratics."""
import torch


@torch.no_grad()
def warm_start(batch, hessian, variable, beta, weights, smoothness, tolerance, enabled):
    diagonal = batch.jacobi(beta, weights, smoothness).clamp_min(1e-20)
    current = variable.clone()
    used = torch.zeros(current.shape[0], dtype=torch.long, device=current.device)
    reference = batch.rhs.norm(dim=1).clamp_min(1e-15)
    ones = torch.ones_like(reference)
    gradient = hessian(current) - batch.rhs
    for _ in range(30):
        active = enabled & (batch.relative_kkt(current, gradient, ones) > tolerance)
        if not bool(active.any()):
            break
        slack = 4 * torch.finfo(current.dtype).eps * batch.bound
        blocked = ((current <= -batch.bound + slack) & (gradient > 0)) | (
            (current >= batch.bound - slack) & (gradient < 0))
        free = (~blocked) & active[:, None]
        residual = torch.where(free, -gradient, torch.zeros_like(gradient))
        target = 0.05 * residual.norm(dim=1)
        scaled = residual / diagonal
        direction = scaled.clone()
        step = torch.zeros_like(current)
        gamma = (residual * scaled).sum(dim=1)
        cg_active = active.clone()
        for _ in range(200):
            product = torch.where(free, hessian(direction), torch.zeros_like(direction))
            denominator = (direction * product).sum(dim=1).clamp_min(1e-30)
            length = torch.where(cg_active, gamma / denominator, torch.zeros_like(gamma))
            step += length[:, None] * direction
            residual -= length[:, None] * product
            used += cg_active
            scaled = residual / diagonal
            new_gamma = (residual * scaled).sum(dim=1)
            cg_active &= residual.norm(dim=1) > target
            ratio = torch.where(cg_active, new_gamma / gamma.clamp_min(1e-30), torch.zeros_like(gamma))
            direction = torch.where(cg_active[:, None] & free,
                                     scaled + ratio[:, None] * direction, torch.zeros_like(direction))
            gamma = new_gamma
            if not bool(cg_active.any()):
                break
        projected_step = batch.project(current + step) - current
        descent = (gradient * projected_step).sum(dim=1) < 0
        step = torch.where(descent[:, None], step, -gradient / diagonal)
        objective = (0.5 * current * (gradient - batch.rhs)).sum(dim=1)
        length = torch.ones_like(reference)
        for _ in range(40):
            candidate = batch.project(current + length[:, None] * step)
            candidate = torch.where(active[:, None], candidate, current)
            candidate_gradient = hessian(candidate) - batch.rhs
            candidate_objective = (0.5 * candidate * (candidate_gradient - batch.rhs)).sum(dim=1)
            decrease = 1e-4 * (gradient * (candidate - current)).sum(dim=1)
            roundoff = 16 * torch.finfo(current.dtype).eps * (
                objective.abs() + candidate_objective.abs() + 1)
            bad = active & (candidate_objective > objective + decrease + roundoff)
            if not bool(bad.any()):
                break
            length = torch.where(bad, length * 0.5, length)
        else:
            return current, used
        current, gradient = candidate, candidate_gradient
    return current, used
