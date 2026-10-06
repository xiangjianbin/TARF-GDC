"""Skip inactive independent systems without changing the IRLS updates."""
import time

import torch


class NonzeroRows:
    def __init__(self, operator, n_model, n_data):
        self.operator, self.n_model, self.n_data = operator, n_model, n_data

    def forward(self, density):
        active = density.ne(0).any(1)
        if bool(active.all()):
            return self.operator.forward(density)
        output = density.new_zeros(len(density), self.n_data)
        if bool(active.any()):
            output[active] = self.operator.forward(density[active])
        return output

    def adjoint(self, values):
        active = values.ne(0).any(1)
        if bool(active.all()):
            return self.operator.adjoint(values)
        output = values.new_zeros(len(values), self.n_model)
        if bool(active.any()):
            output[active] = self.operator.adjoint(values[active])
        return output


@torch.no_grad()
def solve(context, beta, max_iterations=6000, tolerance=1e-5, max_outer=80,
          epsilon=.01, outer_tolerance=.001, smoothness=1.0):
    started = time.perf_counter()
    p = context.problem
    density, info = context.quadratic(beta, max_iterations=max_iterations,
                                      tolerance=tolerance, smoothness=smoothness)
    total = info["iterations"].clone()
    outer_count = torch.zeros_like(total)
    active = torch.ones(len(density), dtype=torch.bool, device=density.device)
    change = torch.zeros_like(info["relative_kkt"])
    kkt = info["relative_kkt"].clone()
    for _ in range(max_outer):
        indices = torch.where(active)[0]
        sub = p.batch(context.observed[indices])
        old = density[indices]
        candidate, inner = sub.quadratic(
            beta, sub.weights(old, epsilon), initial=old, max_iterations=max_iterations,
            tolerance=tolerance * .1, smoothness=smoothness,
        )
        density[indices] = candidate
        total[indices] += inner["iterations"]
        outer_count[indices] += 1
        change[indices] = (candidate - old).norm(dim=1) / old.norm(dim=1).clamp_min(1e-10)
        variable = candidate * p.sensitivity
        gradient = sub.hessian(variable, beta, sub.weights(candidate, epsilon), smoothness) - sub.rhs
        kkt[indices] = sub.relative_kkt(variable, gradient, inner["lipschitz"])
        active &= (change > outer_tolerance) | (kkt > tolerance)
        if not bool(active.any()):
            break
    if density.is_cuda:
        torch.cuda.synchronize()
    return density.reshape(-1, *p.shape), {
        "total_iterations": total, "outer_iterations": outer_count,
        "relative_kkt": kkt, "relative_change": change, "converged": kkt <= tolerance,
        "seconds": time.perf_counter() - started,
    }
