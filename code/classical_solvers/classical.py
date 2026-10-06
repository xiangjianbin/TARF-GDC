"""Sensitivity-weighted, signed, box-constrained L2 and smoothed-L1 IRLS.

The GPU batches share only matrix reads. Objectives, line searches, convergence
tests, and iterates remain independent for each observation.
"""
from __future__ import annotations

import math
import time

import torch
from projected_newton import warm_start


def differences(model, axis, spacing):
    lower = [slice(None)] * model.ndim
    upper = [slice(None)] * model.ndim
    lower[axis], upper[axis] = slice(None, -1), slice(1, None)
    return (model[tuple(upper)] - model[tuple(lower)]) / spacing


def add_difference_adjoint(out, values, axis, spacing):
    lower = [slice(None)] * out.ndim
    upper = [slice(None)] * out.ndim
    lower[axis], upper[axis] = slice(None, -1), slice(1, None)
    out[tuple(lower)] -= values / spacing
    out[tuple(upper)] += values / spacing


class Problem:
    def __init__(self, matrix, shape=(16, 32, 32), channels=5,
                 spacing=(0.5, 1.0, 1.0), floor=0.001):
        self.matrix = matrix.contiguous()
        self.shape = tuple(shape)
        self.channels = channels
        self.convolution = None
        self.spacing = tuple(spacing)
        self.n_data, self.n_model = matrix.shape
        if self.n_model != math.prod(self.shape) or self.n_data % channels:
            raise ValueError("Incompatible operator, channels, or model shape")
        self.component_diagonal = matrix.reshape(-1, channels, self.n_model).square().sum(dim=0)
        sensitivity = matrix.square().sum(dim=0).sqrt()
        # SimPEG set_weights(sensitivity=wr) places sqrt(wr) in the norm.
        self.sensitivity = (sensitivity / sensitivity.max()).clamp_min(floor).sqrt()
        self.sensitivity2 = self.sensitivity.reshape(1, *shape).square()
        self.face_weights = []
        for axis in range(1, 4):
            lo, hi = [slice(None)] * 4, [slice(None)] * 4
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            self.face_weights.append(
                0.5 * (self.sensitivity2[tuple(lo)] + self.sensitivity2[tuple(hi)])
            )

    def batch(self, observed, max_density=0.85):
        return Batch(self, observed, max_density)

    def forward(self, density):
        return density @ self.matrix.T if self.convolution is None else self.convolution.forward(density)

    def adjoint(self, data):
        return data @ self.matrix if self.convolution is None else self.convolution.adjoint(data)


class Batch:
    def __init__(self, problem, observed, max_density):
        self.problem = problem
        self.observed = observed.to(problem.matrix.dtype).reshape(-1, problem.n_data)
        self.scale = self.observed.reshape(
            -1, problem.n_data // problem.channels, problem.channels
        ).square().mean(dim=1).sqrt().clamp_min(1e-6)
        self.row_weights = self.scale.reciprocal().square().repeat(
            1, problem.n_data // problem.channels
        )
        self.rhs = (problem.adjoint(self.observed * self.row_weights)
                    / problem.n_data / problem.sensitivity)
        self.bound = max_density * problem.sensitivity
        self.data_diagonal = (self.scale.reciprocal().square() @ problem.component_diagonal
                              / problem.n_data / problem.sensitivity.square())

    def weights(self, density=None, epsilon=0.01, density_scale=0.5):
        if density is None:
            return [1.0, 1.0, 1.0, 1.0]
        volume = density.reshape(-1, *self.problem.shape)
        terms = [volume] + [differences(volume, a, h)
                            for a, h in enumerate(self.problem.spacing, 1)]
        return [density_scale / (term.square() + epsilon**2).sqrt() for term in terms]

    def regularization_hessian(self, variable, weights, smoothness=1.0):
        p = self.problem
        density = (variable / p.sensitivity).reshape(-1, *p.shape)
        result = density * p.sensitivity2 * weights[0]
        for axis, (spacing, face_weight, irls_weight) in enumerate(
                zip(p.spacing, p.face_weights, weights[1:]), 1):
            diff = differences(density, axis, spacing)
            add_difference_adjoint(
                result, smoothness * face_weight * irls_weight * diff, axis, spacing
            )
        return result.reshape_as(variable) / p.sensitivity / p.n_model

    def hessian(self, variable, beta, weights, smoothness=1.0):
        p = self.problem
        density = variable / p.sensitivity
        data_term = (p.adjoint(p.forward(density) * self.row_weights)
                     / p.n_data / p.sensitivity)
        return data_term + beta * self.regularization_hessian(variable, weights, smoothness)

    def project(self, variable):
        return torch.maximum(torch.minimum(variable, self.bound), -self.bound)

    def relative_kkt(self, variable, gradient, lipschitz):
        boundary_slack = 4 * torch.finfo(variable.dtype).eps * self.bound
        blocked = ((variable <= -self.bound + boundary_slack) & (gradient > 0)) | (
            (variable >= self.bound - boundary_slack) & (gradient < 0))
        projected = torch.where(blocked, torch.zeros_like(gradient), gradient)
        return projected.norm(dim=1) / self.rhs.norm(dim=1).clamp_min(1e-15)

    def jacobi(self, beta, weights, smoothness):
        p = self.problem
        diagonal = torch.ones_like(self.rhs).reshape(-1, *p.shape) * p.sensitivity2 * weights[0]
        for axis, (spacing, face, weight) in enumerate(zip(p.spacing, p.face_weights, weights[1:]), 1):
            lo, hi = [slice(None)] * 4, [slice(None)] * 4
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            value = smoothness * face * weight / spacing**2
            diagonal[tuple(lo)] += value
            diagonal[tuple(hi)] += value
        return self.data_diagonal + beta * diagonal.reshape_as(self.rhs) / p.n_model / p.sensitivity.square()

    def cg_initial(self, hessian, variable, beta, weights, smoothness, tolerance, enabled):
        return warm_start(self, hessian, variable, beta, weights, smoothness, tolerance, enabled)

    @torch.no_grad()
    def quadratic(self, beta, weights=None, initial=None, max_iterations=2000,
                  tolerance=1e-5, smoothness=1.0, enabled=None):
        weights = self.weights() if weights is None else weights
        hessian = lambda value: self.hessian(value, beta, weights, smoothness)
        variable = torch.zeros_like(self.rhs) if initial is None else self.project(
            initial.reshape_as(self.rhs) * self.problem.sensitivity
        )
        enabled = (torch.ones(variable.shape[0], device=variable.device, dtype=torch.bool)
                   if enabled is None else enabled.clone())
        variable, cg_iterations = self.cg_initial(hessian, variable, beta, weights,
                                                 smoothness, tolerance, enabled)
        gradient = hessian(variable) - self.rhs
        # Identical deterministic initial vector for every independent system.
        vector = torch.cos(torch.arange(self.problem.n_model, device=variable.device,
                                        dtype=variable.dtype) * 0.731)[None].expand_as(variable)
        vector = vector / vector.norm(dim=1, keepdim=True)
        for _ in range(25):
            product = hessian(vector)
            vector = product / product.norm(dim=1, keepdim=True).clamp_min(1e-30)
        lipschitz = (vector * hessian(vector)).sum(dim=1).abs().clamp_min(1e-12) * 1.15
        previous, previous_gradient = variable.clone(), gradient.clone()
        momentum = torch.ones(variable.shape[0], device=variable.device, dtype=variable.dtype)
        active = enabled & (self.relative_kkt(variable, gradient, lipschitz) > tolerance)
        iterations = torch.zeros_like(momentum, dtype=torch.long)
        backtracks = torch.zeros_like(iterations)
        for iteration in range(max_iterations):
            if not bool(active.any()):
                break
            next_momentum = (1.0 + (1.0 + 4.0 * momentum.square()).sqrt()) / 2.0
            coefficient = (momentum - 1.0) / next_momentum
            extrapolated = variable + coefficient[:, None] * (variable - previous)
            extrapolated_gradient = gradient + coefficient[:, None] * (gradient - previous_gradient)
            for _ in range(30):
                candidate = self.project(extrapolated - extrapolated_gradient / lipschitz[:, None])
                candidate = torch.where(active[:, None], candidate, variable)
                candidate_gradient = hessian(candidate) - self.rhs
                step = candidate - extrapolated
                curvature = (step * (candidate_gradient - extrapolated_gradient)).sum(dim=1)
                upper = lipschitz * step.square().sum(dim=1)
                roundoff = (10 * torch.finfo(variable.dtype).eps * step.norm(dim=1)
                            * (candidate_gradient.norm(dim=1) + extrapolated_gradient.norm(dim=1)
                               + self.rhs.norm(dim=1)))
                slack = 1e-6 * (curvature.abs() + upper.abs()) + roundoff + 1e-15
                bad = active & (curvature > upper + slack)
                if not bool(bad.any()):
                    break
                lipschitz = torch.where(bad, lipschitz * 2.0, lipschitz)
                backtracks += bad
            else:
                raise RuntimeError("Quadratic line search failed")
            iterations += active
            restart = ((extrapolated - candidate) * (candidate - variable)).sum(dim=1) > 0
            previous, previous_gradient = variable, gradient
            variable, gradient = candidate, candidate_gradient
            momentum = torch.where(restart, torch.ones_like(momentum), next_momentum)
            if iteration % 20 == 19 or iteration == max_iterations - 1:
                kkt = self.relative_kkt(variable, gradient, lipschitz)
                active &= kkt > tolerance
                momentum = torch.where(active, momentum, torch.ones_like(momentum))
                if not bool(active.any()):
                    break
        kkt = self.relative_kkt(variable, hessian(variable) - self.rhs, lipschitz)
        density = variable / self.problem.sensitivity
        if not bool(torch.isfinite(density).all()):
            raise RuntimeError("Non-finite density")
        return density, {
            "iterations": iterations + cg_iterations, "relative_kkt": kkt,
            "converged": kkt <= tolerance, "backtracks": backtracks,
            "lipschitz": lipschitz,
        }

    @torch.no_grad()
    def solve(self, method, beta, max_iterations=2000, tolerance=1e-5,
              max_outer=12, epsilon=0.01, outer_tolerance=1e-3, smoothness=1.0):
        if method not in {"L2", "IRLS"}:
            raise ValueError(method)
        started = time.perf_counter()
        density, info = self.quadratic(beta, max_iterations=max_iterations,
                                       tolerance=tolerance, smoothness=smoothness)
        total_iterations = info["iterations"].clone()
        outer_iterations = torch.zeros_like(total_iterations)
        relative_change = torch.zeros_like(info["relative_kkt"])
        if method == "IRLS":
            outer_active = torch.ones_like(total_iterations, dtype=torch.bool)
            for outer in range(max_outer):
                old = density
                candidate, info = self.quadratic(
                    beta, self.weights(old, epsilon), initial=old,
                    max_iterations=max_iterations, tolerance=tolerance * 0.1,
                    smoothness=smoothness, enabled=outer_active,
                )
                density = torch.where(outer_active[:, None], candidate, old)
                total_iterations += info["iterations"] * outer_active
                outer_iterations += outer_active
                change = (density - old).norm(dim=1) / old.norm(dim=1).clamp_min(1e-10)
                relative_change = torch.where(outer_active, change, relative_change)
                variable = density * self.problem.sensitivity
                final_gradient = self.hessian(variable, beta, self.weights(density, epsilon),
                                               smoothness) - self.rhs
                kkt = self.relative_kkt(variable, final_gradient, info["lipschitz"])
                outer_active &= (relative_change > outer_tolerance) | (kkt > tolerance)
                if not bool(outer_active.any()):
                    break
            info["relative_kkt"] = kkt
            info["converged"] = kkt <= tolerance
        if density.is_cuda:
            torch.cuda.synchronize()
        info.update(total_iterations=total_iterations, outer_iterations=outer_iterations,
                    relative_change=relative_change,
                    seconds=time.perf_counter() - started)
        return density.reshape(-1, *self.problem.shape), info
