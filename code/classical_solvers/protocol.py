"""Unchanged solver/calibration selection routines from the selected experiment."""
import hashlib
import torch
SOLVER = dict(max_iterations=6000,tolerance=1e-5,max_outer=80,outer_tolerance=1e-3,
              epsilon=.01,smoothness=1.)
CALIBRATION_PER_TRACK = 4
CALIBRATION_SEED = b"v030-v023-irls-reoptimized-20260930"
def stable_rank(value: str) -> int:
    return int(hashlib.sha256(CALIBRATION_SEED + value.encode()).hexdigest(), 16)


def calibration_indices(dataset):
    selected = []
    tracks = {str(row["generation_track"]) for row in dataset.records}
    for track in sorted(tracks):
        candidates = [
            (index, row) for index, row in enumerate(dataset.records)
            if str(row["generation_track"]) == track
        ]
        candidates.sort(key=lambda item: (stable_rank(item[1]["sample_id"]), item[0]))
        selected.extend(index for index, _ in candidates[:CALIBRATION_PER_TRACK])
    return sorted(selected)


def solve_full(context, method, beta):
    density, info = context.solve(
        method,
        beta,
        max_iterations=SOLVER["max_iterations"],
        tolerance=SOLVER["tolerance"],
        max_outer=SOLVER["max_outer"],
        epsilon=SOLVER["epsilon"],
        outer_tolerance=SOLVER["outer_tolerance"],
        smoothness=SOLVER["smoothness"],
    )
    if method == "IRLS":
        info["kkt_converged"] = info["relative_kkt"] <= SOLVER["tolerance"]
        info["outer_converged"] = info["relative_change"] <= SOLVER["outer_tolerance"]
        info["converged"] = info["kkt_converged"] & info["outer_converged"]
    else:
        info["kkt_converged"] = info["converged"]
        info["outer_converged"] = torch.ones_like(info["converged"])
    return density, info


@torch.no_grad()
def continue_irls(context, initial_density, beta):
    """Continue IRLS from a saved density with the V026 update equations."""
    p = context.problem
    density = initial_density.reshape(len(initial_density), -1).to(
        context.observed.device, dtype=p.matrix.dtype
    ).clone()
    total = torch.zeros(len(density), dtype=torch.long, device=density.device)
    outer_count = torch.zeros_like(total)
    change = torch.zeros(len(density), dtype=p.matrix.dtype, device=density.device)
    kkt = torch.full_like(change, float("inf"))
    active = torch.ones(len(density), dtype=torch.bool, device=density.device)
    for _ in range(SOLVER["max_outer"]):
        indices = torch.where(active)[0]
        if len(indices) == 0:
            break
        sub = p.batch(context.observed[indices])
        old = density[indices]
        candidate, inner = sub.quadratic(
            beta,
            sub.weights(old, SOLVER["epsilon"]),
            initial=old,
            max_iterations=SOLVER["max_iterations"],
            tolerance=SOLVER["tolerance"] * 0.1,
            smoothness=SOLVER["smoothness"],
            enabled=torch.ones(len(indices), dtype=torch.bool, device=density.device),
        )
        density[indices] = candidate
        total[indices] += inner["iterations"]
        outer_count[indices] += 1
        local_change = (candidate - old).norm(dim=1) / old.norm(dim=1).clamp_min(1e-10)
        change[indices] = local_change
        variable = candidate * p.sensitivity
        gradient = sub.hessian(
            variable,
            beta,
            sub.weights(candidate, SOLVER["epsilon"]),
            SOLVER["smoothness"],
        ) - sub.rhs
        kkt[indices] = sub.relative_kkt(variable, gradient, inner["lipschitz"])
        active[indices] = (
            (local_change > SOLVER["outer_tolerance"]) | (kkt[indices] > SOLVER["tolerance"])
        )
    both = (kkt <= SOLVER["tolerance"]) & (change <= SOLVER["outer_tolerance"])
    return density.reshape(-1, *p.shape), {
        "total_iterations": total,
        "outer_iterations": outer_count,
        "relative_kkt": kkt,
        "relative_change": change,
        "kkt_converged": kkt <= SOLVER["tolerance"],
        "outer_converged": change <= SOLVER["outer_tolerance"],
        "converged": both,
    }
