from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np

from .drives import ParamSpec

Objective = Callable[[np.ndarray], float]


@dataclass
class OptResult:
    theta: np.ndarray
    value: float
    n_evals: int
    history: List[float] = field(default_factory=list)


def _clip(theta: np.ndarray, spec: ParamSpec) -> np.ndarray:
    return np.minimum(np.maximum(theta, spec.lower), spec.upper)


def _penalized(objective: Objective, spec: ParamSpec) -> Objective:
    def f(theta: np.ndarray) -> float:
        clipped = _clip(theta, spec)
        penalty = float(np.sum((theta - clipped) ** 2))
        return objective(clipped) + 1e3 * penalty
    return f


def coord_descent(objective: Objective, spec: ParamSpec, x0: np.ndarray,
                  budget: int) -> Tuple[np.ndarray, float, int]:
    x = _clip(np.asarray(x0, dtype=float), spec)
    val = objective(x)
    nevals = 1
    step = (spec.upper - spec.lower) * 0.25
    tol = 1e-3 * (spec.upper - spec.lower + 1e-9)
    while nevals < budget and np.any(step > tol):
        improved = False
        for d in range(spec.dim):
            for s in (+1.0, -1.0):
                cand = x.copy()
                cand[d] = min(max(cand[d] + s * step[d], spec.lower[d]), spec.upper[d])
                v = objective(cand)
                nevals += 1
                if v < val:
                    val, x, improved = v, cand, True
                if nevals >= budget:
                    break
            if nevals >= budget:
                break
        if not improved:
            step *= 0.5
    return x, val, nevals


def grid_seed_starts(objective: Objective, spec: ParamSpec,
                     n_per_dim: int = 12) -> List[np.ndarray]:
    """Evaluate a coarse grid over the parameter space and return the best
    n_per_dim starting points for subsequent local search.  Only practical for
    dim <= 2 (grid size n_per_dim^dim stays manageable); higher dimensions
    return an empty list.
    """
    if spec.dim > 2:
        return []
    axes = [np.linspace(spec.lower[d], spec.upper[d], n_per_dim)
            for d in range(spec.dim)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(spec.dim, -1).T
    vals = np.array([objective(p) for p in grid])
    order = np.argsort(vals)
    return [grid[i] for i in order[:n_per_dim]]


def minimize_drive(objective: Objective, spec: ParamSpec, *, n_restarts: int = 8,
                   max_evals: int = 200, seed: int = 0,
                   extra_starts=None, grid_seed: bool = False) -> OptResult:
    rng = np.random.default_rng(seed)
    best_theta, best_val, history, total = None, float("inf"), [], 0
    grid_starts = grid_seed_starts(objective, spec) if grid_seed else []
    n_extra = len(extra_starts or []) + len(grid_starts)
    per = max(10, max_evals // max(1, n_restarts + n_extra))
    try:
        from scipy.optimize import minimize as scipy_min
        have_scipy = True
    except Exception:  # pragma: no cover - exercised only without scipy
        have_scipy = False

    starts = [np.asarray(s, dtype=float) for s in (extra_starts or [])]
    starts += [np.asarray(s, dtype=float) for s in grid_starts]
    starts += [spec.lower + rng.random(spec.dim) * (spec.upper - spec.lower)
               for _ in range(n_restarts)]

    def consider(cand, val):
        nonlocal best_theta, best_val
        history.append(val)
        if val < best_val:
            best_val, best_theta = val, cand

    for x0 in starts:
        c0 = _clip(x0, spec)
        consider(c0, float(objective(c0)))
        total += 1
        if have_scipy:
            res = scipy_min(_penalized(objective, spec), x0, method="Nelder-Mead",
                            options={"maxfev": per, "xatol": 1e-3, "fatol": 1e-4})
            cand = _clip(np.asarray(res.x, dtype=float), spec)
            consider(cand, float(objective(cand)))
            total += int(res.nfev) + 1
        else:
            cand, val, nevals = coord_descent(objective, spec, x0, per)
            consider(cand, val)
            total += nevals

    return OptResult(theta=best_theta, value=best_val, n_evals=total, history=history)
