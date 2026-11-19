from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dnls import DNLSParameters, effective_alpha, gaussian_state, simulate, zero_phase
from .config import RunConfig
from .drives import build_drive, null_theta, param_spec
from .metrics import relocalized_flag, restoration_score
from .optimize import minimize_drive


@dataclass
class ScenarioResult:
    scenario: str
    beta: float
    ansatz: str
    theta_star: np.ndarray
    r: float
    w_spread: float
    w_driven: float
    w_target: float
    relocalized: bool
    times: np.ndarray
    width_undriven: np.ndarray
    width_driven: np.ndarray
    mu_driven: np.ndarray


def _sim(params: DNLSParameters, cfg: RunConfig, phase, psi0=None):
    return simulate(params, t_max=cfg.t_max, dt=cfg.dt, phase=phase,
                    psi0=psi0, sample_every=cfg.sample_every, store_states=False,
                    renormalize=True, classify=False)


def _params(cfg: RunConfig, lambda_: float, beta: float) -> DNLSParameters:
    return DNLSParameters(n_sites=cfg.n_sites, lambda_=lambda_, beta=beta,
                          alpha=cfg.alpha, hopping=cfg.hopping)


def _optimize(objective, ansatz, cfg):
    spec = param_spec(ansatz, n_segments=cfg.n_segments)
    return minimize_drive(objective, spec, n_restarts=cfg.n_restarts,
                          max_evals=cfg.max_evals, seed=cfg.seed,
                          extra_starts=[null_theta(ansatz, n_segments=cfg.n_segments)],
                          grid_seed=(spec.dim <= 2))


def run_scenario_A(beta: float, ansatz: str, cfg: RunConfig,
                   w_ref: float | None = None) -> ScenarioResult:
    """Suppress nonlinear subdiffusion at lambda>2; drive on from t=0.

    w_ref: fixed normalization width (beta=0 undriven final width).  When
    provided, R uses the same absolute scale as Scenario B, avoiding the
    ill-conditioned fractional-restoration form when w_spread ~ w_target.
    If not provided it is computed here (costs one extra simulation at beta=0).
    """
    psi0 = gaussian_state(cfg.n_sites, sigma=cfg.initial_sigma_a)
    target = _sim(_params(cfg, cfg.lambda_a, 0.0), cfg, zero_phase, psi0=psi0)
    w_target = float(target.diagnostics["width"][-1])

    if w_ref is None:
        w_ref = w_target  # beta=0 undriven final width is the target itself

    pb = _params(cfg, cfg.lambda_a, beta)
    undriven = _sim(pb, cfg, zero_phase, psi0=psi0)
    w_spread = float(undriven.diagnostics["width"][-1])

    def objective(theta):
        phase = build_drive(ansatz, theta, control_time=0.0, window=cfg.t_max,
                            n_segments=cfg.n_segments)
        return float(_sim(pb, cfg, phase, psi0=psi0).diagnostics["width"][-1])

    opt = _optimize(objective, ansatz, cfg)
    driven = _sim(pb, cfg, build_drive(ansatz, opt.theta, control_time=0.0,
                                       window=cfg.t_max, n_segments=cfg.n_segments), psi0=psi0)
    w_driven = float(driven.diagnostics["width"][-1])
    return ScenarioResult(
        "A", beta, ansatz, np.asarray(opt.theta),
        restoration_score(w_spread, w_driven, w_target, w_ref=w_ref),
        w_spread, w_driven, w_target,
        relocalized_flag(driven.times, driven.diagnostics["width"], 0.0),
        driven.times, undriven.diagnostics["width"], driven.diagnostics["width"],
        effective_alpha(driven.times, driven.diagnostics["width"]),
    )


def run_scenario_B(beta: float, ansatz: str, cfg: RunConfig,
                   w_ref: float | None = None) -> ScenarioResult:
    """Reverse an already-spread interacting packet at lambda<2; drive after T1.

    w_ref: the beta=0 undriven final width used as the fixed normalization for R.
    If not provided it is computed here (costs one extra simulation at beta=0).
    """
    pb = _params(cfg, cfg.lambda_b, beta)
    undriven = _sim(pb, cfg, zero_phase)
    w_spread = float(undriven.diagnostics["width"][-1])
    w_target = float(undriven.diagnostics["width"][0])
    window = cfg.t_max - cfg.control_time

    if w_ref is None:
        p0 = _params(cfg, cfg.lambda_b, 0.0)
        w_ref = float(_sim(p0, cfg, zero_phase).diagnostics["width"][-1])

    def objective(theta):
        phase = build_drive(ansatz, theta, control_time=cfg.control_time, window=window,
                            n_segments=cfg.n_segments)
        return float(_sim(pb, cfg, phase).diagnostics["width"][-1])

    opt = _optimize(objective, ansatz, cfg)
    driven = _sim(pb, cfg, build_drive(ansatz, opt.theta, control_time=cfg.control_time,
                                       window=window, n_segments=cfg.n_segments))
    w_driven = float(driven.diagnostics["width"][-1])
    return ScenarioResult(
        "B", beta, ansatz, np.asarray(opt.theta),
        restoration_score(w_spread, w_driven, w_target, w_ref=w_ref),
        w_spread, w_driven, w_target,
        relocalized_flag(driven.times, driven.diagnostics["width"], cfg.control_time),
        driven.times, undriven.diagnostics["width"], driven.diagnostics["width"],
        effective_alpha(driven.times, driven.diagnostics["width"]),
    )
