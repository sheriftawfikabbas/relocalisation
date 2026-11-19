from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

from .config import RunConfig
from .scenarios import run_scenario_A, run_scenario_B


def _run_one(task) -> Dict[str, object]:
    scenario, beta, ansatz, cfg, w_ref = task
    print(f"[start] scenario={scenario} beta={beta:.2f} ansatz={ansatz}", flush=True)
    try:
        if scenario == "A":
            res = run_scenario_A(beta, ansatz, cfg, w_ref=w_ref)
        else:
            res = run_scenario_B(beta, ansatz, cfg, w_ref=w_ref)
        row = {
            "scenario": scenario, "beta": beta, "ansatz": ansatz, "r": res.r,
            "w_spread": res.w_spread, "w_driven": res.w_driven, "w_target": res.w_target,
            "relocalized": res.relocalized,
            "theta_star": [float(x) for x in res.theta_star],
        }
        print(f"[done]  scenario={scenario} beta={beta:.2f} ansatz={ansatz} R={res.r:.4f}", flush=True)
        return row
    except Exception:
        print(f"[ERROR] scenario={scenario} beta={beta:.2f} ansatz={ansatz}\n{traceback.format_exc()}", flush=True)
        raise


def _compute_w_ref_b(cfg: RunConfig) -> float:
    """Simulate beta=0 undriven scenario B once to get the fixed R denominator."""
    from .scenarios import _params, _sim
    from .dnls import zero_phase
    p0 = _params(cfg, cfg.lambda_b, 0.0)
    return float(_sim(p0, cfg, zero_phase).diagnostics["width"][-1])


def _compute_w_ref_a(cfg: RunConfig) -> float:
    """Simulate beta=0 undriven scenario A once to get the fixed R denominator."""
    from .scenarios import _params, _sim
    from .dnls import gaussian_state, zero_phase
    p0 = _params(cfg, cfg.lambda_a, 0.0)
    psi0 = gaussian_state(cfg.n_sites, sigma=cfg.initial_sigma_a)
    return float(_sim(p0, cfg, zero_phase, psi0=psi0).diagnostics["width"][-1])


def beta_sweep(scenario: str, ansatz_list: Sequence[str], betas: Sequence[float],
               cfg: RunConfig, jobs: int = 1) -> List[Dict[str, object]]:
    w_ref: Optional[float] = None
    if scenario == "B":
        print("[info] computing w_ref (beta=0 undriven) for scenario B ...", flush=True)
        w_ref = _compute_w_ref_b(cfg)
        print(f"[info] w_ref = {w_ref:.4f}", flush=True)
    elif scenario == "A":
        print("[info] computing w_ref (beta=0 undriven) for scenario A ...", flush=True)
        w_ref = _compute_w_ref_a(cfg)
        print(f"[info] w_ref = {w_ref:.4f}", flush=True)

    tasks = [(scenario, float(b), a, cfg, w_ref) for a in ansatz_list for b in betas]
    if jobs == 1:
        return [_run_one(t) for t in tasks]
    results = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(_run_one, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            row = future.result()
            results[id(task)] = row
    return [results[id(t)] for t in tasks if id(t) in results]
