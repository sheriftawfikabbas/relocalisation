import numpy as np

from relocalization.drives import param_spec
from relocalization.optimize import minimize_drive, coord_descent


def _bowl(target):
    def objective(theta):
        return float(np.sum((np.asarray(theta) - target) ** 2))
    return objective


def test_minimize_recovers_convex_optimum():
    spec = param_spec("single_sinusoid")  # bounds [0,4] x [0.1,4]
    target = np.array([2.0, 1.5])
    res = minimize_drive(_bowl(target), spec, n_restarts=3, max_evals=150, seed=0)
    assert np.allclose(res.theta, target, atol=0.1)
    assert res.value < 0.05


def test_minimize_is_deterministic_under_seed():
    spec = param_spec("single_sinusoid")
    obj = _bowl(np.array([1.0, 2.0]))
    a = minimize_drive(obj, spec, n_restarts=2, max_evals=80, seed=7)
    b = minimize_drive(obj, spec, n_restarts=2, max_evals=80, seed=7)
    assert np.allclose(a.theta, b.theta)


def test_minimize_respects_bounds():
    spec = param_spec("single_sinusoid")
    target = np.array([100.0, 100.0])  # far outside upper bound
    res = minimize_drive(_bowl(target), spec, n_restarts=2, max_evals=80, seed=1)
    assert np.all(res.theta <= spec.upper + 1e-9)
    assert np.all(res.theta >= spec.lower - 1e-9)


def test_coord_descent_fallback_finds_optimum():
    spec = param_spec("single_sinusoid")
    target = np.array([3.0, 2.0])
    x0 = np.array([0.5, 0.5])
    theta, val, nevals = coord_descent(_bowl(target), spec, x0, budget=400)
    assert np.allclose(theta, target, atol=0.2)


def test_minimize_honors_extra_start_baseline():
    spec = param_spec("single_sinusoid")
    target = np.array([2.0, 1.5])
    baseline = np.array([0.0, 1.0])
    baseline_val = float(np.sum((baseline - target) ** 2))
    res = minimize_drive(_bowl(target), spec, n_restarts=1, max_evals=40, seed=3,
                         extra_starts=[baseline])
    assert res.value <= baseline_val + 1e-9


def test_minimize_budget_accounts_for_extra_starts():
    spec = param_spec("single_sinusoid")
    target = np.array([2.0, 1.5])
    # 1 extra start + 1 restart => 2 starts; each ~per evals + 2 bookkeeping evals.
    res = minimize_drive(_bowl(target), spec, n_restarts=1, max_evals=40, seed=0,
                         extra_starts=[np.array([0.0, 1.0])])
    starts = 1 + 1
    per = max(10, 40 // starts)
    assert res.n_evals <= starts * (per + 2) + 1
