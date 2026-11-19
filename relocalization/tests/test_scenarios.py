import numpy as np

from relocalization.config import dev_config
from relocalization.scenarios import run_scenario_A, run_scenario_B, ScenarioResult


def _tiny():
    return dev_config(n_sites=101, t_max=30.0, dt=0.05, sample_every=10,
                      max_evals=20, n_restarts=1, ansatze=("single_sinusoid",))


def test_scenario_A_beta_positive_has_spreading_and_scores():
    cfg = _tiny()
    res = run_scenario_A(beta=2.0, ansatz="single_sinusoid", cfg=cfg)
    assert isinstance(res, ScenarioResult) and res.scenario == "A"
    assert res.w_spread > res.w_target
    assert res.w_driven <= res.w_spread + 1e-9
    assert -1.0 <= res.r <= 1.5


def test_scenario_B_beta_zero_spreads_and_reduces():
    cfg = _tiny()
    res = run_scenario_B(beta=0.0, ansatz="single_sinusoid", cfg=cfg)
    assert res.scenario == "B"
    assert res.w_spread > res.w_target
    assert res.times.shape == res.width_driven.shape == res.mu_driven.shape


def test_scenario_is_deterministic_under_seed():
    cfg = _tiny()
    a = run_scenario_A(beta=2.0, ansatz="single_sinusoid", cfg=cfg)
    b = run_scenario_A(beta=2.0, ansatz="single_sinusoid", cfg=cfg)
    assert np.allclose(a.theta_star, b.theta_star)


def test_scenario_A_uses_finite_width_initial_state():
    cfg = _tiny()
    res = run_scenario_A(beta=1.0, ansatz="single_sinusoid", cfg=cfg)
    assert res.width_undriven[0] > 3.0  # Gaussian init (single-site would be 0)


def test_scenario_B_uses_single_site_initial_state():
    import pytest
    cfg = _tiny()
    res = run_scenario_B(beta=0.0, ansatz="single_sinusoid", cfg=cfg)
    assert res.width_undriven[0] == pytest.approx(0.0, abs=1e-9)
