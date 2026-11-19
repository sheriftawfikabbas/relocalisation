import pytest

from relocalization.dnls import DNLSParameters, simulate, zero_phase
from relocalization.drives import build_drive


def _final_width(phase):
    p = DNLSParameters(n_sites=201, lambda_=1.0, beta=0.0)
    res = simulate(p, t_max=40.0, dt=0.02, phase=phase, sample_every=20,
                   store_states=False, renormalize=True, classify=False)
    return float(res.diagnostics["width"][-1])


def test_zero_phase_matches_current_code_baseline():
    assert _final_width(zero_phase) == pytest.approx(27.96096, abs=1e-3)


def test_single_sinusoid_reproduces_rebaselined_arrest():
    drive = build_drive("single_sinusoid", [1.0, 1.0], control_time=15.0)
    assert _final_width(drive) == pytest.approx(15.05789, abs=1e-3)
