import math
import numpy as np
import pytest

from relocalization.metrics import (
    restoration_score,
    extract_beta_star,
    extract_beta_star_half_peak,
    relocalized_flag,
)


def test_restoration_score_endpoints_and_midpoint():
    assert restoration_score(10.0, 10.0, 2.0) == pytest.approx(0.0)
    assert restoration_score(10.0, 2.0, 2.0) == pytest.approx(1.0)
    assert restoration_score(10.0, 6.0, 2.0) == pytest.approx(0.5)


def test_restoration_score_negligible_spread_is_nan():
    # w_spread ~ w_target => nothing to restore, R undefined (not a blown-up value)
    assert math.isnan(restoration_score(5.0, 3.0, 5.0))       # denom 0
    assert math.isnan(restoration_score(5.02, 1.0, 5.0))      # denom 0.02 < 5% of 5
    # meaningful spread still scores normally
    assert restoration_score(10.0, 6.0, 2.0) == pytest.approx(0.5)


def test_extract_beta_star_half_peak_finds_collapse():
    betas = [0.0, 0.5, 1.0, 2.0, 3.0]
    r = [0.31, 0.36, 0.38, 0.0, 0.0]   # peak 0.38 at 1.0, half 0.19, collapse 1->2
    # crossing (0.38-0.19)/(0.38-0) = 0.5 -> beta*=1.5
    assert extract_beta_star_half_peak(betas, r) == pytest.approx(1.5)


def test_extract_beta_star_half_peak_nan_when_no_restoration():
    assert math.isnan(extract_beta_star_half_peak([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]))


def test_extract_beta_star_half_peak_ignores_nan_points():
    betas = [0.0, 0.5, 1.0, 2.0]
    r = [0.0, float("nan"), 0.4, 0.0]  # peak 0.4 at 1.0, half 0.2, collapse 1->2 -> 1.5
    assert extract_beta_star_half_peak(betas, r) == pytest.approx(1.5)


def test_extract_beta_star_interpolates_crossing():
    betas = [0.0, 1.0, 2.0, 3.0]
    r = [1.0, 0.8, 0.4, 0.1]
    assert extract_beta_star(betas, r, threshold=0.5) == pytest.approx(1.75)


def test_extract_beta_star_all_pass_returns_max():
    assert extract_beta_star([0.0, 1.0, 2.0], [0.9, 0.8, 0.7], 0.5) == pytest.approx(2.0)


def test_extract_beta_star_first_fails_returns_nan():
    assert math.isnan(extract_beta_star([0.0, 1.0], [0.3, 0.1], 0.5))


def test_extract_beta_star_ignores_degenerate_beta_zero():
    # beta=0 is a degenerate reference (R=0 in Scenario A); beta* is measured
    # over beta>0. Despite R(0)=0, the beta>0 points start above threshold.
    betas = [0.0, 1.0, 2.0, 3.0]
    r = [0.0, 0.9, 0.8, 0.2]
    # crossing between beta=2 (0.8) and beta=3 (0.2): frac=(0.8-0.5)/(0.8-0.2)=0.5 -> 2.5
    assert extract_beta_star(betas, r, 0.5) == pytest.approx(2.5)


def test_extract_beta_star_all_beta_zero_returns_nan():
    assert math.isnan(extract_beta_star([0.0], [0.0], 0.5))


def test_relocalized_flag_true_when_peak_then_drop():
    times = np.linspace(0.0, 40.0, 41)
    width = np.concatenate([np.linspace(0.0, 20.0, 21), np.linspace(20.0, 3.0, 20)])
    assert relocalized_flag(times, width, t_control=20.0) is True
