import math
import numpy as np
import pytest

from relocalization.dnls import sinusoidal_phase
from relocalization.drives import build_drive, param_spec, ANSATZE, MULTIPLY_PI, null_theta


def test_single_sinusoid_matches_dnls_sinusoidal_phase():
    drive = build_drive("single_sinusoid", [1.0, 1.0], control_time=15.0)
    ref = sinusoidal_phase(1.0, omega=1.0, start_time=15.0, multiply_pi=MULTIPLY_PI)
    for t in (0.0, 10.0, 15.0, 15.5, 30.0):
        assert drive(t) == pytest.approx(ref(t))


def test_drives_are_zero_before_control_time():
    for ansatz, theta in [
        ("single_sinusoid", [1.0, 1.0]),
        ("two_tone", [1.0, 1.0, 0.5, 2.0, 0.3]),
        ("chirp", [1.0, 1.0, 0.01]),
        ("bang_bang", [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]),
    ]:
        drive = build_drive(ansatz, theta, control_time=5.0, window=35.0, n_segments=8)
        assert drive(4.999) == 0.0


def test_two_tone_value_after_control():
    drive = build_drive("two_tone", [1.0, 1.0, 0.5, 2.0, 0.0], control_time=0.0)
    tau = 0.7
    expected = math.pi * (1.0 * math.sin(1.0 * tau) + 0.5 * math.sin(2.0 * tau))
    assert drive(tau) == pytest.approx(expected)


def test_chirp_instantaneous_frequency_in_argument():
    gamma, w0, k = 1.0, 1.0, 0.02
    drive = build_drive("chirp", [gamma, w0, k], control_time=0.0)
    tau = 3.0
    expected = math.pi * gamma * math.sin((w0 + 0.5 * k * tau) * tau)
    assert drive(tau) == pytest.approx(expected)


def test_bang_bang_is_piecewise_constant_and_holds_last():
    vals = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    drive = build_drive("bang_bang", vals, control_time=0.0, window=8.0, n_segments=8)
    assert drive(0.5) == pytest.approx(0.1)
    assert drive(2.5) == pytest.approx(0.3)
    assert drive(100.0) == pytest.approx(-0.8)


def test_bang_bang_requires_window():
    with pytest.raises(ValueError):
        build_drive("bang_bang", [0.0] * 8, control_time=0.0, window=None)


def test_bang_bang_does_not_alias_input_array():
    import numpy as np
    arr = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
    drive = build_drive("bang_bang", arr, control_time=0.0, window=8.0, n_segments=8)
    arr[0] = 999.9
    assert drive(0.5) == pytest.approx(0.1)  # unaffected by later mutation


def test_param_spec_equality_does_not_raise():
    a = param_spec("two_tone")
    b = param_spec("two_tone")
    # eq=False: identity-based equality, must not raise on ndarray fields
    assert (a == b) in (True, False)


def test_param_spec_dims_and_bounds():
    assert set(ANSATZE) == {"single_sinusoid", "two_tone", "chirp", "bang_bang"}
    assert param_spec("single_sinusoid").dim == 2
    assert param_spec("two_tone").dim == 5
    assert param_spec("chirp").dim == 3
    assert param_spec("bang_bang", n_segments=8).dim == 8
    spec = param_spec("single_sinusoid")
    assert np.all(spec.lower <= spec.upper)
    assert spec.labels == ("gamma", "omega")


def test_null_theta_gives_zero_drive():
    from relocalization.drives import null_theta
    for ansatz in ("single_sinusoid", "two_tone", "chirp", "bang_bang"):
        drive = build_drive(ansatz, null_theta(ansatz, n_segments=8),
                            control_time=0.0, window=10.0, n_segments=8)
        for t in (0.0, 1.0, 3.3, 9.9):
            assert drive(t) == pytest.approx(0.0)
