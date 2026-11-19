from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .dnls import PhaseFunction, sinusoidal_phase

MULTIPLY_PI = True
_PI = math.pi
ANSATZE = ("single_sinusoid", "two_tone", "chirp", "bang_bang")


@dataclass(frozen=True, eq=False)
class ParamSpec:
    name: str
    lower: np.ndarray
    upper: np.ndarray
    labels: Tuple[str, ...]

    @property
    def dim(self) -> int:
        return len(self.labels)


def param_spec(ansatz: str, *, n_segments: int = 8, gamma_max: float = 4.0,
               omega_min: float = 0.1, omega_max: float = 4.0, k_max: float = 0.05) -> ParamSpec:
    if ansatz == "single_sinusoid":
        return ParamSpec(ansatz, np.array([0.0, omega_min]), np.array([gamma_max, omega_max]),
                         ("gamma", "omega"))
    if ansatz == "two_tone":
        return ParamSpec(ansatz,
                         np.array([0.0, omega_min, 0.0, omega_min, -_PI]),
                         np.array([gamma_max, omega_max, gamma_max, omega_max, _PI]),
                         ("g1", "o1", "g2", "o2", "delta"))
    if ansatz == "chirp":
        return ParamSpec(ansatz, np.array([0.0, omega_min, -k_max]),
                         np.array([gamma_max, omega_max, k_max]), ("gamma", "omega0", "k"))
    if ansatz == "bang_bang":
        return ParamSpec(ansatz, np.full(n_segments, -_PI), np.full(n_segments, _PI),
                         tuple(f"phi{i}" for i in range(n_segments)))
    raise ValueError(f"unknown ansatz: {ansatz}")


def build_drive(ansatz: str, theta: Sequence[float], *, control_time: float = 0.0,
                window: float | None = None, n_segments: int = 8) -> PhaseFunction:
    theta = np.asarray(theta, dtype=float)
    amp = _PI if MULTIPLY_PI else 1.0

    if ansatz == "single_sinusoid":
        return sinusoidal_phase(float(theta[0]), omega=float(theta[1]),
                                start_time=control_time, multiply_pi=MULTIPLY_PI)

    if ansatz == "two_tone":
        g1, o1, g2, o2, delta = (float(x) for x in theta)

        def phase_tt(t: float) -> float:
            if t < control_time:
                return 0.0
            tau = t - control_time
            return amp * (g1 * math.sin(o1 * tau) + g2 * math.sin(o2 * tau + delta))

        return phase_tt

    if ansatz == "chirp":
        gamma, w0, k = (float(x) for x in theta)

        def phase_ch(t: float) -> float:
            if t < control_time:
                return 0.0
            tau = t - control_time
            return amp * gamma * math.sin((w0 + 0.5 * k * tau) * tau)

        return phase_ch

    if ansatz == "bang_bang":
        if window is None or window <= 0.0:
            raise ValueError("bang_bang requires a positive window")
        seg = window / n_segments
        vals = theta.copy()

        def phase_bb(t: float) -> float:
            if t < control_time:
                return 0.0
            idx = int((t - control_time) // seg)
            idx = max(0, min(n_segments - 1, idx))
            return float(vals[idx])

        return phase_bb

    raise ValueError(f"unknown ansatz: {ansatz}")


def null_theta(ansatz: str, *, n_segments: int = 8) -> np.ndarray:
    """Parameter vector that yields the identically-zero drive phi(t) = 0."""
    if ansatz == "single_sinusoid":
        return np.array([0.0, 1.0])
    if ansatz == "two_tone":
        return np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    if ansatz == "chirp":
        return np.array([0.0, 1.0, 0.0])
    if ansatz == "bang_bang":
        return np.zeros(n_segments)
    raise ValueError(f"unknown ansatz: {ansatz}")
