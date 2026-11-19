"""Discrete nonlinear Schrodinger Aubry-Andre simulations.

This module implements the dimensionless equation used in Tawfik (2012):

    i d psi_j / dt = -J (psi_{j+1} + psi_{j-1})
                     + lambda cos(2 pi alpha j + phi(t)) psi_j
                     + beta |psi_j|^2 psi_j

with open boundary conditions.  The helpers below also cover the follow-up
questions in the attached note: time-periodic phase drives, piecewise
re-localization protocols, participation/IPR diagnostics, and transport
classification from width scaling.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PhaseFunction = Callable[[float], float]
GOLDEN_ALPHA = (math.sqrt(5.0) - 1.0) / 2.0
EPS = 1.0e-14


@dataclass(frozen=True)
class DNLSParameters:
    """Dimensionless DNLS/Aubry-Andre parameters."""

    n_sites: int = 201
    lambda_: float = 3.0
    beta: float = 0.0
    alpha: float = GOLDEN_ALPHA
    hopping: float = 1.0

    def sites(self) -> np.ndarray:
        """Integer site labels centered on the initially occupied site."""
        return np.arange(self.n_sites, dtype=float) - (self.n_sites // 2)


@dataclass(frozen=True)
class PhaseWindow:
    """A local-time phase protocol active inside a time window."""

    start: float
    end: Optional[float]
    phase: PhaseFunction


@dataclass
class ScalingFit:
    """Late-time scaling fit used for regime labels."""

    power_alpha: float
    power_r2: float
    log_slope: float
    log_r2: float
    tail_width_mean: float
    tail_width_std: float
    tail_participation_mean: float
    tail_ipr_mean: float


@dataclass
class SimulationResult:
    """Sampled simulation output."""

    params: DNLSParameters
    times: np.ndarray
    diagnostics: Dict[str, np.ndarray]
    states: Optional[np.ndarray] = None
    label: Optional[str] = None
    scaling: Optional[ScalingFit] = None

    def save_csv(self, path: str | Path) -> None:
        """Write diagnostics to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = ["time"] + sorted(self.diagnostics.keys())
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(keys)
            for i, time in enumerate(self.times):
                row = [time]
                for key in keys[1:]:
                    row.append(self.diagnostics[key][i])
                writer.writerow(row)


def zero_phase(_: float) -> float:
    return 0.0


def constant_phase(value: float) -> PhaseFunction:
    value = float(value)

    def phase(_: float) -> float:
        return value

    return phase


def sinusoidal_phase(
    amplitude: float,
    omega: float = 1.0,
    phase_offset: float = 0.0,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    multiply_pi: bool = False,
) -> PhaseFunction:
    """Return phi(t) = amplitude * sin(omega * (t - start) + offset).

    Tawfik writes the sinusoidal protocol as pi * c * sin(t).  Set
    multiply_pi=True to use that convention.
    """

    amp = float(amplitude) * (math.pi if multiply_pi else 1.0)
    omega = float(omega)
    phase_offset = float(phase_offset)
    start_time = float(start_time)

    def phase(t: float) -> float:
        if t < start_time:
            return 0.0
        if end_time is not None and t > end_time:
            return 0.0
        return amp * math.sin(omega * (t - start_time) + phase_offset)

    return phase


def step_phase(
    value: float,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> PhaseFunction:
    """Return a rectangular phase step or pulse."""

    value = float(value)
    start_time = float(start_time)

    def phase(t: float) -> float:
        if t < start_time:
            return 0.0
        if end_time is not None and t > end_time:
            return 0.0
        return value

    return phase


def piecewise_phase(windows: Sequence[PhaseWindow], default: float = 0.0) -> PhaseFunction:
    """Build a phase function from local-time windows.

    The window phase receives local time tau = t - start.
    """

    ordered = sorted(windows, key=lambda item: item.start)
    default = float(default)

    def phase(t: float) -> float:
        for window in ordered:
            if t < window.start:
                continue
            if window.end is not None and t > window.end:
                continue
            return float(window.phase(t - window.start))
        return default

    return phase


def delta_state(n_sites: int, center: Optional[int] = None) -> np.ndarray:
    """Single-site initial state."""
    if center is None:
        center = n_sites // 2
    psi = np.zeros(n_sites, dtype=np.complex128)
    psi[int(center)] = 1.0 + 0.0j
    return psi


def gaussian_state(
    n_sites: int,
    sigma: float = 3.0,
    center: Optional[float] = None,
    phase_gradient: float = 0.0,
) -> np.ndarray:
    """Normalized Gaussian initial state on the lattice."""
    sites = np.arange(n_sites, dtype=float)
    if center is None:
        center = n_sites // 2
    envelope = np.exp(-((sites - center) ** 2) / (4.0 * sigma**2))
    phase = np.exp(1j * phase_gradient * (sites - center))
    return normalize(envelope * phase)


def normalize(psi: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(psi) ** 2))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero state.")
    return np.asarray(psi, dtype=np.complex128) / norm


def onsite_potential(params: DNLSParameters, phi: float) -> np.ndarray:
    return params.lambda_ * np.cos(2.0 * math.pi * params.alpha * params.sites() + phi)


def linear_hamiltonian(params: DNLSParameters, phi: float) -> np.ndarray:
    """Linear Aubry-Andre Hamiltonian for beta=0."""
    hamiltonian = np.zeros((params.n_sites, params.n_sites), dtype=np.complex128)
    np.fill_diagonal(hamiltonian, onsite_potential(params, phi))
    offdiag = -params.hopping * np.ones(params.n_sites - 1, dtype=np.complex128)
    hamiltonian[np.arange(params.n_sites - 1), np.arange(1, params.n_sites)] = offdiag
    hamiltonian[np.arange(1, params.n_sites), np.arange(params.n_sites - 1)] = offdiag
    return hamiltonian


def dnls_rhs(
    t: float,
    psi: np.ndarray,
    params: DNLSParameters,
    phase: PhaseFunction = zero_phase,
) -> np.ndarray:
    """Right-hand side d psi / dt for the DNLS equation."""
    neighbors = np.zeros_like(psi)
    neighbors[:-1] += psi[1:]
    neighbors[1:] += psi[:-1]

    phi = phase(t)
    onsite = onsite_potential(params, phi)
    nonlinear = params.beta * np.abs(psi) ** 2
    h_psi = -params.hopping * neighbors + (onsite + nonlinear) * psi
    return -1j * h_psi


def rk4_step(
    t: float,
    psi: np.ndarray,
    dt: float,
    params: DNLSParameters,
    phase: PhaseFunction = zero_phase,
) -> np.ndarray:
    """One explicit fourth-order Runge-Kutta step."""
    k1 = dnls_rhs(t, psi, params, phase)
    k2 = dnls_rhs(t + 0.5 * dt, psi + 0.5 * dt * k1, params, phase)
    k3 = dnls_rhs(t + 0.5 * dt, psi + 0.5 * dt * k2, params, phase)
    k4 = dnls_rhs(t + dt, psi + dt * k3, params, phase)
    return psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def diagnostics_from_density(
    density: np.ndarray,
    sites: np.ndarray,
    reference_center: float = 0.0,
) -> Dict[str, float]:
    """Width, participation, IPR, and second moment for a normalized density."""
    rho = np.asarray(density, dtype=float)
    norm = float(np.sum(rho))
    if norm <= EPS:
        raise ValueError("Density norm is zero.")
    mean_site = float(np.sum(sites * rho) / norm)
    centered_width = float(np.sqrt(np.sum(((sites - mean_site) ** 2) * rho) / norm))
    m2 = float(np.sum(((sites - reference_center) ** 2) * rho) / norm)
    rho2 = float(np.sum(rho**2))
    participation = float((norm**2) / max(rho2, EPS))
    ipr = float(rho2 / max(norm**2, EPS))
    return {
        "norm": norm,
        "center": mean_site,
        "width": centered_width,
        "m2": m2,
        "participation": participation,
        "ipr": ipr,
    }


def diagnostics_for_state(
    psi: np.ndarray,
    sites: np.ndarray,
    reference_center: float = 0.0,
) -> Dict[str, float]:
    return diagnostics_from_density(np.abs(psi) ** 2, sites, reference_center)


def simulate(
    params: DNLSParameters,
    t_max: float,
    dt: float,
    phase: PhaseFunction = zero_phase,
    psi0: Optional[np.ndarray] = None,
    sample_every: int = 10,
    store_states: bool = True,
    renormalize: bool = True,
    classify: bool = True,
) -> SimulationResult:
    """Integrate the DNLS equation and sample diagnostics."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive.")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive.")

    psi = delta_state(params.n_sites) if psi0 is None else normalize(psi0)
    sites = params.sites()
    reference_center = 0.0
    n_steps = int(round(t_max / dt))

    times: List[float] = []
    states: List[np.ndarray] = []
    diag_rows: List[Dict[str, float]] = []

    for step in range(n_steps + 1):
        t = step * dt
        if step % sample_every == 0 or step == n_steps:
            times.append(t)
            diag_rows.append(diagnostics_for_state(psi, sites, reference_center))
            if store_states:
                states.append(psi.copy())

        if step == n_steps:
            break

        psi = rk4_step(t, psi, dt, params, phase)
        if renormalize:
            psi = normalize(psi)

    times_array = np.asarray(times, dtype=float)
    diagnostics = {
        key: np.asarray([row[key] for row in diag_rows], dtype=float)
        for key in diag_rows[0]
    }
    states_array = np.asarray(states, dtype=np.complex128) if store_states else None

    scaling = fit_transport_scaling(
        times_array,
        diagnostics["width"],
        diagnostics["participation"],
        diagnostics["ipr"],
    )
    label = classify_transport(params, scaling) if classify else None
    return SimulationResult(params, times_array, diagnostics, states_array, label, scaling)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= EPS:
        return 1.0
    return 1.0 - ss_res / ss_tot


def fit_transport_scaling(
    times: np.ndarray,
    width: np.ndarray,
    participation: np.ndarray,
    ipr: np.ndarray,
    tail_fraction: float = 0.5,
) -> ScalingFit:
    """Fit late-time width to power-law and logarithmic forms."""
    if len(times) < 4:
        return ScalingFit(0.0, 0.0, 0.0, 0.0, float(width[-1]), 0.0, float(participation[-1]), float(ipr[-1]))

    start = max(1, int((1.0 - tail_fraction) * len(times)))
    t_tail = np.asarray(times[start:], dtype=float)
    w_tail = np.asarray(width[start:], dtype=float)
    p_tail = np.asarray(participation[start:], dtype=float)
    ipr_tail = np.asarray(ipr[start:], dtype=float)

    valid = (t_tail > 0.0) & (w_tail > 0.0)
    if int(np.sum(valid)) < 3:
        return ScalingFit(0.0, 0.0, 0.0, 0.0, float(np.mean(w_tail)), float(np.std(w_tail)), float(np.mean(p_tail)), float(np.mean(ipr_tail)))

    t_fit = t_tail[valid]
    w_fit = np.maximum(w_tail[valid], EPS)

    log_t = np.log(t_fit)
    log_w = np.log(w_fit)
    power_slope, power_intercept = np.polyfit(log_t, log_w, 1)
    power_pred = power_slope * log_t + power_intercept
    power_r2 = _r2(log_w, power_pred)

    log_slope, log_intercept = np.polyfit(log_t, w_fit, 1)
    log_pred = log_slope * log_t + log_intercept
    log_r2 = _r2(w_fit, log_pred)

    return ScalingFit(
        power_alpha=float(power_slope),
        power_r2=float(power_r2),
        log_slope=float(log_slope),
        log_r2=float(log_r2),
        tail_width_mean=float(np.mean(w_tail)),
        tail_width_std=float(np.std(w_tail)),
        tail_participation_mean=float(np.mean(p_tail)),
        tail_ipr_mean=float(np.mean(ipr_tail)),
    )


def classify_transport(params: DNLSParameters, fit: ScalingFit) -> str:
    """Heuristic regime label from late-time scaling.

    The labels are meant for scans and triage, not as a substitute for visual
    inspection.  Thresholds are intentionally conservative.
    """
    n_sites = params.n_sites
    alpha = fit.power_alpha
    width = fit.tail_width_mean
    width_std = fit.tail_width_std
    participation = fit.tail_participation_mean

    if participation <= 2.5 and width <= 2.5:
        return "self_trapping"
    if width > 0.22 * n_sites and alpha >= 0.20:
        return "diffusion"
    if alpha >= 0.35:
        return "diffusion"
    if 0.07 <= alpha < 0.35:
        if fit.log_r2 >= fit.power_r2 - 0.03 and participation <= 0.25 * n_sites:
            return "logarithmic_localization"
        return "subdiffusion"
    if alpha < 0.07 and width_std <= max(0.5, 0.08 * width):
        return "ordinary_localization"
    return "weak_spreading"


def effective_alpha(times: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Local d log(width) / d log(time), useful for scaling plots."""
    safe_t = np.maximum(times, EPS)
    safe_w = np.maximum(width, EPS)
    return np.gradient(np.log(safe_w), np.log(safe_t))


def scan_phase_diagram(
    lambda_values: Iterable[float],
    beta_values: Iterable[float],
    gamma_values: Iterable[float],
    omega_values: Iterable[float],
    base_params: DNLSParameters,
    t_max: float,
    dt: float,
    sample_every: int,
    phase_scale_pi: bool = False,
) -> List[Dict[str, float | str]]:
    """Run a lambda/beta/gamma/omega scan for the sinusoidal phase drive."""
    rows: List[Dict[str, float | str]] = []
    for lambda_value in lambda_values:
        for beta_value in beta_values:
            params = DNLSParameters(
                n_sites=base_params.n_sites,
                lambda_=float(lambda_value),
                beta=float(beta_value),
                alpha=base_params.alpha,
                hopping=base_params.hopping,
            )
            for gamma_value in gamma_values:
                for omega_value in omega_values:
                    phase = sinusoidal_phase(
                        gamma_value,
                        omega=omega_value,
                        multiply_pi=phase_scale_pi,
                    )
                    result = simulate(
                        params,
                        t_max=t_max,
                        dt=dt,
                        phase=phase,
                        sample_every=sample_every,
                        store_states=False,
                    )
                    assert result.scaling is not None
                    fit = result.scaling
                    rows.append(
                        {
                            "lambda": float(lambda_value),
                            "beta": float(beta_value),
                            "gamma": float(gamma_value),
                            "omega": float(omega_value),
                            "label": result.label or "",
                            "power_alpha": fit.power_alpha,
                            "power_r2": fit.power_r2,
                            "log_slope": fit.log_slope,
                            "log_r2": fit.log_r2,
                            "tail_width_mean": fit.tail_width_mean,
                            "tail_participation_mean": fit.tail_participation_mean,
                            "tail_ipr_mean": fit.tail_ipr_mean,
                        }
                    )
    return rows


def write_rows_csv(path: str | Path, rows: Sequence[Dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def detect_relocalization(times: np.ndarray, width: np.ndarray, t_control: float) -> Dict[str, float | bool]:
    """Summarize whether a protocol arrests or reverses earlier spreading."""
    before = width[times <= t_control]
    after = width[times >= t_control]
    if len(before) == 0 or len(after) == 0:
        return {"arrested": False, "relocalized": False, "peak_before": float("nan"), "final_width": float(width[-1])}

    peak_before = float(np.max(before))
    peak_after = float(np.max(after))
    final_width = float(width[-1])
    initial_width = float(width[0])
    arrested = final_width <= 0.90 * peak_after
    relocalized = peak_before > max(initial_width + 1.0, 1.5 * max(initial_width, 1.0)) and final_width <= 0.75 * peak_before
    return {
        "arrested": bool(arrested),
        "relocalized": bool(relocalized),
        "peak_before": peak_before,
        "peak_after": peak_after,
        "final_width": final_width,
    }
