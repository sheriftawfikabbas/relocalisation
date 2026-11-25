"""Truncated Wigner approximation (TWA) for the driven Bose-Hubbard chain.

Bose-Hubbard model (hopping J, onsite lambda*cos(2 pi alpha j + phi(t)), U/2 n(n-1))
with N_b bosons is mapped to the mean-field DNLS of Eq. (dnls) via beta = U N_b.
TWA evolves classical fields alpha_j(t) from the Wigner distribution of the
initial state under the *Weyl symbol* of H and estimates symmetrically ordered
moments by trajectory averages.

Conventions
-----------
* Per-atom coupling g = U = beta / N_b.  Fields are NOT normalized: sum|alpha|^2 ~ N_b.
* Physical density (the Weyl symbol of n_j is |alpha_j|^2 - 1/2):
      rho_j(t) = mean_traj |alpha_j|^2 - 1/2,
  a *signed* estimator.  It is never clipped; moments are ratios of sums of the
  signed density over a fixed analysis window |j| <= W (see `window_moments`).
  W must be fixed from the mean-field run of the same protocol (`choose_window`),
  never from TWA data.
* Initial state.  For a Fock state |N_b> on one site ("fock_ring") the Wigner
  function is approximated by the ring |alpha_0|^2 = N_b + 1/2 with uniform random
  phase (reproduces <n> = N_b exactly); every other mode is vacuum noise
  alpha_j = eta_j, eta_j ~ CN with <|eta_j|^2> = 1/2.  For a general
  (normalized) classical profile psi0 ("coherent") alpha_j = sqrt(N_b) psi0_j + eta_j
  is used (exact for a coherent state, an O(1/N_b) approximation otherwise).

Weyl ordering
-------------
The Weyl symbol of a^dag^2 a^2 is |alpha|^4 - 2|alpha|^2 + 1/2, hence
    H_int = (U/2) sum n(n-1)  ->  (U/2) sum (|alpha|^4 - 2|alpha|^2 + 1/2),
    i d alpha / dt = dH/dalpha* = ... + U (|alpha|^2 - 1) alpha.
The exact Weyl drift therefore differs from the plain GPE drift g|alpha|^2 alpha only
by the uniform onsite shift -U, which commutes with the rest of H and multiplies
every alpha_j by the same phase exp(+iUt).  Densities |alpha_j|^2 are unaffected, so
the code evolves with g|alpha|^2 alpha; `weyl=True` switches to the explicit
U(|alpha|^2-1) drift and tests confirm identical densities.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np
from numba import njit, prange

from .dnls import DNLSParameters, PhaseFunction, zero_phase

VACUUM = 0.5  # Weyl vacuum occupation <|eta|^2> per mode


@njit(parallel=True, cache=True, fastmath=False)
def _rk4_batch(a, v0, v1, v2, hop, g, dt):
    """In-place RK4 step on a (B, N) complex array; v0,v1,v2 are the onsite
    potentials (incl. any Weyl shift) at t, t+dt/2, t+dt."""
    B, N = a.shape
    for b in prange(B):
        row = a[b]
        k1 = np.empty(N, np.complex128)
        k2 = np.empty(N, np.complex128)
        k3 = np.empty(N, np.complex128)
        k4 = np.empty(N, np.complex128)
        tmp = np.empty(N, np.complex128)
        for stage in range(4):
            if stage == 0:
                src = row
                v = v0
                out = k1
            elif stage == 1:
                for j in range(N):
                    tmp[j] = row[j] + 0.5 * dt * k1[j]
                src = tmp
                v = v1
                out = k2
            elif stage == 2:
                for j in range(N):
                    tmp[j] = row[j] + 0.5 * dt * k2[j]
                src = tmp
                v = v1
                out = k3
            else:
                for j in range(N):
                    tmp[j] = row[j] + dt * k3[j]
                src = tmp
                v = v2
                out = k4
            for j in range(N):
                nb = 0.0 + 0.0j
                if j > 0:
                    nb += src[j - 1]
                if j < N - 1:
                    nb += src[j + 1]
                x = src[j]
                dens = x.real * x.real + x.imag * x.imag
                hpsi = -hop * nb + (v[j] + g * dens) * x
                out[j] = -1j * hpsi
        for j in range(N):
            row[j] = row[j] + (dt / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])


def _onsite(params: DNLSParameters, sites: np.ndarray, phi: float, shift: float) -> np.ndarray:
    return params.lambda_ * np.cos(2.0 * math.pi * params.alpha * sites + phi) - shift


def evolve_batch(alpha: np.ndarray, params: DNLSParameters, t_max: float, dt: float,
                 phase: PhaseFunction = zero_phase, sample_every: int = 20, *,
                 t_start: float = 0.0, weyl: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evolve a batch of classical fields (B, N) with params.beta used as the
    per-atom coupling g.  Returns (times, mean |alpha|^2 at sampled times, final fields).
    `weyl=True` uses the explicit Weyl drift g(|alpha|^2 - 1) alpha."""
    a = np.ascontiguousarray(alpha, dtype=np.complex128).copy()
    sites = params.sites()
    g = float(params.beta)
    shift = g if weyl else 0.0
    n_steps = int(round(t_max / dt))
    times = []
    dens = []
    for step in range(n_steps + 1):
        t = t_start + step * dt
        if step % sample_every == 0 or step == n_steps:
            times.append(t)
            dens.append(np.mean(np.abs(a) ** 2, axis=0))
        if step == n_steps:
            break
        v0 = _onsite(params, sites, phase(t), shift)
        v1 = _onsite(params, sites, phase(t + 0.5 * dt), shift)
        v2 = _onsite(params, sites, phase(t + dt), shift)
        _rk4_batch(a, v0, v1, v2, float(params.hopping), g, dt)
    return np.asarray(times), np.asarray(dens), a


def sample_initial(psi0_norm: np.ndarray, n_atoms: float, n_traj: int,
                   rng: np.random.Generator, mode: str = "auto") -> np.ndarray:
    """Initial classical fields, shape (n_traj, N).  See module docstring."""
    psi0 = np.asarray(psi0_norm, dtype=np.complex128)
    n = psi0.size
    eta = (rng.standard_normal((n_traj, n)) + 1j * rng.standard_normal((n_traj, n))) / 2.0
    single = np.count_nonzero(np.abs(psi0) > 1e-12) == 1
    if mode == "auto":
        mode = "fock_ring" if single else "coherent"
    if mode == "fock_ring":
        if not single:
            raise ValueError("fock_ring requires a single-site psi0")
        c = int(np.argmax(np.abs(psi0)))
        theta = rng.uniform(0.0, 2.0 * math.pi, n_traj)
        eta[:, c] = math.sqrt(n_atoms + VACUUM) * np.exp(1j * theta)
        return eta
    if mode == "coherent":
        return math.sqrt(n_atoms) * psi0[None, :] + eta
    raise ValueError(f"unknown mode {mode}")


def window_moments(density: np.ndarray, sites: np.ndarray, window: Optional[float] = None
                   ) -> Dict[str, float]:
    """Norm, mean and width of the signed density inside |j| <= window (ratios of sums)."""
    rho = np.asarray(density, dtype=float)
    if window is not None:
        m = np.abs(sites) <= window
        rho, s = rho[m], sites[m]
    else:
        s = sites
    norm = float(np.sum(rho))
    if abs(norm) < 1e-12:
        return {"norm": norm, "mean": 0.0, "width": 0.0}
    mean = float(np.sum(s * rho) / norm)
    var = float(np.sum((s - mean) ** 2 * rho) / norm)
    return {"norm": norm, "mean": mean, "width": math.sqrt(max(var, 0.0))}


def central_mass(density: np.ndarray, sites: np.ndarray, r_center: float,
                 window: Optional[float] = None) -> float:
    """Signed-density fraction inside |j|<=r_center relative to the window total."""
    rho = np.asarray(density, dtype=float)
    mask = np.ones_like(sites, dtype=bool) if window is None else (np.abs(sites) <= window)
    total = float(np.sum(rho[mask]))
    if abs(total) < 1e-12:
        return 0.0
    return float(np.sum(rho[mask & (np.abs(sites) <= r_center)]) / total)


def choose_window(mf_densities, sites: np.ndarray, tol: float = 1e-3) -> Tuple[float, float]:
    """Smallest W with mean-field mass outside |j|<=W below `tol` at every sampled time
    of every supplied run (iterable of (n_times, N) normalized-density arrays).
    Returns (W, worst outside-mass at W).  If tol cannot be met inside the lattice,
    the full half-width is returned together with its (>tol) outside mass of 0."""
    order = np.abs(sites)
    half = float(order.max())
    worst = np.zeros(int(half) + 1)
    for dens in mf_densities:
        d = np.asarray(dens, dtype=float)
        d = d / d.sum(axis=1, keepdims=True)
        for w in range(int(half) + 1):
            out = d[:, order > w].sum(axis=1).max()
            worst[w] = max(worst[w], out)
    ok = np.where(worst < tol)[0]
    w = float(ok[0]) if ok.size else half
    return w, float(worst[int(w)])


def choose_window_vs_time(mf_density_t, sites: np.ndarray, tol: float = 1e-3,
                          floor: float = 10.0) -> np.ndarray:
    """Per-sample-time analysis half-width W(t) from ONE mean-field run
    (n_times, N): smallest W >= floor with mean-field mass outside |j|<=W below tol.
    Uses mean-field data only, so the TWA estimator is never tuned on its own noise."""
    d = np.asarray(mf_density_t, dtype=float)
    d = d / d.sum(axis=1, keepdims=True)
    order = np.abs(sites)
    out = np.empty(d.shape[0])
    for i in range(d.shape[0]):
        # tail[w] = mass with |j| > w
        srt = np.argsort(order)
        cs = np.cumsum(d[i][srt])
        ok = np.where(1.0 - cs < tol)[0]
        out[i] = max(floor, order[srt][ok[0]] if ok.size else order.max())
    return out


def twa_width_curve(psi0_norm: np.ndarray, base_params: DNLSParameters, *,
                    n_atoms: float, n_traj: int, t_max: float, dt: float,
                    phase: PhaseFunction = zero_phase, sample_every: int = 20,
                    seed: int = 0, window=None,
                    mode: str = "auto", chunk: int = 200, weyl: bool = False
                    ) -> Dict[str, np.ndarray]:
    """Ensemble TWA width vs time.

    base_params.beta is the mean-field interaction; the per-atom coupling is
    beta / n_atoms.  Returns times, width, norm (~N_b), mean, density (signed
    physical density at the last sample time), and `density_t` (all sample times).
    """
    rng = np.random.default_rng(seed)
    sites = base_params.sites()
    params = replace(base_params, beta=base_params.beta / n_atoms)
    acc = None
    times = None
    done = 0
    while done < n_traj:
        b = min(chunk, n_traj - done)
        a0 = sample_initial(psi0_norm, n_atoms, b, rng, mode)
        times, d, _ = evolve_batch(a0, params, t_max, dt, phase, sample_every, weyl=weyl)
        acc = d * b if acc is None else acc + d * b
        done += b
    dens_t = acc / n_traj - VACUUM
    w_arr = np.broadcast_to(np.asarray(window if window is not None else np.nan, dtype=float),
                            (dens_t.shape[0],))
    mom = [window_moments(dens_t[i], sites, None if np.isnan(w_arr[i]) else float(w_arr[i]))
           for i in range(dens_t.shape[0])]
    return {"times": times,
            "width": np.array([m["width"] for m in mom]),
            "norm": np.array([m["norm"] for m in mom]),
            "mean": np.array([m["mean"] for m in mom]),
            "density": dens_t[-1], "density_t": dens_t}
