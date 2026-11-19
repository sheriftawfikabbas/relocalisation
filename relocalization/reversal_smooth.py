"""Smooth, bandwidth-limited phase-only reversal, and its robustness.

Same problem as reversal_phase.py (lambda=1, delta packet spread freely for T1=15,
then phase-only control for T_c, objective F = |psi_0|^2), but the control is now
smooth and band-limited:

    phi(t) = sum_{k=1}^{K} b_k sin(pi k t / T_c),   0 <= t <= T_c,

which is continuous, starts and ends at phi=0 (no jump at switch-on), and has no
spectral content above Omega_max = pi K / T_c.  We scan Omega_max, warm-starting
each K from the previous one, and then test the best protocols against
  * coloured (Ornstein-Uhlenbeck) phase noise of rms sigma and correlation time tau;
  * static calibration errors: drive-amplitude scale, global phase offset, drive
    rate (time stretch), lattice depth lambda, interaction beta, spread time T1.
The piecewise-constant optimum of reversal_phase.py (0.5-time-unit segments,
unlimited bandwidth) is tested alongside for comparison.

Run: python -m relocalization.reversal_smooth BETA [OMEGA_MAX ...]
(extra OMEGA_MAX values continue the default scan, warm-started from its last point)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from .reversal_phase import ALPHA, DT, K_FULL, K_HALF, LAM, N, OUT, SEG, T1, x

CX, SX = torch.cos(2 * math.pi * ALPHA * x), torch.sin(2 * math.pi * ALPHA * x)
TC = 60.0
N_STEPS = int(round(TC / DT))
OMEGA_MAX = (0.25, 0.5, 1.0, 2.0)
F64 = torch.float64


def propagate(psi, phi, beta, lam=LAM, record_every=0):
    """Strang split-step.  psi (N,B); phi (n_steps,B) at step midpoints; beta, lam scalars or (B,)."""
    beta = torch.as_tensor(beta, dtype=F64).reshape(-1)[None, :]
    lam = torch.as_tensor(lam, dtype=F64).reshape(-1)[None, :]
    c, s = torch.cos(phi), torch.sin(phi)
    n = phi.shape[0]
    traj = []
    psi = K_HALF @ psi
    for k in range(n):
        v = lam * (CX[:, None] * c[k][None, :] - SX[:, None] * s[k][None, :])
        psi = psi * torch.exp(-1j * DT * (v + beta * psi.abs() ** 2))
        if record_every and (k + 1) % record_every == 0:
            traj.append((K_HALF @ psi).detach()[N // 2].abs() ** 2)
        psi = (K_FULL if k < n - 1 else K_HALF) @ psi
    return (psi, torch.stack(traj)) if record_every else psi


def spread(beta, lam=LAM, t1=T1):
    psi = torch.zeros(N, 1, dtype=torch.complex128)
    psi[N // 2] = 1.0
    n1 = int(round(t1 / DT))
    with torch.no_grad():
        return propagate(psi, torch.zeros(n1, 1, dtype=F64), beta, lam)[:, 0]


def step_times(stretch=1.0):
    return (torch.arange(N_STEPS, dtype=F64) + 0.5) * DT * stretch


def sine_phase(b, stretch=1.0):
    """b (B,K) -> phi (n_steps,B)."""
    k = torch.arange(1, b.shape[1] + 1, dtype=F64)
    t = step_times(stretch).clamp(max=TC)
    return torch.sin(math.pi * t[:, None] * k[None, :] / TC) @ b.T


def piecewise_phase(phis, stretch=1.0):
    """phis (n_seg,) -> phi (n_steps,1)."""
    idx = (step_times(stretch) / SEG).long().clamp(max=phis.numel() - 1)
    return phis[idx][:, None]


def optimise(psi1, beta, K, warm=None, *, n_batch=8, iters=250, seed=0, lr=0.05):
    g = torch.Generator().manual_seed(seed + K)
    b = 2.0 / math.sqrt(K) * torch.randn(n_batch, K, generator=g, dtype=F64)
    if warm is not None:
        kw = warm.numel()
        b[: n_batch // 2] = 0.05 * b[: n_batch // 2]
        b[: n_batch // 2, :kw] += warm
    b.requires_grad_()
    psi_b = psi1[:, None].expand(N, n_batch).clone()
    opt = torch.optim.Adam([b], lr=lr)
    best_F = torch.full((n_batch,), -1.0, dtype=F64)
    best_b = b.detach().clone()
    for _ in range(iters):
        opt.zero_grad()
        F = propagate(psi_b, sine_phase(b), beta)[N // 2].abs() ** 2
        better = F.detach() > best_F
        best_F = torch.where(better, F.detach(), best_F)
        best_b[better] = b.detach()[better]
        (-torch.log(F + 1e-12).sum()).backward()
        opt.step()
    i = int(torch.argmax(best_F))
    return float(best_F[i]), best_b[i]


def ou_noise(sigma, tau, n_real, seed):
    """Stationary OU process, rms sigma, correlation time tau: (n_steps, n_real)."""
    g = torch.Generator().manual_seed(seed)
    a = math.exp(-DT / tau)
    xi = torch.randn(N_STEPS, n_real, generator=g, dtype=F64)
    out = torch.empty_like(xi)
    out[0] = xi[0]
    for k in range(1, N_STEPS):
        out[k] = a * out[k - 1] + math.sqrt(1 - a * a) * xi[k]
    return sigma * out


def robustness(phase_fn, beta, *, n_real=32):
    """phase_fn(stretch) -> phi (n_steps,1).  Returns noise and static-error tables."""
    res = {"noise": [], "static": []}
    psi1 = spread(beta)
    phi0 = phase_fn(1.0)
    with torch.no_grad():
        for tau in (0.5, 5.0):
            for sigma in (0.02, 0.05, 0.1, 0.2, 0.4):
                phi = phi0 + ou_noise(sigma, tau, n_real, seed=int(1000 * sigma + tau))
                F = propagate(psi1[:, None].expand(N, n_real).clone(), phi, beta)[N // 2].abs() ** 2
                res["noise"].append({"tau": tau, "sigma": sigma, "F_mean": float(F.mean()),
                                     "F_std": float(F.std()), "F_min": float(F.min())})
        cases = [("ideal", 0.0, dict())]
        for e in (-0.1, -0.05, -0.02, 0.02, 0.05, 0.1):
            cases += [("amplitude", e, dict(amp=1 + e)), ("rate", e / 2, dict(stretch=1 + e / 2)),
                      ("lambda", e, dict(lam=LAM * (1 + e))), ("beta", 2 * e, dict(beta=beta * (1 + 2 * e)))]
        for d in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2):
            cases.append(("offset", d, dict(offset=d)))
        for d in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
            cases.append(("T1", d, dict(t1=T1 + d)))
        psis, phis, lams, betas = [], [], [], []
        for _, _, c in cases:
            lam, bb = c.get("lam", LAM), c.get("beta", beta)
            psis.append(spread(bb, lam, c.get("t1", T1)))
            phis.append(c.get("amp", 1.0) * phase_fn(c.get("stretch", 1.0)) + c.get("offset", 0.0))
            lams.append(lam)
            betas.append(bb)
        F = propagate(torch.stack(psis, 1), torch.cat(phis, 1), torch.tensor(betas, dtype=F64),
                      torch.tensor(lams, dtype=F64))[N // 2].abs() ** 2
        for (kind, err, _), f in zip(cases, F):
            res["static"].append({"kind": kind, "error": err, "F": float(f)})
    return res


def main() -> None:
    torch.set_num_threads(5)
    beta = float(sys.argv[1])
    omegas = tuple(float(a) for a in sys.argv[2:]) or OMEGA_MAX
    tag = "" if omegas == OMEGA_MAX else "_om" + "-".join(f"{o:g}" for o in omegas)
    psi1 = spread(beta)
    out = {"beta": beta, "Tc": TC, "smooth": []}
    warm = None
    if tag:  # warm-start from the highest Omega_max of the default scan
        prev = json.loads((OUT / f"smooth_beta{beta:g}.json").read_text())["smooth"]
        warm = torch.tensor(max(prev, key=lambda r: r["Omega_max"])["b"], dtype=F64)
    for om in omegas:
        K = int(round(om * TC / math.pi))
        F, b = optimise(psi1, beta, K, warm)
        warm = b
        phi = sine_phase(b[None, :])[:, 0]
        dphi = torch.diff(phi) / DT
        out["smooth"].append({"Omega_max": om, "K": K, "F": F, "b": b.tolist(),
                              "phi_rms": float(phi.pow(2).mean().sqrt()),
                              "phi_max": float(phi.abs().max()),
                              "dphidt_max": float(dphi.abs().max())})
        print(f"beta={beta} Omega_max={om} K={K}: F={F:.3f} max|phi|={float(phi.abs().max()):.2f} "
              f"max|dphi/dt|={float(dphi.abs().max()):.2f}", flush=True)

    # robustness: smallest Omega_max reaching F>=0.9 (else the best), plus the piecewise optimum
    ok = [r for r in out["smooth"] if r["F"] >= 0.9]
    pick = ok[0] if ok else max(out["smooth"], key=lambda r: r["F"])
    out["robust_pick"] = pick["Omega_max"]
    b_pick = torch.tensor(pick["b"], dtype=F64)[None, :]
    _, traj = propagate(psi1[:, None], sine_phase(b_pick), beta, record_every=20)
    out["smooth_F_traj"] = traj[:, 0].tolist()
    out["smooth_phi"] = sine_phase(b_pick)[::20, 0].tolist()
    out["robust_smooth"] = robustness(lambda s: sine_phase(b_pick, s), beta)
    pw = json.loads((OUT / f"phase_only_beta{beta:g}.json").read_text())["phase_Tc60"]
    pw_phis = torch.tensor(pw["phis_over_pi"], dtype=F64) * math.pi
    out["robust_piecewise"] = robustness(lambda s: piecewise_phase(pw_phis, s), beta)
    (OUT / f"smooth_beta{beta:g}{tag}.json").write_text(json.dumps(out, indent=1))
    for tag in ("robust_smooth", "robust_piecewise"):
        print(tag, json.dumps([(r["tau"], r["sigma"], round(r["F_mean"], 3))
                               for r in out[tag]["noise"]]), flush=True)


if __name__ == "__main__":
    main()
