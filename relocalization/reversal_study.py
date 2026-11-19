"""Phase-only reversal of spreading: follow-up study.

Setting as in reversal_phase.py: lambda=1 Harper chain, a delta packet spreads freely
for T1 with phi=0, then a phase control phi(t) acts for T_c; the objective is the
return fidelity F = |psi_{j0}(T1+T_c)|^2 (probability back on the starting site).

Sub-studies (each writes results/relocalization/reversal_study/<name>*.json):

  ensemble  one control optimized for a whole ensemble of initial conditions
            (spread-time jitter, start-site shifts, beta miscalibration), with
            held-out members, to measure how far the control is from universal.
  scaling   minimal control time vs spread time T1 (F >= 0.9 threshold).
  bandwidth smooth band-limited controls, Omega_max scan at several lambda, compared
            with the spectral width of the Harper Hamiltonian.
  manybody  the mean-field-designed controls applied (i) in the truncated Wigner
            approximation and (ii) exactly for N_b = 2, 3 bosons; plus a control
            optimized directly on the exact N_b = 2 problem.

Run: python -m relocalization.reversal_study <study> [beta ...]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/relocalization/reversal_study"
ALPHA = (math.sqrt(5.0) - 1.0) / 2.0
DT, SEG = 0.05, 0.5
F64, C128 = torch.float64, torch.complex128
import os  # noqa: E402

DEV = torch.device(os.environ.get("REV_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
torch.set_num_threads(int(os.environ.get("REV_THREADS", "3")))


def _hop(n):
    h = np.diag(-np.ones(n - 1), 1)
    return h + h.T


class Chain:
    """Open Harper chain of N sites, centred at j=0, Strang split-step propagator."""

    def __init__(self, n, lam=1.0, dt=DT):
        self.n, self.lam, self.dt = n, lam, dt
        self.x = torch.arange(n, dtype=F64, device=DEV) - n // 2
        self.cx = torch.cos(2 * math.pi * ALPHA * self.x)
        self.sx = torch.sin(2 * math.pi * ALPHA * self.x)
        ev, u = np.linalg.eigh(_hop(n))
        u = torch.tensor(u, dtype=C128, device=DEV)
        ev = torch.tensor(ev, dtype=F64, device=DEV)
        kin = lambda t: u @ torch.diag(torch.exp(-1j * t * ev)) @ u.conj().T  # noqa: E731
        self.kh, self.kf = kin(0.5 * dt), kin(dt)

    def site(self, j):
        return self.n // 2 + j

    def evolve(self, psi, phi, beta, lam=None):
        """psi (N,B); phi (n_steps,) or (n_steps,B) at step midpoints; beta, lam scalar or (B,)."""
        lam = self.lam if lam is None else lam
        beta = torch.as_tensor(beta, dtype=F64, device=DEV).reshape(1, -1)
        lam = torch.as_tensor(lam, dtype=F64, device=DEV).reshape(1, -1)
        if phi.dim() == 1:
            phi = phi[:, None]
        c, s = torch.cos(phi), torch.sin(phi)
        n = phi.shape[0]
        if n == 0:
            return psi
        psi = self.kh @ psi
        for k in range(n):
            v = lam * (self.cx[:, None] * c[k][None, :] - self.sx[:, None] * s[k][None, :])
            psi = psi * torch.exp(-1j * self.dt * (v + beta * psi.abs() ** 2))
            psi = (self.kf if k < n - 1 else self.kh) @ psi
        return psi

    def spread(self, betas, t1s, j0s, lams=None):
        """Batch of delta packets at j0, free (phi=0) evolution for t1 each; returns (N,B)."""
        b = len(betas)
        lams = [self.lam] * b if lams is None else lams
        out = torch.zeros(self.n, b, dtype=C128, device=DEV)
        with torch.no_grad():
            for i, (be, t1, j0, la) in enumerate(zip(betas, t1s, j0s, lams)):
                psi = torch.zeros(self.n, 1, dtype=C128, device=DEV)
                psi[self.site(j0)] = 1.0
                n1 = int(round(t1 / self.dt))
                out[:, i] = self.evolve(psi, torch.zeros(n1, dtype=F64, device=DEV), be, la)[:, 0]
        return out


def pw_steps(phis, dt=DT, seg=SEG):
    """Piecewise-constant segments (n_seg,) or (n_seg,B) -> per-step phase."""
    return torch.repeat_interleave(phis, int(round(seg / dt)), dim=0)


def metrics(chain, psi, j0s):
    p = psi.abs() ** 2
    idx = torch.tensor([chain.site(j) for j in j0s], device=DEV)
    F = p[idx, torch.arange(p.shape[1], device=DEV)]
    edge = p[(chain.x.abs() > chain.n // 2 - 20)].sum(0)
    return F, edge


def optimise_shared(chain, psi0, j0s, betas, tc, *, n_starts=4, iters=300, lr=0.08, seed=0,
                    init=None, lams=None):
    """One piecewise phase shared by all ensemble members; n_starts random restarts in batch.

    Objective per start: mean_i -log F_i over the members.  Returns best (phis, F_members).
    """
    m = psi0.shape[1]
    n_seg = int(round(tc / SEG))
    g = torch.Generator(device="cpu").manual_seed(seed)
    phis = 2 * math.pi * torch.rand(n_seg, n_starts, generator=g, dtype=F64)
    if init is not None:
        phis[:, 0] = init.cpu()
        phis[:, 1:] = init.cpu()[:, None] + 0.3 * torch.randn(n_seg, n_starts - 1, generator=g, dtype=F64)
    phis = phis.to(DEV).requires_grad_()
    psi_b = psi0.repeat(1, n_starts)                          # column = start*m + member
    beta_b = torch.tensor(list(betas) * n_starts, dtype=F64, device=DEV)
    lam_b = None if lams is None else torch.tensor(list(lams) * n_starts, dtype=F64, device=DEV)
    idx = torch.tensor([chain.site(j) for j in j0s] * n_starts, device=DEV)
    cols = torch.arange(m * n_starts, device=DEV)
    opt = torch.optim.Adam([phis], lr=lr)
    best_obj = torch.full((n_starts,), float("inf"), dtype=F64)
    best_phi = phis.detach().clone()
    best_F = torch.zeros(n_starts, m, dtype=F64)
    for _ in range(iters):
        opt.zero_grad()
        phi_steps = pw_steps(phis).repeat_interleave(m, dim=1)
        psi = chain.evolve(psi_b, phi_steps, beta_b, lam_b)
        F = (psi.abs() ** 2)[idx, cols].reshape(n_starts, m)
        obj = (-torch.log(F + 1e-12)).mean(1)
        better = obj.detach().cpu() < best_obj
        best_obj = torch.where(better, obj.detach().cpu(), best_obj)
        best_phi[:, better.to(DEV)] = phis.detach()[:, better.to(DEV)]
        best_F[better] = F.detach().cpu()[better]
        obj.sum().backward()
        opt.step()
    i = int(torch.argmin(best_obj))
    return best_phi[:, i].detach(), best_F[i]


def evaluate(chain, phis, betas, t1s, j0s, lams=None):
    with torch.no_grad():
        psi0 = chain.spread(betas, t1s, j0s, lams)
        psi = chain.evolve(psi0, pw_steps(phis), torch.tensor(betas, dtype=F64, device=DEV),
                           None if lams is None else torch.tensor(lams, dtype=F64, device=DEV))
        F, edge = metrics(chain, psi, j0s)
    return F.cpu().numpy(), float(edge.max())


def _save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- ensemble
def ensemble_members(kind, width, beta, t1=15.0):
    """(train, test) lists of (beta, t1, j0)."""
    if kind == "T1":
        tr = [(beta, t1 + d, 0) for d in np.linspace(-width, width, 5)]
        te = [(beta, t1 + d, 0) for d in (np.linspace(-width, width, 5)[:-1] + width / 4)]
    elif kind == "site":
        tr = [(beta, t1, j) for j in range(-int(width), int(width) + 1)]
        te = [(beta, t1, j) for j in (-int(width) - 1, int(width) + 1)]
    elif kind == "beta":
        tr = [(beta * (1 + d), t1, 0) for d in np.linspace(-width, width, 5)]
        te = [(beta * (1 + d), t1, 0) for d in (np.linspace(-width, width, 5)[:-1] + width / 4)]
    else:
        raise ValueError(kind)
    return tr, te


def run_ensemble(betas):
    chain = Chain(301)
    plan = [("T1", w) for w in (0.25, 0.5, 1.0, 2.0)] + [("site", w) for w in (1, 2, 4)] + \
           [("beta", w) for w in (0.1, 0.25, 0.5)]
    for beta in betas:
        rows = []
        for tc in (60.0, 120.0):
            for kind, w in plan:
                if kind == "beta" and beta == 0.0:
                    continue
                tr, te = ensemble_members(kind, w, beta)
                b_tr, t_tr, j_tr = map(list, zip(*tr))
                psi0 = chain.spread(b_tr, t_tr, j_tr)
                phis, F_tr = optimise_shared(chain, psi0, j_tr, b_tr, tc)
                F_tr2, edge = evaluate(chain, phis, b_tr, t_tr, j_tr)
                b_te, t_te, j_te = map(list, zip(*te))
                F_te, edge_te = evaluate(chain, phis, b_te, t_te, j_te)
                row = {"beta": beta, "Tc": tc, "kind": kind, "width": w,
                       "train": [list(m) for m in tr], "F_train": F_tr2.tolist(),
                       "test": [list(m) for m in te], "F_test": F_te.tolist(),
                       "F_train_mean": float(F_tr2.mean()), "F_train_min": float(F_tr2.min()),
                       "F_test_mean": float(F_te.mean()), "F_test_min": float(F_te.min()),
                       "edge_max": max(edge, edge_te), "phis": phis.cpu().tolist()}
                rows.append(row)
                _log(f"ensemble beta={beta} Tc={tc:g} {kind}+-{w}: train mean {row['F_train_mean']:.3f} "
                     f"min {row['F_train_min']:.3f} | test mean {row['F_test_mean']:.3f} "
                     f"min {row['F_test_min']:.3f} edge {row['edge_max']:.1e}")
                _save(f"ensemble_beta{beta:g}.json", rows)


# --------------------------------------------------------------------------- scaling
def run_scaling(betas):
    chain = Chain(1001)
    for beta in betas:
        rows = []
        for t1 in (15.0, 30.0, 60.0):
            psi0 = chain.spread([beta], [t1], [0])
            F0, _ = metrics(chain, psi0, [0])
            init = None
            for tc in (15.0, 30.0, 60.0, 120.0, 240.0):
                if init is not None:  # warm start: previous optimum, then zeros (phi=0 continuation)
                    init = torch.cat([init, torch.zeros(int(round(tc / SEG)) - init.numel(),
                                                        dtype=F64, device=DEV)])
                phis, _ = optimise_shared(chain, psi0, [0], [beta], tc, iters=400, init=init)
                F, edge = evaluate(chain, phis, [beta], [t1], [0])
                init = phis
                rows.append({"beta": beta, "T1": t1, "Tc": tc, "F_spread": float(F0[0]),
                             "F": float(F[0]), "edge": edge, "phis": phis.cpu().tolist()})
                _log(f"scaling beta={beta} T1={t1:g} Tc={tc:g}: F={F[0]:.3f} (spread {float(F0[0]):.3f}) "
                     f"edge {edge:.1e}")
                _save(f"scaling_beta{beta:g}.json", rows)
                if F[0] >= 0.97:
                    break


# --------------------------------------------------------------------------- bandwidth
def sine_phase(b, tc, dt=DT):
    """b (K,B) -> phi (n_steps,B), phi = sum_k b_k sin(pi k t / tc), zero at both ends."""
    n = int(round(tc / dt))
    t = (torch.arange(n, dtype=F64, device=DEV) + 0.5) * dt
    k = torch.arange(1, b.shape[0] + 1, dtype=F64, device=DEV)
    return torch.sin(math.pi * t[:, None] * k[None, :] / tc) @ b


def optimise_smooth(chain, psi0, beta, tc, K, warm=None, *, n_starts=8, iters=300, lr=0.05, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed + K)
    b = 2.0 / math.sqrt(K) * torch.randn(K, n_starts, generator=g, dtype=F64)
    if warm is not None:
        kw = warm.numel()
        b[:, : n_starts // 2] *= 0.05
        b[:kw, : n_starts // 2] += warm.cpu()[:, None]
    b = b.to(DEV).requires_grad_()
    psi_b = psi0.repeat(1, n_starts)
    i0 = chain.site(0)
    opt = torch.optim.Adam([b], lr=lr)
    best = torch.full((n_starts,), -1.0, dtype=F64)
    best_b = b.detach().clone()
    for _ in range(iters):
        opt.zero_grad()
        F = chain.evolve(psi_b, sine_phase(b, tc), beta)[i0].abs() ** 2
        better = F.detach().cpu() > best
        best = torch.where(better, F.detach().cpu(), best)
        best_b[:, better.to(DEV)] = b.detach()[:, better.to(DEV)]
        (-torch.log(F + 1e-12)).sum().backward()
        opt.step()
    i = int(torch.argmax(best))
    return float(best[i]), best_b[:, i]


def spectral_scales(chain, psi0):
    """Spectral width of H(phi=0) and the population-weighted energy spread of psi0."""
    h = _hop(chain.n) + np.diag(chain.lam * chain.cx.cpu().numpy())
    ev, u = np.linalg.eigh(h)
    w = np.abs(u.T @ psi0[:, 0].cpu().numpy()) ** 2
    occ = ev[w > 1e-6 * w.max()]
    return {"spec_width": float(ev.max() - ev.min()),
            "occupied_width": float(occ.max() - occ.min()),
            "energy_std": float(math.sqrt(max((w * ev ** 2).sum() - (w * ev).sum() ** 2, 0.0)))}


def run_bandwidth(betas):
    tc = 60.0
    omegas = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    for beta in betas:
        rows = []
        for lam in (0.5, 1.0, 1.5):
            chain = Chain(301, lam=lam)
            psi0 = chain.spread([beta], [15.0], [0])
            scales = spectral_scales(chain, psi0)
            warm = None
            for om in omegas:
                K = int(round(om * tc / math.pi))
                F, b = optimise_smooth(chain, psi0, beta, tc, K, warm)
                warm = b
                rows.append({"beta": beta, "lambda": lam, "Tc": tc, "Omega_max": om, "K": K, "F": F,
                             "b": b.cpu().tolist(), **scales})
                _log(f"bandwidth beta={beta} lam={lam} Om={om}: F={F:.3f}  "
                     f"(spec width {scales['spec_width']:.2f}, occupied {scales['occupied_width']:.2f})")
                _save(f"bandwidth_beta{beta:g}.json", rows)


# --------------------------------------------------------------------------- many-body
def twa_apply(chain, phis, beta, n_atoms, n_traj=400, t1=15.0, seed=1):
    """Fock-ring TWA: classical fields alpha = sqrt(N_b) psi, nonlinearity beta |psi|^2."""
    from .twa import sample_initial
    rng = np.random.default_rng(seed)
    psi0 = np.zeros(chain.n)
    psi0[chain.site(0)] = 1.0
    a0 = sample_initial(psi0, n_atoms, n_traj, rng, mode="fock_ring") / math.sqrt(n_atoms)
    psi = torch.tensor(a0.T, dtype=C128, device=DEV)
    with torch.no_grad():
        psi = chain.evolve(psi, torch.zeros(int(round(t1 / DT)), dtype=F64, device=DEV), beta)
        rho_spread = (psi.abs() ** 2).mean(1) - 0.5 / n_atoms
        psi = chain.evolve(psi, pw_steps(phis), beta)
        rho = (psi.abs() ** 2).mean(1) - 0.5 / n_atoms
    i0 = chain.site(0)
    se = float((psi[i0].abs() ** 2).std() / math.sqrt(n_traj))
    return float(rho_spread[i0]), float(rho[i0]), se


class FewBody:
    """Exact N_b = 2 or 3 bosons, first quantization on L sites, Bose-Hubbard on-site U."""

    def __init__(self, nb, L, lam=1.0, dt=DT):
        self.nb, self.L, self.dt, self.lam = nb, L, dt, lam
        x = torch.arange(L, dtype=F64, device=DEV) - L // 2
        self.cx, self.sx = torch.cos(2 * math.pi * ALPHA * x), torch.sin(2 * math.pi * ALPHA * x)
        ev, u = np.linalg.eigh(_hop(L))
        u = torch.tensor(u, dtype=C128, device=DEV)
        ev = torch.tensor(ev, dtype=F64, device=DEV)
        kin = lambda t: u @ torch.diag(torch.exp(-1j * t * ev)) @ u.conj().T  # noqa: E731
        self.kh, self.kf = kin(0.5 * dt), kin(dt)
        eye = torch.eye(L, dtype=F64, device=DEV)
        if nb == 2:
            self.dbl = eye
        else:
            d = torch.zeros(L, L, L, dtype=F64, device=DEV)
            d += eye[:, :, None] + eye[:, None, :] + eye[None, :, :]
            self.dbl = d
        self.c = L // 2

    def kin(self, psi, k):
        # apply single-particle propagator k on each coordinate in turn; psi (L,..,L,B)
        for ax in range(self.nb):
            psi = torch.movedim(torch.tensordot(k, psi, dims=([1], [ax])), 0, ax)
        return psi

    def pot(self, c, s):
        v1 = self.lam * (self.cx[:, None] * c[None, :] - self.sx[:, None] * s[None, :])  # (L,B)
        if self.nb == 2:
            return v1[:, None, :] + v1[None, :, :]
        return v1[:, None, None, :] + v1[None, :, None, :] + v1[None, None, :, :]

    def evolve(self, psi, phi, U):
        if phi.dim() == 1:
            phi = phi[:, None].expand(-1, psi.shape[-1])
        c, s = torch.cos(phi), torch.sin(phi)
        n = phi.shape[0]
        psi = self.kin(psi, self.kh)
        inter = (U * self.dbl)[..., None]
        for k in range(n):
            psi = psi * torch.exp(-1j * self.dt * (self.pot(c[k], s[k]) + inter))
            psi = self.kin(psi, self.kf if k < n - 1 else self.kh)
        return psi

    def initial(self, b=1):
        shape = (self.L,) * self.nb + (b,)
        psi = torch.zeros(shape, dtype=C128, device=DEV)
        psi[(self.c,) * self.nb] = 1.0
        return psi

    def observables(self, psi):
        """P(all bosons on the start site), <n_0>/N_b, and the mass within 10 sites of an edge."""
        p = psi.abs() ** 2
        F = p[(self.c,) * self.nb]
        marg = p.sum(dim=tuple(range(1, self.nb)))       # (L,B) single-coordinate marginal
        edge = marg[:10].sum(0) + marg[-10:].sum(0)
        return F, marg[self.c], edge


def run_manybody(betas):
    chain = Chain(301)
    rows = []
    for beta in betas:
        src = json.loads((ROOT / f"results/relocalization/reversal/phase_only_beta{beta:g}.json").read_text())
        ens = OUT / f"ensemble_beta{beta:g}.json"
        controls = {"single_Tc60": torch.tensor(src["phase_Tc60"]["phis_over_pi"], dtype=F64) * math.pi}
        if ens.exists():
            for r in json.loads(ens.read_text()):
                if r["kind"] == "T1" and r["width"] == 1.0 and r["Tc"] == 120.0:
                    controls["T1robust_Tc120"] = torch.tensor(r["phis"], dtype=F64)
        for name, phis in controls.items():
            phis = phis.to(DEV)
            F_mf, _ = evaluate(chain, phis, [beta], [15.0], [0])
            base = {"beta": beta, "control": name, "F_mf": float(F_mf[0])}
            for nat in (10, 30, 100, 1000):
                r_s, r_f, se = twa_apply(chain, phis, beta, nat)
                rows.append({**base, "method": "TWA", "N_b": nat, "n0_spread": r_s, "n0_final": r_f, "se": se})
                _log(f"manybody beta={beta} {name} TWA N_b={nat}: n0/N_b {r_s:.3f} -> {r_f:.3f} (+-{se:.3f}); "
                     f"MF F={F_mf[0]:.3f}")
                _save("manybody.json", rows)
            for nb, L in ((2, 201), (3, 141)):
                fb = FewBody(nb, L)
                U = beta / nb                                 # beta = U N_b, as in the manuscript
                with torch.no_grad():
                    psi = fb.evolve(fb.initial(), torch.zeros(int(round(15.0 / DT)), dtype=F64, device=DEV), U)
                    F_s, n0_s, _ = fb.observables(psi)
                    psi = fb.evolve(psi, pw_steps(phis), U)
                    F_f, n0_f, edge = fb.observables(psi)
                    edge = float(edge[0])
                rows.append({**base, "method": "exact", "N_b": nb, "L": L, "U": U,
                             "P_all_spread": float(F_s[0]), "P_all_final": float(F_f[0]),
                             "n0_spread": float(n0_s[0]), "n0_final": float(n0_f[0]), "edge": edge,
                             "F_mf_pow_Nb": float(F_mf[0]) ** nb})
                _log(f"manybody beta={beta} {name} exact N_b={nb} (U={U:.3f}, L={L}): P(all home) "
                     f"{float(F_s[0]):.3f} -> {float(F_f[0]):.3f}; <n0>/N_b {float(n0_s[0]):.3f} -> "
                     f"{float(n0_f[0]):.3f}; MF F^Nb={float(F_mf[0]) ** nb:.3f}; edge {edge:.1e}")
                _save("manybody.json", rows)
        # direct optimization on the exact two-boson problem
        fb = FewBody(2, 201)
        U = beta / 2
        with torch.no_grad():
            psi_s = fb.evolve(fb.initial(), torch.zeros(int(round(15.0 / DT)), dtype=F64, device=DEV), U)
        for tc in (60.0,):
            n_seg, n_starts = int(round(tc / SEG)), 4
            init = controls["single_Tc60"].to(DEV)
            g = torch.Generator(device="cpu").manual_seed(0)
            phis = init[:, None].repeat(1, n_starts).cpu()
            phis[:, 1:] += 0.3 * torch.randn(n_seg, n_starts - 1, generator=g, dtype=F64)
            phis = phis.to(DEV).requires_grad_()
            opt = torch.optim.Adam([phis], lr=0.05)
            psi_b = psi_s.repeat(1, 1, n_starts)
            best, best_phi = -1.0, None
            for it in range(300):
                opt.zero_grad()
                P, _, _ = fb.observables(fb.evolve(psi_b, pw_steps(phis), U))
                j = int(torch.argmax(P.detach()))
                if float(P[j]) > best:
                    best, best_phi = float(P[j]), phis.detach()[:, j].clone()
                (-torch.log(P + 1e-12)).sum().backward()
                opt.step()
                if it % 50 == 0:
                    _log(f"  exact N_b=2 direct opt beta={beta} it {it}: best P(all home) {best:.3f}")
            rows.append({"beta": beta, "control": "exact_Nb2_optimized", "method": "exact_opt", "N_b": 2,
                         "U": U, "Tc": tc, "P_all_final": best, "phis": best_phi.cpu().tolist()})
            _log(f"manybody beta={beta} exact N_b=2 optimized directly: P(all home) = {best:.3f}")
            _save("manybody.json", rows)


def main() -> None:
    study = sys.argv[1]
    betas = [float(a) for a in sys.argv[2:]] or [0.0, 1.0]
    _log(f"device {DEV}; study {study}; betas {betas}")
    {"ensemble": run_ensemble, "scaling": run_scaling, "bandwidth": run_bandwidth,
     "manybody": run_manybody}[study](betas)
    _log("done")


if __name__ == "__main__":
    main()
