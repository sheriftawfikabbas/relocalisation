"""Completion runs for the open items named in the reversal manuscript.

Same model, propagator and optimizer as reversal_study.py.  Sub-studies (each writes
results/relocalization/reversal_completion/<name>.json):

  bw_lam     lambda=0.5 bandwidth run to convergence (Omega_max up to 12, 1500 iters),
             extra depths lambda=0.75, 1.25, and the spectral weight of the control
             operator dH/dphi between occupied eigenstates (why lambda matters).
  scaling120 control-time scaling extended to T1=120 (N=1501).
  ens_beta   beta(1 +- 50%) ensemble with a longer control and more effort.
  site_cross start-site ensemble (w=4) at intermediate beta, to locate the crossover.
  fewbody    controls optimized for finite N_b: exact N_b=2,3 and TWA N_b=10,30,100.
  states     other initial states and the localized regime (lambda=2.5).
  lie        Lie rank on larger chains.
  starts     32 random starts for the headline controls (spread of local optima).
  modelfree  gradient-free (SPSA) learning from noisy measured F only.

Run: python -m relocalization.reversal_completion <study>
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

from .reversal_study import (
    DEV, F64, C128, DT, SEG, ROOT, Chain, _hop, evaluate, metrics, optimise_shared,
    optimise_smooth, pw_steps, sine_phase, spectral_scales,
)

import os  # noqa: E402

OUT = ROOT / os.environ.get("REV_OUT", "results/relocalization/reversal_completion")


def _save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- bw_lam
def control_spectrum(chain, psi0, beta=0.0):
    """Spectral weight of the control generator dH/dphi = -lam(sin phi A + cos phi B) at phi=0,
    i.e. lam*B, between eigenstates of H(phi=0) weighted by the occupation of psi0.

    Returns frequencies omega_mn = |E_m-E_n| and weights p_n |<m|lam B|n>|^2 (linear, beta=0).
    """
    lam = chain.lam
    x = chain.x.cpu().numpy()
    h = _hop(chain.n) + np.diag(lam * np.cos(2 * math.pi * ((math.sqrt(5) - 1) / 2) * x))
    ev, u = np.linalg.eigh(h)
    b = u.T @ np.diag(lam * np.sin(2 * math.pi * ((math.sqrt(5) - 1) / 2) * x)) @ u
    p = np.abs(u.T @ psi0[:, 0].cpu().numpy()) ** 2
    w = (b ** 2) * p[None, :]
    om = np.abs(ev[:, None] - ev[None, :])
    edges = np.arange(0, 6.01, 0.25)
    hist, _ = np.histogram(om.ravel(), bins=edges, weights=w.ravel())
    cum = np.cumsum(hist) / hist.sum()
    return {"edges": edges.tolist(), "weight": hist.tolist(), "cum": cum.tolist(),
            "total": float(w.sum()),
            "omega_half": float(edges[1:][np.searchsorted(cum, 0.5)]),
            "omega_90": float(edges[1:][np.searchsorted(cum, 0.9)])}


def run_bw_lam():
    tc, beta = 60.0, 1.0
    rows = []
    prev = json.loads((ROOT / "results/relocalization/reversal_study/bandwidth_beta1.json").read_text())
    for lam in (0.5, 0.75, 1.0, 1.25, 1.5):
        chain = Chain(301, lam=lam)
        psi0 = chain.spread([beta], [15.0], [0])
        spec = control_spectrum(chain, psi0)
        scales = spectral_scales(chain, psi0)
        rows.append({"lambda": lam, "kind": "spectrum", **spec, **scales})
        _log(f"lam={lam}: control weight total {spec['total']:.3f}, omega_50 {spec['omega_half']:.2f}, "
             f"omega_90 {spec['omega_90']:.2f}")
        _save("bw_lam.json", rows)
        if lam in (1.0, 1.5):
            continue  # already in reversal_study
        if lam == 0.5:
            omegas, iters = (8.0, 10.0, 12.0), 1500
            warm = torch.tensor([r for r in prev if r["lambda"] == 0.5 and r["Omega_max"] == 6.0][0]["b"],
                                dtype=F64)
        else:
            omegas, iters, warm = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0), 300, None
        for om in omegas:
            K = int(round(om * tc / math.pi))
            F, b = optimise_smooth(chain, psi0, beta, tc, K, warm, iters=iters)
            warm = b
            rows.append({"lambda": lam, "kind": "smooth", "Omega_max": om, "K": K, "F": F, "iters": iters,
                         "b": b.cpu().tolist()})
            _log(f"bw lam={lam} Om={om} iters={iters}: F={F:.3f}")
            _save("bw_lam.json", rows)


# --------------------------------------------------------------------------- scaling120
def run_scaling120():
    chain, beta, t1 = Chain(1501), 1.0, 120.0
    psi0 = chain.spread([beta], [t1], [0])
    F0, _ = metrics(chain, psi0, [0])
    rows, init = [], None
    for tc in (120.0, 240.0, 480.0):
        if init is not None:
            init = torch.cat([init, torch.zeros(int(round(tc / SEG)) - init.numel(), dtype=F64, device=DEV)])
        phis, _ = optimise_shared(chain, psi0, [0], [beta], tc, iters=300, init=init)
        F, edge = evaluate(chain, phis, [beta], [t1], [0])
        init = phis
        rows.append({"beta": beta, "T1": t1, "Tc": tc, "F_spread": float(F0[0]), "F": float(F[0]),
                     "edge": edge, "phis": phis.cpu().tolist()})
        _log(f"scaling T1=120 Tc={tc:g}: F={F[0]:.3f} (spread {float(F0[0]):.3f}) edge {edge:.1e}")
        _save("scaling120.json", rows)


# --------------------------------------------------------------------------- ensembles
def _ens(chain, tr, te, tc, n_starts, iters, seed=0):
    b_tr, t_tr, j_tr = map(list, zip(*tr))
    psi0 = chain.spread(b_tr, t_tr, j_tr)
    phis, _ = optimise_shared(chain, psi0, j_tr, b_tr, tc, n_starts=n_starts, iters=iters, seed=seed)
    F_tr, _ = evaluate(chain, phis, b_tr, t_tr, j_tr)
    b_te, t_te, j_te = map(list, zip(*te))
    F_te, _ = evaluate(chain, phis, b_te, t_te, j_te)
    return phis, F_tr, F_te


def run_ens_beta():
    chain, beta = Chain(301), 1.0
    rows = []
    for n_mem, tc, iters in ((5, 120.0, 600), (9, 120.0, 600), (9, 240.0, 600)):
        d = np.linspace(-0.5, 0.5, n_mem)
        tr = [(beta * (1 + x), 15.0, 0) for x in d]
        te = [(beta * (1 + x), 15.0, 0) for x in (d[:-1] + (d[1] - d[0]) / 2)]
        phis, F_tr, F_te = _ens(chain, tr, te, tc, 8, iters)
        rows.append({"n_members": n_mem, "Tc": tc, "iters": iters, "train": [m[0] for m in tr],
                     "F_train": F_tr.tolist(), "test": [m[0] for m in te], "F_test": F_te.tolist(),
                     "phis": phis.cpu().tolist()})
        _log(f"ens beta+-50% members={n_mem} Tc={tc:g}: train {F_tr.mean():.3f} (min {F_tr.min():.3f}) "
             f"test {F_te.mean():.3f} (min {F_te.min():.3f})")
        _save("ens_beta.json", rows)


def run_site_cross():
    chain, w = Chain(301), 4
    rows = []
    for beta in (0.0, 0.1, 0.25, 0.5, 1.0):
        tr = [(beta, 15.0, j) for j in range(-w, w + 1)]
        te = [(beta, 15.0, j) for j in (-w - 1, w + 1, -w - 2, w + 2)]
        phis, F_tr, F_te = _ens(chain, tr, te, 120.0, 4, 300)
        rows.append({"beta": beta, "width": w, "F_train": F_tr.tolist(), "F_test": F_te.tolist(),
                     "test_sites": [m[2] for m in te], "phis": phis.cpu().tolist()})
        _log(f"site w={w} beta={beta}: train {F_tr.mean():.3f} test {F_te.mean():.3f} {np.round(F_te, 3)}")
        _save("site_cross.json", rows)


# --------------------------------------------------------------------------- few-body controls
def _ckpt_evolve(fn, psi, phi, chunk=100):
    """Gradient-checkpointed evolution in chunks of steps (Strang halves merge exactly)."""
    from torch.utils.checkpoint import checkpoint
    for k in range(0, phi.shape[0], chunk):
        psi = checkpoint(fn, psi, phi[k:k + chunk], use_reentrant=False)
    return psi


def _mf_control(beta):
    src = json.loads((ROOT / f"results/relocalization/reversal/phase_only_beta{beta:g}.json").read_text())
    return torch.tensor(src["phase_Tc60"]["phis_over_pi"], dtype=F64, device=DEV) * math.pi


def _adam(phis, loss_fn, iters, lr):
    phis = phis.clone().requires_grad_()
    opt = torch.optim.Adam([phis], lr=lr)
    best, best_phi = -1.0, phis.detach()[:, 0].clone()
    for it in range(iters):
        opt.zero_grad()
        val = loss_fn(phis)                       # (n_starts,) figure of merit, maximize
        j = int(torch.argmax(val.detach()))
        if float(val[j]) > best:
            best, best_phi = float(val[j]), phis.detach()[:, j].clone()
        (-torch.log(val + 1e-12)).sum().backward()
        opt.step()
        if it % 50 == 0:
            _log(f"    it {it}: best {best:.4f}")
    return best, best_phi


def run_fewbody():
    from .reversal_study import FewBody, twa_apply
    from .twa import sample_initial
    beta = 1.0
    rows = []
    mf = _mf_control(beta)
    parts = os.environ.get("REV_FB_PARTS", "exact,twa").split(",")
    fname = "fewbody.json" if len(parts) == 2 else f"fewbody_{parts[0]}.json"
    # exact two bosons: P(all home), control optimized on the exact state, Tc = 60, 120, 240
    fb = FewBody(2, 201)
    U = beta / 2
    with torch.no_grad():
        psi_s = fb.evolve(fb.initial(), torch.zeros(int(round(15.0 / DT)), dtype=F64, device=DEV), U)
    init = mf
    for tc in ((60.0, 120.0, 240.0) if "exact" in parts else ()):
        n_seg = int(round(tc / SEG))
        init = torch.cat([init, torch.zeros(n_seg - init.numel(), dtype=F64, device=DEV)])
        g = torch.Generator(device="cpu").manual_seed(1)
        phis = init[:, None].repeat(1, 2).cpu()
        phis[:, 1] += 0.3 * torch.randn(n_seg, generator=g, dtype=F64)
        psi_b = psi_s.repeat(1, 1, 2)

        def merit(p):
            out = _ckpt_evolve(lambda s, ph: fb.evolve(s, ph, U), psi_b, pw_steps(p))
            return fb.observables(out)[0]
        best, init = _adam(phis.to(DEV), merit, 300, 0.05)
        with torch.no_grad():
            P, n0, edge = fb.observables(fb.evolve(psi_s, pw_steps(init), U))
        rows.append({"method": "exact_opt", "N_b": 2, "U": U, "beta": beta, "Tc": tc,
                     "P_all": float(P[0]), "n0": float(n0[0]), "edge": float(edge[0]),
                     "phis": init.cpu().tolist()})
        _log(f"fewbody exact N_b=2 Tc={tc:g}: P_all={float(P[0]):.3f} <n0>/N_b={float(n0[0]):.3f} "
             f"edge {float(edge[0]):.1e}")
        _save(fname, rows)
    # TWA-optimized controls for N_b = 10, 30, 100 (objective <n_0>/N_b), fresh trajectories to test
    chain = Chain(301)
    for nat in ((10, 30, 100) if "twa" in parts else ()):
        rng = np.random.default_rng(7)
        psi0 = np.zeros(chain.n)
        psi0[chain.site(0)] = 1.0
        a0 = sample_initial(psi0, nat, 200, rng, mode="fock_ring") / math.sqrt(nat)
        psi = torch.tensor(a0.T, dtype=C128, device=DEV)
        with torch.no_grad():
            psi = chain.evolve(psi, torch.zeros(int(round(15.0 / DT)), dtype=F64, device=DEV), beta)
        i0 = chain.site(0)
        for tc in (60.0, 120.0):
            n_seg = int(round(tc / SEG))
            init = torch.cat([mf, torch.zeros(n_seg - mf.numel(), dtype=F64, device=DEV)])

            def merit(p):
                ph = pw_steps(p[:, 0])
                out = _ckpt_evolve(lambda s, q: chain.evolve(s, q, beta), psi, ph)
                return ((out[i0].abs() ** 2).mean() - 0.5 / nat).clamp(min=1e-6).reshape(1)
            best, phis = _adam(init[:, None], merit, 200, 0.05)
            _, n0_mf, se_mf = twa_apply(chain, init, beta, nat, n_traj=1000, seed=11)
            _, n0, se = twa_apply(chain, phis, beta, nat, n_traj=1000, seed=11)
            F_mf, _ = evaluate(chain, phis, [beta], [15.0], [0])
            rows.append({"method": "TWA_opt", "N_b": nat, "beta": beta, "Tc": tc, "n0_train": best,
                         "n0_test": n0, "se": se, "n0_mf_control": n0_mf, "F_meanfield_of_new": float(F_mf[0]),
                         "phis": phis.cpu().tolist()})
            _log(f"fewbody TWA N_b={nat} Tc={tc:g}: <n0>/N_b {n0_mf:.3f} (MF control) -> {n0:.3f} "
                 f"(+-{se:.3f}, train {best:.3f}); new control in MF: {float(F_mf[0]):.3f}")
            _save(fname, rows)


# --------------------------------------------------------------------------- other states, regimes
def _opt_overlap(chain, psi_init, psi_spread, beta, tc, n_starts=8, iters=300, seed=0):
    """Maximize |<psi_init|psi(T1+Tc)>|^2 over a piecewise phase."""
    n_seg = int(round(tc / SEG))
    g = torch.Generator(device="cpu").manual_seed(seed)
    phis = (2 * math.pi * torch.rand(n_seg, n_starts, generator=g, dtype=F64)).to(DEV)
    psi_b = psi_spread.repeat(1, n_starts)
    tgt = psi_init.conj()

    def merit(p):
        return (tgt * chain.evolve(psi_b, pw_steps(p), beta)).sum(0).abs() ** 2
    return _adam(phis, merit, iters, 0.08)


def run_states():
    rows = []
    beta = 1.0
    chain = Chain(301)
    x = chain.x
    states = {
        "gaussian_s2": torch.exp(-x ** 2 / (2 * 2.0 ** 2)),
        "gaussian_s2_k0.5": torch.exp(-x ** 2 / (2 * 2.0 ** 2)) * torch.exp(0.5j * x),
        "two_site": ((x == 0) | (x == 1)).to(F64),
        "site_j0=7": (x == 7).to(F64),
    }
    for name, s in states.items():
        s = s.to(C128)
        s = (s / s.abs().pow(2).sum().sqrt())[:, None]
        with torch.no_grad():
            sp = chain.evolve(s, torch.zeros(int(round(15.0 / DT)), dtype=F64, device=DEV), beta)
        f0 = float((s.conj() * sp).sum().abs() ** 2)
        F, _ = _opt_overlap(chain, s, sp, beta, 60.0)
        rows.append({"state": name, "lambda": 1.0, "beta": beta, "T1": 15.0, "Tc": 60.0, "F_spread": f0, "F": F})
        _log(f"states {name}: overlap after spread {f0:.3f} -> controlled {F:.3f}")
        _save("states.json", rows)
    # critical and localized regimes, site packet; long spread time at lambda=2.5 (subdiffusive)
    for lam, t1 in ((2.0, 15.0), (2.5, 15.0), (2.5, 200.0)):
        chain = Chain(301, lam=lam)
        for beta in (0.0, 1.0, 2.0):
            psi0 = chain.spread([beta], [t1], [0])
            F0, _ = metrics(chain, psi0, [0])
            w = float(torch.sqrt(((psi0[:, 0].abs() ** 2) * chain.x ** 2).sum()))
            phis, _ = optimise_shared(chain, psi0, [0], [beta], 60.0, n_starts=8)
            F, edge = evaluate(chain, phis, [beta], [t1], [0])
            rows.append({"state": "site", "lambda": lam, "beta": beta, "T1": t1, "Tc": 60.0,
                         "F_spread": float(F0[0]), "rms_spread": w, "F": float(F[0]), "edge": edge})
            _log(f"states lam={lam} T1={t1:g} beta={beta}: F {float(F0[0]):.3f} (rms {w:.2f}) -> {F[0]:.3f}")
            _save("states.json", rows)


# --------------------------------------------------------------------------- Lie rank, starts
def run_lie():
    from .reversal_phase import lie_rank
    rows = []
    for n in (15, 17, 19, 21):
        r = lie_rank(n)
        rows.append(r)
        _log(f"lie n={n}: rank {r['rank']} / {r['dim_u(n)']}")
        _save("lie.json", rows)


def run_starts():
    chain = Chain(201)
    rows = []
    for beta in (0.0, 1.0, 4.0):
        psi0 = chain.spread([beta], [15.0], [0])
        Fs = []
        for seed in range(4):  # 4 x 8 = 32 random starts, each kept separately
            n_seg = 120
            g = torch.Generator(device="cpu").manual_seed(100 + seed)
            phis = (2 * math.pi * torch.rand(n_seg, 8, generator=g, dtype=F64)).to(DEV).requires_grad_()
            opt = torch.optim.Adam([phis], lr=0.08)
            best = torch.zeros(8, dtype=F64)
            for _ in range(300):
                opt.zero_grad()
                F = chain.evolve(psi0.repeat(1, 8), pw_steps(phis), beta)[chain.site(0)].abs() ** 2
                best = torch.maximum(best, F.detach().cpu())
                (-torch.log(F + 1e-12)).sum().backward()
                opt.step()
            Fs += best.tolist()
        Fs = np.array(Fs)
        rows.append({"beta": beta, "F_starts": Fs.tolist(), "max": float(Fs.max()),
                     "median": float(np.median(Fs)), "min": float(Fs.min()),
                     "frac_ge_0.9": float((Fs >= 0.9).mean())})
        _log(f"starts beta={beta}: max {Fs.max():.3f} median {np.median(Fs):.3f} min {Fs.min():.3f} "
             f"frac>=0.9 {(Fs >= 0.9).mean():.2f}")
        _save("starts.json", rows)


# --------------------------------------------------------------------------- model-free
def _spsa(chain, psi_true, beta, lam, theta, to_phase, iters, shots, a0, c0, seed):
    """SPSA ascent on F estimated from `shots` binomial measurements of the start-site occupation."""
    rng = np.random.default_rng(seed)
    i0 = chain.site(0)
    hist = []

    def measure(thetas):  # thetas (P,B)
        with torch.no_grad():
            F = chain.evolve(psi_true.repeat(1, thetas.shape[1]), to_phase(thetas), beta, lam)[i0].abs() ** 2
        F = F.clamp(0, 1).cpu().numpy()
        return rng.binomial(shots, F) / shots if shots else F, F

    for k in range(iters):
        ak, ck = a0 / (k + 1 + 50) ** 0.602, c0 / (k + 1) ** 0.101
        d = torch.tensor(rng.choice([-1.0, 1.0], size=theta.numel()), dtype=F64, device=DEV)
        est, _ = measure(torch.stack([theta + ck * d, theta - ck * d], 1))
        theta = theta + ak * (est[0] - est[1]) / (2 * ck) * d
        if k % 100 == 0 or k == iters - 1:
            _, Ft = measure(theta[:, None])
            hist.append({"iter": k, "evaluations": 2 * (k + 1), "F_true": float(Ft[0])})
            _log(f"    spsa it {k}: F_true {Ft[0]:.3f}")
    return theta, hist


def run_modelfree():
    rows = []
    beta_true, lam_true = 1.2, 1.03          # true system differs from the model (beta=1, lam=1)
    chain = Chain(201)
    psi_true = chain.spread([beta_true], [15.0], [0], [lam_true])
    lam_t = torch.tensor([lam_true], dtype=F64, device=DEV)
    pw = lambda th: pw_steps(th)  # noqa: E731
    mf = _mf_control(1.0)
    with torch.no_grad():
        F_nom = float(chain.evolve(psi_true, pw(mf[:, None]), beta_true, lam_t)[chain.site(0)].abs() ** 2)
    _log(f"modelfree: nominal model control on the true system F={F_nom:.3f}")
    for shots in (0, 1000):
        th, hist = _spsa(chain, psi_true, beta_true, lam_t, mf.clone(), pw, 2000, shots, 0.6, 0.05, 1)
        rows.append({"case": "refine_model_control", "beta_true": beta_true, "lam_true": lam_true,
                     "shots": shots, "F_start": F_nom, "history": hist, "F_final": hist[-1]["F_true"]})
        _log(f"modelfree refine shots={shots}: {F_nom:.3f} -> {hist[-1]['F_true']:.3f}")
        _save("modelfree.json", rows)
    # from scratch, smooth band-limited (Omega_max=4, K=76), no model at all
    K, tc = 76, 60.0
    th, hist = _spsa(chain, psi_true, beta_true, lam_t, torch.zeros(K, dtype=F64, device=DEV),
                     lambda b: sine_phase(b, tc), 6000, 1000, 2.0, 0.1, 2)
    rows.append({"case": "from_scratch_smooth", "K": K, "Omega_max": 4.0, "shots": 1000,
                 "beta_true": beta_true, "lam_true": lam_true, "history": hist, "F_final": hist[-1]["F_true"]})
    _log(f"modelfree from scratch: F={hist[-1]['F_true']:.3f}")
    _save("modelfree.json", rows)


def run_modelfree_long():
    """From-scratch smooth control learned from 1000-shot measurements only, 30000 SPSA iterations."""
    beta_true, lam_true = 1.2, 1.03
    chain = Chain(201)
    psi_true = chain.spread([beta_true], [15.0], [0], [lam_true])
    lam_t = torch.tensor([lam_true], dtype=F64, device=DEV)
    K, tc = 76, 60.0
    rows = []
    for a0, c0 in ((1.0, 0.1), (3.0, 0.15)):
        th, hist = _spsa(chain, psi_true, beta_true, lam_t, torch.zeros(K, dtype=F64, device=DEV),
                         lambda b: sine_phase(b, tc), 30000, 1000, a0, c0, 3)
        rows.append({"case": "from_scratch_smooth_long", "K": K, "Omega_max": 4.0, "shots": 1000,
                     "a0": a0, "c0": c0, "beta_true": beta_true, "lam_true": lam_true,
                     "history": hist, "F_final": hist[-1]["F_true"], "b": th.cpu().tolist()})
        _log(f"modelfree long a0={a0} c0={c0}: F={hist[-1]['F_true']:.3f}")
        _save("modelfree_long.json", rows)


STUDIES = {"modelfree_long": run_modelfree_long,
"bw_lam": run_bw_lam, "scaling120": run_scaling120, "ens_beta": run_ens_beta,
           "site_cross": run_site_cross, "fewbody": run_fewbody, "states": run_states, "lie": run_lie,
           "starts": run_starts, "modelfree": run_modelfree}


if __name__ == "__main__":
    for s in sys.argv[1:]:
        _log(f"device {DEV}; study {s}")
        STUDIES[s]()
        _log(f"done {s}")
