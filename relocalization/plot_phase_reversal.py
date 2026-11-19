"""Figure: mean-field reversal of spreading by a phase-only control.

Panels
  (a,b) space-time density, free continuation vs optimized phase control (beta=1)
  (c)   probability on the starting site through the control, beta = 0 and 1
  (d)   return fidelity vs control time in units of the spread time
  (e)   return fidelity vs the bandwidth of a smooth control, three lattice depths
  (f)   one control for an ensemble of states: held-out return fidelity

Run: REV_DEVICE=cpu python -m relocalization.plot_phase_reversal
Needs results/relocalization/reversal/phase_only_beta*.json (reversal_phase.py) and
results/relocalization/reversal_study/*.json (reversal_study.py).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("REV_DEVICE", "cpu")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .reversal_study import DEV, DT, F64, SEG, Chain, pw_steps  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/relocalization"
OUT = RES / "figures/phase_reversal.png"

# validated categorical slots 1-3 (light); text in ink tokens, grid recessive
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e3e2de", "#9a9993"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cfe0f5", "#7fb0e8", "#2a78d6", "#123c70"])

T1, CHUNK = 15.0, 10          # chunk = 10 steps = one 0.5 control segment (exact for Strang splitting)


def control(beta: float) -> torch.Tensor:
    d = json.loads((RES / f"reversal/phase_only_beta{beta:g}.json").read_text())["phase_Tc60"]
    return torch.tensor(d["phis_over_pi"], dtype=F64, device=DEV) * math.pi


def trajectory(chain: Chain, beta: float, phis: torch.Tensor | None, t_total: float):
    """Density (n_t, N) sampled every 0.5 time units; control starts at T1 (phi=0 if phis is None)."""
    n_seg_total = int(round(t_total / SEG))
    n_free = int(round(T1 / SEG))
    seg = torch.zeros(n_seg_total, dtype=F64, device=DEV)
    if phis is not None:
        seg[n_free:n_free + phis.numel()] = phis
    steps = pw_steps(seg)                                 # (n_steps,)
    psi = torch.zeros(chain.n, 1, dtype=torch.complex128, device=DEV)
    psi[chain.site(0)] = 1.0
    dens = [(psi.abs() ** 2)[:, 0].cpu().numpy()]
    with torch.no_grad():
        for s in range(n_seg_total):
            psi = chain.evolve(psi, steps[s * CHUNK:(s + 1) * CHUNK], beta)
            dens.append((psi.abs() ** 2)[:, 0].cpu().numpy())
    return np.array(dens)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, color=GRID)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.title.set_color(INK)
    ax.title.set_size(9.5)


def label(ax, x, y, text, color, **kw):
    ax.text(x, y, text, color=color, fontsize=8, va="center", **kw)


def supp_figure() -> None:
    """Waveforms and calibration / noise sensitivity of the beta=1 controls (JSON only)."""
    sm = json.loads((RES / "reversal/smooth_beta1.json").read_text())
    pw = json.loads((RES / "reversal/phase_only_beta1.json").read_text())["phase_Tc60"]
    bw = [r for r in json.loads((RES / "reversal_study/bandwidth_beta1.json").read_text())
          if r["lambda"] == 1.0 and r["Omega_max"] == 4.0][0]
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.0), layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axs.ravel()
    # (a) piecewise control, wrapped to (-pi, pi]
    phis = (np.array(pw["phis_over_pi"]) + 1.0) % 2.0 - 1.0
    tt = np.arange(len(phis) + 1) * SEG
    style(ax_a)
    ax_a.stairs(phis, tt, color=BLUE, lw=1.6)
    ax_a.set_xlim(0, 60)
    ax_a.set_ylim(-1.05, 1.05)
    ax_a.set_xlabel("control time $t-T_1$")
    ax_a.set_ylabel(r"$\phi/\pi$ (mod 2)")
    ax_a.set_title("(a) piecewise control, $\\beta=1$, $F=0.97$", loc="left")
    # (b) smooth control
    b = np.array(bw["b"])
    t = np.linspace(0, 60, 1200)
    phi = np.sin(np.pi * np.outer(t, np.arange(1, len(b) + 1)) / 60.0) @ b
    style(ax_b)
    ax_b.plot(t, phi / np.pi, color=ORANGE, lw=1.6)
    ax_b.set_xlim(0, 60)
    ax_b.set_xlabel("control time $t-T_1$")
    ax_b.set_ylabel(r"$\phi/\pi$")
    ax_b.set_title(f"(b) smooth control, $\\Omega_{{\\max}}=4$, $F={bw['F']:.2f}$", loc="left")
    # (c) calibration errors, piecewise control
    st = sm["robust_piecewise"]["static"]
    style(ax_c)
    for kind, name, col, xt in (("amplitude", "amplitude", BLUE, 0.04), ("rate", "timing", ORANGE, 0.0),
                                ("lambda", "lattice depth $\\lambda$", AQUA, -0.05)):
        pts = sorted((r["error"] * 100, r["F"]) for r in st if r["kind"] == kind)
        pts = [q for q in pts if abs(q[0]) <= 10.01]
        ax_c.plot([q[0] for q in pts], [q[1] for q in pts], color=col, lw=2, marker="o", ms=5,
                  mec="#fcfcfb", mew=1.5, solid_capstyle="round")
    ax_c.set_xlabel("fractional calibration error (%)")
    ax_c.set_ylabel("return fidelity $F$")
    ax_c.set_ylim(0, 1.05)
    ax_c.set_title("(c) static errors", loc="left")
    label(ax_c, 1.0, 0.33, "amplitude", BLUE, ha="left")
    label(ax_c, -0.2, 0.60, "timing", ORANGE, ha="right")
    label(ax_c, -3.2, 0.93, r"depth $\lambda$", AQUA, ha="right")
    # (d) coloured noise
    nz = sm["robust_piecewise"]["noise"]
    style(ax_d)
    for tau, col in ((0.5, BLUE), (5.0, ORANGE)):
        pts = sorted((r["sigma"], r["F_mean"]) for r in nz if r["tau"] == tau)
        ax_d.plot([q[0] for q in pts], [q[1] for q in pts], color=col, lw=2, marker="o", ms=5,
                  mec="#fcfcfb", mew=1.5, solid_capstyle="round")
    ax_d.set_xlabel(r"rms phase noise $\sigma$ (rad)")
    ax_d.set_ylabel("mean return fidelity")
    ax_d.set_ylim(0, 1.05)
    ax_d.set_title("(d) coloured phase noise", loc="left")
    label(ax_d, 0.2, 0.74, r"$\tau=0.5$", BLUE, ha="left")
    label(ax_d, 0.21, 0.44, r"$\tau=5$", ORANGE, ha="left")
    out = RES / "figures/phase_reversal_supp.png"
    fig.savefig(out, dpi=170, facecolor="#fcfcfb")
    print("Wrote", out)


def main() -> None:
    chain = Chain(201)
    t_total = T1 + 60.0
    ctrl1, ctrl0 = control(1.0), control(0.0)
    free1 = trajectory(chain, 1.0, None, t_total)
    drv1 = trajectory(chain, 1.0, ctrl1, t_total)
    drv0 = trajectory(chain, 0.0, ctrl0, t_total)
    free0 = trajectory(chain, 0.0, None, t_total)
    t = np.arange(free1.shape[0]) * SEG
    c = chain.site(0)

    fig = plt.figure(figsize=(11.0, 6.9), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))
    ax_d, ax_e, ax_f = (fig.add_subplot(gs[1, k]) for k in range(3))

    # (a, b) space-time density
    jmax = 30
    sl = slice(c - jmax, c + jmax + 1)
    vmin = -4.0
    for ax, dens, ttl in ((ax_a, free1, "(a) free evolution, $\\beta=1$"),
                          (ax_b, drv1, "(b) phase control from $t=15$, $\\beta=1$")):
        im = ax.imshow(np.log10(np.maximum(dens[:, sl], 10 ** vmin)).T, origin="lower", aspect="auto",
                       extent=[t[0], t[-1], -jmax - 0.5, jmax + 0.5], cmap=SEQ, vmin=vmin, vmax=0)
        ax.axvline(T1, color=INK, lw=0.8, ls=":")
        ax.set_xlabel("time $t$")
        ax.set_title(ttl, loc="left")
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.xaxis.label.set_color(INK2)
        ax.xaxis.label.set_size(9)
        ax.title.set_color(INK)
        ax.title.set_size(9.5)
    ax_a.set_ylabel("site $j$")
    ax_a.yaxis.label.set_color(INK2)
    ax_a.yaxis.label.set_size(9)
    ax_b.set_yticklabels([])
    cb = fig.colorbar(im, ax=[ax_a, ax_b], location="bottom", shrink=0.5, pad=0.02, aspect=40)
    cb.set_label(r"$\log_{10}|\psi_j|^2$", color=INK2, fontsize=8)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_edgecolor(GRID)

    # (c) probability on the starting site
    style(ax_c)
    ax_c.plot(t, free1[:, c], color=GRAY, lw=2, solid_capstyle="round")
    ax_c.plot(t, drv1[:, c], color=BLUE, lw=2, solid_capstyle="round")
    ax_c.plot(t, drv0[:, c], color=ORANGE, lw=2, solid_capstyle="round")
    ax_c.axvline(T1, color=INK, lw=0.8, ls=":")
    ax_c.set_xlim(0, t[-1])
    ax_c.set_ylim(0, 1.05)
    ax_c.set_xlabel("time $t$")
    ax_c.set_ylabel("probability on the starting site")
    ax_c.set_title("(c) return to the starting site", loc="left")
    label(ax_c, 4, 0.9, "spread", INK2, ha="left")
    label(ax_c, t[-1] - 1, 0.74, f"$\\beta=0$: {drv0[-1, c]:.2f}", ORANGE, ha="right")
    label(ax_c, t[-1] - 1, 0.62, f"$\\beta=1$: {drv1[-1, c]:.2f}", BLUE, ha="right")
    label(ax_c, 21, 0.27, f"undriven $\\beta=1$: {free1[-1, c]:.3f}", INK2, ha="left")

    # (d) control time in units of the spread time
    style(ax_d)
    sc = json.loads((RES / "reversal_study/scaling_beta1.json").read_text())
    cols = {15.0: BLUE, 30.0: ORANGE, 60.0: AQUA}
    for t1, col in cols.items():
        rows = [r for r in sc if r["T1"] == t1]
        x = [r["Tc"] / t1 for r in rows]
        y = [r["F"] for r in rows]
        ax_d.plot(x, y, color=col, lw=2, marker="o", ms=5, mec="#fcfcfb", mew=1.5, solid_capstyle="round")
    ax_d.set_xscale("log", base=2)
    ax_d.set_xticks([0.25, 0.5, 1, 2, 4])
    ax_d.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
    ax_d.set_ylim(0, 1.05)
    ax_d.set_xlabel("control time / spread time, $T_c/T_1$")
    ax_d.set_ylabel("return fidelity $F$")
    ax_d.set_title("(d) control time vs spread time", loc="left")
    ax_d.legend(handles=[plt.Line2D([], [], color=c_, lw=2, marker="o", ms=5, mec="#fcfcfb", label=f"$T_1={int(t1)}$")
                         for t1, c_ in cols.items()], loc="lower right", frameon=False, fontsize=8,
                labelcolor=INK2, title="curves nearly coincide", title_fontsize=7.5)

    # (e) bandwidth
    style(ax_e)
    bw = json.loads((RES / "reversal_study/bandwidth_beta1.json").read_text())
    for lam, col in ((0.5, BLUE), (1.0, ORANGE), (1.5, AQUA)):
        rows = [r for r in bw if r["lambda"] == lam]
        ax_e.plot([r["Omega_max"] for r in rows], [r["F"] for r in rows], color=col, lw=2, marker="o",
                  ms=5, mec="#fcfcfb", mew=1.5, solid_capstyle="round")
    ax_e.set_ylim(0, 1.05)
    ax_e.set_xlabel(r"control bandwidth $\Omega_{\max}/J$")
    ax_e.set_ylabel("return fidelity $F$")
    ax_e.set_title("(e) bandwidth needed, $T_c=60$", loc="left")
    label(ax_e, 8.2, 0.817, r"$\lambda=0.5$", BLUE, ha="left")
    label(ax_e, 8.2, 0.955, r"$\lambda=1$", ORANGE, ha="left")
    label(ax_e, 8.2, 1.03, r"$\lambda=1.5$", AQUA, ha="left")
    ax_e.set_xlim(0.5, 10.2)
    ax_e.set_ylim(0, 1.08)

    # (f) ensemble robustness (held-out)
    style(ax_f)
    en = [r for r in json.loads((RES / "reversal_study/ensemble_beta1.json").read_text()) if r["Tc"] == 120.0]
    groups = (("T1", "spread time $T_1\\pm$", BLUE, (0.25, 0.5, 1.0, 2.0)),
              ("beta", "interaction $\\beta\\pm$", ORANGE, (0.1, 0.25, 0.5)),
              ("site", "start site $\\pm$", AQUA, (1, 2, 4)))
    y, yt, yl = 0.0, [], []
    for kind, name, col, widths in groups:
        for w in widths:
            r = next(q for q in en if q["kind"] == kind and q["width"] == w)
            wl = f"{w:g}" if kind != "beta" else f"{int(round(w * 100))}%"
            ax_f.plot([0, r["F_test_mean"]], [y, y], color=col, lw=2, solid_capstyle="round")
            ax_f.plot([r["F_train_mean"]], [y], marker="o", ms=6, mfc="#fcfcfb", mec=col, mew=1.5)
            ax_f.plot([r["F_test_mean"]], [y], marker="o", ms=6, color=col, mec="#fcfcfb", mew=1.5)
            ax_f.text(r["F_test_mean"] + 0.03, y, f"{r['F_test_mean']:.2f}", color=INK2, fontsize=8, va="center")
            yt.append(y)
            yl.append(f"{name}{wl}")
            y -= 1
        y -= 0.5
    ax_f.set_yticks(yt)
    ax_f.set_yticklabels(yl, fontsize=8, color=INK2)
    ax_f.set_xlim(0, 1.12)
    ax_f.grid(False, axis="y")
    ax_f.set_xlabel("return fidelity $F$\n(open: training members, filled: held-out)")
    ax_f.set_title("(f) one control, many states", loc="left")


    fig.suptitle("Mean-field reversal of spreading by a phase-only control ($\\lambda=1$, $T_1=15$ unless noted)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor="#fcfcfb")
    print("Wrote", OUT)
    print(f"final center prob: free beta=1 {free1[-1, c]:.4f}  driven beta=1 {drv1[-1, c]:.4f}  "
          f"driven beta=0 {drv0[-1, c]:.4f}  free beta=0 {free0[-1, c]:.4f}")


if __name__ == "__main__":
    supp_figure()
    main()
