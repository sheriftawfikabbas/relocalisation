from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

from .metrics import extract_beta_star_inflection


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_r_vs_beta(rows: Sequence[Dict[str, object]], path: str | Path,
                   smooth_sigma: float = 0.3) -> None:
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ansatze = sorted({str(r["ansatz"]) for r in rows})
    scenario = str(rows[0].get("scenario", "B")) if rows else "B"
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    colors = {}
    for ansatz in ansatze:
        pts = sorted([r for r in rows if r["ansatz"] == ansatz], key=lambda r: float(r["beta"]))
        betas = [float(r["beta"]) for r in pts]
        rvals = [float(r["r"]) for r in pts]
        line, = ax.plot(betas, rvals, marker="o", ms=3, ls="none", alpha=0.45, label=ansatz)
        colors[ansatz] = line.get_color()
        # smooth and overplot
        from scipy.ndimage import gaussian_filter1d
        import numpy as _np
        finite = [(b, r) for b, r in zip(betas, rvals) if b > 0 and math.isfinite(r)]
        if len(finite) >= 2:
            bf, rf = zip(*finite)
            db = float(_np.median(_np.diff(bf)))
            keep = [i for i, b in enumerate(bf) if b <= bf[0] + 60 * db]
            bf, rf = [bf[i] for i in keep], [rf[i] for i in keep]
            rs = gaussian_filter1d(list(rf), sigma=smooth_sigma/db if db > 0 else 1.0)
            ax.plot(list(bf), rs, color=line.get_color(), lw=1.5)
        bstar = extract_beta_star_inflection(betas, rvals)
        if scenario == "B" and math.isfinite(bstar):
            ax.axvline(bstar, color=line.get_color(), ls="--", lw=0.8, alpha=0.8)
    ax.set_xlabel(r"nonlinearity $\beta$")
    if scenario == "B":
        ax.set_ylabel(r"arrest score $R$")
        ax.set_title(r"Arrest, $\lambda=1$ (dashed: operational $\beta^\star$)")
    else:
        ax.set_ylabel(r"suppression score $R$")
        ax.set_title(r"Suppression, $\lambda=3$ (no $\beta^\star$ defined)")
    ax.legend(frameon=False, fontsize=8, title="drive family (points: runs; lines: smoothed)",
              title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_beta_star_table(rows: Sequence[Dict[str, object]], threshold: float,
                          path: str | Path) -> List[Dict[str, object]]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out: List[Dict[str, object]] = []
    groups = {}
    for r in rows:
        groups.setdefault((str(r["scenario"]), str(r["ansatz"])), []).append(r)
    for (scenario, ansatz), grp in sorted(groups.items()):
        grp = sorted(grp, key=lambda r: float(r["beta"]))
        betas = [float(r["beta"]) for r in grp]
        rvals = [float(r["r"]) for r in grp]
        finite_pos = [(bb, rr) for bb, rr in zip(betas, rvals) if bb > 0.0 and math.isfinite(rr)]
        zero_pts = [(bb, rr) for bb, rr in zip(betas, rvals) if bb == 0.0 and math.isfinite(rr)]
        r_zero = zero_pts[0][1] if zero_pts else (finite_pos[0][1] if finite_pos else float("nan"))
        peak_r = max((rr for _, rr in finite_pos), default=float("nan"))
        r_report = r_zero if scenario == "B" else peak_r
        bstar = extract_beta_star_inflection(betas, rvals)
        out.append({"scenario": scenario, "ansatz": ansatz, "r_initial": r_report, "beta_star": bstar})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "ansatz", "r_initial", "beta_star"])
        writer.writeheader()
        writer.writerows(out)
    return out


def plot_scenario_panels(result, path: str | Path) -> None:
    """Width (undriven vs driven, log-y) and local exponent mu(t).

    The first mu(t) sample is dropped because effective_alpha uses log-time and
    the t=0 sample produces a meaningless first value.
    """
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    axes[0].plot(result.times, result.width_undriven, label="undriven", lw=1.2)
    axes[0].plot(result.times, result.width_driven, label="driven (optimized)", lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("width")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title(f"Scenario {result.scenario}, beta={result.beta:g}, {result.ansatz}")
    axes[1].plot(result.times[1:], result.mu_driven[1:], lw=1.2)
    axes[1].axhline(0.0, color="black", ls=":", lw=1.0)
    axes[1].set_ylabel("local exponent mu(t)")
    axes[1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
