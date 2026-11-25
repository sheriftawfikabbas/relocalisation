"""Print a compact summary of the reversal_completion runs (CPU copy preferred, GPU copy as fallback).

Run: python -m relocalization.summarize_completion
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results/relocalization"
DIRS = [ROOT / "reversal_completion_cpu", ROOT / "reversal_completion"]


def load(name):
    for d in DIRS:
        p = d / name
        if p.exists():
            return json.loads(p.read_text()), p.parent.name
    return None, None


def main():
    r, src = load("lie.json")
    if r:
        print(f"[lie] ({src})", [(x["n"], x["rank"], x["dim_u(n)"]) for x in r])
    r, src = load("starts.json")
    if r:
        for x in r:
            print(f"[starts] ({src}) beta={x['beta']:g}: max {x['max']:.3f} median {x['median']:.3f} "
                  f"min {x['min']:.3f} frac>=0.9 {x['frac_ge_0.9']:.2f}")
    r, src = load("bw_lam.json")
    if r:
        for x in r:
            if x["kind"] == "spectrum":
                print(f"[bw spectrum] ({src}) lam={x['lambda']:g}: total {x['total']:.3f} om50 {x['omega_half']:g} "
                      f"om90 {x['omega_90']:g} occupied {x['occupied_width']:.2f} Estd {x['energy_std']:.2f}")
            else:
                print(f"[bw] ({src}) lam={x['lambda']:g} Om={x['Omega_max']:g} iters={x['iters']}: F={x['F']:.3f}")
    r, src = load("states.json")
    if r:
        for x in r:
            extra = f" rms {x['rms_spread']:.2f}" if "rms_spread" in x else ""
            print(f"[states] ({src}) {x['state']} lam={x['lambda']:g} beta={x['beta']:g} T1={x['T1']:g}: "
                  f"{x['F_spread']:.3f} -> {x['F']:.3f}{extra}")
    r, src = load("ens_beta.json")
    if r:
        for x in r:
            ft, fe = x["F_train"], x["F_test"]
            print(f"[ens beta+-50%] ({src}) members={x['n_members']} Tc={x['Tc']:g}: train {sum(ft)/len(ft):.3f} "
                  f"(min {min(ft):.3f}) test {sum(fe)/len(fe):.3f} (min {min(fe):.3f}); test F "
                  + " ".join(f"{b:.3f}:{f:.2f}" for b, f in zip(x["test"], fe)))
    r, src = load("site_cross.json")
    if r:
        for x in r:
            ft, fe = x["F_train"], x["F_test"]
            print(f"[site w=4] ({src}) beta={x['beta']:g}: train {sum(ft)/len(ft):.3f} test {sum(fe)/len(fe):.3f} "
                  + str([round(f, 3) for f in fe]))
    r, src = load("scaling120.json")
    if r:
        for x in r:
            print(f"[scaling T1=120] ({src}) Tc={x['Tc']:g} (Tc/T1={x['Tc']/x['T1']:g}): F={x['F']:.3f} "
                  f"spread {x['F_spread']:.3f} edge {x['edge']:.1e}")
    r, src = load("fewbody.json")
    if r:
        for x in r:
            if x["method"] == "exact_opt":
                print(f"[fewbody exact N_b=2] ({src}) Tc={x['Tc']:g}: P_all {x['P_all']:.3f} n0 {x['n0']:.3f}")
            else:
                print(f"[fewbody TWA N_b={x['N_b']}] ({src}) Tc={x['Tc']:g}: MF control {x['n0_mf_control']:.3f} -> "
                      f"optimized {x['n0_test']:.3f} (+-{x['se']:.3f}); new control in MF {x['F_meanfield_of_new']:.3f}")
    for name in ("modelfree.json", "modelfree_long.json"):
        r, src = load(name)
        if r:
            for x in r:
                h = x["history"]
                print(f"[{name[:-5]}] ({src}) {x['case']} shots={x['shots']} "
                      f"{'a0=' + str(x['a0']) if 'a0' in x else ''}: F {h[0]['F_true']:.3f} -> {x['F_final']:.3f} "
                      f"after {h[-1]['evaluations']} evaluations")


if __name__ == "__main__":
    main()
