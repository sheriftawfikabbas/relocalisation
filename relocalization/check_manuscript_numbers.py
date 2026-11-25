"""Assert that numbers quoted in the manuscript match the stored reversal-study results.

Run: MANUSCRIPT_DIR=/path/to/manuscript python -m relocalization.check_manuscript_numbers
Exits non-zero and lists every mismatch.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Paper sources are not tracked in this repository; point MANUSCRIPT_DIR at the folder with main.tex and supp.tex.
MS = Path(os.environ.get("MANUSCRIPT_DIR", ROOT / "manuscript"))
if not (MS / "main.tex").exists() or not (MS / "supp.tex").exists():
    sys.exit(f"main.tex / supp.tex not found in {MS}; set MANUSCRIPT_DIR to the manuscript folder.")
MAIN = (MS / "main.tex").read_text()
SUPP = (MS / "supp.tex").read_text()
TEX = MAIN + SUPP


def rows(rel):
    return list(csv.DictReader((ROOT / rel).open()))


def pick(rs, **kw):
    out = [r for r in rs if all(str(r[k]) == str(v) for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]


CHECKS = []


def check(label, value, quoted, tol, needle):
    CHECKS.append((label, float(value), float(quoted), tol, needle))


# ---- phase-only reversal (main text and Supplemental Material) ----
RV = ROOT / "results/relocalization/reversal"
RS = ROOT / "results/relocalization/reversal_study"


def js(p):
    return json.loads(p.read_text())


for b, q0, qk, q60, q4 in [("0", 0.051, None, 0.963, None), ("0.5", None, 0.82, 0.969, None),
                           ("1", 0.097, 0.42, 0.966, None), ("2", None, None, 0.966, None),
                           ("4", None, None, 0.931, None)]:
    d = js(RV / f"phase_only_beta{b}.json")
    check(f"reversal F beta={b} Tc=60", d["phase_Tc60"]["F"], q60, 6e-4, f"{q60:.3f}")
    if q0 is not None:
        check(f"free F0 beta={b}", d["after_spread"]["F"], q0, 6e-4, f"{q0:.3f}")
    if qk is not None:
        check(f"pi-kick echo F beta={b}", d["kick_echo"]["F"], qk, 6e-3, f"{qk:.2f}")
d1 = js(RV / "phase_only_beta1.json")
check("width before control beta=1", d1["after_spread"]["width"], 9.6, 0.05, "9.6")
check("width after control beta=1", d1["phase_Tc60"]["width"], 7.1, 0.05, "7.1")
check("hybrid Tc=30 beta=1", d1["kick+phase_Tc30"]["F"], 0.95, 6e-3, "0.95")
check("phase-only Tc=30 beta=1", d1["phase_Tc30"]["F"], 0.90, 6e-3, "0.90")

sc1 = js(RS / "scaling_beta1.json")
for t1, tc, q in [(15.0, 60.0, 0.97), (30.0, 120.0, 0.98), (60.0, 240.0, 0.99)]:
    r = [x for x in sc1 if x["T1"] == t1 and x["Tc"] == tc][0]
    check(f"scaling 4*T1 T1={t1:g}", r["F"], q, 6e-3, f"{q:.2f}")

bw = js(RS / "bandwidth_beta1.json")
for lam, om, q in [(1.0, 3.0, 0.90), (1.0, 4.0, 0.94), (1.5, 2.0, 0.90)]:
    r = [x for x in bw if x["lambda"] == lam and x["Omega_max"] == om][0]
    check(f"bandwidth lam={lam:g} Om={om:g}", r["F"], q, 6e-3, f"{q:.2f}")

# ---- completion studies (CPU copy preferred) ----
COMP = [ROOT / "results/relocalization/reversal_completion_cpu", ROOT / "results/relocalization/reversal_completion"]


def cj(name):
    for d in COMP:
        if (d / name).exists():
            return js(d / name)
    raise FileNotFoundError(name)


bwl = cj("bw_lam.json")
for om, q in [(8.0, 0.825), (10.0, 0.840), (12.0, 0.852)]:
    r = [x for x in bwl if x["kind"] == "smooth" and x["lambda"] == 0.5 and x["Omega_max"] == om][0]
    check(f"bandwidth lam=0.5 long Om={om:g}", r["F"], q, 6e-4, f"{q:.3f}")
for lam, om in [(1.25, 3.0), (0.75, 5.0)]:   # smallest bandwidth reaching 0.9 (text: 3 and 5)
    rr = sorted([x for x in bwl if x["kind"] == "smooth" and x["lambda"] == lam], key=lambda x: x["Omega_max"])
    first = [x["Omega_max"] for x in rr if x["F"] >= 0.9][0]
    check(f"bandwidth lam={lam:g} first Om with F>=0.9", first, om, 0, "$1$ and $0.75$")
for lam, q in [(0.5, 0.12), (1.0, 0.45), (1.5, 0.79)]:
    r = [x for x in bwl if x["kind"] == "spectrum" and x["lambda"] == lam][0]
    check(f"control spectral weight lam={lam:g}", r["total"], q, 6e-3, f"{q:.2f}")
    check(f"control omega_90 lam={lam:g} in [4,4.5]", r["omega_90"], 4.25, 0.25, "$4$--$4.5J$")

stt = cj("starts.json")
s0, s1, s4 = (pick(stt, beta=b) for b in (0.0, 1.0, 4.0))
check("starts beta=0 all >=0.9", s0["frac_ge_0.9"], 1.0, 0, "all reach $F\\ge0.90$")
check("starts beta=1 frac>=0.9", s1["frac_ge_0.9"], 0.97, 6e-3, "$97\\%$")
check("starts beta=1 median", s1["median"], 0.95, 6e-3, "median $0.95$")
check("starts beta=1 best", s1["max"], 0.97, 6e-3, "best $0.97$")
check("starts beta=4 median", s4["median"], 0.90, 6e-3, "the median is $0.90$")
check("starts beta=4 frac>=0.9", s4["frac_ge_0.9"], 0.47, 6e-3, "$47\\%$")

sts = cj("states.json")
for name, q0, q in [("gaussian_s2", 0.163, 0.971), ("gaussian_s2_k0.5", 0.023, 0.964),
                    ("two_site", 0.005, 0.964), ("site_j0=7", None, 0.966)]:
    r = pick(sts, state=name)
    check(f"state {name} controlled", r["F"], q, 6e-4, f"{q:.3f}")
    if q0 is not None:
        check(f"state {name} after spread", r["F_spread"], q0, 6e-4, f"{q0:.3f}")
crit = [x for x in sts if x["state"] == "site"]
check("crit/loc min controlled F", min(x["F"] for x in crit), 0.993, 6e-4, "0.993")
check("crit/loc max controlled F", max(x["F"] for x in crit), 1.000, 6e-4, "1.000")
check("crit/loc min rms", min(x["rms_spread"] for x in crit), 0.5, 0.06, "$0.5$--$3.5$")
check("crit/loc max rms", max(x["rms_spread"] for x in crit), 3.5, 0.05, "$0.5$--$3.5$")
check("crit/loc min F0", min(x["F_spread"] for x in crit), 0.38, 6e-3, "$F_0=0.38$--$0.88$")
check("crit/loc max F0", max(x["F_spread"] for x in crit), 0.88, 6e-3, "$F_0=0.38$--$0.88$")

mfr = cj("modelfree.json")
ref0 = [x for x in mfr if x["case"] == "refine_model_control" and x["shots"] == 0][0]
ref1 = [x for x in mfr if x["case"] == "refine_model_control" and x["shots"] == 1000][0]
check("modelfree nominal F", ref1["F_start"], 0.42, 6e-3, "only $F=0.42$")
check("modelfree refine 1000 shots", ref1["F_final"], 0.84, 6e-3, "to $0.84$")
check("modelfree refine exact", ref0["F_final"], 0.87, 6e-3, "$0.87$ if $F$ is known exactly")
check("modelfree refine evaluations", ref1["history"][-1]["evaluations"], 4000, 0, "$4000$ runs")
mfl = max(cj("modelfree_long.json"), key=lambda x: x["F_final"])
check("modelfree from scratch best", mfl["F_final"], 0.75, 6e-3, "$F=0.75$ after $60\\,000$ runs")
check("modelfree from scratch evaluations", mfl["history"][-1]["evaluations"], 60000, 0, "$60\\,000$ runs")

scan = cj("ens_beta_scan.json")


def _bad(key, thr=0.8):
    return [b for b, x in zip(scan["betas"], scan[key]) if x < thr]


f240 = scan["m9_Tc240"]
check("ens beta 9m Tc240 frac>=0.8", sum(x >= 0.8 for x in f240) / len(f240), 0.89, 6e-3, "$89\\%$")
check("ens beta 9m Tc240 window lo", min(_bad("m9_Tc240")), 1.39, 6e-3, "1.39$--$1.49")
check("ens beta 9m Tc240 window hi", max(_bad("m9_Tc240")), 1.49, 6e-3, "1.39$--$1.49")
check("ens beta 9m Tc240 min", min(f240), 0.004, 6e-4, "falls to $0.004$")
check("ens beta F_T1 peak", max(scan["F_T1"]), 0.50, 6e-3, "peaks\nat $0.50$")
check("ens beta F_T1 peak beta", scan["betas"][scan["F_T1"].index(max(scan["F_T1"]))], 1.46, 6e-3, "1.46")
check("ens beta 5m Tc120 window lo", min(_bad("m5_Tc120")), 1.26, 6e-3, "$1.26$--$1.5$")
check("ens beta 9m Tc120 window lo", min(_bad("m9_Tc120")), 1.31, 6e-3, "$1.31$--$1.5$")

twa = {(r["N_b"], r["Tc"]): r for r in cj("fewbody_twa.json") if r["method"] == "TWA_opt"}
for nb, mf, opt in [(30, 0.57, 0.67), (100, 0.83, 0.85)]:
    r = twa[(nb, 60.0)]
    check(f"TWA mf control Nb={nb}", r["n0_mf_control"], mf, 6e-3, f"{mf:.2f}")
    check(f"TWA optimized Nb={nb}", r["n0_test"], opt, 6e-3, f"{opt:.2f}")
check("TWA mean-field F Nb=30", twa[(30, 60.0)]["F_meanfield_of_new"], 0.92, 6e-3, "$0.92$ and")
check("TWA mean-field F Nb=100", twa[(100, 60.0)]["F_meanfield_of_new"], 0.96, 6e-3, "$0.96$")
check("TWA Nb=10 train", twa[(10, 60.0)]["n0_train"], 0.37, 6e-3, "value is $0.37$")
check("TWA Nb=10 fresh", twa[(10, 60.0)]["n0_test"], 0.20, 6e-3, "value is $0.20$")
check("TWA Nb=10 mf control", twa[(10, 60.0)]["n0_mf_control"], 0.18, 6e-3, "$0.18$\nfor")

for tc, q in [(120.0, 0.32), (240.0, 0.41)]:
    r = [x for x in cj("fewbody.json") if x["method"] == "exact_opt" and x["Tc"] == tc][0]
    check(f"exact Nb=2 re-optimized Tc={tc:g}", r["P_all"], q, 6e-3, f"{q:.2f}")
check("exact Nb=2 Tc=240 edge", [x for x in cj("fewbody.json") if x["method"] == "exact_opt" and x["Tc"] == 240.0][0]["edge"],
      0.02, 5e-3, "$2\\%$ of the density")
check("TWA Nb=30 Tc=120", twa[(30, 120.0)]["n0_test"], 0.48, 6e-3, "$0.48$ for $N_b=30$")
check("TWA Nb=100 Tc=120", twa[(100, 120.0)]["n0_test"], 0.80, 6e-3, "$0.80$ for $N_b=100$")

sc = cj("site_cross.json")
for b, q in [(0.0, 0.68), (0.1, 0.63), (0.25, 0.63), (0.5, 0.35), (1.0, 0.04)]:
    r = pick(sc, beta=b)
    check(f"site crossover beta={b:g} adjacent sites", (r["F_test"][0] + r["F_test"][1]) / 2, q, 6e-3, f"{q:.2f}")
check("site crossover train beta=0", pick(sc, beta=0.0)["F_train"] and sum(pick(sc, beta=0.0)["F_train"]) / 9, 0.81, 6e-3, "from $0.81$ to $0.54$")
check("site crossover train beta=1", sum(pick(sc, beta=1.0)["F_train"]) / 9, 0.54, 6e-3, "from $0.81$ to $0.54$")
r0 = pick(sc, beta=0.0)["F_test"]
check("site crossover beta=0 two sites out min", min(r0[2:]), 0.31, 6e-3, "$0.31$--$0.48$")
check("site crossover beta=0 two sites out max", max(r0[2:]), 0.48, 6e-3, "$0.31$--$0.48$")

s120 = {r["Tc"]: r["F"] for r in cj("scaling120.json")}
check("scaling T1=120 Tc=T1 in 0.72-0.79", s120[120.0], 0.755, 0.035, "$F=0.72$--$0.79$ at $T_c=T_1$")
check("scaling T1=120 Tc=4T1", s120[480.0], 0.99, 6e-3, "$0.99$, $0.99$ at")
check("scaling T1=120 Tc=2T1 in 0.89-0.93", s120[240.0], 0.91, 0.02, "$0.89$--$0.93$ at")

lie = cj("lie.json")
check("Lie rank full up to n=21", max(x["n"] for x in lie if x["rank"] == x["dim_u(n)"]), 21, 0, "n=5,7,\\dots,21")

ens = [r for r in js(RS / "ensemble_beta1.json") if r["Tc"] == 120.0]
for kind, w, q in [("T1", 0.25, 0.96), ("T1", 0.5, 0.90), ("T1", 1.0, 0.81), ("T1", 2.0, 0.59),
                   ("beta", 0.1, 0.97), ("beta", 0.25, 0.93)]:
    r = [x for x in ens if x["kind"] == kind and x["width"] == w][0]
    check(f"ensemble held-out {kind} w={w}", r["F_test_mean"], q, 6e-3, f"{q:.2f}")

mb = js(RS / "manybody.json")
for nb, q in [(10, 0.17), (30, 0.55), (100, 0.83), (1000, 0.95)]:
    r = [x for x in mb if x["beta"] == 1.0 and x["method"] == "TWA" and x["N_b"] == nb
         and x["control"] == "single_Tc60"][0]
    check(f"TWA n0/Nb beta=1 Nb={nb}", r["n0_final"], q, 6e-3, f"{q:.2f}")
for nb, q in [(2, 0.12), (3, 0.06)]:
    r = [x for x in mb if x["beta"] == 1.0 and x["method"] == "exact" and x["N_b"] == nb
         and x["control"] == "single_Tc60"][0]
    check(f"exact P_all beta=1 Nb={nb}", r["P_all_final"], q, 6e-3, f"{q:.2f}")
r = [x for x in mb if x["beta"] == 1.0 and x["method"] == "exact_opt"][0]
check("exact Nb=2 re-optimized P_all", r["P_all_final"], 0.40, 6e-3, "0.40")

st = js(RV / "smooth_beta1.json")["robust_piecewise"]["static"]
for kind, e, q in [("amplitude", 0.02, 0.87), ("amplitude", 0.05, 0.51), ("rate", -0.01, 0.48),
                   ("rate", 0.01, 0.55), ("offset", 0.05, 0.93)]:
    r = [x for x in st if x["kind"] == kind and abs(x["error"] - e) < 1e-9][0]
    check(f"static error {kind} {e:+g}", r["F"], q, 6e-3, f"{q:.2f}")


def main() -> int:
    bad = 0
    for label, value, quoted, tol, needle in CHECKS:
        ok_val = abs(value - quoted) <= tol
        ok_txt = needle in TEX
        if not (ok_val and ok_txt):
            bad += 1
            print(f"FAIL {label}: data={value:.5g} quoted={quoted} "
                  f"(value {'ok' if ok_val else 'MISMATCH'}, text {'found' if ok_txt else 'MISSING'})")
    print(f"{len(CHECKS) - bad}/{len(CHECKS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
