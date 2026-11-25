"""Fine scan of the beta-ensemble controls over beta in [0.5, 1.5] (supplement, ensemble-robustness figure/table).

Re-evaluates the controls stored in results/relocalization/reversal_completion_cpu/ens_beta.json (produced by
`python -m relocalization.reversal_completion ens_beta`) on a 0.0125-spaced beta grid and writes
reversal_completion_cpu/ens_beta_scan.json.  The scan exposes the narrow failure window near the self-trapping onset.

Run: REV_DEVICE=cpu python -m relocalization.ens_beta_scan
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .reversal_study import DEV, F64, Chain, evaluate, metrics

DIR = Path(__file__).resolve().parent.parent / "results/relocalization/reversal_completion_cpu"


def main() -> None:
    rows = json.loads((DIR / "ens_beta.json").read_text())
    chain = Chain(301)
    betas = [float(b) for b in np.round(np.arange(0.5, 1.5001, 0.0125), 4)]
    out = {"betas": betas}
    psi0 = chain.spread(betas, [15.0] * len(betas), [0] * len(betas))
    F0, _ = metrics(chain, psi0, [0] * len(betas))
    out["F_T1"] = [float(x) for x in F0]
    for r in rows:
        phis = torch.tensor(r["phis"], dtype=F64, device=DEV)
        F, _ = evaluate(chain, phis, betas, [15.0] * len(betas), [0] * len(betas))
        out[f"m{r['n_members']}_Tc{r['Tc']:g}"] = [float(x) for x in F]
    (DIR / "ens_beta_scan.json").write_text(json.dumps(out))
    for i, b in enumerate(betas):
        print(f"{b:.4f} F_T1 {out['F_T1'][i]:.3f} "
              + " ".join(f"{k} {v[i]:.3f}" for k, v in out.items() if k.startswith("m")))


if __name__ == "__main__":
    main()
