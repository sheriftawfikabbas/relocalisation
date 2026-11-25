# relocalisation

Code and stored results for the paper on **reversing wavepacket spreading in the nonlinear Harper (Aubry–André) lattice with phase-only control** (mean-field DNLS, α = (√5−1)/2, λ = 1, β = U·N_b), submitted to *Communications in Nonlinear Science and Numerical Simulation*.

A delta packet spreads freely for T₁ = 15; a spatially uniform quasi-periodic phase φ(t) is then applied for T_c, and the return fidelity F = |ψ_{j₀}(T₁+T_c)|² is maximised by back-propagation through the Strang split-step propagator (dt = 0.05). Everything in the paper's tables and quoted numbers is regenerated from the JSON files in `results/`.

## Layout

| Path | Content |
|---|---|
| `relocalization/dnls.py` | DNLS/Harper split-step simulator |
| `relocalization/reversal_phase.py` | piecewise-constant phase-only control, π-kick echo, Lie-rank (controllability) test |
| `relocalization/reversal_smooth.py` | smooth band-limited control φ = Σ b_k sin(πkt/T_c), noise and calibration robustness |
| `relocalization/reversal_study.py` | `ensemble`, `scaling`, `bandwidth`, `manybody` studies; shared `Chain` / `evaluate` / TWA / exact few-body code |
| `relocalization/reversal_completion.py` | follow-up studies: `bw_lam`, `scaling120`, `ens_beta`, `site_cross`, `fewbody`, `states`, `lie`, `starts`, `modelfree`, `modelfree_long` |
| `relocalization/ens_beta_scan.py` | fine β-scan (β ∈ [0.5, 1.5]) of the ensemble controls; writes `ens_beta_scan.json` |
| `relocalization/twa.py` | truncated-Wigner sampling used for the many-body (N_b) checks |
| `relocalization/plot_phase_reversal.py` | main and supplement figures (`results/relocalization/figures/phase_reversal*.png`) |
| `relocalization/make_reversal_tables.py` | writes the LaTeX tables (`reversal_tables.tex`, `reversal_main_table.tex`) from the JSON results |
| `relocalization/check_manuscript_numbers.py` | asserts that every number quoted in the manuscript matches the stored results (108 checks) |
| `relocalization/summarize_completion.py` | prints a compact summary of the completion studies |
| `results/relocalization/reversal/` | phase-only and smooth controls (β = 0, 0.5, 1, 2, 4), Lie rank, summary |
| `results/relocalization/reversal_study/` | ensemble, scaling (T_c), bandwidth, many-body results |
| `results/relocalization/reversal_completion_cpu/` | completion studies (CPU runs; preferred where both copies exist) |
| `results/relocalization/reversal_completion/` | completion studies (GPU runs: `fewbody` incl. exact N_b = 2, 3, `lie`, and duplicates) |
| `run_relocalization.sh` | legacy launcher for the earlier (arrest/relocalization) sweep; not needed for the reversal paper |

## Setup

```bash
pip install -r requirements.txt     # numpy scipy matplotlib torch numba pytest
```

Set `REV_DEVICE=cpu` (or `cuda`) to choose the device; the default is CUDA if available. All optimisation runs use float64.

## Reproducing the paper

Regenerate tables and check the manuscript against the stored results (seconds, CPU):

```bash
MANUSCRIPT_DIR=/path/to/manuscript python -m relocalization.check_manuscript_numbers   # needs main.tex, supp.tex
MANUSCRIPT_DIR=out/ python -m relocalization.make_reversal_tables                      # writes the .tex tables there
python -m relocalization.summarize_completion
REV_DEVICE=cpu python -m relocalization.plot_phase_reversal                            # figures
```

The manuscript sources are not part of this repository, so the checker needs `MANUSCRIPT_DIR`.

Re-running the optimisations (hours; run detached, e.g. in `tmux`):

```bash
python -m relocalization.reversal_phase 0 0.5 1 2 4          # phase-only controls, Lie rank
python -m relocalization.reversal_smooth 1                   # smooth controls + robustness (per β)
python -m relocalization.reversal_study ensemble 0 1         # also: scaling, bandwidth, manybody
python -m relocalization.reversal_completion bw_lam ens_beta site_cross fewbody states lie starts modelfree modelfree_long scaling120
REV_DEVICE=cpu python -m relocalization.ens_beta_scan        # after ens_beta
```

Optimisation is seed- and hardware-dependent at the level of local optima, so a re-run reproduces the qualitative results and approximate fidelities, not necessarily bit-identical files; the stored JSON files are the ones the paper's numbers come from.

## Tests

```bash
python -m pytest -q relocalization/tests
```

## Data availability

All data underlying the figures and tables are in `results/`. License: MIT.
