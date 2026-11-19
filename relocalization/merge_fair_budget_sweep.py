"""Merge per-ansatz fair-budget sweeps into combined sweep_A.csv / sweep_B.csv,
beta_star.csv, and per-scenario figures.

Companion to run_fair_budget_sweep.sh: that script sweeps each ansatz
independently, with its own evaluation budget scaled by its free-parameter
count, into ``<out>/by_ansatz/<ansatz>/{sweep_A,sweep_B}.csv``. This script
concatenates those per-ansatz results into ``<out>/sweep_A.csv``,
``<out>/sweep_B.csv``, and regenerates the aggregate ``beta_star.csv`` and
``figures/r_vs_beta_{A,B}.png`` using the same functions the production
pipeline uses (relocalization.figures), so the merged output is a drop-in
comparison against results/relocalization/.

Run:
    python -m relocalization.merge_fair_budget_sweep <output_dir> [--evals-per-param N]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .cli import _git_commit, _write_rows_csv
from .config import production_config
from .drives import param_spec
from .figures import plot_r_vs_beta, write_beta_star_table

SCENARIOS = ("A", "B")


def _read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def merge(out_dir: Path, evals_per_param: Optional[int]) -> None:
    by_ansatz = out_dir / "by_ansatz"
    if not by_ansatz.is_dir():
        raise SystemExit(f"no per-ansatz sweeps found under {by_ansatz}")
    ansatz_dirs = sorted(p.name for p in by_ansatz.iterdir() if p.is_dir())
    if not ansatz_dirs:
        raise SystemExit(f"no per-ansatz sweeps found under {by_ansatz}")

    merged: Dict[str, List[Dict[str, object]]] = {s: [] for s in SCENARIOS}
    per_ansatz_config: Dict[str, Dict[str, object]] = {}
    for ansatz in ansatz_dirs:
        adir = by_ansatz / ansatz
        for scenario in SCENARIOS:
            csv_path = adir / f"sweep_{scenario}.csv"
            if csv_path.exists():
                merged[scenario].extend(_read_rows(csv_path))
        manifest_path = adir / "manifest.json"
        if manifest_path.exists():
            cfg = json.loads(manifest_path.read_text())["config"]
            per_ansatz_config[ansatz] = {
                "max_evals": cfg["max_evals"],
                "n_restarts": cfg["n_restarts"],
                "dim": param_spec(ansatz).dim,
            }

    all_rows: List[Dict[str, object]] = []
    for scenario in SCENARIOS:
        rows = merged[scenario]
        if not rows:
            continue
        _write_rows_csv(out_dir / f"sweep_{scenario}.csv", rows)
        plot_r_vs_beta(rows, out_dir / "figures" / f"r_vs_beta_{scenario}.png")
        all_rows.extend(rows)

    threshold = production_config().beta_star_threshold
    write_beta_star_table(all_rows, threshold, out_dir / "beta_star.csv")

    manifest = {
        "kind": "fair_budget_sweep_merge",
        "git_commit": _git_commit(),
        "evals_per_param": evals_per_param,
        "per_ansatz": per_ansatz_config,
        "merged_at": time.time(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=list))
    print(f"Wrote merged fair-budget sweep to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--evals-per-param", type=int, default=None)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    merge(Path(args.output_dir), args.evals_per_param)


if __name__ == "__main__":
    main()
