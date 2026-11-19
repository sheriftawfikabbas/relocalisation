from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import List

from .config import RunConfig, dev_config, production_config
from .figures import plot_r_vs_beta, write_beta_star_table
from .sweep import beta_sweep


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _apply_overrides(cfg: RunConfig, args) -> RunConfig:
    changes = {}
    if args.n_sites is not None:
        changes["n_sites"] = args.n_sites
    if args.t_max is not None:
        changes["t_max"] = args.t_max
    if args.dt is not None:
        changes["dt"] = args.dt
    if args.sample_every is not None:
        changes["sample_every"] = args.sample_every
    if args.max_evals is not None:
        changes["max_evals"] = args.max_evals
    if args.n_restarts is not None:
        changes["n_restarts"] = args.n_restarts
    if args.seed is not None:
        changes["seed"] = args.seed
    if args.betas is not None:
        changes["betas"] = tuple(float(x) for x in args.betas.split(","))
    if args.ansatze is not None:
        changes["ansatze"] = tuple(args.ansatze.split(","))
    return replace(cfg, **changes)


def _write_rows_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def cmd_sweep(args) -> None:
    cfg = production_config() if args.profile == "production" else dev_config()
    cfg = _apply_overrides(cfg, args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = ["A", "B"] if args.scenario == "both" else [args.scenario]

    started = time.time()
    all_rows: List[dict] = []
    for scenario in scenarios:
        rows = beta_sweep(scenario, cfg.ansatze, cfg.betas, cfg, jobs=args.jobs)
        _write_rows_csv(output / f"sweep_{scenario}.csv", rows)
        plot_r_vs_beta(rows, output / "figures" / f"r_vs_beta_{scenario}.png")
        all_rows.extend(rows)
    write_beta_star_table(all_rows, cfg.beta_star_threshold, output / "beta_star.csv")

    manifest = {
        "profile": cfg.profile,
        "git_commit": _git_commit(),
        "scenarios": scenarios,
        "wall_seconds": round(time.time() - started, 2),
        "config": asdict(cfg),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=list))
    print(f"Wrote sweep outputs to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nonlinear re-localization study.")
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep", help="Run a beta sweep for one or both scenarios.")
    sweep.add_argument("--scenario", choices=["A", "B", "both"], default="both")
    sweep.add_argument("--profile", choices=["dev", "production"], default="dev")
    sweep.add_argument("--output", default="results/relocalization")
    sweep.add_argument("--jobs", type=int, default=1)
    for name in ("n-sites", "sample-every", "max-evals", "n-restarts", "seed"):
        sweep.add_argument(f"--{name}", type=int, default=None)
    for name in ("t-max", "dt"):
        sweep.add_argument(f"--{name}", type=float, default=None)
    sweep.add_argument("--betas", default=None, help="comma-separated, e.g. 0,1,2,5")
    sweep.add_argument("--ansatze", default=None, help="comma-separated ansatz names")
    sweep.set_defaults(func=cmd_sweep)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
