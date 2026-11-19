"""Regenerate the arrest-score figures of the Supplemental Material.

Reads results/relocalization/sweep_{A,B}.csv (production sweep; see README) and
writes the arrest score r-versus-beta figures.  The scenario-B figure is the
one included in the Supplemental Material.

Run: python -m relocalization.plot_arrest
"""

from __future__ import annotations

import csv
from pathlib import Path

from .figures import plot_r_vs_beta

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/relocalization"


def main() -> None:
    for scenario, name in (("B", "r_vs_beta_B.png"), ("A", "r_vs_beta_A.png")):
        with (RES / f"sweep_{scenario}.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        out = RES / "figures" / name
        plot_r_vs_beta(rows, out)
        print("wrote", out)


if __name__ == "__main__":
    main()
