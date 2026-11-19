"""Regenerate the arrest-score figure of the Supplemental Material.

Reads results/relocalization/sweep_B.csv (production sweep; see README) and
writes the arrest score r-versus-beta figure included in the Supplemental
Material.

Run: python -m relocalization.plot_arrest
"""

from __future__ import annotations

import csv
from pathlib import Path

from .figures import plot_r_vs_beta

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/relocalization"


def main() -> None:
    with (RES / "sweep_B.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    out = RES / "figures" / "r_vs_beta_B.png"
    plot_r_vs_beta(rows, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
