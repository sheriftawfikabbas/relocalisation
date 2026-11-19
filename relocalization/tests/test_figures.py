from pathlib import Path

from relocalization.config import dev_config
from relocalization.figures import plot_r_vs_beta, write_beta_star_table


def _rows():
    return [
        {"scenario": "A", "ansatz": "single_sinusoid", "beta": 0.0, "r": 1.0},
        {"scenario": "A", "ansatz": "single_sinusoid", "beta": 1.0, "r": 0.8},
        {"scenario": "A", "ansatz": "single_sinusoid", "beta": 2.0, "r": 0.3},
        {"scenario": "A", "ansatz": "two_tone", "beta": 0.0, "r": 1.0},
        {"scenario": "A", "ansatz": "two_tone", "beta": 1.0, "r": 0.9},
        {"scenario": "A", "ansatz": "two_tone", "beta": 2.0, "r": 0.6},
    ]


def test_plot_r_vs_beta_writes_png(tmp_path):
    out = tmp_path / "r_vs_beta.png"
    plot_r_vs_beta(_rows(), out)
    assert out.exists() and out.stat().st_size > 0


def test_beta_star_table_written_and_ordered(tmp_path):
    out = tmp_path / "beta_star.csv"
    rows = write_beta_star_table(_rows(), threshold=0.5, path=out)
    assert out.exists()
    assert all("r_initial" in r for r in rows)
    bstar = {r["ansatz"]: r["beta_star"] for r in rows}
    assert bstar["two_tone"] == bstar["single_sinusoid"] == 1.5
