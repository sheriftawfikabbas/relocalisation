import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run_ansatz_sweep(out: Path, ansatz: str, max_evals: int) -> None:
    cmd = [
        sys.executable, "-m", "relocalization.cli", "sweep",
        "--scenario", "both", "--profile", "dev", "--ansatze", ansatz,
        "--n-sites", "61", "--t-max", "16", "--dt", "0.05", "--sample-every", "10",
        "--max-evals", str(max_evals), "--n-restarts", "1",
        "--betas", "0,1", "--output", str(out), "--jobs", "1",
    ]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr


def test_merge_combines_per_ansatz_sweeps(tmp_path):
    out = tmp_path / "fair_budget"
    _run_ansatz_sweep(out / "by_ansatz" / "single_sinusoid", "single_sinusoid", max_evals=12)
    _run_ansatz_sweep(out / "by_ansatz" / "two_tone", "two_tone", max_evals=20)

    cmd = [
        sys.executable, "-m", "relocalization.merge_fair_budget_sweep",
        str(out), "--evals-per-param", "6",
    ]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr

    sweep_a = out / "sweep_A.csv"
    sweep_b = out / "sweep_B.csv"
    beta_star = out / "beta_star.csv"
    manifest = out / "manifest.json"
    assert sweep_a.exists() and sweep_b.exists() and beta_star.exists() and manifest.exists()
    assert (out / "figures" / "r_vs_beta_A.png").exists()
    assert (out / "figures" / "r_vs_beta_B.png").exists()

    text_a = sweep_a.read_text()
    assert "single_sinusoid" in text_a and "two_tone" in text_a

    data = json.loads(manifest.read_text())
    assert data["per_ansatz"]["single_sinusoid"]["max_evals"] == 12
    assert data["per_ansatz"]["two_tone"]["max_evals"] == 20
    assert data["per_ansatz"]["two_tone"]["dim"] == 5


def test_merge_requires_existing_by_ansatz_dir(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    cmd = [
        sys.executable, "-m", "relocalization.merge_fair_budget_sweep",
        str(out),
    ]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=60)
    assert res.returncode != 0
