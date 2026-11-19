import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_cli_sweep_dev_overrides_writes_outputs(tmp_path):
    out = tmp_path / "reloc"
    cmd = [
        sys.executable, "-m", "relocalization.cli", "sweep",
        "--scenario", "A", "--profile", "dev", "--output", str(out), "--jobs", "1",
        "--n-sites", "61", "--t-max", "16", "--dt", "0.05", "--sample-every", "10",
        "--max-evals", "12", "--n-restarts", "1",
        "--betas", "0,1", "--ansatze", "single_sinusoid",
    ]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr
    assert (out / "manifest.json").exists()
    assert (out / "sweep_A.csv").exists()
    assert (out / "beta_star.csv").exists()
    assert (out / "figures" / "r_vs_beta_A.png").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["profile"] == "dev"
    assert "git_commit" in manifest and "config" in manifest
