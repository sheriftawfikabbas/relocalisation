import stat
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_launcher_exists_and_is_executable():
    script = REPO / "run_relocalization.sh"
    assert script.exists()
    mode = script.stat().st_mode
    assert mode & stat.S_IXUSR, "launcher must be executable"
    text = script.read_text()
    assert "tmux new-session" in text
    assert "reloc-prod" in text
    assert "--scenario both" in text
    assert "--profile production" in text
