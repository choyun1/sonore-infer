"""The committed plots in results/plots/ are what tools/results_plots.py draws from results/data/."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_committed_plots_match_their_data(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "results_plots.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    drawn = sorted(path.name for path in tmp_path.glob("*.svg"))
    committed = sorted(path.name for path in (ROOT / "results" / "plots").glob("*.svg"))
    assert drawn == committed
    for name in drawn:
        # Compared as text so that a checkout with Windows line endings still matches.
        expected = (ROOT / "results" / "plots" / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert (tmp_path / name).read_text(encoding="utf-8") == expected, name
