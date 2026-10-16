import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_simind,
]

# The examples plot their results; matplotlib is the "examples" extra.
pytest.importorskip("matplotlib")


@pytest.mark.slow
def test_run_all_examples():
    """Run the core examples through their command-line runner."""
    script_path = Path(__file__).resolve().parents[1] / "scripts/run_all_examples.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        input="y\n",
        capture_output=True,
        text=True,
        timeout=3600,
    )

    assert result.returncode == 0, (
        f"Script failed with return code {result.returncode}\n"
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
