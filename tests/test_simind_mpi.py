"""Run the serial vs MPI SIMIND agreement check inside the container suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.requires_simind]

ROOT = Path(__file__).resolve().parents[1]


def test_serial_and_mpi_simind_agree(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_simind_mpi.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
