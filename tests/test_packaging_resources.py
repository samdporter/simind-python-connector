"""Packaged resource inventory and wheel contents."""

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from simind_python_connector.configs import get
from simind_python_connector.configs import list as list_configs


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit

REQUIRED_CONFIGS = (
    "AnyScan.yaml",
    "Example.yaml",
    "MLD001_SCAN0.yaml",
    "Discovery670.yaml",
    "input.smc",
)


def test_configs_inventory_includes_presets():
    assert set(REQUIRED_CONFIGS) <= set(list_configs())


def test_configs_get_resolves_existing_resources():
    assert all(get(name).is_file() for name in REQUIRED_CONFIGS)


@pytest.mark.slow
def test_built_wheel_contains_packaged_resources(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    assert {
        "simind_python_connector/configs/input.smc",
        "simind_python_connector/configs/AnyScan.yaml",
        "simind_python_connector/data/h2o.atn",
        "simind_python_connector/data/bone.atn",
        "simind_python_connector/data/Schneider2000.json",
    } <= names
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(sdists) == 1
    with tarfile.open(sdists[0]) as archive:
        assert any(
            name.endswith("/docs/simind_validation.md") for name in archive.getnames()
        )
