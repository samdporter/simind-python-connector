"""Packaged-resource inventory and wheel-content guarantees."""

import subprocess
import sys
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
    names = set(list_configs())
    for required in REQUIRED_CONFIGS:
        assert required in names, (required, sorted(names))


def test_configs_get_resolves_existing_resources():
    for name in REQUIRED_CONFIGS:
        resource = get(name)
        # Traversable: has to expose the packaged bytes either directly or
        # via a materialised path.
        assert resource.is_file(), name


@pytest.mark.slow
def test_built_wheel_contains_packaged_resources(tmp_path: Path):
    """Build a wheel from tracked sources and assert package data ships."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
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
        names = archive.namelist()
    for packaged in (
        "simind_python_connector/configs/input.smc",
        "simind_python_connector/configs/AnyScan.yaml",
        "simind_python_connector/data/h2o.atn",
        "simind_python_connector/data/bone.atn",
        "simind_python_connector/data/Schneider2000.json",
    ):
        assert packaged in names, packaged


@pytest.mark.slow
def test_configs_load_from_zipped_install(tmp_path: Path):
    """SimulationConfig must accept Traversables from zipped wheel installs."""
    built = tmp_path / "build"
    built.mkdir()
    build_result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(built), str(ROOT)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert build_result.returncode == 0, build_result.stderr[-2000:]
    wheel_path = next(built.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as source:
        with zipfile.ZipFile(tmp_path / "pkg.zip", "w") as target:
            for name in source.namelist():
                if not name.startswith("simind_python_connector-1.0.1.dist-info"):
                    target.writestr(name, source.read(name))

    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path / 'pkg.zip')!r})\n"
        "from simind_python_connector import SimulationConfig\n"
        "from simind_python_connector.configs import get\n"
        "cfg = SimulationConfig(get('AnyScan.yaml'))\n"
        "assert cfg.get_value('photon_energy') > 0\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert "OK" in result.stdout
