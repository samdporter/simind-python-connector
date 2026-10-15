"""Packaged-resource inventory and wheel-content guarantees."""

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from simind_python_connector.configs import get
from simind_python_connector.configs import list as list_configs
from simind_python_connector.data import load_table


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


@pytest.mark.parametrize(
    ("name", "first_row"),
    [
        ("h2o.atn", (1.0e-3, 4.077e3)),
        ("bone.atn", (1.0e-3, 3.442e3)),
    ],
)
def test_load_table_reads_packaged_attenuation_data(name, first_row):
    table = load_table(name)

    assert table.ndim == 2
    assert table.shape[1] == 2
    assert table.shape[0] > 10
    assert table[0] == pytest.approx(first_row)


@pytest.mark.slow
def test_built_artifacts_contain_packaged_resources(tmp_path: Path):
    """Build wheel and sdist artifacts and assert their required files ship."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
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
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1
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

    with tarfile.open(sdists[0], "r:gz") as archive:
        names = archive.getnames()
    sdist_root = sdists[0].name.removesuffix(".tar.gz")
    for packaged in (
        "tests/fixtures/minimal_variable_sections.smc",
        "examples/run_all_examples.sh",
        "scripts/run_all_examples.py",
        "scripts/run_container_examples.sh",
        "scripts/run_container_validation.sh",
        "docker/compose.yaml",
        "docker/constraints.txt",
        "docker/import_guard/sitecustomize.py",
        "docker/python/Dockerfile",
        "docker/stir/Dockerfile",
        ".dockerignore",
    ):
        assert f"{sdist_root}/{packaged}" in names, packaged
