"""Packaged resource inventory, wheel contents and sdist policy."""

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
REQUIRED_DATA = (
    "bone.atn",
    "h2o.atn",
    "Schneider2000.json",
)

# The sdist ships the sources needed to install, build the documentation, run
# the examples and run the shipped tests. The rendered gallery and other
# generated output are excluded; rendered documentation is built from the
# tagged repository.
SDIST_REQUIRED = {
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "CHANGELOG.md",
    "CITATION.cff",
    ".zenodo.json",
    "uv.lock",
    "docs/conf.py",
    "docs/_gallery.py",
    "docs/index.rst",
    "docs/changelog.rst",
    "docs/contributing.rst",
    "docs/simind_validation.md",
    "docs/spectub_cache_interference.md",
    "examples/README.rst",
    "examples/_python_connector_helpers.py",
    "examples/getting_started/README.rst",
    "examples/getting_started/plot_01_basic_simulation.py",
    "scripts/make_y90_bundle.py",
    "scripts/check_wheel_resources.py",
    "scripts/run_container_examples.sh",
    "scripts/run_container_validation.sh",
    "tests/test_make_y90_bundle.py",
    "tests/fixtures/minimal_variable_sections.smc",
    "tests/fixtures/golden/circular.hs",
    "docker/compose.yaml",
    "docker/import_guard/sitecustomize.py",
    ".github/workflows/tests.yml",
    ".github/workflows/publish.yml",
}
SDIST_FORBIDDEN_PREFIXES = (
    "docs/auto_examples/",
    "docs/examples/",
    "docs/superpowers/",
    "docs/_build/",
    "docs/_generated/",
    ".superpowers/",
    "tmp/",
    "output/",
    "simind/",
    "dist/",
    "build/",
)
SDIST_FORBIDDEN = {
    "AGENTS.md",
    "CLAUDE.md",
    "opencode.jsonc",
    "simind_manual.pdf",
    "docs/sg_execution_times.rst",
}


def _run_build(outdir: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(outdir),
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.fixture(scope="module")
def built_archives(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("archives")
    _run_build(outdir)
    wheels = list(outdir.glob("*.whl"))
    sdists = list(outdir.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1
    return wheels[0], sdists[0]


def _sdist_names(archive: Path) -> set[str]:
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    root = names[0].split("/", 1)[0]
    return {name[len(root) + 1 :] for name in names if name.startswith(f"{root}/")}


def _assert_no_forbidden_sdist_paths(names: set[str]) -> None:
    for name in names:
        assert name not in SDIST_FORBIDDEN, name
        assert not name.startswith(SDIST_FORBIDDEN_PREFIXES), name
        assert not name.endswith((".pyc", ".pyo")), name


def test_configs_inventory_includes_presets():
    assert set(REQUIRED_CONFIGS) <= set(list_configs())


def test_configs_get_resolves_existing_resources():
    assert all(get(name).is_file() for name in REQUIRED_CONFIGS)


@pytest.mark.slow
def test_built_wheel_contains_packaged_resources(built_archives):
    wheel, _ = built_archives
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    expected = {
        *(f"simind_python_connector/configs/{name}" for name in REQUIRED_CONFIGS),
        *(f"simind_python_connector/data/{name}" for name in REQUIRED_DATA),
    }
    assert expected <= names
    assert not any(
        name.startswith(("tests/", "docs/", "examples/", "scripts/")) for name in names
    )


@pytest.mark.slow
def test_built_sdist_ships_source_payload_and_excludes_generated_output(built_archives):
    _, sdist = built_archives
    names = _sdist_names(sdist)

    assert SDIST_REQUIRED <= names
    _assert_no_forbidden_sdist_paths(names)
