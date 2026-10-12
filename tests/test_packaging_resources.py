"""Packaged-resource inventory and wheel-content guarantees."""

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

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

CACHE_LIMITATION_DOC = "docs/spectub_cache_interference.md"

# Local-only files that must never reach a published artifact.
FORBIDDEN_LOCAL_FILES = (
    "AGENTS.md",
    "CLAUDE.md",
    "opencode.jsonc",
    "simind_manual.pdf",
)

# intersphinx is dropped from the docs build: fetching inventories needs the
# network, and an offline fetch warns and fails the strict (-W) build.
SDIST_SPHINX_EXTENSIONS = ",".join(
    [
        "sphinx.ext.autodoc",
        "sphinx.ext.napoleon",
        "sphinx.ext.viewcode",
        "sphinx.ext.githubpages",
        "sphinx_autodoc_typehints",
        "myst_parser",
    ]
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
            # Extract version from wheel filename to find dist-info
            wheel_name = wheel_path.name
            version = wheel_name.replace("simind_python_connector-", "").replace(
                "-py3-none-any.whl", ""
            )
            dist_info_prefix = f"simind_python_connector-{version}.dist-info"
            for name in source.namelist():
                if not name.startswith(dist_info_prefix):
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


def _copy_project(destination: Path) -> None:
    shutil.copytree(
        ROOT,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "__pycache__",
            "*.egg-info",
            "dist",
            "build",
            "_build",
            "superpowers",
            "simind",
            "simind_manual.pdf",
            ".pytest_cache",
            ".ruff_cache",
        ),
    )


def _plant_local_material(project: Path) -> None:
    (project / "docs" / "superpowers").mkdir()
    (project / "docs" / "superpowers" / "handoff.md").write_text("handoff")
    (project / "docs" / "superpowers" / "plan.rst").write_text("plan")
    (project / "docs" / "superpowers" / "notes.py").write_text("notes = 1\n")
    (project / "docs" / "_build").mkdir()
    (project / "docs" / "_build" / "stale.rst").write_text("stale")
    (project / "AGENTS.md").write_text("local instructions")
    (project / "CLAUDE.md").write_text("local instructions")
    (project / "opencode.jsonc").write_text("local config")
    (project / "simind_manual.pdf").write_bytes(b"%PDF-1.4 canary")
    (project / "simind").mkdir()
    (project / "simind" / "simind").write_text("licensed binary")


def _extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(destination, filter="data")
        else:
            tar.extractall(destination)


def _is_forbidden(relative: PurePosixPath) -> bool:
    if relative.name in FORBIDDEN_LOCAL_FILES:
        return True
    if relative.parts[0] == "simind":
        return True
    return relative.parts[:2] in {("docs", "superpowers"), ("docs", "_build")}


@pytest.fixture(scope="module")
def sdist(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    work = tmp_path_factory.mktemp("sdist")
    project = work / "project"
    _copy_project(project)
    _plant_local_material(project)

    dist = work / "dist"
    dist.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--outdir",
            str(dist),
            str(project),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    archive = next(dist.glob("*.tar.gz"))
    with tarfile.open(archive) as tar:
        members = tar.getnames()
    extracted_parent = work / "extracted"
    extracted_parent.mkdir()
    _extract(archive, extracted_parent)
    extracted = next(extracted_parent.iterdir())
    return SimpleNamespace(
        archive=archive, members=members, project=project, extracted=extracted
    )


@pytest.mark.slow
def test_sdist_ships_the_cache_limitation_page(sdist: SimpleNamespace):
    member = f"{sdist.extracted.name}/{CACHE_LIMITATION_DOC}"
    assert member in sdist.members, sorted(sdist.members)
    with tarfile.open(sdist.archive) as tar:
        assert (
            tar.extractfile(member).read() == (ROOT / CACHE_LIMITATION_DOC).read_bytes()
        )


@pytest.mark.slow
def test_sdist_excludes_local_material_and_canaries(sdist: SimpleNamespace):
    assert (sdist.project / "docs" / "superpowers" / "handoff.md").is_file()
    assert (sdist.project / "docs" / "_build" / "stale.rst").is_file()
    assert (sdist.project / "simind" / "simind").is_file()

    root = sdist.extracted.name
    offenders = [
        name
        for name in sdist.members
        if name != root and _is_forbidden(PurePosixPath(name).relative_to(root))
    ]
    assert offenders == []


@pytest.mark.slow
def test_extracted_sdist_docs_build_with_strict_sphinx(
    sdist: SimpleNamespace, tmp_path: Path
):
    extracted = sdist.extracted
    env = {**os.environ, "PYTHONPATH": str(extracted)}

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import simind_python_connector as p; print(p.__file__)",
        ],
        cwd=extracted,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert probe.returncode == 0, probe.stderr[-1000:]
    assert Path(probe.stdout.strip()).resolve().is_relative_to(extracted.resolve())

    html = tmp_path / "html"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            "-d",
            str(tmp_path / "doctrees"),
            "-D",
            f"extensions={SDIST_SPHINX_EXTENSIONS}",
            str(extracted / "docs"),
            str(html),
        ],
        cwd=extracted,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    page = html / "spectub_cache_interference.html"
    assert page.is_file()
    assert "SPECTUB all-views-cache interference" in page.read_text()
    changelog = (html / "changelog.html").read_text()
    assert 'href="spectub_cache_interference.html"' in changelog
