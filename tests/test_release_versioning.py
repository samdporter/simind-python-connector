"""Check shared versions and release preparation in an isolated repository."""

import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

from simind_python_connector import __version__


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit


def test_documentation_uses_package_version():
    assert runpy.run_path(str(ROOT / "docs/conf.py"))["release"] == __version__


@pytest.mark.parametrize(
    ("bump", "expected"),
    [
        ("patch", "1.2.4"),
        ("minor", "1.3.0"),
        ("major", "2.0.0"),
        ("2.1.0", "2.1.0"),
        ("1.2.3", "1.2.3"),
    ],
)
def test_release_updates_metadata_and_tags(tmp_path, bump, expected):
    def run(*args):
        return subprocess.run(
            args, cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout.strip()

    (tmp_path / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/release.sh", tmp_path / "scripts/release.sh")
    (tmp_path / "simind_python_connector").mkdir()
    version_file = tmp_path / "simind_python_connector/_version.py"
    version_file.write_text('__version__ = "1.2.3"\n')
    (tmp_path / ".zenodo.json").write_text('{"version": "1.2.3"}\n')
    (tmp_path / "CITATION.cff").write_text(
        "version: 1.2.3\ndate-released: 2000-01-01\n"
    )
    (tmp_path / "CHANGELOG.md").write_text(f"## [{expected}] - 2026-10-16\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/changelog.rst").write_text(f"{expected} - 2026-10-16\n")
    run("git", "init")
    run("git", "config", "user.name", "Release Test")
    run("git", "config", "user.email", "release@example.com")
    run("git", "add", ".")
    run("git", "-c", "commit.gpgsign=false", "commit", "-m", "Initial")
    run("git", "config", "commit.gpgsign", "false")
    run("git", "config", "tag.gpgsign", "false")

    run("bash", "scripts/release.sh", bump)

    assert runpy.run_path(str(version_file))["__version__"] == expected
    assert json.loads((tmp_path / ".zenodo.json").read_text())["version"] == expected
    citation = (tmp_path / "CITATION.cff").read_text()
    assert f"version: {expected}\n" in citation
    assert "date-released: 2000-01-01" not in citation
    assert run("git", "tag", "--list") == f"v{expected}"
    assert run("git", "status", "--porcelain") == ""
