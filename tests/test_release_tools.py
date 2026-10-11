"""Tests for the release-preparation scripts and publication safeguards.

Everything here runs in temporary fixtures or reads the repository read-only; it
never mutates the checkout the tests live in.
"""

from __future__ import annotations

import datetime
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import docutils.core
import pytest
import yaml


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
WORKFLOWS = ROOT / ".github" / "workflows"

VERSION = "2.0.1"
DATE = "2026-10-11"
OLD_VERSION = "3.0.0"
OLD_DATE = "2026-10-09"


def _changelog_md(version: str = VERSION, date: str = DATE) -> str:
    return f"""# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [{version}] - {date}

### Fixed
- Something.

## [3.0.0] - 2026-10-09

### Added
- Historical note that must survive.

## [2.0.0] - 2026-10-09

### Fixed
- Older note.
"""


def _changelog_rst(version: str = VERSION, date: str = DATE) -> str:
    title = f"{version} - {date}"
    return f""".. _changelog:

Changelog
=========

{title}
{"-" * len(title)}

Fixed
~~~~~

- Top-level item with ``inline code`` and `a link <https://example.org/docs>`__.

  - Nested bullet one.
  - Nested bullet two.

3.0.0 - 2026-10-09
------------------

Added
~~~~~

- Historical note that must survive.

2.0.0 - 2026-10-09
------------------

Fixed
~~~~~

- Older note.
"""


def _pyproject(version: str) -> str:
    return (
        "[build-system]\n"
        'requires = ["setuptools>=77.0.3", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "\n"
        "[project]\n"
        'name = "simind-python-connector"\n'
        f'version = "{version}"\n'
    )


def _uv_lock(version: str) -> str:
    return (
        "[[package]]\n"
        'name = "markdown-it-py"\n'
        'version = "3.0.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        "\n"
        "[[package]]\n"
        'name = "simind-python-connector"\n'
        f'version = "{version}"\n'
        'source = { editable = "." }\n'
    )


def _citation(version: str, date: str) -> str:
    return (
        "cff-version: 1.2.0\n"
        'title: "simind-python-connector"\n'
        "authors:\n"
        "- family-names: Porter\n"
        "  given-names: Sam\n"
        '  orcid: "https://orcid.org/0000-0002-1815-4716"\n'
        f"date-released: {date}\n"
        f"version: {version}\n"
        "doi: 10.5281/zenodo.19605198\n"
        "license: Apache-2.0\n"
    )


def _zenodo(version: str) -> str:
    return (
        "{\n"
        '  "title": "simind-python-connector",\n'
        '  "license": "apache-2.0",\n'
        f'  "version": "{version}"\n'
        "}\n"
    )


def _package_init(version: str) -> str:
    return (
        '"""Package."""\n'
        "\n"
        "from importlib import metadata as _meta\n"
        "\n"
        "\n"
        'for _dist_name in ("simind-python-connector", __name__):\n'
        "    try:\n"
        "        __version__ = _meta.version(_dist_name)\n"
        "        break\n"
        "    except _meta.PackageNotFoundError:\n"
        "        continue\n"
        "else:\n"
        f'    __version__ = "{version}"\n'
    )


def _docs_conf(version: str) -> str:
    return (
        '"""Sphinx configuration."""\n'
        "\n"
        "from importlib import metadata as importlib_metadata\n"
        "\n"
        "\n"
        f'release = "{version}"\n'
        'for dist_name in ("simind-python-connector",):\n'
        "    try:\n"
        "        release = importlib_metadata.version(dist_name)\n"
        "        break\n"
        "    except importlib_metadata.PackageNotFoundError:\n"
        "        continue\n"
    )


def _build_project(
    tmp_path: Path,
    *,
    version: str = OLD_VERSION,
    date: str = OLD_DATE,
    markdown: str | None = None,
    rst: str | None = None,
    pyproject: str | None = None,
    citation: str | None = None,
    zenodo: str | None = None,
    package_init: str | None = None,
    docs_conf: str | None = None,
    uv_lock: str | None = None,
) -> Path:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "docs").mkdir()
    (project / "simind_python_connector").mkdir()
    for name in ("release.sh", "prepare_release.py"):
        shutil.copy(SCRIPTS / name, project / "scripts" / name)

    (project / "pyproject.toml").write_text(pyproject or _pyproject(version))
    (project / "uv.lock").write_text(uv_lock or _uv_lock(version))
    (project / "CITATION.cff").write_text(citation or _citation(version, date))
    (project / ".zenodo.json").write_text(zenodo or _zenodo(version))
    (project / "simind_python_connector" / "__init__.py").write_text(
        package_init or _package_init(version)
    )
    (project / "docs" / "conf.py").write_text(docs_conf or _docs_conf(version))
    (project / "CHANGELOG.md").write_text(markdown or _changelog_md())
    (project / "docs" / "changelog.rst").write_text(rst or _changelog_rst())
    return project


def _run(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/prepare_release.py", *args],
        cwd=project,
        capture_output=True,
        text=True,
    )


def _read(path: Path) -> str:
    return path.read_text()


def _snapshot(project: Path) -> dict[str, str]:
    return {
        path.relative_to(project).as_posix(): path.read_text()
        for path in sorted(project.rglob("*"))
        if path.is_file() and ".git" not in path.parts
    }


def _git_env() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.org",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.org",
    }


def _git(project: Path, *args: str, env: dict[str, str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=project, capture_output=True, text=True, env=env
    ).stdout


def _on(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))


def _needs(job: dict) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def test_prepare_rewrites_every_machine_surface(tmp_path: Path) -> None:
    project = _build_project(tmp_path)

    result = _run(project, VERSION, "--date", DATE)

    assert result.returncode == 0, result.stderr
    assert f'version = "{VERSION}"' in _read(project / "pyproject.toml")
    lock = _read(project / "uv.lock")
    assert f'name = "simind-python-connector"\nversion = "{VERSION}"' in lock
    assert 'name = "markdown-it-py"\nversion = "3.0.0"' in lock
    citation = _read(project / "CITATION.cff")
    assert f"version: {VERSION}" in citation
    assert f"date-released: {DATE}" in citation
    assert '"version": "2.0.1"' in _read(project / ".zenodo.json")
    assert f'__version__ = "{VERSION}"' in _read(
        project / "simind_python_connector" / "__init__.py"
    )
    assert f'release = "{VERSION}"' in _read(project / "docs" / "conf.py")


def test_historical_release_notes_are_untouched(tmp_path: Path) -> None:
    project = _build_project(tmp_path)
    markdown_before = _read(project / "CHANGELOG.md")
    rst_before = _read(project / "docs" / "changelog.rst")

    assert _run(project, VERSION, "--date", DATE).returncode == 0

    assert _read(project / "CHANGELOG.md") == markdown_before
    assert _read(project / "docs" / "changelog.rst") == rst_before
    assert "## [3.0.0] - 2026-10-09" in markdown_before
    assert "Historical note that must survive." in markdown_before


def test_citation_metadata_is_preserved(tmp_path: Path) -> None:
    project = _build_project(tmp_path)
    before = _read(project / "CITATION.cff").splitlines()

    assert _run(project, VERSION, "--date", DATE).returncode == 0

    after = _read(project / "CITATION.cff").splitlines()
    assert len(after) == len(before)
    changed = {old: new for old, new in zip(before, after) if old != new}
    assert changed == {
        "date-released: 2026-10-09": f"date-released: {DATE}",
        "version: 3.0.0": f"version: {VERSION}",
    }
    assert "cff-version: 1.2.0" in after
    assert "doi: 10.5281/zenodo.19605198" in after
    assert "license: Apache-2.0" in after


def test_lockfile_only_touches_the_root_package(tmp_path: Path) -> None:
    project = _build_project(tmp_path)
    before = _read(project / "uv.lock").splitlines()

    assert _run(project, VERSION, "--date", DATE).returncode == 0

    after = _read(project / "uv.lock").splitlines()
    changed = [(old, new) for old, new in zip(before, after) if old != new]
    assert changed == [('version = "3.0.0"', f'version = "{VERSION}"')]
    assert 'name = "markdown-it-py"\nversion = "3.0.0"' in "\n".join(after)


def test_installed_version_lookups_are_kept(tmp_path: Path) -> None:
    project = _build_project(tmp_path)

    assert _run(project, VERSION, "--date", DATE).returncode == 0

    package_init = _read(project / "simind_python_connector" / "__init__.py")
    assert "_meta.version(_dist_name)" in package_init
    assert "_meta.PackageNotFoundError" in package_init
    assert f'__version__ = "{VERSION}"' in package_init
    docs_conf = _read(project / "docs" / "conf.py")
    assert "importlib_metadata.version(dist_name)" in docs_conf
    assert "importlib_metadata.PackageNotFoundError" in docs_conf
    assert f'release = "{VERSION}"' in docs_conf


def test_missing_release_note_changes_nothing(tmp_path: Path) -> None:
    markdown = (
        "# Changelog\n\n## [Unreleased]\n\n"
        "## [3.0.0] - 2026-10-09\n\n### Added\n- Historical.\n"
    )
    project = _build_project(tmp_path, markdown=markdown)
    before = _snapshot(project)

    result = _run(project, VERSION, "--date", DATE)

    assert result.returncode != 0
    assert VERSION in result.stderr
    assert _snapshot(project) == before


def test_nonempty_unreleased_section_changes_nothing(tmp_path: Path) -> None:
    markdown = _changelog_md().replace(
        "## [Unreleased]\n\n## [2.0.1]",
        "## [Unreleased]\n\n### Added\n- Still pending.\n\n## [2.0.1]",
    )
    project = _build_project(tmp_path, markdown=markdown)
    before = _snapshot(project)

    result = _run(project, VERSION, "--date", DATE)

    assert result.returncode != 0
    assert "Unreleased" in result.stderr
    assert _snapshot(project) == before


def test_malformed_version_field_changes_nothing(tmp_path: Path) -> None:
    project = _build_project(
        tmp_path, pyproject='[project]\nname = "simind-python-connector"\n'
    )
    before = _snapshot(project)

    result = _run(project, VERSION, "--date", DATE)

    assert result.returncode != 0
    assert _snapshot(project) == before


def test_late_malformed_field_changes_nothing(tmp_path: Path) -> None:
    docs_conf = (
        '"""Sphinx configuration."""\n\n'
        "from importlib import metadata as importlib_metadata\n\n"
        'release = importlib_metadata.version("simind-python-connector")\n'
    )
    project = _build_project(tmp_path, docs_conf=docs_conf)
    before = _snapshot(project)

    result = _run(project, VERSION, "--date", DATE)

    assert result.returncode != 0
    assert "docs/conf.py" in result.stderr
    assert _snapshot(project) == before


def test_impossible_calendar_date_changes_nothing(tmp_path: Path) -> None:
    impossible = "2026-02-30"
    project = _build_project(
        tmp_path,
        markdown=_changelog_md(date=impossible),
        rst=_changelog_rst(date=impossible),
    )
    before = _snapshot(project)

    result = _run(project, VERSION, "--date", impossible)

    assert result.returncode != 0
    assert impossible in result.stderr
    assert _snapshot(project) == before


def test_check_passes_without_mutating(tmp_path: Path) -> None:
    project = _build_project(tmp_path, version=VERSION, date=DATE)
    before = _snapshot(project)

    result = _run(project, VERSION, "--date", DATE, "--check")

    assert result.returncode == 0, result.stderr
    assert _snapshot(project) == before


def test_check_fails_on_a_stale_surface_without_mutating(tmp_path: Path) -> None:
    project = _build_project(tmp_path)
    before = _snapshot(project)

    result = _run(project, VERSION, "--date", DATE, "--check")

    assert result.returncode != 0
    assert "pyproject.toml" in result.stderr
    assert _snapshot(project) == before


def test_hand_written_nested_rst_notes_are_valid(tmp_path: Path) -> None:
    project = _build_project(tmp_path)
    rst = _read(project / "docs" / "changelog.rst")
    assert "``inline code``" in rst
    assert "Nested bullet one." in rst

    result = _run(project, VERSION, "--date", DATE)

    assert result.returncode == 0, result.stderr
    docutils.core.publish_doctree(
        rst, settings_overrides={"report_level": 2, "halt_level": 2}
    )


def test_release_wrapper_has_no_git_side_effects(tmp_path: Path) -> None:
    today = datetime.date.today().isoformat()
    project = _build_project(
        tmp_path, markdown=_changelog_md(date=today), rst=_changelog_rst(date=today)
    )
    env = _git_env()
    for command in (["init", "-q"], ["add", "."], ["commit", "-qm", "init"]):
        subprocess.run(["git", *command], cwd=project, check=True, env=env)
    head_before = _git(project, "rev-parse", "HEAD", env=env)

    result = subprocess.run(
        ["bash", "scripts/release.sh", VERSION],
        cwd=project,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert _git(project, "rev-parse", "HEAD", env=env) == head_before
    assert _git(project, "tag", env=env).strip() == ""
    assert _git(project, "diff", "--cached", "--name-only", env=env).strip() == ""
    assert f'version = "{VERSION}"' in _read(project / "pyproject.toml")
    assert _git(project, "status", "--porcelain", env=env).strip() != ""


def test_publish_workflow_tests_the_release_tag_then_publishes() -> None:
    text = _read(WORKFLOWS / "publish.yml")
    workflow = yaml.safe_load(text)
    jobs = workflow["jobs"]

    assert jobs["tests"]["uses"] == "./.github/workflows/tests.yml"
    assert "tests" in _needs(jobs["publish"])
    assert "tomllib" in text
    assert 'if [ "$GITHUB_REF_NAME" != "v${version}" ]; then' in text
    assert "${{ github.event.release.tag_name }}" in text


def test_publish_workflow_rejects_a_tag_without_the_v_prefix(tmp_path: Path) -> None:
    workflow = yaml.safe_load(_read(WORKFLOWS / "publish.yml"))
    run = next(
        step["run"]
        for step in workflow["jobs"]["publish"]["steps"]
        if step.get("name") == "Check the tag matches the package version"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\necho 2.0.1\n")
    fake_python.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    bare = subprocess.run(
        ["bash", "-c", run],
        capture_output=True,
        text=True,
        env={**env, "GITHUB_REF_NAME": "2.0.1"},
    )
    prefixed = subprocess.run(
        ["bash", "-c", run],
        capture_output=True,
        text=True,
        env={**env, "GITHUB_REF_NAME": "v2.0.1"},
    )

    assert bare.returncode != 0
    assert "does not match" in bare.stdout
    assert prefixed.returncode == 0


def test_tests_workflow_preserves_matrix_and_checks() -> None:
    text = _read(WORKFLOWS / "tests.yml")
    workflow = yaml.safe_load(text)

    assert "workflow_call" in _on(workflow)
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    assert matrix == ["3.10", "3.11", "3.12", "3.13"]
    assert (
        "not integration and not requires_sirf and not requires_stir and "
        "not requires_simind and not requires_pytomography and not requires_cil "
        "and not ci_skip"
    ) in text.replace("\n", " ").replace("  ", " ")
    assert "from simind_python_connector.configs import get" in text
    assert "from simind_python_connector.data import data_path" in text
    assert "scripts/run_container_validation.sh --only-core" in text
    assert "scripts/run_container_examples.sh --only-core" in text


def test_build_requires_spdx_capable_setuptools() -> None:
    assert re.search(r"setuptools>=77\.0\.3", _read(ROOT / "pyproject.toml"))
