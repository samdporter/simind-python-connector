#!/usr/bin/env python
"""Prepare the release metadata for VERSION without committing or publishing.

The release notes themselves are hand-written: CHANGELOG.md carries a dated,
non-empty ``## [VERSION] - DATE`` section with exactly one empty
``## [Unreleased]`` section above it, and docs/changelog.rst carries the matching
dated section. This script only validates those notes and rewrites the machine
version surfaces.

Usage: python scripts/prepare_release.py VERSION --date YYYY-MM-DD [--check]
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PYPROJECT = ROOT / "pyproject.toml"
UV_LOCK = ROOT / "uv.lock"
CITATION = ROOT / "CITATION.cff"
ZENODO = ROOT / ".zenodo.json"
PACKAGE_INIT = ROOT / "simind_python_connector" / "__init__.py"
DOCS_CONF = ROOT / "docs" / "conf.py"
CHANGELOG_MD = ROOT / "CHANGELOG.md"
CHANGELOG_RST = ROOT / "docs" / "changelog.rst"

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MD_HEADING = re.compile(
    r"^## \[([^\]]+)\](?: - (\d{4}-\d{2}-\d{2}))?[ \t]*$", re.MULTILINE
)
_RST_TITLE = re.compile(
    r"^(?P<title>\d+\.\d+\.\d+ - \d{4}-\d{2}-\d{2})\n-+[ \t]*$", re.MULTILINE
)


class PrepareError(Exception):
    """A release surface is missing or malformed."""


def _md_sections(markdown: str) -> list[tuple[str, str | None, str]]:
    matches = list(_MD_HEADING.finditer(markdown))
    sections = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections.append((match.group(1), match.group(2), markdown[match.end() : end]))
    return sections


def _rst_release_bodies(rst: str) -> dict[str, str]:
    matches = list(_RST_TITLE.finditer(rst))
    bodies: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(rst)
        bodies[match.group("title")] = rst[match.end() : end]
    return bodies


def _validate_changelogs(markdown: str, rst: str, version: str, date: str) -> None:
    sections = _md_sections(markdown)
    unreleased = [body for name, _, body in sections if name == "Unreleased"]
    if len(unreleased) != 1:
        raise PrepareError(
            "CHANGELOG.md must have exactly one '## [Unreleased]' heading"
        )
    if unreleased[0].strip():
        raise PrepareError(
            "the '## [Unreleased]' section of CHANGELOG.md must be empty"
        )

    release = [
        (entry_date, body) for name, entry_date, body in sections if name == version
    ]
    if not release:
        raise PrepareError(f"CHANGELOG.md has no '## [{version}]' release-notes entry")
    entry_date, entry_body = release[0]
    if entry_date != date:
        raise PrepareError(
            f"CHANGELOG.md '[{version}]' entry is dated {entry_date}, expected {date}"
        )
    if not entry_body.strip():
        raise PrepareError(f"CHANGELOG.md '[{version}]' entry is empty")

    title = f"{version} - {date}"
    rst_body = _rst_release_bodies(rst).get(title)
    if rst_body is None:
        raise PrepareError(f"docs/changelog.rst has no '{title}' release-notes entry")
    if not rst_body.strip():
        raise PrepareError(f"docs/changelog.rst '{title}' entry is empty")


def _set_pyproject(text: str, version: str) -> str:
    text, changes = re.subn(r'(?m)^version = "[^"]*"$', f'version = "{version}"', text)
    if changes != 1:
        raise PrepareError(
            "pyproject.toml: expected exactly one 'version = \"...\"' line"
        )
    return text


def _set_uv_lock(text: str, version: str) -> str:
    text, changes = re.subn(
        r'(?m)^(name = "simind-python-connector"\nversion = ")[^"]*(")$',
        rf"\g<1>{version}\g<2>",
        text,
    )
    if changes != 1:
        raise PrepareError(
            'uv.lock: expected exactly one editable "simind-python-connector" package'
        )
    return text


def _set_citation(text: str, version: str, date: str) -> str:
    text, version_changes = re.subn(r"(?m)^version: .*$", f"version: {version}", text)
    text, date_changes = re.subn(
        r"(?m)^date-released: .*$", f"date-released: {date}", text
    )
    if version_changes != 1:
        raise PrepareError(
            "CITATION.cff: expected exactly one top-level 'version:' line"
        )
    if date_changes != 1:
        raise PrepareError("CITATION.cff: expected exactly one 'date-released:' line")
    return text


def _set_zenodo(text: str, version: str) -> str:
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise PrepareError(f".zenodo.json is not valid JSON: {exc}") from exc
    text, changes = re.subn(r'"version": "[^"]*"', f'"version": "{version}"', text)
    if changes != 1:
        raise PrepareError(".zenodo.json: expected exactly one '\"version\"' field")
    return text


def _set_package_init(text: str, version: str) -> str:
    text, changes = re.subn(
        r'(?m)^    __version__ = "[^"]*"$', f'    __version__ = "{version}"', text
    )
    if changes != 1:
        raise PrepareError(
            "simind_python_connector/__init__.py: expected exactly one fallback "
            "'__version__ = \"...\"' line"
        )
    return text


def _set_docs_conf(text: str, version: str) -> str:
    text, changes = re.subn(r'(?m)^release = "[^"]*"$', f'release = "{version}"', text)
    if changes != 1:
        raise PrepareError(
            "docs/conf.py: expected exactly one 'release = \"...\"' line"
        )
    return text


def _machine_surfaces(version: str, date: str) -> dict[Path, str]:
    return {
        PYPROJECT: _set_pyproject(PYPROJECT.read_text(), version),
        UV_LOCK: _set_uv_lock(UV_LOCK.read_text(), version),
        CITATION: _set_citation(CITATION.read_text(), version, date),
        ZENODO: _set_zenodo(ZENODO.read_text(), version),
        PACKAGE_INIT: _set_package_init(PACKAGE_INIT.read_text(), version),
        DOCS_CONF: _set_docs_conf(DOCS_CONF.read_text(), version),
    }


def prepare(version: str, date: str, *, check: bool) -> None:
    if not _VERSION_RE.match(version):
        raise PrepareError(f"invalid version {version!r}; expected X.Y.Z")
    if not _DATE_RE.match(date):
        raise PrepareError(f"invalid date {date!r}; expected YYYY-MM-DD")
    try:
        datetime.date.fromisoformat(date)
    except ValueError as exc:
        raise PrepareError(f"invalid date {date!r}; {exc}") from exc

    _validate_changelogs(
        CHANGELOG_MD.read_text(), CHANGELOG_RST.read_text(), version, date
    )

    surfaces = _machine_surfaces(version, date)
    if check:
        stale = [path for path, text in surfaces.items() if text != path.read_text()]
        if stale:
            names = ", ".join(str(path.relative_to(ROOT)) for path in stale)
            raise PrepareError(f"release surfaces are not at {version}: {names}")
        return

    for path, text in surfaces.items():
        path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="release version, X.Y.Z")
    parser.add_argument("--date", required=True, help="release date, YYYY-MM-DD")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the eight release surfaces are already at VERSION without writing",
    )
    args = parser.parse_args(argv)

    try:
        prepare(args.version, args.date, check=args.check)
    except PrepareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(f"{args.version} is consistent across the release surfaces")
    else:
        print(f"prepared release metadata for {args.version} (nothing committed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
