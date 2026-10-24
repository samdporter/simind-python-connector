"""Verify that an installed wheel exposes every packaged resource.

Run this from a temporary directory with the virtual environment's interpreter::

    cd "$(mktemp -d)"
    "$venv/bin/python" "$repo/scripts/check_wheel_resources.py" --repo "$repo"

It fails when the imported package comes from the checkout (or any interpreter
other than the virtual environment), when a bundled preset or data file is
missing, or when a resource is present but unreadable.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

import simind_python_connector
from simind_python_connector import configs, data


EXPECTED_CONFIGS = frozenset(
    {
        "AnyScan.yaml",
        "Discovery670.yaml",
        "Example.yaml",
        "MLD001_SCAN0.yaml",
        "input.smc",
    }
)
EXPECTED_DATA = frozenset({"bone.atn", "h2o.atn", "Schneider2000.json"})


def _fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def _read(resource) -> bytes:
    if not resource.is_file():
        _fail(f"missing resource: {resource}")
    with resource.open("rb") as handle:
        contents = handle.read()
    if not contents:
        _fail(f"empty resource: {resource}")
    return contents


def _check_import_location(repo: Path | None) -> Path:
    package_dir = Path(simind_python_connector.__file__).resolve().parent
    prefix = Path(sys.prefix).resolve()
    if prefix not in package_dir.parents:
        _fail(f"package imported from {package_dir}, outside virtualenv {prefix}")
    if repo is not None:
        checkout = repo.resolve()
        if checkout == package_dir or checkout in package_dir.parents:
            _fail(f"package imported from the checkout: {package_dir}")
    return package_dir


def _check_configs() -> None:
    names = set(configs.list())
    if names != EXPECTED_CONFIGS:
        _fail(
            "packaged presets differ: "
            f"missing={sorted(EXPECTED_CONFIGS - names)}, "
            f"unexpected={sorted(names - EXPECTED_CONFIGS)}"
        )
    for name in sorted(names):
        _read(configs.get(name))


def _check_data() -> None:
    package = resources.files(data.__name__)
    names = {
        item.name
        for item in package.iterdir()
        if item.is_file() and item.suffix in {".atn", ".json"}
    }
    if names != EXPECTED_DATA:
        _fail(
            "packaged data differs: "
            f"missing={sorted(EXPECTED_DATA - names)}, "
            f"unexpected={sorted(names - EXPECTED_DATA)}"
        )
    for name in sorted(names):
        contents = _read(data.data_path(name))
        if name.endswith(".json"):
            json.loads(contents)
        else:
            table = data.load_table(name)
            if table.size == 0:
                _fail(f"empty attenuation table: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, help="checkout that must not shadow the wheel"
    )
    args = parser.parse_args(argv)

    package_dir = _check_import_location(args.repo)
    _check_configs()
    _check_data()
    print(
        f"ok: {package_dir} ({len(EXPECTED_CONFIGS)} presets, "
        f"{len(EXPECTED_DATA)} data files)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
