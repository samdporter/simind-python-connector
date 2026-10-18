import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pytest


class Engine(NamedTuple):
    label: str
    version: int


def _output_dir() -> Path:
    return Path(os.environ.get("SIMIND_VALIDATION_OUT", "output/validation"))


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_result(
    path: Path,
    engine: Engine,
    feature: str,
    case: str,
    measured: float,
    *,
    expected: float | None = None,
    rel: float | None = None,
    abs_tol: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    smoke: bool = False,
    note: str = "",
) -> None:
    has_bounds = lower is not None or upper is not None
    is_bounds_form = has_bounds and expected is None and rel is None and abs_tol is None
    is_expected_form = (
        expected is not None and not has_bounds and (rel is None) != (abs_tol is None)
    )
    if not (is_bounds_form or is_expected_form):
        raise ValueError(
            "Give exactly one tolerance form: expected with rel, expected with "
            "abs_tol, or lower and/or upper"
        )

    if has_bounds:
        passed = (lower is None or measured >= lower) and (
            upper is None or measured <= upper
        )
    else:
        tolerance = abs_tol if abs_tol is not None else rel * abs(expected)
        passed = abs(measured - expected) <= tolerance

    _append(
        path,
        {
            "feature": feature,
            "case": case,
            "engine": engine.label,
            "simind_version": engine.version,
            "measured": measured,
            "expected": expected,
            "rel": rel,
            "abs_tol": abs_tol,
            "lower": lower,
            "upper": upper,
            "passed": passed,
            "smoke": smoke,
            "note": note,
            "status": "run",
            "timestamp": _timestamp(),
        },
    )
    assert passed, (
        f"{case}: measured {measured}, expected {expected} "
        f"(rel={rel}, abs_tol={abs_tol}, lower={lower}, upper={upper})"
    )


def _engine() -> Engine:
    version = int(os.environ.get("SIMIND_VERSION") or 8)
    label = os.environ.get("SIMIND_VALIDATION_ENGINE") or (
        f"v{version}-{platform.system().lower()}"
    )
    return Engine(label, version)


@pytest.fixture(scope="session")
def engine() -> Engine:
    return _engine()


def _feature(request) -> str:
    return request.module.__name__.rsplit(".", 1)[-1].removeprefix("test_")


def _record_not_run(path: Path, engine: Engine, feature: str, case: str, reason: str):
    _append(
        path,
        {
            "feature": feature,
            "case": case,
            "engine": engine.label,
            "simind_version": engine.version,
            "note": reason,
            "status": "not run",
            "timestamp": _timestamp(),
        },
    )
    pytest.skip(reason)


V9_LINUX_SCATTWIN_DEFECT = (
    "v9 Linux SCATTWIN engine defect: aborts or silently corrupts projections"
)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Record scattwin_data tests as not run on v9 Linux before any fixture runs.

    The marker's optional argument names the case (default: the test name
    without "test_"); it is formatted with the test's parameters and "id".
    """
    marker = item.get_closest_marker("scattwin_data")
    engine = _engine()
    if marker is None or engine.label != "v9-linux":
        return
    callspec = getattr(item, "callspec", None)
    params = {} if callspec is None else {"id": callspec.id, **callspec.params}
    case = marker.args[0] if marker.args else item.originalname.removeprefix("test_")
    _record_not_run(
        _output_dir() / f"{engine.label}.jsonl",
        engine,
        _feature(item),
        case.format(**params),
        V9_LINUX_SCATTWIN_DEFECT,
    )


@pytest.fixture
def validate(request, engine):
    path = _output_dir() / f"{engine.label}.jsonl"
    feature = _feature(request)

    def _validate(case: str, measured: float, **kwargs) -> None:
        record_result(path, engine, feature, case, measured, **kwargs)

    return _validate


@pytest.fixture
def not_run(request, engine):
    path = _output_dir() / f"{engine.label}.jsonl"
    feature = _feature(request)

    def _not_run(case: str, reason: str) -> None:
        _record_not_run(path, engine, feature, case, reason)

    return _not_run
