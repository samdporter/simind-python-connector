import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.validation import conftest as validation


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
V8 = validation.Engine("v8-linux", 8)
V9 = validation.Engine("v9-linux", 9)


def _load_report():
    spec = importlib.util.spec_from_file_location(
        "validation_report", ROOT / "scripts" / "validation_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_validate_records_then_passes(tmp_path):
    path = tmp_path / "v8-linux.jsonl"
    validation.record_result(
        path, V8, "sensitivity", "lehr", 0.40, expected=0.3957, rel=0.04
    )
    [record] = _lines(path)
    assert record["passed"] is True
    assert record["feature"] == "sensitivity"
    assert record["engine"] == "v8-linux"
    assert record["simind_version"] == 8
    assert record["status"] == "run"


def test_validate_records_then_fails(tmp_path):
    path = tmp_path / "v8-linux.jsonl"
    with pytest.raises(AssertionError, match="lehr"):
        validation.record_result(
            path, V8, "sensitivity", "lehr", 0.30, expected=0.3957, rel=0.04
        )
    [record] = _lines(path)
    assert record["passed"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected": 1, "rel": 0.1, "abs_tol": 0.1},
        {"expected": 1},
        {},
    ],
)
def test_exactly_one_tolerance_form_is_required(tmp_path, kwargs):
    with pytest.raises(ValueError):
        validation.record_result(tmp_path / "x.jsonl", V8, "f", "c", 1.0, **kwargs)


def test_bounds_form(tmp_path):
    path = tmp_path / "v8-linux.jsonl"
    validation.record_result(path, V8, "f", "in", 0.75, lower=0.70, upper=0.80)
    with pytest.raises(AssertionError):
        validation.record_result(path, V8, "f", "out", 0.81, lower=0.70, upper=0.80)
    assert [r["passed"] for r in _lines(path)] == [True, False]


def test_report_marks_missing_engines_with_a_dash(tmp_path):
    report = _load_report()
    path = tmp_path / "v8-linux.jsonl"
    validation.record_result(
        path, V8, "sensitivity", "a", 0.4, expected=0.3957, rel=0.04
    )
    with pytest.raises(AssertionError):
        validation.record_result(
            path, V8, "sensitivity", "b", 0.37, expected=0.3957, rel=0.01
        )
    validation.record_result(
        tmp_path / "v9-linux.jsonl", V9, "other", "c", 1.0, expected=1, abs_tol=0.1
    )
    text = report.render(tmp_path)
    assert "### sensitivity" in text
    assert "| case | expected | v8-linux | v9-linux | smoke |" in text
    assert "| a | 0.3957 | pass 0.4 | — | no |" in text
    assert "FAIL 0.37" in text


def test_report_writes_between_markers_and_keeps_the_rest(tmp_path):
    report = _load_report()
    doc = tmp_path / "doc.md"
    doc.write_text(
        "before\n<!-- validation:begin -->\nold\n<!-- validation:end -->\nafter\n"
    )
    report.write_section(doc, "new")
    assert doc.read_text() == (
        "before\n<!-- validation:begin -->\nnew\n<!-- validation:end -->\nafter\n"
    )

    bare = tmp_path / "bare.md"
    bare.write_text("# Title\n")
    report.write_section(bare, "new")
    assert bare.read_text() == (
        "# Title\n\n## Validation results\n\n"
        "<!-- validation:begin -->\nnew\n<!-- validation:end -->\n"
    )


def test_validation_tests_are_excluded_from_the_ci_selection():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"validation: ' in pyproject
    assert '"macos_v9: ' in pyproject

    for path in sorted((ROOT / "tests" / "validation").glob("test_*.py")):
        tree = ast.parse(path.read_text())
        marks = [
            node.attr
            for stmt in tree.body
            if isinstance(stmt, ast.Assign)
            and any(getattr(t, "id", None) == "pytestmark" for t in stmt.targets)
            for node in ast.walk(stmt.value)
            if isinstance(node, ast.Attribute)
        ]
        assert "requires_simind" in marks, path.name


class _FakeItem:
    module = SimpleNamespace(__name__="tests.validation.test_mpi")
    originalname = "test_serial_vs_split"
    callspec = None

    def __init__(self, marker):
        self.marker = marker

    def get_closest_marker(self, name):
        return self.marker


def test_scattwin_data_cases_are_not_run_on_v9_linux(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMIND_VALIDATION_OUT", str(tmp_path))
    monkeypatch.setenv("SIMIND_VALIDATION_ENGINE", "v9-linux")
    monkeypatch.setenv("SIMIND_VERSION", "9")
    marker = pytest.mark.scattwin_data.mark

    with pytest.raises(pytest.skip.Exception):
        validation.pytest_runtest_setup(_FakeItem(marker))

    [record] = _lines(tmp_path / "v9-linux.jsonl")
    assert record["status"] == "not run"
    assert record["feature"] == "mpi"
    assert record["case"] == "serial_vs_split"
    assert "SCATTWIN engine defect" in record["note"]


def test_scattwin_data_cases_run_on_other_engines(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMIND_VALIDATION_OUT", str(tmp_path))
    monkeypatch.setenv("SIMIND_VALIDATION_ENGINE", "v9-macos")
    monkeypatch.setenv("SIMIND_VERSION", "9")

    validation.pytest_runtest_setup(_FakeItem(pytest.mark.scattwin_data.mark))

    assert not list(tmp_path.iterdir())
