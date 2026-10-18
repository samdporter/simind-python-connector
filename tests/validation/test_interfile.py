import re
from pathlib import Path

import numpy as np
import pytest

from simind_python_connector.core.types import ScoringRoutine
from simind_python_connector.utils.interfile import (
    load_interfile_array,
    read_header,
    read_projection_geometry,
)
from tests.test_release_regressions_simind import _connector


pytestmark = [pytest.mark.validation, pytest.mark.requires_simind]


def _run(tmp_path: Path, routine: ScoringRoutine, ca: int | None = None):
    connector = _connector(tmp_path, "if", scoring_routine=routine)
    if ca is not None:
        connector.add_runtime_switch("CA", ca)
    return connector, connector.run()


def _parse_problem(connector, outputs) -> str:
    rows = int(connector.get_config().get_value(77))
    cols = int(connector.get_config().get_value(76))
    views = int(connector.get_config().get_value(29))
    try:
        for result in outputs.values():
            read_header(result.header_path)
            array = load_interfile_array(result.header_path).array
            if array.shape != (views, rows, cols):
                return f"{result.header_path.name}: shape {array.shape}"
            if array.dtype != np.float32 or not np.isfinite(array).all():
                return f"{result.header_path.name}: {array.dtype}, non-finite"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


@pytest.mark.parametrize(
    "case, routine, ca",
    [
        ("dummy", ScoringRoutine.DUMMY, None),
        pytest.param(
            "scattwin_ca0",
            ScoringRoutine.SCATTWIN,
            0,
            marks=pytest.mark.scattwin_data("outputs_parse_{case}"),
        ),
        pytest.param(
            "scattwin_ca1",
            ScoringRoutine.SCATTWIN,
            1,
            marks=pytest.mark.scattwin_data("outputs_parse_{case}"),
        ),
        ("penetrate", ScoringRoutine.PENETRATE, None),
    ],
)
def test_outputs_parse(validate, tmp_path, case, routine, ca):
    connector, outputs = _run(tmp_path, routine, ca)
    problem = _parse_problem(connector, outputs)

    validate(
        f"outputs_parse_{case}",
        float(not problem),
        expected=1.0,
        abs_tol=0,
        note=problem,
    )


def test_truncated_payload_rejected(validate, tmp_path):
    _, outputs = _run(tmp_path, ScoringRoutine.PENETRATE)
    result = next(iter(outputs.values()))
    truncated = tmp_path / "truncated.a00"
    truncated.write_bytes(result.data_path.read_bytes()[:-2])
    header = re.sub(
        r"(?im)^(!?name of data file\s*:=).*$",
        rf"\1 {truncated.name}",
        result.header_path.read_text(),
    )
    header_path = tmp_path / "truncated.hs"
    header_path.write_text(header)

    try:
        load_interfile_array(header_path)
        rejected = 0.0
    except ValueError:
        rejected = 1.0

    validate("truncated_payload_rejected", rejected, expected=1.0, abs_tol=0)


@pytest.mark.scattwin_data
def test_projection_geometry_from_header(validate, tmp_path):
    connector, outputs = _run(tmp_path, ScoringRoutine.SCATTWIN, ca=0)
    result = next(iter(outputs.values()))

    geometry = read_projection_geometry(read_header(result.header_path))

    radius_cm = float(connector.get_config().get_value(12))
    validate(
        "projection_geometry_from_header",
        abs(geometry.radius_mm - 10 * radius_cm),
        expected=0,
        abs_tol=0.1,
    )


def test_penetrate_components_labelled(validate, tmp_path):
    _, outputs = _run(tmp_path, ScoringRoutine.PENETRATE)

    required = {
        "all_interactions",
        "geom_coll_primary",
        "septal_pen_primary",
        "coll_scatter_primary",
    }
    unlabelled = [k for k in outputs if re.fullmatch(r"b\d{2}", k) and k != "b20"]

    validate(
        "penetrate_components_labelled",
        float(required <= outputs.keys() and not unlabelled),
        expected=1.0,
        abs_tol=0,
        note=f"keys={sorted(outputs)}",
    )
