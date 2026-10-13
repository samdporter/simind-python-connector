"""Real-SIMIND regression smokes for the diagnostics/boundary port.

These run the actual SIMIND V8 binary (Linux x86-64, under Docker) and cover the
successful-run diagnostics exclusion (SIMIND echoes the configured comment under
OTHER INFORMATION) and the exit-0 missing-spectra detection. They are
integration tests and are skipped whenever SIMIND is not on PATH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simind_python_connector import RuntimeOperator, SimindPythonConnector
from simind_python_connector.configs import get
from simind_python_connector.core.types import ScoringRoutine, SimulationError


pytestmark = [pytest.mark.integration, pytest.mark.requires_simind]

GRID = 16
VIEWS = 4
VOXEL_MM = 4.0
HISTORIES = 0.02
MISSING_NUCLIDE = "not_a_real_nuclide"


def _connector(
    output_dir: Path, prefix: str, scoring_routine: ScoringRoutine | None = None
) -> SimindPythonConnector:
    """A centred point source with a uniform attenuating body (smoke)."""
    connector = SimindPythonConnector(get("Example.yaml"), output_dir, prefix)
    connector.get_config().set_flag(11, True)
    connector.add_config_value(26, HISTORIES)
    connector.add_config_value(29, VIEWS)
    connector.add_config_value(53, 0)
    connector.add_config_value(28, VOXEL_MM / 10)
    connector.add_config_value(76, GRID)
    connector.add_config_value(77, GRID)
    source = np.zeros((GRID, GRID, GRID), np.float32)
    source[GRID // 2, GRID // 2, GRID // 2] = 1.0
    mu_map = np.full((GRID, GRID, GRID), 0.15, np.float32)
    connector.configure_voxel_phantom(
        source, mu_map, VOXEL_MM, scoring_routine=scoring_routine
    )
    connector.set_energy_windows([126], [154], [0])
    for switch, value in (("FI", "tc99m"), ("CC", "ma-lehr"), ("RR", 7)):
        connector.add_runtime_switch(switch, value)
    return connector


def _assert_readable(outputs: dict) -> None:
    assert "tot_w1" in outputs
    projection = outputs["tot_w1"].projection
    assert projection.ndim == 3
    assert projection.shape[0] == VIEWS
    assert np.isfinite(projection).all()
    assert projection.sum() > 0


def test_valid_run_with_missing_spectra_comment_succeeds(tmp_path: Path) -> None:
    connector = _connector(tmp_path, "echo")
    connector.get_config().set_comment("No input file found for the spectra:test.isd")

    _assert_readable(connector.run())


@pytest.mark.parametrize(
    "comment",
    [
        "SIMIND error number: 0",
        "SIMIND error number: 7",
        "========================= Error",
        "OTHER INFORMATION",
    ],
    ids=["error0", "error7", "divider", "banner"],
)
def test_valid_run_with_fatal_looking_comment_succeeds(
    tmp_path: Path, comment: str
) -> None:
    connector = _connector(tmp_path, "fatalc")
    connector.get_config().set_comment(comment)

    _assert_readable(connector.run())


def test_missing_nuclide_reports_the_missing_isd_file(tmp_path: Path) -> None:
    connector = _connector(tmp_path, "iso")

    with pytest.raises(SimulationError) as excinfo:
        connector.run(RuntimeOperator(switches={"FI": MISSING_NUCLIDE}))

    message = str(excinfo.value)
    assert f"{MISSING_NUCLIDE}.isd" in message
    assert "No projection headers" not in message


def _assert_outputs_usable(outputs: dict, routine: ScoringRoutine) -> None:
    assert outputs, f"{routine.name} produced no outputs"
    for key, result in outputs.items():
        projection = result.projection
        assert projection.ndim == 3, key
        assert projection.shape[0] == VIEWS, key
        assert np.isfinite(projection).all(), key
    assert sum(float(result.projection.sum()) for result in outputs.values()) > 0


@pytest.mark.parametrize(
    "routine",
    [ScoringRoutine.DUMMY, ScoringRoutine.SCATTWIN, ScoringRoutine.PENETRATE],
    ids=lambda routine: routine.name,
)
def test_supported_scoring_routines_produce_readable_outputs(
    tmp_path: Path, routine: ScoringRoutine
) -> None:
    connector = _connector(tmp_path, f"r{routine.value}", scoring_routine=routine)

    outputs = connector.run(RuntimeOperator(switches={"RR": 12345}))

    _assert_outputs_usable(outputs, routine)
    if routine is ScoringRoutine.SCATTWIN:
        # /CA:1 makes the engine write air/tot/sca and the connector derives
        # primary as tot - sca for the order-0 window.
        assert {"tot_w1", "sca_w1", "pri_w1"} <= set(outputs)
        assert np.isfinite(outputs["pri_w1"].projection).all()
    else:
        assert not [key for key in outputs if key.startswith("pri")]


def test_small_valid_phantom_produces_meaningful_primary_outputs(
    tmp_path: Path,
) -> None:
    connector = _connector(tmp_path, "small")

    outputs = connector.run(RuntimeOperator(switches={"RR": 12345}))

    assert "tot_w1" in outputs and "sca_w1" in outputs and "pri_w1" in outputs
    total = outputs["tot_w1"].projection
    primary = outputs["pri_w1"].projection
    assert primary.sum() > 0
    assert primary.sum() <= total.sum() + 1e-6
