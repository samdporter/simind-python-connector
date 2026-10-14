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
from simind_python_connector.core.types import (
    PenetrateOutputType,
    ScoringRoutine,
    SimulationError,
)
from simind_python_connector.utils.interfile import ProjectionGeometry


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


def _prefix_bytes(output_dir: Path, prefix: str) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
        if path.is_file()
        and (path.stem == prefix or path.stem.startswith(f"{prefix}_"))
    }


def test_penetrate_prefix_isolation(tmp_path: Path) -> None:
    """Overlapping prefixes must not claim each other's PENETRATE outputs."""
    expected = {component.slug for component in PenetrateOutputType}

    s = _connector(tmp_path, "s", scoring_routine=ScoringRoutine.PENETRATE)
    s_outputs = s.run(RuntimeOperator(switches={"RR": 12345}))
    assert set(s_outputs) == expected

    s2 = _connector(tmp_path, "s2", scoring_routine=ScoringRoutine.PENETRATE)
    s2_outputs = s2.run(RuntimeOperator(switches={"RR": 12345}))
    assert set(s2_outputs) == expected
    s2_bytes = _prefix_bytes(tmp_path, "s2")

    sx = _connector(tmp_path, "sx", scoring_routine=ScoringRoutine.PENETRATE)
    sx_outputs = sx.run(RuntimeOperator(switches={"RR": 12345}))
    assert set(sx_outputs) == expected
    sx_bytes = _prefix_bytes(tmp_path, "sx")

    s_again = _connector(tmp_path, "s", scoring_routine=ScoringRoutine.PENETRATE)
    s_again_outputs = s_again.run(RuntimeOperator(switches={"RR": 12345}))

    assert set(s_again_outputs) == expected
    assert _prefix_bytes(tmp_path, "s2") == s2_bytes
    assert _prefix_bytes(tmp_path, "sx") == sx_bytes
    found = s_again.converter.find_penetrate_h00_file("s", str(tmp_path))
    assert Path(found).name == "s.h00"


def test_saved_config_with_negative_zero_runs(tmp_path: Path) -> None:
    connector = _connector(tmp_path, "negzero")
    # Index 16 (source shift) is not overridden by configure_voxel_phantom.
    connector.get_config().set_value(16, -0.0)

    outputs = connector.run(RuntimeOperator(switches={"RR": 12345}))

    _assert_readable(outputs)
    basic_lines = (tmp_path / "negzero.smc").read_text().splitlines()[3:27]
    assert len(basic_lines) == 24
    for line in basic_lines:
        assert len(line) == 60
        for start in range(0, 60, 12):
            assert len(line[start : start + 12]) == 12


def _acquisition_geometry(radii_mm: tuple[float, ...]) -> ProjectionGeometry:
    return ProjectionGeometry(
        num_projections=VIEWS,
        extent_deg=360.0,
        direction="CW",
        start_angle_deg=180.0,
        radius_mm=None,
        radii_mm=radii_mm,
        num_bins=GRID,
        num_axial=GRID,
        bin_size_mm=VOXEL_MM,
        axial_size_mm=VOXEL_MM,
        image_duration_s=None,
    )


def test_explicit_orbit_overrides_the_generated_acquisition_orbit(
    tmp_path: Path,
) -> None:
    connector = _connector(tmp_path, "orb")
    generated_radii_mm = tuple(240.0 + 5.0 * i for i in range(VIEWS))
    connector.configure_acquisition(_acquisition_geometry(generated_radii_mm))

    explicit_radii_cm = [35.0 + 0.5 * i for i in range(VIEWS)]
    explicit = tmp_path / "caller-orbit.cor"
    centre_bin = GRID // 2
    explicit.write_text(
        "".join(f"{r:12.3f}{centre_bin:6d}\n" for r in explicit_radii_cm)
    )
    explicit_bytes = explicit.read_bytes()

    outputs = connector.run(RuntimeOperator(orbit_file=explicit))

    _assert_readable(outputs)
    # The explicit orbit was staged unchanged and the acquisition orbit, which
    # would have used the different generated radii, was not created.
    assert explicit.read_bytes() == explicit_bytes
    assert not (tmp_path / "orb_acquisition.cor").exists()

    header = Path(next(iter(outputs.values())).header_path).read_text()
    expected = ", ".join(f"{r * 10:.1f}" for r in explicit_radii_cm)
    assert f"Radii := {{{expected}}}" in header
