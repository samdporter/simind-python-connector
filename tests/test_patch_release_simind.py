"""Real-SIMIND regression checks for the 1.1.2 bug-hunt fixes.

These run the actual SIMIND V8 binary (Linux x86-64, under Docker in CI) and
cover the behaviours fixed in this patch: SIMIND's exit-0 error diagnostics, the
orbit basename collision, output-prefix validation versus directory spaces, the
PENETRATE no-backend fallback and the MPI switch. They are integration tests and
are skipped whenever SIMIND is not on PATH.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import simind_python_connector.core.executor as executor_module
from simind_python_connector import RuntimeOperator, SimindPythonConnector
from simind_python_connector.configs import get
from simind_python_connector.core.executor import SimindExecutor
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    PenetrateOutputType,
    ScoringRoutine,
    SimulationError,
)


pytestmark = [pytest.mark.integration, pytest.mark.requires_simind]

RUN_TIMEOUT = 600.0
GRID = 32
VIEWS = 8
HISTORIES = 0.02
ORBIT_RADIUS_CM = 10.0


def _connector(
    output_dir: Path,
    prefix: str,
    *,
    scoring_routine: ScoringRoutine | None = None,
    energy_windows: bool = True,
) -> SimindPythonConnector:
    """A centred point source with a uniform attenuating body."""
    connector = SimindPythonConnector(get("Example.yaml"), output_dir, prefix)
    connector.add_config_value(26, HISTORIES)
    connector.add_config_value(29, VIEWS)
    source = np.zeros((GRID, GRID, GRID), np.float32)
    source[GRID // 2, GRID // 2, GRID // 2] = 1.0
    connector.configure_voxel_phantom(
        source,
        np.full((GRID, GRID, GRID), 0.15, np.float32),
        scoring_routine=scoring_routine,
    )
    if energy_windows:
        connector.set_energy_windows([126], [154], [0])
    return connector


def _orbit_bytes(count: int = VIEWS, radius_cm: float = ORBIT_RADIUS_CM) -> bytes:
    """A valid SIMIND non-uniform orbit file: one detector radius (cm) per view."""
    return "".join(f"{radius_cm:.1f}\n" for _ in range(count)).encode("ascii")


def _assert_readable(outputs: dict, views: int = VIEWS) -> np.ndarray:
    assert "tot_w1" in outputs
    projection = outputs["tot_w1"].projection
    assert projection.ndim == 3
    assert projection.shape[0] == views
    assert np.isfinite(projection).all()
    assert projection.sum() > 0
    return projection


def _stir_or_sirf_importable() -> bool:
    for module in ("stir", "stirextra", "sirf.STIR"):
        try:
            importlib.import_module(module)
        except Exception:
            continue
        return True
    return False


def _simind_mpi_binary() -> Path | None:
    configured = os.environ.get("SIMIND_MPI_BIN")
    if configured:
        path = Path(configured)
        return path if path.is_file() else None
    simind = os.environ.get("SIMIND_BIN")
    if simind and os.sep in simind:
        path = Path(simind).with_name("simind_mpi")
        return path if path.is_file() else None
    found = shutil.which("simind_mpi")
    return Path(found) if found else None


def _mpi_runtime_skip_reason() -> str | None:
    """Why the SIMIND MPI smoke test cannot run, or None when it can.

    ``mpirun`` and ``simind_mpi`` being present is not enough: SIMIND links
    against historical MPICH SONAMEs (libmpi.so.12, libmpifort.so.12) that an
    Open MPI host (the SIRF image) does not provide, so the binary is present
    but cannot be loaded. Probe ``ldd`` for unresolved MPI libraries so those
    hosts skip instead of failing.
    """
    if shutil.which("mpirun") is None:
        return "mpirun is not on PATH"

    binary = _simind_mpi_binary()
    if binary is None:
        return "the SIMIND MPI binary is not available"
    if not os.access(binary, os.X_OK):
        return f"the SIMIND MPI binary {binary} is not executable"

    try:
        probe = subprocess.run(
            ["ldd", str(binary)], capture_output=True, text=True, check=False
        )
    except OSError as error:
        return f"ldd could not inspect the SIMIND MPI binary ({error})"

    unresolved = sorted(
        {
            line.partition("=>")[0].strip()
            for line in probe.stdout.splitlines()
            if "not found" in line
            and line.partition("=>")[0].strip().startswith("libmpi")
        }
    )
    if unresolved:
        return "SIMIND MPI runtime not usable: " + ", ".join(unresolved)

    return None


def test_missing_smc_file_raises_with_simind_diagnostics(tmp_path: Path):
    """A valid prefix with no .smc on disk must fail with SIMIND diagnostics.

    SIMIND V8 prints its error banner and still exits 0, so the executor has to
    detect the error output; a future binary that exits non-zero is equally
    acceptable.
    """
    executor = SimindExecutor()
    missing = "absent_config"

    with pytest.raises(SimulationError) as excinfo:
        executor.run_simulation(missing, cwd=tmp_path, timeout=RUN_TIMEOUT)

    message = str(excinfo.value)
    assert "SIMIND" in message
    assert missing in message
    assert re.search(r"error|failed", message, re.IGNORECASE)


def test_orbit_with_colliding_basename_does_not_clobber_inputs(tmp_path: Path):
    orbit_bytes = _orbit_bytes()

    # 1) The same valid orbit under a non-colliding name is the happy path.
    happy_dir = tmp_path / "happy"
    happy_orbit = tmp_path / "valid_orbit.cor"
    happy_orbit.write_bytes(orbit_bytes)
    happy = _connector(happy_dir, "s")
    happy_outputs = happy.run(
        RuntimeOperator(orbit_file=happy_orbit, timeout=RUN_TIMEOUT)
    )
    _assert_readable(happy_outputs)
    assert (happy_dir / "s_orbit.cor").read_bytes() == orbit_bytes

    # 2) Identical bytes, now named like the protected source input.
    external = tmp_path / "external"
    external.mkdir()
    colliding_orbit = external / "s_src.smi"
    colliding_orbit.write_bytes(orbit_bytes)

    collide_dir = tmp_path / "collide"
    collide = _connector(collide_dir, "s")
    source_path = collide_dir / "s_src.smi"
    density_path = collide_dir / "s_dns.dmi"
    source_before = source_path.read_bytes()
    density_before = density_path.read_bytes()

    outputs = collide.run(
        RuntimeOperator(orbit_file=colliding_orbit, timeout=RUN_TIMEOUT)
    )

    assert (collide_dir / "s_orbit.cor").read_bytes() == orbit_bytes
    assert colliding_orbit.read_bytes() == orbit_bytes
    assert source_path.read_bytes() == source_before
    assert density_path.read_bytes() == density_before
    _assert_readable(outputs)


def test_underscore_prefix_in_directory_with_spaces(tmp_path: Path):
    with pytest.raises(ValueError, match="output_prefix"):
        SimindPythonConnector(get("Example.yaml"), tmp_path / "unused", "hyphen-prefix")

    spaced_dir = tmp_path / "output dir with spaces"
    connector = _connector(spaced_dir, "my_run")
    outputs = connector.run(RuntimeOperator(timeout=RUN_TIMEOUT))
    _assert_readable(outputs)


def test_penetrate_without_backends_has_no_fallback_warnings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    if _stir_or_sirf_importable():
        pytest.skip("STIR or SIRF is importable, so the no-backend path is not run")

    connector = _connector(tmp_path, "pen", scoring_routine=ScoringRoutine.PENETRATE)
    with caplog.at_level(logging.WARNING):
        outputs = connector.run(RuntimeOperator(timeout=RUN_TIMEOUT))

    assert set(outputs) == {component.slug for component in PenetrateOutputType}
    for key, result in outputs.items():
        projection = result.projection
        assert projection.ndim == 3, key
        assert projection.shape[0] == VIEWS, key
        assert np.isfinite(projection).all(), key

    noisy = [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and "Falling back to file path" in record.getMessage()
    ]
    assert noisy == []


@pytest.mark.parametrize(
    "processes",
    [pytest.param(1, id="mp1"), pytest.param(2, id="mp2"), pytest.param(4, id="mp4")],
)
def test_mpi_views_and_projections_are_readable(tmp_path: Path, processes: int):
    reason = _mpi_runtime_skip_reason()
    if reason is not None:
        pytest.skip(reason)

    connector = _connector(tmp_path, "mpi")
    outputs = connector.run(
        RuntimeOperator(switches={"MP": processes}, timeout=RUN_TIMEOUT)
    )

    for key, result in outputs.items():
        projection = result.projection
        assert projection.ndim == 3, key
        assert projection.shape[0] == VIEWS, key
        assert np.isfinite(projection).all(), key
    assert outputs["tot_w1"].projection.sum() > 0


MISSING_NUCLIDE = "not_a_real_nuclide"
# The executor cap derived from SIMIND's fixed 60-character output-name buffer:
# the longest engine-written suffix is 11 characters ("_tot_w1.a00"), so 49 is
# the measured SCATTWIN boundary and the executor keeps a two-character margin.
OUTPUT_PREFIX_CAP = 47
# Raw SIMIND V8.0 boundary measured against the real binary: a 49-character
# SCATTWIN prefix still writes the full ".h00"/".a00" names, 50 truncates them.
ENGINE_RAW_PREFIX_LIMIT = 49


def _point_connector(output_dir: Path, prefix: str, shape: tuple[int, ...]):
    """A centred point source in an all-air phantom of the requested shape."""
    connector = SimindPythonConnector(get("Example.yaml"), output_dir, prefix)
    connector.add_config_value(26, HISTORIES)
    connector.add_config_value(29, VIEWS)
    source = np.zeros(shape, np.float32)
    source[tuple(dim // 2 for dim in shape)] = 1.0
    connector.configure_voxel_phantom(source, np.zeros(shape, np.float32))
    connector.set_energy_windows([126], [154], [0])
    return connector


def _prefix_bytes(output_dir: Path, prefix: str) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
        if path.is_file()
        and (path.stem == prefix or path.stem.startswith(f"{prefix}_"))
    }


def test_missing_isotope_reports_the_missing_isd_file(tmp_path: Path):
    """A bad /FI isotope names the missing .isd, not a missing-header error."""
    connector = _connector(tmp_path, "iso")

    with pytest.raises(SimulationError) as excinfo:
        connector.run(
            RuntimeOperator(switches={"FI": MISSING_NUCLIDE}, timeout=RUN_TIMEOUT)
        )

    message = str(excinfo.value)
    assert f"{MISSING_NUCLIDE}.isd" in message
    assert "No SPECT projection headers" not in message


def test_negative_index_1_with_missing_isotope_reports_the_missing_isd_file(
    tmp_path: Path,
):
    connector = _connector(tmp_path, "negiso")
    connector.add_config_value(1, -140.0)

    with pytest.raises(SimulationError) as excinfo:
        connector.run(
            RuntimeOperator(switches={"FI": MISSING_NUCLIDE}, timeout=RUN_TIMEOUT)
        )

    message = str(excinfo.value)
    assert f"{MISSING_NUCLIDE}.isd" in message
    assert "No SPECT projection headers" not in message


def test_valid_run_with_missing_spectra_text_in_comment_succeeds(tmp_path: Path):
    """An echoed comment cannot be mistaken for the fatal missing-spectra line.

    SIMIND echoes the configured comment verbatim under OTHER INFORMATION, so a
    valid run whose comment spells the diagnostic must still succeed.
    """
    connector = _connector(tmp_path, "echo")
    connector.get_config().set_comment("No input file found for the spectra:test.isd")

    outputs = connector.run(RuntimeOperator(timeout=RUN_TIMEOUT))

    _assert_readable(outputs)


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
def test_valid_run_with_fatal_looking_comment_succeeds(tmp_path: Path, comment: str):
    """An echoed comment that spells a diagnostic cannot fail a valid run.

    SIMIND echoes the configured comment verbatim under OTHER INFORMATION, so
    the executor must exclude only that echoed line from its error scans while
    still detecting the same text elsewhere.
    """
    connector = _connector(tmp_path, "fatalc")
    connector.get_config().set_comment(comment)

    outputs = connector.run(RuntimeOperator(timeout=RUN_TIMEOUT))

    _assert_readable(outputs)


@pytest.mark.parametrize(
    "routine",
    [ScoringRoutine.DUMMY, ScoringRoutine.SCATTWIN, ScoringRoutine.PENETRATE],
    ids=lambda routine: routine.name,
)
def test_supported_scoring_routines_produce_readable_outputs(
    tmp_path: Path, routine: ScoringRoutine
):
    connector = _connector(tmp_path, f"r{routine.value}", scoring_routine=routine)

    outputs = connector.run(
        RuntimeOperator(switches={"RR": 12345}, timeout=RUN_TIMEOUT)
    )

    assert outputs
    assert not [key for key in outputs if key.startswith("pri")]
    for key, result in outputs.items():
        projection = result.projection
        assert projection.ndim == 3, key
        assert projection.shape[0] == VIEWS, key
        assert np.isfinite(projection).all(), key
    assert sum(float(result.projection.sum()) for result in outputs.values()) > 0


def test_penetrate_prefix_isolation(tmp_path: Path):
    """Overlapping prefixes must not claim each other's PENETRATE outputs."""
    expected = {component.slug for component in PenetrateOutputType}

    s = _connector(tmp_path, "s", scoring_routine=ScoringRoutine.PENETRATE)
    s_outputs = s.run(RuntimeOperator(switches={"RR": 12345}, timeout=RUN_TIMEOUT))
    assert set(s_outputs) == expected

    s2 = _connector(tmp_path, "s2", scoring_routine=ScoringRoutine.PENETRATE)
    s2_outputs = s2.run(RuntimeOperator(switches={"RR": 12345}, timeout=RUN_TIMEOUT))
    assert set(s2_outputs) == expected
    s2_bytes = _prefix_bytes(tmp_path, "s2")

    sx = _connector(tmp_path, "sx", scoring_routine=ScoringRoutine.PENETRATE)
    sx_outputs = sx.run(RuntimeOperator(switches={"RR": 12345}, timeout=RUN_TIMEOUT))
    assert set(sx_outputs) == expected
    sx_bytes = _prefix_bytes(tmp_path, "sx")

    s_again = _connector(tmp_path, "s", scoring_routine=ScoringRoutine.PENETRATE)
    s_again_outputs = s_again.run(
        RuntimeOperator(switches={"RR": 12345}, timeout=RUN_TIMEOUT)
    )

    assert set(s_again_outputs) == expected
    assert _prefix_bytes(tmp_path, "s2") == s2_bytes
    assert _prefix_bytes(tmp_path, "sx") == sx_bytes
    found = s_again.converter.find_penetrate_h00_file("s", str(tmp_path))
    assert Path(found).name == "s.h00"


def test_engine_raw_prefix_limit_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pin the real SIMIND V8.0 limit the executor's cap is derived from.

    SCATTWIN writes ``{prefix}_tot_w1.a00`` (11-character suffix) into the
    engine's fixed 60-character buffer, so 49 is the largest prefix that keeps
    the full name and 50 truncates it. The public cap is two characters below
    for safety; lift it here to exercise the raw engine boundary, and invoke
    the engine directly for the over-limit case the public APIs would reject.
    """
    at_limit = tmp_path / "at_limit"
    over_limit = tmp_path / "over_limit"
    prefix49 = "a" * ENGINE_RAW_PREFIX_LIMIT
    prefix50 = "a" * (ENGINE_RAW_PREFIX_LIMIT + 1)

    # Stage the over-limit inputs with the cap lifted only far enough to write
    # them; the 50-character run itself never goes through the validators.
    monkeypatch.setattr(
        executor_module, "_OUTPUT_PREFIX_MAX_LENGTH", ENGINE_RAW_PREFIX_LIMIT + 1
    )
    over = _connector(over_limit, prefix50, energy_windows=False)
    # run() saves the .smc just before launching; stage it here for the direct
    # engine invocation below.
    over.get_config().save_file(over_limit / prefix50)
    monkeypatch.setattr(
        executor_module, "_OUTPUT_PREFIX_MAX_LENGTH", ENGINE_RAW_PREFIX_LIMIT
    )
    connector = _connector(at_limit, prefix49, energy_windows=False)

    outputs = connector.run(
        RuntimeOperator(switches={"RR": 12345}, timeout=RUN_TIMEOUT)
    )

    _assert_readable(outputs)
    assert (at_limit / f"{prefix49}_tot_w1.a00").is_file()
    assert (at_limit / f"{prefix49}_tot_w1.h00").is_file()

    # Direct engine invocation: the public APIs reject a 50-character prefix,
    # so bypass every validator and pin the measured truncation behaviour.
    # The command mirrors what run_simulation issues, switches included.
    # Reuse the executor's own resolution so relative, tilde and bare-name
    # SIMIND_BIN values behave exactly as they do under run_simulation; a bare
    # name stays bare and the subprocess resolves it against its own cwd.
    binary = SimindExecutor()._resolve_executable()
    switch_blob = "".join(
        f"/{key}:{value}" for key, value in over.runtime_switches.switches.items()
    )
    subprocess.run(
        [binary, prefix50, prefix50, f"{switch_blob}/RR:12345"],
        check=True,
        shell=False,
        cwd=over_limit,
        timeout=RUN_TIMEOUT,
        capture_output=True,
        text=True,
    )

    # The 61-character name loses its final character in the 60-character
    # buffer, so the truncated names exist but no readable header remains.
    assert (over_limit / f"{prefix50}_tot_w1.a0").is_file()
    assert (over_limit / f"{prefix50}_tot_w1.h0").is_file()
    assert not (over_limit / f"{prefix50}_tot_w1.a00").exists()
    assert not (over_limit / f"{prefix50}_tot_w1.h00").exists()


def test_prefix_at_cap_runs_through_conversion(tmp_path: Path):
    connector = _connector(tmp_path, "a" * OUTPUT_PREFIX_CAP, energy_windows=False)

    outputs = connector.run(
        RuntimeOperator(switches={"RR": 12345}, timeout=RUN_TIMEOUT)
    )

    _assert_readable(outputs)


def test_single_slice_phantom_produces_nonzero_totals(tmp_path: Path):
    connector = _point_connector(tmp_path, "slice", (2, 1, 1))

    outputs = connector.run(
        RuntimeOperator(switches={"NN": 20, "RR": 12345}, timeout=RUN_TIMEOUT)
    )

    _assert_readable(outputs)


def test_valid_quantization_control_runs(tmp_path: Path):
    connector = SimindPythonConnector(
        get("Example.yaml"),
        tmp_path,
        "quant",
        quantization_scale=0.5 * DEFAULT_QUANTIZATION_SCALE,
    )
    connector.add_config_value(26, HISTORIES)
    connector.add_config_value(29, VIEWS)
    source = np.zeros((GRID, GRID, GRID), np.float32)
    source[GRID // 2, GRID // 2, GRID // 2] = 1.0
    connector.configure_voxel_phantom(
        source, np.full((GRID, GRID, GRID), 0.15, np.float32)
    )
    connector.set_energy_windows([126], [154], [0])

    outputs = connector.run(
        RuntimeOperator(switches={"RR": 12345}, timeout=RUN_TIMEOUT)
    )

    _assert_readable(outputs)


def test_saved_config_with_negative_zero_runs(tmp_path: Path):
    connector = _connector(tmp_path, "negzero")
    # Index 16 (source shift) is not overridden by configure_voxel_phantom.
    connector.get_config().set_value(16, -0.0)

    outputs = connector.run(
        RuntimeOperator(switches={"RR": 12345}, timeout=RUN_TIMEOUT)
    )

    _assert_readable(outputs)
    basic_lines = (tmp_path / "negzero.smc").read_text().splitlines()[3:27]
    assert len(basic_lines) == 24
    for line in basic_lines:
        assert len(line) == 60
        for start in range(0, 60, 12):
            assert len(line[start : start + 12]) == 12
