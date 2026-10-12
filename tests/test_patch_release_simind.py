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

from simind_python_connector import RuntimeOperator, SimindPythonConnector
from simind_python_connector.configs import get
from simind_python_connector.core.executor import SimindExecutor
from simind_python_connector.core.types import (
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
