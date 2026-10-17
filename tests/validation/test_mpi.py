import numpy as np
import pytest

from simind_python_connector import SimindPythonConnector
from simind_python_connector.configs import get
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
)


pytestmark = [pytest.mark.validation, pytest.mark.requires_simind]

VIEWS = 24


def _run(
    tmp_path,
    prefix: str,
    processes: int | None = None,
    split: bool = False,
    scoring_routine: ScoringRoutine | None = None,
    seed: int | None = 12345,
    ca: int | None = None,
):
    connector = SimindPythonConnector(
        get("Example.yaml"),
        tmp_path,
        prefix,
        quantization_scale=0.05 * DEFAULT_QUANTIZATION_SCALE,
    )
    source = np.zeros((32, 32, 32), np.float32)
    source[12:20, 12:20, 12:20] = 1.0
    connector.configure_voxel_phantom(
        source,
        np.where(source > 0, 0.15, 0.0),
        4.0,
        scoring_routine=scoring_routine,
    )
    connector.set_energy_windows([126], [154], [0])
    connector.add_runtime_switch("FI", "tc99m")
    connector.add_runtime_switch("CC", "ma-lehr")
    if seed is not None:
        connector.add_runtime_switch("RR", seed)
    if ca is not None:
        connector.add_runtime_switch("CA", ca)
    connector.add_config_value(29, VIEWS)
    if processes:
        connector.set_mpi(processes, split)
    return connector.run()


def _total(outputs, key: str = "tot_w1") -> float:
    return float(outputs[key].projection.sum())


def test_serial_vs_replicate(validate, tmp_path):
    """Replicate MPI cannot take RR, so both runs use SIMIND's default seed."""
    serial = _total(_run(tmp_path, "serial", seed=None))
    replicate = _total(_run(tmp_path, "replicate", 2, seed=None))

    validate(
        "serial_vs_replicate", replicate / serial, expected=1.0, rel=0.05, smoke=True
    )


def test_serial_vs_split(validate, tmp_path):
    serial = _total(_run(tmp_path, "serial"))
    split = _total(_run(tmp_path, "split", 2, True))

    validate("serial_vs_split", split / serial, expected=1.0, rel=0.05, smoke=True)


def test_penetrate_split_refused(validate, tmp_path):
    """simind_mpi /MP writes only the primary process's PENETRATE projections
    (12 of 24 views with 2 processes on v8 and v9), so the connector refuses
    the combination; the refusal is the expected behaviour."""
    try:
        _run(tmp_path, "psplit", 2, True, ScoringRoutine.PENETRATE)
    except ValueError as error:
        refused, note = float("PENETRATE" in str(error)), str(error)
    else:
        refused, note = 0.0, "run was not refused"

    validate("penetrate_split_refused", refused, expected=1.0, abs_tol=0, note=note)


def test_replicate_fixed_rr_refused(validate, tmp_path):
    """With a fixed RR every simind_mpi process repeats the same histories and
    the result is bin-for-bin the serial run, so the connector refuses it."""
    try:
        _run(tmp_path, "rrfixed", 2)
    except ValueError as error:
        refused, note = float("RR" in str(error)), str(error)
    else:
        refused, note = 0.0, "run was not refused"

    validate("replicate_fixed_rr_refused", refused, expected=1.0, abs_tol=0, note=note)


def test_replicate_adds_statistics(validate, tmp_path):
    """Without RR, SIMIND seeds each MPI process differently, so a replicate
    run must differ from the serial run that uses the same (.num) seed."""
    kwargs = {"scoring_routine": ScoringRoutine.SCATTWIN, "ca": 0, "seed": None}
    serial = _run(tmp_path, "rserial", **kwargs)["tot_w1"].projection
    replicate = _run(tmp_path, "rrep", 2, **kwargs)["tot_w1"].projection
    populated = serial > 0

    validate(
        "replicate_adds_statistics",
        float(
            np.mean(np.abs(replicate - serial)[populated]) / serial[populated].mean()
        ),
        lower=1e-3,
        smoke=True,
    )
