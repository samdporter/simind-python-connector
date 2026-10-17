import numpy as np
import pytest

from simind_python_connector import SimindPythonConnector
from simind_python_connector.configs import get
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
    SimulationError,
)


pytestmark = [pytest.mark.validation, pytest.mark.requires_simind]

VIEWS = 24


def _run(
    tmp_path,
    prefix: str,
    processes: int | None = None,
    split: bool = False,
    scoring_routine: ScoringRoutine | None = None,
    seed: int = 12345,
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
    serial = _total(_run(tmp_path, "serial"))
    replicate = _total(_run(tmp_path, "replicate", 2))

    validate(
        "serial_vs_replicate", replicate / serial, expected=1.0, rel=0.05, smoke=True
    )


def test_serial_vs_split(validate, tmp_path):
    serial = _total(_run(tmp_path, "serial"))
    split = _total(_run(tmp_path, "split", 2, True))

    validate("serial_vs_split", split / serial, expected=1.0, rel=0.05, smoke=True)


def test_penetrate_split_complete(validate, tmp_path):
    serial = _run(tmp_path, "pserial", scoring_routine=ScoringRoutine.PENETRATE)
    try:
        split = _run(tmp_path, "psplit", 2, True, ScoringRoutine.PENETRATE)
    except SimulationError as error:
        complete, total_ratio, note = 0.0, 0.0, str(error)[-300:]
    else:
        views = split["all_interactions"].projection.sum(axis=(1, 2))
        complete = float(np.mean(views > 0))
        total_ratio = _total(split, "all_interactions") / _total(
            serial, "all_interactions"
        )
        note = ""

    try:
        validate(
            "penetrate_split_complete",
            complete,
            expected=1.0,
            abs_tol=0,
            smoke=True,
            note=note,
        )
    finally:
        validate(
            "penetrate_split_total",
            total_ratio,
            expected=1.0,
            rel=0.05,
            smoke=True,
            note=note,
        )


def test_replicate_adds_statistics(validate, tmp_path):
    """A failure means the replicate ranks repeat identical histories, so MPI
    replicate adds no statistics over a serial run with the same seed."""
    kwargs = {"scoring_routine": ScoringRoutine.SCATTWIN, "ca": 0}
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
