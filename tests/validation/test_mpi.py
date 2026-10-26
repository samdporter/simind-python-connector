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
NN = 4
# Without RR, simind_mpi seeds process r exactly like RR = r + 1, so the
# serial reference seeds stay clear of the replicate processes' seeds.
SERIAL_SEEDS = (101, 102, 103, 104)
SPLIT_SEED = 11


def _run(
    directory,
    prefix: str,
    processes: int | None = None,
    split: bool = False,
    scoring_routine: ScoringRoutine = ScoringRoutine.SCATTWIN,
    seed: int | None = None,
):
    connector = SimindPythonConnector(
        get("Example.yaml"),
        directory,
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
    connector.add_runtime_switch("CA", 0)
    connector.add_runtime_switch("NN", NN)
    if seed is not None:
        connector.add_runtime_switch("RR", seed)
    connector.add_config_value(29, VIEWS)
    if processes:
        connector.set_mpi(processes, split)
    return connector.run()


def _image(outputs) -> np.ndarray:
    return outputs["tot_w1"].projection


@pytest.fixture(scope="module")
def serial_images(tmp_path_factory):
    directory = tmp_path_factory.mktemp("serial")
    return [
        _image(_run(directory, f"serial{seed}", seed=seed)) for seed in SERIAL_SEEDS
    ]


@pytest.fixture(scope="module")
def replicate_image(tmp_path_factory):
    return _image(_run(tmp_path_factory.mktemp("replicate"), "replicate", 2))


@pytest.mark.scattwin_data
def test_serial_vs_replicate(validate, serial_images, replicate_image):
    """Replicate MPI cannot take RR, so the replicate run uses SIMIND's own
    per-process seeding; the reference is the mean of the serial totals."""
    reference = np.mean([image.sum() for image in serial_images])
    replicate = replicate_image.sum()

    validate(
        "serial_vs_replicate",
        float(replicate / reference),
        expected=1.0,
        rel=0.05,
        smoke=True,
    )


@pytest.mark.scattwin_data
def test_serial_vs_split(validate, tmp_path, serial_images):
    reference = np.mean([image.sum() for image in serial_images])
    split = _image(_run(tmp_path, "split", 2, True, seed=SPLIT_SEED)).sum()

    validate(
        "serial_vs_split", float(split / reference), expected=1.0, rel=0.05, smoke=True
    )


@pytest.mark.scattwin_data
def test_replicate_halves_variance(validate, serial_images, replicate_image):
    """Two replicate processes sum independent histories, so the replicate
    image has half the per-bin variance of a serial run. Identical ranks
    (one shared seed) would give a ratio of 1.

    M, the mean of the K serial images, is itself noisy: the serial
    deviations from M underestimate the variance by (K - 1) / K and the
    replicate deviation includes var(M) = var / K, so both are corrected.
    """
    k = len(serial_images)
    mean_image = np.mean(serial_images, axis=0)
    populated = mean_image > 0
    serial_variance = (
        k
        / (k - 1)
        * np.mean([np.mean((s - mean_image)[populated] ** 2) for s in serial_images])
    )
    replicate_variance = (
        np.mean((replicate_image - mean_image)[populated] ** 2) - serial_variance / k
    )

    validate(
        "replicate_halves_variance",
        float(replicate_variance / serial_variance),
        upper=0.75,
        smoke=True,
        note="expected about 0.5 for 2 processes",
    )


@pytest.mark.scattwin_data
def test_replicate_seeds_processes_as_rr_1_to_n(validate, tmp_path, replicate_image):
    """The replicate run is the mean of the serial runs with RR=1 and RR=2."""
    rr1 = _image(_run(tmp_path, "rr1", seed=1))
    rr2 = _image(_run(tmp_path, "rr2", seed=2))

    validate(
        "replicate_seeds_processes_as_rr_1_to_n",
        float(np.max(np.abs(2 * replicate_image - rr1 - rr2))),
        expected=0.0,
        abs_tol=1e-5,
    )


@pytest.mark.scattwin_data
def test_replicate_three_processes_runs(validate, tmp_path):
    try:
        _run(tmp_path, "replicate3", 3)
    except Exception as error:
        completed, note = 0.0, str(error)[-300:]
    else:
        completed, note = 1.0, ""

    validate(
        "replicate_three_processes_runs",
        completed,
        expected=1.0,
        abs_tol=0,
        smoke=True,
        note=note,
    )


def test_penetrate_split_refused(validate, tmp_path):
    """simind_mpi /MP writes only the primary process's PENETRATE projections
    (12 of 24 views with 2 processes on v8 and v9), so the connector refuses
    the combination; the refusal is the expected behaviour."""
    try:
        _run(tmp_path, "psplit", 2, True, ScoringRoutine.PENETRATE, seed=SPLIT_SEED)
    except ValueError as error:
        refused, note = float("PENETRATE" in str(error)), str(error)
    else:
        refused, note = 0.0, "run was not refused"

    validate("penetrate_split_refused", refused, expected=1.0, abs_tol=0, note=note)


def test_replicate_fixed_rr_refused(validate, tmp_path):
    """With a fixed RR every simind_mpi process repeats the same histories and
    the result is bin-for-bin the serial run, so the connector refuses it."""
    try:
        _run(tmp_path, "rrfixed", 2, seed=12345)
    except ValueError as error:
        refused, note = float("RR" in str(error)), str(error)
    else:
        refused, note = 0.0, "run was not refused"

    validate("replicate_fixed_rr_refused", refused, expected=1.0, abs_tol=0, note=note)
