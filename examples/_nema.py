"""Shared NEMA/IEC setup for the reconstruction tutorials (examples 11-15).

Every tutorial uses the same phantom, projection template and simulated
acquisition, so the comparisons share one high-precision expectation instead
of each running its own Monte Carlo simulation. The expectation is cached in
the examples output directory, keyed by the settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from simind_python_connector import configs
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
)
from simind_python_connector.phantoms import nema_iec_phantom


MPI_PROCESSES = 8
GRID = (48, 48, 48)
VOXEL_MM = 7.0
VIEWS = 60
RADIUS_MM = 250
ENERGY_WINDOW_KEV = (126.0, 154.0)
COLLIMATOR = "ma-lehr"
PHANTOM_PRESET = "pet"
SUPERSAMPLE = 4

# The expectation is simulated at this acquisition time and scaled to any
# other time per view, because the counts are linear in the acquisition time.
REFERENCE_TIME_S = 2.0
# Default quantization is the legacy 500-history budget at full signed-range
# resolution; four times that budget raises the histories per projection. The
# history-to-count ratio does not establish the variance of SIMIND's weighted
# scatter or primary estimator, so these tutorials keep the accounting but do
# not turn it into a Monte Carlo precision claim.
HISTORY_MULTIPLIER = 4
QUANTIZATION_SCALE = DEFAULT_QUANTIZATION_SCALE

# One stopping rule for every OSEM arm: six subsets, four full iterations.
NUM_SUBSETS = 6
OSEM_ITERATIONS = 4

# Independent Poisson realisations used for the uncertainty bars.
SEEDS = (12345, 12346, 12347)

SPHERES = {"sphere_3": 17, "sphere_4": 22, "sphere_5": 28, "sphere_6": 37}
TRUE_RATIO = 8.0

_CACHE_VERSION = 2


def _resolve_executable(value: str) -> str:
    """Expand ``~`` and anchor a path-bearing executable like the executor."""
    if os.sep in value or (os.altsep and os.altsep in value):
        return str(Path(value).expanduser().absolute())
    return value


def _which_executable(value: str, *, role: str) -> Path:
    resolved = shutil.which(value)
    if resolved is None:
        raise RuntimeError(
            f"SIMIND {role} {value!r} not found; set "
            f"{'SIMIND_MPI_BIN' if role == 'MPI executable' else 'SIMIND_BIN'} "
            "or install it on PATH before loading the shared expectation."
        )
    return Path(resolved).resolve()


def _change_version(change: Path) -> str | None:
    """Read the distribution version from SIMIND's ``change`` banner.

    The simulation binaries have no version flag, but the ``change`` program
    in the same directory prints ``SIMIND version V8.0`` before its menu.
    ``change`` pauses for input, so it is run from the distribution directory
    (where ``simind.smc`` lives) and killed after the banner is captured.
    """
    if not change.is_file():
        return None
    try:
        process = subprocess.Popen(
            [str(change)],
            cwd=str(change.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except OSError:
        return None
    try:
        output, _ = process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
    match = re.search(r"SIMIND\s+version\s+(V\d+\.\d+)", output)
    return match.group(1) if match else None


def detect_engine(
    executable: str | None = None, mpi_executable: str | None = None
) -> str:
    """Identify the installed SIMIND engine by version and both binary hashes.

    NEMA always runs under MPI, so the identity covers the selected
    ``simind_mpi`` as well as ``simind``. The version comes from the sibling
    ``change`` program; if that program is missing or its banner is not
    recognised, detection fails rather than reusing a cache under an unknown
    engine. The SHA-256 prefixes keep two builds of the same nominal version
    apart, so a cache cannot be reused across binary changes.
    """
    serial_value = executable or os.environ.get("SIMIND_BIN") or "simind"
    anchored = _resolve_executable(serial_value)
    serial = _which_executable(anchored, role="executable")
    mpi_value = mpi_executable or os.environ.get("SIMIND_MPI_BIN")
    if mpi_value:
        mpi = _which_executable(_resolve_executable(mpi_value), role="MPI executable")
    elif os.sep in anchored:
        # Match SimindExecutor: choose the sibling from the anchored but
        # unresolved serial path, so a symlinked SIMIND_BIN keeps its own
        # directory's simind_mpi rather than the target's.
        mpi = _which_executable(
            str(Path(anchored).with_name("simind_mpi")), role="MPI executable"
        )
    else:
        mpi = _which_executable("simind_mpi", role="MPI executable")

    version = _change_version(serial.parent / "change")
    if version is None:
        raise RuntimeError(
            f"Cannot read the SIMIND version from {serial.parent / 'change'!s}. "
            "The NEMA cache is keyed by the engine, so a complete SIMIND "
            "distribution (including the change program and its version banner) "
            "is required; point SIMIND_BIN at one."
        )
    serial_digest = hashlib.sha256(serial.read_bytes()).hexdigest()[:12]
    mpi_digest = hashlib.sha256(mpi.read_bytes()).hexdigest()[:12]
    return (
        f"SIMIND {version} "
        f"[simind:sha256:{serial_digest};simind_mpi:sha256:{mpi_digest}]"
    )


def engine_label(engine: str) -> str:
    """Short human label for a detected engine, e.g. ``SIMIND v8``."""
    match = re.search(r"V(\d+)\.", engine)
    if match:
        return f"SIMIND v{match.group(1)}"
    return engine.split(" [", 1)[0]


@dataclass(frozen=True)
class Phantom:
    """NEMA/IEC body phantom with the masks the tutorials score."""

    nema: Any
    mu: np.ndarray
    slice_z: int
    profile_row: int
    background: np.ndarray

    @property
    def activity(self) -> np.ndarray:
        return self.nema.activity_mbq

    @property
    def total_mbq(self) -> float:
        return float(self.nema.activity_mbq.sum())

    def relative_to_background(self, array: np.ndarray) -> np.ndarray:
        return array / array[self.background].mean()


def phantom() -> Phantom:
    """Build the phantom and the attenuation map used by every tutorial."""
    from simind_python_connector.converters.attenuation import (
        get_attenuation_coefficient,
    )

    nema = nema_iec_phantom(
        GRID, (VOXEL_MM,) * 3, preset=PHANTOM_PRESET, supersample=SUPERSAMPLE
    )
    mu = nema.density_g_cm3 * get_attenuation_coefficient("water", 140.0)
    slice_z, profile_row = (
        np.argwhere(nema.masks["sphere_6"]).mean(axis=0).round().astype(int)[:2]
    )
    return Phantom(
        nema=nema,
        mu=mu,
        slice_z=int(slice_z),
        profile_row=int(profile_row),
        background=nema.masks["background"] > 0,
    )


def to_sirf_image(array: np.ndarray) -> Any:
    """Wrap a (z, y, x) array in a SIRF image with the tutorial geometry."""
    builder = _image_builder()
    builder.set_pixel_array(np.asarray(array, dtype=np.float32))
    return builder.build()


def _image_builder() -> Any:
    from simind_python_connector.builders import STIRSPECTImageDataBuilder

    return STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": str(GRID[2]),
            "!matrix size [2]": str(GRID[1]),
            "!matrix size [3]": str(GRID[0]),
            "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [3]": str(VOXEL_MM),
        },
        backend="sirf",
    )


def template(output_dir: Path) -> Any:
    """Acquisition template shared by the simulation and the reconstruction."""
    import sirf.STIR as sirf

    from simind_python_connector.builders import STIRSPECTAcquisitionDataBuilder

    sirf.AcquisitionData.set_storage_scheme("memory")
    return STIRSPECTAcquisitionDataBuilder(
        header_overrides={
            "!matrix size [1]": str(GRID[2]),
            "!matrix size [2]": str(GRID[1]),
            "!number of projections": str(VIEWS),
            "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
            "!direction of rotation": "CW",
            "start angle": "180",
            "Radius": str(RADIUS_MM),
        },
        backend="sirf",
    ).build(output_path=Path(output_dir) / "template")


def make_adaptor(
    output_dir: Path,
    name: str,
    *,
    quantization_scale: float = QUANTIZATION_SCALE,
    histories: int = HISTORY_MULTIPLIER,
) -> Any:
    """SIMIND adaptor with the tutorial nuclide, window, collimator and MPI."""
    from simind_python_connector import SirfSimindAdaptor

    adaptor = SirfSimindAdaptor(
        configs.get("Example.yaml"),
        str(output_dir),
        name,
        scoring_routine=ScoringRoutine.PENETRATE,
        quantization_scale=quantization_scale,
    )
    adaptor.add_config_value(20, ENERGY_WINDOW_KEV[1])
    adaptor.add_config_value(21, ENERGY_WINDOW_KEV[0])
    adaptor.add_runtime_switch("CC", COLLIMATOR)
    adaptor.add_runtime_switch("NN", histories)
    adaptor.add_config_value(1, 140.0)
    adaptor.set_mpi(MPI_PROCESSES)
    return adaptor


def source_map(activity: np.ndarray, quantization_scale: float) -> np.ndarray:
    """Integer source map SIMIND will read, for the quantization-loss check."""
    budget = 32767 * float(quantization_scale)
    scaled = activity / activity.max() * budget
    return np.clip(np.round(scaled), 0, 32767)


def quantization_loss(activity: np.ndarray, quantization_scale: float) -> float:
    """L1 relative error of the integer source map against the float map."""
    quantized = source_map(activity, quantization_scale).astype(np.float64)
    reference = activity / activity.sum() * quantized.sum()
    return float(np.abs(quantized - reference).sum() / reference.sum())


def scaled_counts(counts: np.ndarray, time_per_view_s: float) -> np.ndarray:
    """Expected counts at an acquisition time, linear in the view time."""
    scale = float(time_per_view_s) / REFERENCE_TIME_S
    return np.maximum(np.asarray(counts, dtype=np.float64) * scale, 0.0)


@dataclass(frozen=True)
class Expectation:
    """Simulated expectation of one 2 s projection, and its geometric primary.

    ``histories`` counts the photon histories per projection, including the
    replicated MPI runs. It is kept separate from the counts because it
    describes how much simulation was run, not the variance of a weighted
    estimator built from those histories.
    """

    counts: np.ndarray
    primary: np.ndarray
    histories: float
    quantization_loss: float
    seconds: float
    reused: bool

    @property
    def total(self) -> float:
        return float(self.counts.sum())

    @property
    def views(self) -> int:
        """Number of views in the projection stack."""
        return int(self.counts.shape[-2])

    @property
    def counts_per_projection(self) -> float:
        """Mean measured counts in one projection."""
        return self.total / self.views

    @property
    def true_scatter(self) -> np.ndarray:
        return self.counts - self.primary

    def measured(self, template: Any, seed: int, time_per_view_s: float) -> Any:
        """Poisson realisation of the expectation at the requested view time."""
        rng = np.random.default_rng(int(seed))
        counts = rng.poisson(scaled_counts(self.counts, time_per_view_s)).astype(
            np.float32
        )
        data = template.clone()
        data.fill(counts)
        return data


def _expectation_settings(
    phantom: Phantom, *, quantization_scale: float, histories: int, engine: str
) -> dict:
    return {
        "version": _CACHE_VERSION,
        "engine": engine,
        "grid": GRID,
        "voxel_mm": VOXEL_MM,
        "views": VIEWS,
        "radius_mm": RADIUS_MM,
        "window_kev": ENERGY_WINDOW_KEV,
        "collimator": COLLIMATOR,
        "preset": PHANTOM_PRESET,
        "supersample": SUPERSAMPLE,
        "total_mbq": float(phantom.total_mbq),
        "reference_time_s": REFERENCE_TIME_S,
        "quantization_scale": float(quantization_scale),
        "histories": int(histories),
    }


def _stored_histories(meta: dict) -> float:
    """Histories per projection from cache metadata, including MPI replication.

    Older cache files recorded the serial source-map budget only. Multiplying
    by the recorded process count puts those files on the same per-projection
    scale without resimulating; the counts themselves are unchanged.
    """
    histories = float(meta["histories"])
    if not meta.get("histories_include_mpi_replicas", False):
        histories *= int(meta.get("mpi_processes", 1))
    return histories


def expectation_cache_file(
    phantom: Phantom,
    *,
    quantization_scale: float = QUANTIZATION_SCALE,
    histories: int = HISTORY_MULTIPLIER,
    cache_dir: Path | None = None,
    engine: str | None = None,
) -> Path:
    """Where the expectation for these settings and engine is cached."""
    engine = engine or detect_engine()
    settings = _expectation_settings(
        phantom,
        quantization_scale=quantization_scale,
        histories=histories,
        engine=engine,
    )
    digest = hashlib.sha1(
        json.dumps(settings, sort_keys=True).encode(), usedforsecurity=False
    ).hexdigest()[:12]
    if cache_dir is None:
        cache_dir = Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "_nema"
    return Path(cache_dir) / f"expectation_{digest}.npz"


def simulate_expectation(
    phantom: Phantom,
    template: Any,
    *,
    quantization_scale: float = QUANTIZATION_SCALE,
    histories: int = HISTORY_MULTIPLIER,
    cache_dir: Path | None = None,
    engine: str | None = None,
) -> Expectation:
    """Simulate (or reuse) the high-precision expectation of one acquisition.

    The cache key holds the engine and every setting that changes the result.
    A reused file must name the same engine and settings; a mismatch is an
    error rather than a silent reuse. A reused file reports its original wall
    time, so the printed summary still shows what the simulation cost.
    """
    engine = engine or detect_engine()
    settings = _expectation_settings(
        phantom,
        quantization_scale=quantization_scale,
        histories=histories,
        engine=engine,
    )
    cache_file = expectation_cache_file(
        phantom,
        quantization_scale=quantization_scale,
        histories=histories,
        cache_dir=cache_dir,
        engine=engine,
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"{engine_label(engine)}: {GRID[0]}x{GRID[1]}x{GRID[2]} grid, {VIEWS} views "
        f"at {RADIUS_MM:.0f} mm, history multiplier {histories}, "
        f"{REFERENCE_TIME_S:g} s/view, {MPI_PROCESSES} MPI processes"
    )

    if cache_file.exists():
        data = np.load(cache_file)
        meta = json.loads(str(data["meta"]))
        if meta.get("engine") != engine:
            raise RuntimeError(
                f"Expectation cache {cache_file} was produced by "
                f"{meta.get('engine')!r}, not {engine!r}. Delete it to resimulate."
            )
        if meta.get("settings") != json.loads(json.dumps(settings)):
            raise RuntimeError(
                f"Expectation cache {cache_file} settings do not match this run. "
                "Delete it to resimulate."
            )
        print(
            f"Reused expectation {cache_file.name}: {meta['total']:.3g} counts, "
            f"{_stored_histories(meta):.3g} histories, "
            f"simulated in {meta['seconds'] / 60:.1f} min"
        )
        return Expectation(
            counts=data["counts"],
            primary=data["primary"],
            histories=_stored_histories(meta),
            quantization_loss=meta["quantization_loss"],
            seconds=meta["seconds"],
            reused=True,
        )

    activity = phantom.activity
    adaptor = make_adaptor(
        cache_file.parent,
        "expectation",
        quantization_scale=quantization_scale,
        histories=histories,
    )
    adaptor.set_template(template)
    adaptor.set_source(to_sirf_image(activity))
    adaptor.set_mu_map(to_sirf_image(phantom.mu))
    adaptor.set_activity(phantom.total_mbq, REFERENCE_TIME_S)
    started = time.perf_counter()
    outputs = adaptor.run()
    seconds = time.perf_counter() - started
    counts = outputs["all_interactions"].as_array()
    primary = outputs["geom_coll_primary"].as_array()
    loss = quantization_loss(activity, quantization_scale)
    histories_run = float(
        source_map(activity, quantization_scale).sum() * histories * MPI_PROCESSES
    )
    meta = {
        "settings": settings,
        "engine": engine,
        "mpi_processes": MPI_PROCESSES,
        "total": float(counts.sum()),
        "histories": histories_run,
        "histories_include_mpi_replicas": True,
        "quantization_loss": loss,
        "seconds": seconds,
    }
    temporary = cache_file.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, counts=counts, primary=primary, meta=json.dumps(meta)
    )
    temporary.replace(cache_file)
    print(
        f"Simulated expectation {cache_file.name}: {counts.sum():.3g} counts, "
        f"{histories_run:.3g} histories, {seconds / 60:.1f} min, "
        f"quantization loss {loss:.2%}"
    )
    return Expectation(
        counts=counts,
        primary=primary,
        histories=histories_run,
        quantization_loss=loss,
        seconds=seconds,
        reused=False,
    )


def to_acquisition(array: np.ndarray, template: Any) -> Any:
    """Wrap a projection array in the tutorial acquisition geometry."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[None]
    data = template.clone()
    data.fill(array)
    return data


def paired_realisations(
    expectation: Expectation,
    template: Any,
    *,
    seeds: tuple[int, ...] = SEEDS,
    time_per_view_s: float = REFERENCE_TIME_S,
) -> Iterator[tuple[int, Any]]:
    """Yield each seed once, so every arm of a comparison sees the same data."""
    for seed in seeds:
        yield seed, expectation.measured(template, seed, time_per_view_s)


def contrast_recovery(image: np.ndarray, phantom: Phantom) -> list[float]:
    """Masked sphere-to-background ratio over the truth, in the tutorial order."""
    relative = phantom.relative_to_background(np.asarray(image))
    return [
        relative[phantom.nema.masks[name] > 0].mean() / TRUE_RATIO for name in SPHERES
    ]


def contrast_recovery_scores(
    image: np.ndarray, phantom: Phantom
) -> tuple[list[float], float]:
    """Contrast recovery and background noise of one reconstruction."""
    return contrast_recovery(image, phantom), background_cov(image, phantom)


def background_cov(image: np.ndarray, phantom: Phantom) -> float:
    """Coefficient of variation in the background mask."""
    values = np.asarray(image)[phantom.background]
    return float(values.std() / values.mean())


def mean_and_sem(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error over the first axis of independent realisations."""
    array = np.asarray(values, dtype=np.float64)
    return array.mean(axis=0), array.std(axis=0, ddof=1) / np.sqrt(array.shape[0])


def mean_and_std(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample standard deviation over independent realisations."""
    array = np.asarray(values, dtype=np.float64)
    return array.mean(axis=0), array.std(axis=0, ddof=1)
