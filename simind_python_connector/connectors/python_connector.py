"""
Backend-agnostic SIMIND connector returning NumPy projection outputs.
"""

from __future__ import annotations

import contextlib
import logging
import math
import numbers
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from simind_python_connector import phantoms
from simind_python_connector.connectors.base import BaseConnector
from simind_python_connector.converters.attenuation import (
    attenuation_to_density,
    hu_to_density_schneider,
)
from simind_python_connector.converters.simind_to_stir import SimindToStirConverter
from simind_python_connector.core.config import RuntimeSwitches, SimulationConfig
from simind_python_connector.core.executor import (
    _OUTPUT_PREFIX_MAX_LENGTH,
    SimindExecutor,
)
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    MAX_SOURCE,
    OUTPUT_PREFIX_PATTERN,
    SIMIND_VOXEL_UNIT_CONVERSION,
    PenetrateOutputType,
    ScoringRoutine,
    SimulationError,
)
from simind_python_connector.phantoms import (
    AnalyticPhantom,
    CardiacSource,
    LibraryPhantom,
    MultipleInserts,
    VoxelPhantom,
)
from simind_python_connector.utils.interfile import (
    InterfileHeader,
    ProjectionGeometry,
    load_interfile_array,
)
from simind_python_connector.utils.simind_utils import create_window_file


ConfigSource = Union[str, os.PathLike[str], SimulationConfig]
PathLike = Union[str, os.PathLike[str]]

ZERO_HISTORY_WARNING_FRACTION = 0.01

# The Interfile connector can only read the projection outputs of these
# routines; LIST_MODE and FORCED_COLLIMATION write formats it cannot consume.
_SUPPORTED_SCORING_ROUTINES = (
    ScoringRoutine.DUMMY,
    ScoringRoutine.SCATTWIN,
    ScoringRoutine.PENETRATE,
)

_MU_MAP_TYPES = ("attenuation", "density", "hu")

_PHANTOM_SWITCHES = ("PX", "TH", "IF", "BG", "HO", "CO", "FZ") + tuple(
    f"{letter}{n}"
    for letter, count in (("A", 3), ("L", 6), ("M", 4))
    for n in range(1, count + 1)
)

_ZUB_TABLE = "phantom.zub"


def _find_zub_table() -> Path:
    """Locate the phantom.zub code table in the installed SIMIND tree."""
    roots = []
    for variable in ("SIMIND_SMC_DIR", "SMC_DIR", "SIMIND_DATA_DIR"):
        directory = os.environ.get(variable)
        if directory:
            roots.append(Path(directory).expanduser().resolve().parent)
    root = os.environ.get("SIMIND_ROOT")
    if root:
        roots.append(Path(root).expanduser().resolve())
    for executable in (os.environ.get("SIMIND_BIN"), "simind"):
        located = (
            shutil.which(SimindExecutor._resolve_executable(executable))
            if executable
            else None
        )
        if located:
            roots.append(Path(located).resolve().parent)

    for base in roots:
        for directory in ("tutorial", "smc_dir"):
            candidate = base / directory / _ZUB_TABLE
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not find SIMIND's {_ZUB_TABLE} code table; look for it in a "
        "tutorial/ or smc_dir/ below SIMIND_ROOT, SMC_DIR or the simind binary"
    )


def _split_voxel_size(voxel_size_mm) -> tuple[float, float]:
    """Return (slice thickness, in-plane size) in mm from a scalar or (z, y, x)."""
    if np.ndim(voxel_size_mm) == 0:
        size = float(voxel_size_mm)
        return size, size
    try:
        size_z, size_y, size_x = (float(v) for v in voxel_size_mm)
    except ValueError as exc:
        raise ValueError("voxel_size_mm must be a scalar or a (z, y, x) tuple") from exc
    if abs(size_y - size_x) > 1e-3:
        raise ValueError(
            f"SIMIND needs square in-plane voxels, got y={size_y} mm and x={size_x} mm"
        )
    return size_z, size_x


def _same_file_entry(a: Path, b: Path) -> bool:
    """True when both paths name the same directory entry, links included."""
    try:
        return os.path.samestat(os.lstat(a), os.lstat(b))
    except OSError:
        return False


def _file_identity(path: Path) -> set[tuple[int, int]]:
    """Device/inode of a path and, when it is a link, of its referent.

    The link's own identity catches a hardlink alias of the file; the
    dereferenced identity catches a symlink alias.
    """
    identities = set()
    for stat_fn in (os.lstat, os.stat):
        try:
            details = stat_fn(path)
        except OSError:
            continue
        identities.add((details.st_dev, details.st_ino))
    return identities


def _same_real_file(a: Path, b: Path) -> bool:
    """True when a and b dereference to the same device and inode.

    A symlink, hardlink or differently-cased spelling of either path is
    caught, because the comparison follows links to the referent.
    """
    try:
        a_stat = os.stat(a)
        b_stat = os.stat(b)
    except OSError:
        return False
    return (a_stat.st_dev, a_stat.st_ino) == (b_stat.st_dev, b_stat.st_ino)


@dataclass(frozen=True)
class ProjectionResult:
    """Projection array together with the header and binary file references."""

    projection: np.ndarray
    header_path: Path
    data_path: Path
    metadata: dict[str, str]


@dataclass
class RuntimeOperator:
    """Runtime modifiers applied when invoking SIMIND."""

    switches: Dict[str, Any] = field(default_factory=dict)
    orbit_file: Optional[PathLike] = None


class SimindPythonConnector(BaseConnector):
    """Pure Python connector for SIMIND with NumPy-first outputs."""

    def __init__(
        self,
        config_source: ConfigSource,
        output_dir: PathLike,
        output_prefix: str = "output",
        quantization_scale: float = DEFAULT_QUANTIZATION_SCALE,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self._validate_output_prefix(output_prefix)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_prefix = output_prefix
        try:
            self.quantization_scale = float(quantization_scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "quantization_scale must be in (0, 1]: the source maximum is "
                f"scaled to {MAX_SOURCE} x quantization_scale before voxel-wise "
                "rounding. Use the NN runtime switch for more statistics."
            ) from exc
        if (
            not math.isfinite(self.quantization_scale)
            or self.quantization_scale <= 0
            or self.quantization_scale > 1
        ):
            raise ValueError(
                "quantization_scale must be in (0, 1]: the source maximum is "
                f"scaled to {MAX_SOURCE} x quantization_scale before voxel-wise "
                "rounding. Use the NN runtime switch for more statistics."
            )

        self.config = self._initialize_config(config_source)
        self.runtime_switches = RuntimeSwitches()
        self.executor = SimindExecutor()
        self.converter = SimindToStirConverter()

        self._outputs: Optional[dict[str, ProjectionResult]] = None
        self._window_file_path: Optional[Path] = None
        self._mpi: Optional[tuple[int, bool]] = None
        self._derived_keys: set[str] = set()
        self._acquisition: Optional[ProjectionGeometry] = None
        self._acquisition_radii_cm: Optional[list[float]] = None
        self._phantom_extent_mm: Optional[tuple[float, float]] = None
        self._phantom_mode: Optional[str] = None
        self._orbit_source_paths: tuple[Path, ...] = ()

    @staticmethod
    def _validate_output_prefix(prefix: str) -> None:
        """Accept only prefixes SIMIND keeps unchanged in its file names.

        SIMIND lowercases file names on Unix, a dot would be taken for a file
        extension, and glob characters would break output discovery. SIMIND
        treats '-' as an argument delimiter, so a hyphenated prefix would
        split the positional filename arguments and produce no output. The
        length cap keeps the engine's output names inside its fixed
        60-character buffer.
        """
        if not isinstance(prefix, str) or not re.fullmatch(
            OUTPUT_PREFIX_PATTERN, prefix
        ):
            raise ValueError(
                "output_prefix must contain only lowercase letters, digits and "
                f"'_', never '-' (SIMIND treats '-' as an argument delimiter), "
                f"got {prefix!r}"
            )
        if len(prefix) > _OUTPUT_PREFIX_MAX_LENGTH:
            raise ValueError(
                f"output_prefix must be at most {_OUTPUT_PREFIX_MAX_LENGTH} "
                f"characters, got {len(prefix)}"
            )

    @staticmethod
    def _initialize_config(config_source: ConfigSource) -> SimulationConfig:
        if isinstance(config_source, SimulationConfig):
            return config_source

        if not isinstance(config_source, (str, os.PathLike)):
            # importlib.resources Traversable from configs.get(): hand it to
            # SimulationConfig untouched (supports zipped installs).
            return SimulationConfig(config_source)

        config_path = Path(config_source).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_source}")

        suffix = config_path.suffix.lower()
        if suffix not in {".smc", ".yaml", ".yml"}:
            raise ValueError(
                f"Unsupported configuration file extension {suffix!r}. "
                "Expected one of .smc, .yaml, .yml"
            )

        return SimulationConfig(str(config_path))

    def add_runtime_switch(self, switch: str, value: Any) -> None:
        """Set a single runtime switch."""
        self.runtime_switches.set_switch(switch, value)

    def add_config_value(self, index: int, value: Any) -> None:
        """Set a SIMIND config value."""
        self.config.set_value(index, value)

    def set_mpi(
        self, processes: Optional[int], split_projections: bool = False
    ) -> None:
        """Run SIMIND under MPI (mpirun + simind_mpi), or turn MPI off with None.

        By default every MPI process simulates all projections and SIMIND
        sums the results. With split_projections=True, /MP shares the
        projections out; the number of projections (Index 29) must then be
        a multiple of processes.
        """
        if processes is None:
            if split_projections:
                raise ValueError("split_projections needs processes")
            self._mpi = None
            return
        if (
            isinstance(processes, bool)
            or not isinstance(processes, (int, np.integer))
            or processes < 1
        ):
            raise ValueError(f"processes must be an integer >= 1, got {processes!r}")
        self._mpi = (int(processes), bool(split_projections))

    def set_activity(
        self, activity_mbq: float, time_per_projection_s: float = 1.0
    ) -> None:
        """Tell SIMIND the source activity, so projections come out on that scale.

        SIMIND Index 25 holds activity x acquisition time per projection
        (MBq x s); the manual assumes 1 s per projection otherwise. Index 25
        does not change the number of photon histories: the source map's
        integer values and NN set those.
        """
        for name, value in (
            ("activity_mbq", activity_mbq),
            ("time_per_projection_s", time_per_projection_s),
        ):
            try:
                valid = (
                    not isinstance(value, (bool, np.bool_))
                    and math.isfinite(float(value))
                    and float(value) > 0
                )
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        product = float(activity_mbq) * float(time_per_projection_s)
        if not math.isfinite(product) or product <= 0:
            raise ValueError(f"product must be finite and > 0, got {product!r}")
        self.config.set_value(25, product)

    def _reset_phantom_settings(self) -> None:
        """Remove the switches and map files of an earlier phantom, and the
        source shifts an earlier analytic phantom set."""
        if self._phantom_mode == "analytic":
            for index in (16, 17, 18):
                self.config.set_value(index, 0.0)
        for switch in _PHANTOM_SWITCHES:
            self.runtime_switches.set_switch(switch, None)
        self.config.set_data_file(5, "none")
        self.config.set_data_file(6, "none")
        self._phantom_extent_mm = None

    def configure_voxel_phantom(
        self,
        source: np.ndarray,
        mu_map: np.ndarray,
        voxel_size_mm: Union[float, tuple[float, float, float]] = 4.0,
        scoring_routine: Optional[Union[ScoringRoutine, int]] = None,
        mu_map_type: str = "attenuation",
        mu_map_energy_kev: Optional[float] = None,
    ) -> tuple[Path, Path]:
        """
        Configure voxel geometry and write source/density input files.

        The projection size (Index 28, 76, 77) comes from the configuration or
        configure_acquisition, not from the phantom. The connector reads SPECT
        Interfile output, so this sets Flag 5 (SPECT) and Flag 14 (Interfile
        headers).

        Args:
            source: Activity distribution in array order (z, y, x).
            mu_map: Attenuator, interpreted according to mu_map_type.
            voxel_size_mm: One size, or (z, y, x); y and x must be equal.
            scoring_routine: Optional override for Index 84; None keeps the
                configuration value.
            mu_map_type: "attenuation" (cm^-1 at mu_map_energy_kev), "density"
                (g/cm^3) or "hu" (CT Hounsfield units, Schneider conversion).
            mu_map_energy_kev: Energy the attenuation map is defined at;
                defaults to abs(Index 1).

        Returns:
            Tuple of (source_file_path, density_file_path).
        """
        self._validate_output_prefix(self.output_prefix)
        if mu_map_type not in _MU_MAP_TYPES:
            raise ValueError(
                f"mu_map_type must be one of {', '.join(_MU_MAP_TYPES)}, "
                f"got {mu_map_type!r}"
            )
        if mu_map_energy_kev is not None and not (
            math.isfinite(float(mu_map_energy_kev)) and float(mu_map_energy_kev) > 0
        ):
            raise ValueError(
                f"mu_map_energy_kev must be > 0, got {mu_map_energy_kev!r}"
            )

        source_array = np.asarray(source, dtype=np.float32)
        mu_map_array = np.asarray(mu_map, dtype=np.float32)

        if source_array.ndim != 3 or mu_map_array.ndim != 3:
            raise ValueError("source and mu_map must both be 3D arrays")
        if source_array.shape != mu_map_array.shape:
            raise ValueError("source and mu_map must have identical shapes")

        slice_mm, pixel_mm = _split_voxel_size(voxel_size_mm)
        slice_cm = slice_mm / SIMIND_VOXEL_UNIT_CONVERSION
        pixel_cm = pixel_mm / SIMIND_VOXEL_UNIT_CONVERSION
        if not all(math.isfinite(v) and v > 0 for v in (slice_cm, pixel_cm)):
            raise ValueError("voxel_size_mm must be > 0")
        if source_array.size == 0 or mu_map_array.size == 0:
            raise ValueError("source and mu_map must not be empty")
        if not np.isfinite(source_array).all() or not np.isfinite(mu_map_array).all():
            raise ValueError("source and mu_map must contain only finite values")
        if (source_array < 0).any() or (
            mu_map_type != "hu" and (mu_map_array < 0).any()
        ):
            raise ValueError("source and mu_map must be non-negative")
        if source_array.size == 1:
            raise ValueError(
                "SIMIND does not support a 1x1x1 voxel phantom; pad it to at "
                "least two voxels in one dimension."
            )

        if scoring_routine is None:
            routine = self._config_scoring_routine()
        elif isinstance(scoring_routine, ScoringRoutine):
            routine = scoring_routine
        elif isinstance(scoring_routine, int) and not isinstance(scoring_routine, bool):
            try:
                routine = ScoringRoutine(scoring_routine)
            except ValueError as exc:
                raise ValueError(
                    f"scoring_routine {scoring_routine!r} is not a valid "
                    "ScoringRoutine value"
                ) from exc
        else:
            raise ValueError(
                "scoring_routine must be None, a ScoringRoutine or int, got "
                f"{type(scoring_routine).__name__}"
            )
        self._validate_supported_scoring_routine(routine)
        dim_z, dim_y, dim_x = (int(v) for v in source_array.shape)

        # Quantize before mutating the configuration or writing inputs so a
        # scale that erases the source cannot leave a half-written run behind.
        source_max = float(source_array.max())
        if source_max > 0:
            source_scaled = (
                source_array / source_max * (MAX_SOURCE * self.quantization_scale)
            )
        else:
            source_scaled = np.zeros_like(source_array)
        source_u16 = np.clip(np.round(source_scaled), 0, MAX_SOURCE).astype(np.uint16)
        if source_max > 0 and not source_u16.any():
            raise ValueError(
                "quantization_scale rounds the non-zero source to all zeros; "
                "increase quantization_scale."
            )

        # Reset after validation so an invalid input cannot clear an earlier
        # phantom's settings or its maps.
        self._reset_phantom_settings()
        cfg = self.config
        cfg.set_flag(5, True)
        cfg.set_value(15, -1)
        cfg.set_value(14, -1)
        cfg.set_flag(14, True)
        if scoring_routine is not None:
            cfg.set_value(84, routine.value)

        # Index 2 and 5 are half-lengths along the slice axis; SIMIND derives
        # the slice thickness from them and the number of slices.
        cfg.set_value(2, dim_z * slice_cm / 2.0)
        cfg.set_value(3, dim_x * pixel_cm / 2.0)
        cfg.set_value(4, dim_y * pixel_cm / 2.0)

        # Density geometry
        cfg.set_value(5, dim_z * slice_cm / 2.0)
        cfg.set_value(6, dim_x * pixel_cm / 2.0)
        cfg.set_value(7, dim_y * pixel_cm / 2.0)
        cfg.set_value(31, pixel_cm)
        cfg.set_value(33, 1)
        cfg.set_value(34, dim_z)
        cfg.set_value(78, dim_x)  # density map i
        cfg.set_value(79, dim_x)  # source map i
        cfg.set_value(81, dim_y)  # density map j
        cfg.set_value(82, dim_y)  # source map j

        self.runtime_switches.set_switch("PX", pixel_cm)
        self.runtime_switches.set_switch(
            "TH", slice_cm if abs(slice_cm - pixel_cm) > 1e-9 else None
        )
        self._phantom_extent_mm = (max(dim_x, dim_y) * pixel_mm, dim_z * slice_mm)

        total_activity = float(source_array.sum())
        if total_activity > 0:
            zero_history_fraction = (
                float(source_array[source_u16 == 0].sum()) / total_activity
            )
            if zero_history_fraction > ZERO_HISTORY_WARNING_FRACTION:
                self.logger.warning(
                    "%.1f%% of the source activity is in voxels below the integer "
                    "resolution and gets no photon histories; increase "
                    "quantization_scale (maximum 1.0) to keep it",
                    100 * zero_history_fraction,
                )

        src_prefix = f"{self.output_prefix}_src"
        source_path = self.output_dir / f"{src_prefix}.smi"
        source_u16.tofile(source_path)
        cfg.set_data_file(6, src_prefix)

        if cfg.get_flag(11):
            density = (
                self._density_from_mu_map(mu_map_array, mu_map_type, mu_map_energy_kev)
                * 1000.0
            )
        else:
            density = np.zeros_like(mu_map_array)

        density_u16 = np.clip(np.round(density), 0, np.iinfo(np.int16).max).astype(
            np.uint16
        )
        dns_prefix = f"{self.output_prefix}_dns"
        density_path = self.output_dir / f"{dns_prefix}.dmi"
        density_u16.tofile(density_path)
        cfg.set_data_file(5, dns_prefix)

        self._phantom_mode = "voxel"
        return source_path, density_path

    def _density_from_mu_map(
        self,
        mu_map_array: np.ndarray,
        mu_map_type: str,
        mu_map_energy_kev: Optional[float],
    ) -> np.ndarray:
        """Convert the mu-map input to density in g/cm^3."""
        if mu_map_type == "density":
            return mu_map_array
        if mu_map_type == "hu":
            return hu_to_density_schneider(mu_map_array)
        if mu_map_energy_kev is None:
            # A negative Index 1 asks SIMIND to read photon energies from the
            # isotope file; its absolute value is still the window energy.
            mu_map_energy_kev = abs(float(self.config.get_value("photon_energy")))
        return attenuation_to_density(mu_map_array, mu_map_energy_kev)

    def configure_phantom(
        self,
        phantom: Union[AnalyticPhantom, LibraryPhantom, VoxelPhantom],
        *,
        time_per_projection_s: float = 1.0,
        scoring_routine: Optional[Union[ScoringRoutine, int]] = None,
    ) -> None:
        """Simulate an analytic, library or voxel phantom (see phantoms).

        A VoxelPhantom has a known activity, so SIMIND is given it through
        Index 25 (normalisation route 1); time_per_projection_s is only used for
        VoxelPhantom. The last configure_phantom/configure_voxel_phantom call wins.
        scoring_routine overrides Index 84 for a VoxelPhantom only; None keeps
        the configuration value. Phantom interactions follow Flag 11 from the
        configuration, except that an AnalyticPhantom sets it from whether it
        has an attenuator.
        """
        self._validate_output_prefix(self.output_prefix)
        if isinstance(phantom, VoxelPhantom):
            self.configure_voxel_phantom(
                source=phantom.activity_mbq,
                mu_map=phantom.density_g_cm3,
                voxel_size_mm=phantom.voxel_size_mm,
                scoring_routine=scoring_routine,
                mu_map_type="density",
            )
            self.set_activity(
                float(np.sum(phantom.activity_mbq)), time_per_projection_s
            )
            return
        if isinstance(phantom, LibraryPhantom):
            self._reset_phantom_settings()
            self._configure_library_phantom(phantom)
            self._phantom_mode = "library"
            return
        if isinstance(phantom, AnalyticPhantom):
            self._reset_phantom_settings()
            self._configure_analytic_phantom(phantom)
            self._phantom_mode = "analytic"
            return
        raise TypeError(
            "phantom must be an AnalyticPhantom, LibraryPhantom or VoxelPhantom, "
            f"got {type(phantom).__name__}"
        )

    def _configure_analytic_phantom(self, phantom: AnalyticPhantom) -> None:
        cfg = self.config
        cfg.set_flag(14, True)
        code, half_dims = phantoms._code_and_half_dims(phantom.source)
        cfg.set_value(15, code)
        for index, value in zip((2, 3, 4), half_dims):
            cfg.set_value(index, value)
        for index, value in zip((16, 17, 18), phantom.source_shift_cm):
            cfg.set_value(index, value)

        if phantom.attenuator is None:
            # A box around the source keeps SIMIND away from the voxel maps of
            # an earlier set-up; with Flag 11 off it does not attenuate.
            phantom_code, phantom_dims = 2, half_dims
            cfg.set_flag(11, False)
        else:
            phantom_code, phantom_dims = phantoms._code_and_half_dims(
                phantom.attenuator
            )
            cfg.set_flag(11, True)
        cfg.set_value(14, phantom_code)
        for index, value in zip((5, 6, 7), phantom_dims):
            cfg.set_value(index, value)

        source = phantom.source
        if isinstance(source, MultipleInserts):
            rows = "".join(
                phantoms._insert_row(insert) + "\n" for insert in source.inserts
            )
            (self.output_dir / f"{self.output_prefix}.inp").write_text(rows)
            self.runtime_switches.set_switch("IF", f"{self.output_prefix}.inp")
            if source.background is not None:
                self.runtime_switches.set_switch("BG", source.background)
            if source.mode == "hot":
                self.runtime_switches.set_switch("HO", True)
            elif source.mode == "cold":
                self.runtime_switches.set_switch("CO", True)
        elif isinstance(source, CardiacSource):
            for switch, value in phantoms._cardiac_switches(source).items():
                self.runtime_switches.set_switch(switch, value)

    def _configure_library_phantom(self, phantom: LibraryPhantom) -> None:
        cfg = self.config
        cfg.set_flag(14, True)
        code, base_name, zub_section = phantom.value
        cfg.set_value(14, code)
        cfg.set_value(15, code)
        cfg.set_data_file(5, base_name)
        cfg.set_data_file(6, base_name)
        cfg.set_flag(15, True)  # write the aligned density, the ground truth
        # SIMIND resolves /FZ against its working directory, so the code table
        # has to sit next to the run's inputs.
        shutil.copy2(_find_zub_table(), self.output_dir / _ZUB_TABLE)
        cfg.set_value(45, zub_section)
        self.runtime_switches.set_switch("FZ", Path(_ZUB_TABLE).stem)
        if phantom is LibraryPhantom.NEMA_IQ:
            # SIMIND manual, voxel phantom table; the file is 364 wide, although
            # the table says 384.
            cfg.set_value(31, 0.1)
            cfg.set_value(2, 11.0)
            cfg.set_value(5, 11.0)
            cfg.set_value(33, 1)
            cfg.set_value(34, 110)
            for index in (78, 79, 81, 82):
                cfg.set_value(index, 364)

    def configure_acquisition(self, geometry: ProjectionGeometry) -> None:
        """Use the projection geometry of a measured acquisition.

        Index 30 is the signed rotation range (negative is counter-clockwise).
        An extent within the 0.5-degree geometry tolerance above 360 is
        written as a full turn, leaving the passed geometry untouched.
        STIR's start angle is SIMIND's plus 180 degrees, which the
        converter adds back to the outputs.
        """
        if abs(geometry.bin_size_mm - geometry.axial_size_mm) > 1e-3:
            raise ValueError(
                "SIMIND has one projection pixel size, but the bin size "
                f"({geometry.bin_size_mm} mm) and axial size "
                f"({geometry.axial_size_mm} mm) differ"
            )
        if not 0.0 < geometry.extent_deg <= 360.5:
            raise ValueError(
                f"extent of rotation must be in (0, 360.5], got {geometry.extent_deg}"
            )
        if (
            geometry.radii_mm is not None
            and len(geometry.radii_mm) != geometry.num_projections
        ):
            raise ValueError(
                f"{len(geometry.radii_mm)} radii given for "
                f"{geometry.num_projections} projections"
            )

        cfg = self.config
        cfg.set_flag(5, True)
        cfg.set_value(29, geometry.num_projections)
        # SIMIND rejects extents beyond a full circle, but STIR templates
        # measured against a full turn report 360.001: that difference sits
        # inside the 0.5-degree geometry tolerance, so SIMIND gets 360.
        extent_deg = min(geometry.extent_deg, 360.0)
        sign = 1.0 if geometry.direction == "CW" else -1.0
        cfg.set_value(30, sign * extent_deg)
        cfg.set_value(41, (geometry.start_angle_deg + 180.0) % 360.0)
        cfg.set_value(28, geometry.bin_size_mm / SIMIND_VOXEL_UNIT_CONVERSION)
        cfg.set_value(76, geometry.num_bins)
        cfg.set_value(77, geometry.num_axial)

        if geometry.radii_mm is not None and len(set(geometry.radii_mm)) == 1:
            cfg.set_value(12, geometry.radii_mm[0] / SIMIND_VOXEL_UNIT_CONVERSION)
            self._acquisition_radii_cm = None
            self._acquisition = geometry
            return

        if geometry.radii_mm is None:
            cfg.set_value(12, geometry.radius_mm / SIMIND_VOXEL_UNIT_CONVERSION)
            self._acquisition_radii_cm = None
        else:
            radii_cm = [r / SIMIND_VOXEL_UNIT_CONVERSION for r in geometry.radii_mm]
            cfg.set_value(12, float(np.mean(radii_cm)))
            self._acquisition_radii_cm = radii_cm
        self._acquisition = geometry

    def set_energy_windows(
        self,
        lower_bounds: Union[float, list[float]],
        upper_bounds: Union[float, list[float]],
        scatter_orders: Union[int, list[int]],
    ) -> None:
        """Write a SIMIND window file for this connector run."""
        self._validate_output_prefix(self.output_prefix)
        window_path = self.output_dir / f"{self.output_prefix}.win"
        create_window_file(
            lower_bounds,
            upper_bounds,
            scatter_orders,
            output_filename=str(window_path),
        )
        self._window_file_path = window_path

    def _validate_runtime_operator(
        self,
        runtime_operator: Optional[RuntimeOperator],
        *,
        write_paths: tuple[Path, ...] = (),
    ) -> RuntimeSwitches:
        """Validate the effective switch set and orbit without writing files.

        Builds the same persistent+runtime switch set run() uses (operator
        values override persistent ones and None removes them), adds the
        automatic CA=1 for the SCATTWIN routine and serialises it through the
        executor, so an illegal switch is rejected before phantom inputs, an
        acquisition orbit or a config are written. Returns the validated
        holder for run().

        write_paths are the files an adaptor is about to write for this run;
        a supplied orbit that is one of them, or the config destination, is
        rejected dereferenced so a real input is never overwritten by staging.
        """
        run_switches_holder = RuntimeSwitches()
        for key, value in self.runtime_switches.switches.items():
            run_switches_holder.set_switch(key, value)
        if runtime_operator is not None:
            for key, value in runtime_operator.switches.items():
                run_switches_holder.set_switch(key, value)

        # SIMIND writes air, tot and sca with /CA:1 (manual, scattwin routine);
        # primary is then tot - sca.
        if self._config_scoring_routine() is ScoringRoutine.SCATTWIN and (
            "CA" not in run_switches_holder.switches
        ):
            run_switches_holder.set_switch("CA", 1)

        self.executor._serialize_switches(run_switches_holder.switches)

        if runtime_operator is not None and runtime_operator.orbit_file is not None:
            orbit_path = Path(runtime_operator.orbit_file).expanduser().resolve()
            if not orbit_path.exists():
                raise FileNotFoundError(f"Orbit file not found: {orbit_path}")
            if not orbit_path.is_file():
                raise ValueError(f"orbit_file must be a regular file: {orbit_path}")
            self._reject_orbit_write_conflict(orbit_path, write_paths)

        return run_switches_holder

    def _reject_orbit_write_conflict(
        self, orbit_path: Path, write_paths: tuple[Path, ...]
    ) -> None:
        """Reject an orbit this run would overwrite as its config or an input.

        The comparison is dereferenced so a symlink, hardlink or differently
        cased spelling of a destination is caught, while the caller's file is
        left in place rather than detached or renamed.
        """
        destinations = (self.output_dir / f"{self.output_prefix}.smc", *write_paths)
        for destination in destinations:
            if _same_real_file(orbit_path, destination):
                raise ValueError(
                    f"orbit_file {orbit_path} is the same file as this run's "
                    f"destination {destination}; refusing to overwrite it"
                )

    def run(
        self, runtime_operator: Optional[RuntimeOperator] = None
    ) -> dict[str, ProjectionResult]:
        """Run SIMIND and return projection outputs as NumPy arrays."""
        self._outputs = None
        self._orbit_source_paths = ()

        # Fail before writing a generated acquisition orbit, staging a supplied
        # orbit, saving the config or clearing stale outputs when the run cannot
        # be supported or the prefix is no longer safe.
        self._validate_output_prefix(self.output_prefix)
        self._config_scoring_routine()
        run_switches_holder = self._validate_runtime_operator(runtime_operator)

        if self._phantom_mode is None:
            self.logger.warning(
                "No phantom configured; SIMIND will use the phantom defined by "
                "the configuration (Index 14/15 and any files it references)."
            )

        mpi_kwargs = {}
        if self._mpi is not None:
            processes, split = self._mpi
            if split:
                raw_projections = self.config.get_value(29)
                try:
                    num_projections = int(round(float(raw_projections)))
                except (TypeError, ValueError, OverflowError) as exc:
                    # Report only the type: interpolating a repr() of a huge
                    # int/Fraction can itself raise the int-string digit limit.
                    raise ValueError(
                        "split_projections needs the number of projections "
                        f"(Index 29) to be an integer, got "
                        f"{type(raw_projections).__name__}"
                    ) from exc
                if num_projections % processes != 0:
                    raise ValueError(
                        f"split_projections needs the number of projections "
                        f"({num_projections}) to be a multiple of the MPI "
                        f"processes ({processes})"
                    )
            mpi_kwargs = {"mpi_processes": processes, "split_projections": split}

        # The effective switches are validated and serialised above, before an
        # explicit orbit is staged or the acquisition orbit below is written.
        # An explicit runtime orbit wins: the acquisition orbit is only
        # generated when none was supplied, so a supplied orbit at (or aliased
        # to) the acquisition pathname is never overwritten before staging.
        orbit_file = None
        if runtime_operator is not None and runtime_operator.orbit_file is not None:
            orbit_file = self._prepare_orbit_file(runtime_operator.orbit_file)
        elif self._acquisition_radii_cm is not None:
            orbit_file = self.output_dir / f"{self.output_prefix}_acquisition.cor"
            # SIMIND's own .cor layout (Index 42 < 0 writes one radius per line
            # in cm at %12.3f followed by the centre bin, Index 76 / 2, at %6d).
            # Written here so a matrix size changed after the template counts.
            centre_bin = int(self.config.get_value(76)) // 2
            orbit_file.write_text(
                "".join(
                    f"{r:12.3f}{centre_bin:6d}\n" for r in self._acquisition_radii_cm
                )
            )

        if self._phantom_extent_mm is not None:
            bin_mm = float(self.config.get_value(28)) * SIMIND_VOXEL_UNIT_CONVERSION
            width_mm, length_mm = self._phantom_extent_mm
            fov_width_mm = float(self.config.get_value(76)) * bin_mm
            fov_length_mm = float(self.config.get_value(77)) * bin_mm
            if width_mm > fov_width_mm:
                self.logger.warning(
                    "The phantom (%.1f mm across) is wider than the projection "
                    "field of view (%.1f mm)",
                    width_mm,
                    fov_width_mm,
                )
            if length_mm > fov_length_mm:
                self.logger.warning(
                    "The phantom (%.1f mm axially) is longer than the projection "
                    "field of view (%.1f mm)",
                    length_mm,
                    fov_length_mm,
                )

        window_file = self._window_file_for_run(run_switches_holder.switches)

        # Outputs are read from SPECT Interfile headers (.h00), which SIMIND
        # writes only with Flag 5 (SPECT) and Flag 14 (Interfile) set. A
        # configuration-defined phantom never goes through configure_phantom,
        # so set them for every run.
        self.config.set_flag(5, True)
        self.config.set_flag(14, True)
        config_path = self.output_dir / self.output_prefix
        self.config.save_file(config_path)

        self._clear_previous_outputs(window_file)
        self.executor.run_simulation(
            self.output_prefix,
            orbit_file,
            run_switches_holder.switches,
            cwd=self.output_dir,
            **mpi_kwargs,
        )

        header_files = self._ensure_interfile_headers()
        header_files = header_files + self._derive_primary_headers(
            header_files, window_file
        )
        self._outputs = self._load_projection_outputs(header_files)
        return self._outputs

    def get_outputs(self) -> dict[str, ProjectionResult]:
        """Return cached outputs from the last completed run."""
        if self._outputs is None:
            raise RuntimeError("No outputs are available. Run the connector first.")
        return self._outputs

    def get_config(self) -> SimulationConfig:
        return self.config

    _OUTPUT_SUFFIXES = {".h00", ".hs", ".a00", ".s", ".win"}

    def _window_file_for_run(self, switches: dict) -> Optional[Path]:
        """Return the window file SIMIND reads for this run, if any."""
        fw = switches.get("FW")
        if fw:
            return self.output_dir / f"{fw}.win"
        if (
            self._window_file_path is not None
            and self._window_file_path.parent == self.output_dir
        ):
            return self._window_file_path
        scattwin = self.output_dir / "scattwin.win"
        return scattwin if scattwin.is_file() else None

    @staticmethod
    def _read_window_orders(window_file: Path) -> list[Optional[int]]:
        """Scatter order of each window line; None for keyword or malformed lines."""
        orders: list[Optional[int]] = []
        for line in window_file.read_text().splitlines():
            lowered = line.lower()
            fields = line.split(",")
            if "dew" in lowered or "tew" in lowered or len(fields) != 3:
                orders.append(None)
                continue
            try:
                value = float(fields[2])
            except ValueError:
                orders.append(None)
                continue
            orders.append(int(value) if value.is_integer() else None)
        return orders

    def _clear_previous_outputs(self, window_file: Optional[Path] = None) -> None:
        """Delete stale outputs from earlier runs sharing this prefix.

        Connector-written inputs (``{prefix}_src.smi`` / ``{prefix}_dns.dmi``)
        are protected because SIMIND still needs them on disk. The
        caller-supplied orbit source is protected too, by name and by file
        identity, so cleanup cannot delete it through a symlink, a hardlink or
        a differently-cased spelling on a case-insensitive filesystem.
        """
        if window_file is None:
            window_file = self._window_file_for_run({})
        protected = {
            f"{self.output_prefix}_src.smi",
            f"{self.output_prefix}_dns.dmi",
        }
        if window_file is not None and window_file.parent == self.output_dir:
            protected.add(window_file.name)
        for slot in (5, 6):
            try:
                protected.add(self.config.get_data_file(slot))
            except KeyError:
                pass
        protected_ids: set[tuple[int, int]] = set()
        for orbit_source in self._orbit_source_paths:
            if orbit_source.parent.resolve() == self.output_dir:
                protected.add(orbit_source.name)
            protected_ids |= _file_identity(orbit_source)

        def is_protected(path: Path) -> bool:
            return path.name in protected or bool(_file_identity(path) & protected_ids)

        for path in sorted(self.output_dir.iterdir()):
            if not path.is_file() or is_protected(path):
                continue
            stem_ok = path.stem == self.output_prefix or path.stem.startswith(
                f"{self.output_prefix}_"
            )
            if not stem_ok:
                continue
            if path.suffix in self._OUTPUT_SUFFIXES or (
                path.suffix.startswith(".b")
                and path.suffix[2:].isdigit()
                and path.stem == self.output_prefix
            ):
                path.unlink()

        for suffix in (".res", ".bis", ".spe", ".cor", ".hct", ".ict"):
            stale = self.output_dir / f"{self.output_prefix}{suffix}"
            if stale.is_file() and not is_protected(stale):
                stale.unlink()

    def _prepare_orbit_file(self, orbit_file: Optional[PathLike]) -> Optional[Path]:
        if orbit_file is None:
            return None

        supplied_path = Path(orbit_file).expanduser()
        orbit_path = supplied_path.resolve()
        if not orbit_path.exists():
            raise FileNotFoundError(f"Orbit file not found: {orbit_path}")
        if not orbit_path.is_file():
            raise ValueError(f"orbit_file must be a regular file: {orbit_path}")
        self._orbit_source_paths = (supplied_path, orbit_path)

        # SIMIND lowercases file names, so pass it a lower-case copy staged
        # under a connector-owned name. The caller's orbit may share its
        # basename with a protected input, so it is never used in place.
        target = self.output_dir / f"{self.output_prefix}_orbit.cor"
        if orbit_path == target or _same_file_entry(target, supplied_path):
            return target

        # Drop a pre-existing alias at the target (for example a symlink to
        # the orbit source) so copy2 writes a real file instead of following
        # the link. Never unlink the caller's own file: on a case-insensitive
        # filesystem, or when the supplied path is an external symlink whose
        # referent is the target, the target is that same directory entry.
        if target.is_symlink() or target.exists():
            if _same_file_entry(target, supplied_path) or _same_file_entry(
                target, orbit_path
            ):
                return target
            target.unlink()
        shutil.copy2(orbit_path, target)
        return target

    def _ensure_interfile_headers(self) -> list[Path]:
        if self._is_penetrate_routine():
            h00_file = self.converter.find_penetrate_h00_file(
                self.output_prefix, str(self.output_dir)
            )
            if h00_file is None:
                raise FileNotFoundError(
                    f"No PENETRATE .h00 file found for prefix {self.output_prefix!r} "
                    f"in {self.output_dir}"
                )
            self.converter.create_penetrate_headers_from_template(
                h00_file, self.output_prefix, str(self.output_dir)
            )
            hs_files = sorted(
                self.output_dir.glob(f"{self.output_prefix}_component_*.hs")
            )
        else:
            h00_files = sorted(
                set(self.output_dir.glob(f"{self.output_prefix}_*.h00"))
                | set(self.output_dir.glob(f"{self.output_prefix}.h00"))
            )
            for h00_file in h00_files:
                hs_file = h00_file.with_suffix(".hs")
                self.converter.convert_file(str(h00_file), str(hs_file))

            hs_files = sorted(
                set(self.output_dir.glob(f"{self.output_prefix}_*.hs"))
                | {h00_file.with_suffix(".hs") for h00_file in h00_files}
            )
        if not hs_files:
            raise FileNotFoundError(
                f"No projection headers (.hs) found for prefix {self.output_prefix!r} "
                f"in {self.output_dir}"
            )

        return hs_files

    def _derive_primary_headers(
        self, header_files: list[Path], window_file: Optional[Path]
    ) -> list[Path]:
        """Write pri_wN = tot_wN - sca_wN for order-0 windows."""
        self._derived_keys = set()
        if window_file is None or not window_file.is_file():
            return []
        orders = self._read_window_orders(window_file)
        names = {path.name for path in header_files}
        tot_prefix = f"{self.output_prefix}_tot_w"
        derived = []
        for tot_header in header_files:
            if not tot_header.stem.startswith(tot_prefix):
                continue
            window = tot_header.stem[len(tot_prefix) :]
            index = int(window) - 1
            if index >= len(orders) or orders[index] != 0:
                continue
            sca_header = tot_header.with_name(f"{self.output_prefix}_sca_w{window}.hs")
            pri_header = tot_header.with_name(f"{self.output_prefix}_pri_w{window}.hs")
            if pri_header.name in names or sca_header.name not in names:
                continue
            try:
                total = load_interfile_array(tot_header).array
            except Exception as exc:
                raise SimulationError(
                    f"Could not load SIMIND output {tot_header}: {exc}"
                ) from exc
            try:
                scatter = load_interfile_array(sca_header).array
            except Exception as exc:
                raise SimulationError(
                    f"Could not load SIMIND output {sca_header}: {exc}"
                ) from exc
            if total.shape != scatter.shape:
                raise SimulationError(
                    f"Could not derive primary from {tot_header} and {sca_header}: "
                    f"shape mismatch ({total.shape} vs {scatter.shape})"
                )
            pri_data = pri_header.with_suffix(".a00")
            (total - scatter).astype("<f4").tofile(pri_data)
            header = InterfileHeader.from_file(tot_header)
            header.set("!name of data file", pri_data.name)
            header.set("!number format", "float")
            header.set("!number of bytes per pixel", 4)
            header.set("imagedata byte order", "LITTLEENDIAN")
            header.set("!data offset in bytes", 0)
            header.write(pri_header)
            derived.append(pri_header)
        self._derived_keys = {self._extract_output_key(path) for path in derived}
        return derived

    def _load_projection_outputs(
        self, header_files: list[Path]
    ) -> dict[str, ProjectionResult]:
        outputs: dict[str, ProjectionResult] = {}

        for header_path in header_files:
            try:
                interfile = load_interfile_array(header_path)
            except Exception as exc:
                raise SimulationError(
                    f"Could not load SIMIND output {header_path}: {exc}"
                ) from exc

            key = self._extract_output_key(header_path)
            outputs[key] = ProjectionResult(
                projection=interfile.array,
                header_path=interfile.header_path,
                data_path=interfile.data_path,
                metadata=(
                    {**interfile.metadata, "derived": "tot - sca"}
                    if key in self._derived_keys
                    else interfile.metadata
                ),
            )

        if not outputs:
            raise RuntimeError(
                f"No valid outputs were parsed from headers in {self.output_dir}"
            )

        return outputs

    def _extract_output_key(self, header_path: Path) -> str:
        stem = header_path.stem

        component_prefix = f"{self.output_prefix}_component_"
        if stem.startswith(component_prefix):
            suffix = stem[len(component_prefix) :]
            if suffix.isdigit():
                component_id = int(suffix)
                with contextlib.suppress(ValueError):
                    return PenetrateOutputType(component_id).slug
                return f"b{component_id:02d}"

        if stem.startswith(self.output_prefix):
            stem = stem[len(self.output_prefix) :]
        return stem.lstrip("_") or header_path.stem

    @staticmethod
    def _coerce_scoring_routine(value: Any, source: str) -> ScoringRoutine:
        """Return a ScoringRoutine for a config value without rounding it."""
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"scoring_routine {value!r} from {source} is not a valid "
                "ScoringRoutine value"
            )
        if isinstance(value, numbers.Integral):
            integral = int(value)
        else:
            try:
                integral = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"scoring_routine {value!r} from {source} is not a valid "
                    "ScoringRoutine value"
                ) from exc
            # Compare against the original value so an exact non-integral
            # Decimal/Fraction is not collapsed to an int by a lossy float.
            if value != integral:
                raise ValueError(
                    f"scoring_routine {value!r} from {source} is not an integral "
                    "ScoringRoutine value"
                )
        try:
            return ScoringRoutine(integral)
        except ValueError as exc:
            raise ValueError(
                f"scoring_routine {value!r} from {source} is not a valid "
                "ScoringRoutine value"
            ) from exc

    @staticmethod
    def _validate_supported_scoring_routine(routine: ScoringRoutine) -> None:
        if routine not in _SUPPORTED_SCORING_ROUTINES:
            raise ValueError(
                f"scoring_routine {routine.name} ({routine.value}) is not "
                "supported by the Interfile connector; supported routines are "
                "DUMMY (0), SCATTWIN (1) and PENETRATE (4)."
            )

    def _config_scoring_routine(self) -> ScoringRoutine:
        try:
            value = self.config.get_value("scoring_routine")
        except IndexError:
            # A short config omits Index 84 entirely. SIMIND's serializer pads
            # it to zero, so an absent routine is effectively DUMMY.
            return ScoringRoutine.DUMMY
        routine = self._coerce_scoring_routine(value, source="the configuration")
        self._validate_supported_scoring_routine(routine)
        return routine

    def _is_penetrate_routine(self) -> bool:
        return self._config_scoring_routine() is ScoringRoutine.PENETRATE


__all__ = [
    "ProjectionResult",
    "RuntimeOperator",
    "SimindPythonConnector",
]
