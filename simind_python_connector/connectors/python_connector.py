"""
Backend-agnostic SIMIND connector returning NumPy projection outputs.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from simind_python_connector.connectors.base import BaseConnector
from simind_python_connector.converters.attenuation import attenuation_to_density
from simind_python_connector.converters.simind_to_stir import SimindToStirConverter
from simind_python_connector.core.config import RuntimeSwitches, SimulationConfig
from simind_python_connector.core.executor import (
    SimindExecutor,
    _validate_output_prefix,
)
from simind_python_connector.core.types import (
    MAX_SOURCE,
    SIMIND_VOXEL_UNIT_CONVERSION,
    OutputError,
    PenetrateOutputType,
    ScoringRoutine,
)
from simind_python_connector.utils.interfile_numpy import load_interfile_array
from simind_python_connector.utils.simind_utils import create_window_file


ConfigSource = Union[str, os.PathLike[str], SimulationConfig]
PathLike = Union[str, os.PathLike[str]]

# The Interfile connector can only read the projection outputs of these
# routines; LIST_MODE and FORCED_COLLIMATION write formats it cannot consume.
_SUPPORTED_SCORING_ROUTINES = (
    ScoringRoutine.DUMMY,
    ScoringRoutine.SCATTWIN,
    ScoringRoutine.PENETRATE,
)


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
    timeout: Optional[float] = None


class SimindPythonConnector(BaseConnector):
    """Pure Python connector for SIMIND with NumPy-first outputs."""

    def __init__(
        self,
        config_source: ConfigSource,
        output_dir: PathLike,
        output_prefix: str = "output",
        quantization_scale: float = 1.0,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self._require_valid_prefix(output_prefix)
        try:
            self.output_dir = Path(output_dir).expanduser().resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OutputError(
                f"Could not create output directory {output_dir!r}: {exc}"
            ) from exc
        self.output_prefix = output_prefix
        try:
            self.quantization_scale = float(quantization_scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"quantization_scale must be in (0, 1], got {quantization_scale!r}"
            ) from exc
        # Above 1 the source would be clipped at MAX_SOURCE, changing its shape.
        if not 0 < self.quantization_scale <= 1:
            raise ValueError(
                f"quantization_scale must be in (0, 1], got {quantization_scale!r}"
            )

        self.config = self._initialize_config(config_source)
        self.runtime_switches = RuntimeSwitches()
        self.executor = SimindExecutor()
        self.converter = SimindToStirConverter()

        self._outputs: Optional[dict[str, ProjectionResult]] = None
        self._window_file_path: Optional[Path] = None
        self._phantom_extent_mm: Optional[tuple[float, float]] = None
        self._orbit_source_paths: tuple[Path, ...] = ()

    @staticmethod
    def _require_valid_prefix(prefix: str) -> None:
        """Accept only prefixes SIMIND keeps unchanged in its file names.

        SIMIND lowercases file names on Unix, a dot would be taken for a file
        extension, and glob characters would break output discovery. A hyphen
        is rejected too: SIMIND V8 treats it as an argument delimiter and
        splits its positional filename arguments, so the run would open the
        wrong file and silently produce nothing. The length cap keeps the
        engine's output names inside its fixed 60-character buffer.
        """
        _validate_output_prefix(prefix)

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

    def configure_voxel_phantom(
        self,
        source: np.ndarray,
        mu_map: np.ndarray,
        voxel_size_mm: float = 4.0,
        scoring_routine: Optional[Union[ScoringRoutine, int]] = None,
    ) -> tuple[Path, Path]:
        """
        Configure voxel geometry and write source/density input files.

        The projection pixel size and matrix (Index 28, 76, 77) come from the
        configuration, not from the phantom. scoring_routine overrides Index 84;
        None keeps the configuration value. The connector reads SPECT Interfile
        output, so this sets Flag 5 (SPECT) and Flag 14 (Interfile headers).
        The mu map is taken to be at abs(Index 1) keV.

        Returns:
            Tuple of (source_file_path, density_file_path).
        """
        self._require_output_dir()
        self._require_valid_prefix(self.output_prefix)
        source_array = np.asarray(source, dtype=np.float32)
        mu_map_array = np.asarray(mu_map, dtype=np.float32)

        if source_array.ndim != 3 or mu_map_array.ndim != 3:
            raise ValueError("source and mu_map must both be 3D arrays")
        if source_array.shape != mu_map_array.shape:
            raise ValueError("source and mu_map must have identical shapes")

        vox_cm = float(voxel_size_mm) / SIMIND_VOXEL_UNIT_CONVERSION
        if not math.isfinite(vox_cm) or vox_cm <= 0:
            raise ValueError("voxel_size_mm must be > 0")
        if source_array.size == 0 or mu_map_array.size == 0:
            raise ValueError("source and mu_map must not be empty")
        if not np.isfinite(source_array).all() or not np.isfinite(mu_map_array).all():
            raise ValueError("source and mu_map must contain only finite values")
        if (source_array < 0).any() or (mu_map_array < 0).any():
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

        cfg = self.config
        cfg.set_flag(5, True)
        cfg.set_value(15, -1)
        cfg.set_value(14, -1)
        cfg.set_flag(14, True)
        if scoring_routine is not None:
            cfg.set_value(84, routine.value)

        # Source geometry
        cfg.set_value(2, dim_z * vox_cm / 2.0)
        cfg.set_value(3, dim_x * vox_cm / 2.0)
        cfg.set_value(4, dim_y * vox_cm / 2.0)

        # Density geometry
        cfg.set_value(5, dim_z * vox_cm / 2.0)
        cfg.set_value(6, dim_x * vox_cm / 2.0)
        cfg.set_value(7, dim_y * vox_cm / 2.0)
        cfg.set_value(31, vox_cm)
        cfg.set_value(33, 1)
        cfg.set_value(34, dim_z)
        cfg.set_value(78, dim_x)  # density map i
        cfg.set_value(79, dim_x)  # source map i
        cfg.set_value(81, dim_y)  # density map j
        cfg.set_value(82, dim_y)  # source map j

        self.runtime_switches.set_switch("PX", vox_cm)
        voxel_mm = float(voxel_size_mm)
        self._phantom_extent_mm = (max(dim_x, dim_y) * voxel_mm, dim_z * voxel_mm)

        src_prefix = f"{self.output_prefix}_src"
        source_path = self.output_dir / f"{src_prefix}.smi"
        self._write_array_file(source_u16, source_path, "source map")
        cfg.set_data_file(6, src_prefix)

        if cfg.get_flag(11):
            # A negative Index 1 asks SIMIND to read photon energies from the
            # isotope file; its absolute value is still the window energy.
            photon_energy = abs(float(cfg.get_value("photon_energy")))
            density = attenuation_to_density(mu_map_array, photon_energy) * 1000.0
        else:
            density = np.zeros_like(mu_map_array)

        density_u16 = np.clip(np.round(density), 0, np.iinfo(np.uint16).max).astype(
            np.uint16
        )
        dns_prefix = f"{self.output_prefix}_dns"
        density_path = self.output_dir / f"{dns_prefix}.dmi"
        self._write_array_file(density_u16, density_path, "density map")
        cfg.set_data_file(5, dns_prefix)

        return source_path, density_path

    def set_energy_windows(
        self,
        lower_bounds: Union[float, list[float]],
        upper_bounds: Union[float, list[float]],
        scatter_orders: Union[int, list[int]],
    ) -> None:
        """Write a SIMIND window file for this connector run."""
        self._require_output_dir()
        self._require_valid_prefix(self.output_prefix)
        window_path = self.output_dir / f"{self.output_prefix}.win"
        try:
            create_window_file(
                lower_bounds,
                upper_bounds,
                scatter_orders,
                output_filename=str(window_path),
            )
        except OSError as exc:
            raise OutputError(
                f"Failed to write energy-window file {window_path}: {exc}"
            ) from exc
        self._window_file_path = window_path

    def _validate_runtime_operator(
        self, runtime_operator: Optional[RuntimeOperator]
    ) -> None:
        """Validate a runtime operator without staging or writing any files.

        Applies the same switch, timeout and orbit checks run() performs, so a
        caller can reject an invalid operator before phantom inputs are
        overwritten.
        """
        if runtime_operator is None:
            return

        SimindExecutor.validate_timeout(runtime_operator.timeout)

        validation_switches = RuntimeSwitches()
        for key, value in runtime_operator.switches.items():
            validation_switches.set_switch(key, value)

        orbit_file = runtime_operator.orbit_file
        if orbit_file is None:
            return
        orbit_path = Path(orbit_file).expanduser().resolve()
        if not orbit_path.exists():
            raise FileNotFoundError(f"Orbit file not found: {orbit_path}")
        if not orbit_path.is_file():
            raise ValueError(f"orbit_file must be a regular file: {orbit_path}")

    def run(
        self, runtime_operator: Optional[RuntimeOperator] = None
    ) -> dict[str, ProjectionResult]:
        """Run SIMIND and return projection outputs as NumPy arrays."""
        self._outputs = None
        self._orbit_source_paths = ()

        # Fail before staging, config save or cleanup when the run cannot be
        # supported, the output directory is no longer there or the prefix has
        # grown past the engine's buffer-safe length.
        self._require_output_dir()
        self._require_valid_prefix(self.output_prefix)
        self._config_scoring_routine()

        # Runtime-operator switches apply to this run only; merge them into
        # a throwaway switch set instead of persistent connector state.
        run_switches_holder = RuntimeSwitches()
        for key, value in self.runtime_switches.switches.items():
            run_switches_holder.set_switch(key, value)
        orbit_file = None
        run_timeout: Optional[float] = None
        if runtime_operator is not None:
            # Validate before staging the orbit or clearing prior outputs so an
            # invalid timeout cannot disturb existing files.
            run_timeout = SimindExecutor.validate_timeout(runtime_operator.timeout)
            for key, value in runtime_operator.switches.items():
                run_switches_holder.set_switch(key, value)
            orbit_file = self._prepare_orbit_file(runtime_operator.orbit_file)

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

        # Outputs are read from SPECT Interfile headers (.h00), which SIMIND
        # writes only with Flag 5 (SPECT) and Flag 14 (Interfile) set.
        self.config.set_flag(5, True)
        self.config.set_flag(14, True)
        config_path = self.output_dir / self.output_prefix
        try:
            self.config.save_file(config_path)
        except OSError as exc:
            raise OutputError(
                f"Failed to write SIMIND configuration "
                f"{config_path.with_suffix('.smc')}: {exc}"
            ) from exc

        self._clear_previous_outputs()
        self.executor.run_simulation(
            self.output_prefix,
            orbit_file,
            run_switches_holder.switches,
            cwd=self.output_dir,
            timeout=run_timeout,
        )

        header_files = self._ensure_interfile_headers()
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

    def _clear_previous_outputs(self) -> None:
        """Delete stale outputs from earlier runs sharing this prefix.

        Connector-written inputs (``{prefix}_src.smi`` / ``{prefix}_dns.dmi``)
        are protected because SIMIND still needs them on disk. A caller-supplied
        orbit source and any symlink or hardlink alias of it are protected by
        file identity.
        """
        protected = {
            f"{self.output_prefix}_src.smi",
            f"{self.output_prefix}_dns.dmi",
        }
        if (
            self._window_file_path is not None
            and self._window_file_path.parent == self.output_dir
        ):
            # Energy-window input written via set_energy_windows(); protect
            # only that exact file, never a pre-existing one at the current
            # prefix/location.
            protected.add(self._window_file_path.name)
        for slot in (5, 6):
            try:
                protected.add(self.config.get_data_file(slot))
            except KeyError:
                pass

        # A caller-supplied orbit may live in output_dir and match the output
        # patterns; never delete it (or an alias of it) out from under SIMIND.
        # Compare directory entries by lstat so a differently-cased spelling on
        # a case-insensitive filesystem is still recognised, and dereferenced
        # files by stat so a symlink or hardlink alias is protected too.
        orbit_entries: list[os.stat_result] = []
        orbit_referents: list[os.stat_result] = []
        for orbit_source in self._orbit_source_paths:
            if orbit_source.parent.resolve() == self.output_dir:
                with contextlib.suppress(OSError):
                    orbit_entries.append(os.lstat(self.output_dir / orbit_source.name))
            with contextlib.suppress(OSError):
                orbit_referents.append(os.stat(orbit_source.resolve()))

        try:
            output_entries = sorted(self.output_dir.iterdir())
        except OSError as exc:
            raise OutputError(
                f"Failed to list output directory {self.output_dir} for stale "
                f"output cleanup: {exc}"
            ) from exc

        for path in output_entries:
            if not path.is_file() or path.name in protected:
                continue
            if any(os.path.samestat(os.lstat(path), entry) for entry in orbit_entries):
                continue
            if any(
                os.path.samestat(os.stat(path), referent)
                for referent in orbit_referents
            ):
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
                try:
                    path.unlink()
                except OSError as exc:
                    raise OutputError(
                        f"Failed to remove stale output {path}: {exc}"
                    ) from exc

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

        # Stage under a connector-owned name so a user-supplied orbit never
        # clobbers a protected input (e.g. {prefix}_src.smi) that happens to
        # share its basename.
        target = self.output_dir / f"{self.output_prefix}_orbit.cor"
        if orbit_path == target:
            return target

        # The supplied path may itself be the canonical destination, possibly
        # as a symlink to somewhere else. Compare its location without following
        # the final link and leave the caller's path alone in that case.
        if supplied_path.parent.resolve() / supplied_path.name == target:
            return target

        # Drop any pre-existing alias (a symlink or hardlink to the orbit
        # source) so copy2 writes a real file instead of following the link.
        if target.is_symlink() or target.exists():
            # On a case-insensitive filesystem a differently-cased spelling is
            # the same directory entry. Compare lstat identities (not the
            # dereferenced file) so a destination symlink to the source is
            # still replaced, but the caller's source is never unlinked. The
            # resolved referent is checked too: an external symlink may point
            # at a differently-cased spelling of the canonical target.
            same_entry = False
            for candidate in (supplied_path, orbit_path):
                try:
                    if os.path.samestat(os.lstat(target), os.lstat(candidate)):
                        same_entry = True
                        break
                except OSError:
                    continue
            if same_entry:
                return target
            try:
                target.unlink()
            except OSError as exc:
                raise OutputError(
                    f"Failed to remove stale orbit staging file {target}: {exc}"
                ) from exc
        try:
            shutil.copy2(orbit_path, target)
        except OSError as exc:
            raise OutputError(
                f"Failed to stage orbit file {orbit_path} to {target}: {exc}"
            ) from exc
        return target

    def _ensure_interfile_headers(self) -> list[Path]:
        if self._is_penetrate_routine():
            try:
                h00_file = self.converter.find_penetrate_h00_file(
                    self.output_prefix, str(self.output_dir)
                )
            except OSError as exc:
                raise OutputError(
                    f"Failed to search for PENETRATE output for prefix "
                    f"{self.output_prefix!r} in {self.output_dir}: {exc}"
                ) from exc
            if h00_file is None:
                raise OutputError(
                    f"No PENETRATE .h00 output found for prefix "
                    f"{self.output_prefix!r} in {self.output_dir}; a PENETRATE "
                    "run should produce a .h00 template and .bNN binaries"
                )
            try:
                self.converter.create_penetrate_headers_from_template(
                    h00_file, self.output_prefix, str(self.output_dir)
                )
            except OSError as exc:
                raise OutputError(
                    f"Failed to convert PENETRATE output {h00_file} in "
                    f"{self.output_dir}: {exc}"
                ) from exc
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
                try:
                    self.converter.convert_file(str(h00_file), str(hs_file))
                except OSError as exc:
                    raise OutputError(
                        f"Failed to convert SIMIND header {h00_file} to "
                        f"{hs_file}: {exc}"
                    ) from exc

            hs_files = sorted(
                set(self.output_dir.glob(f"{self.output_prefix}_*.hs"))
                | {h00_file.with_suffix(".hs") for h00_file in h00_files}
            )
        if not hs_files:
            raise OutputError(
                f"No SPECT projection headers (.hs) found for prefix "
                f"{self.output_prefix!r} in {self.output_dir}; expected the "
                "SIMIND run to write readable SPECT Interfile output"
            )

        return hs_files

    def _load_projection_outputs(
        self, header_files: list[Path]
    ) -> dict[str, ProjectionResult]:
        outputs: dict[str, ProjectionResult] = {}

        for header_path in header_files:
            try:
                interfile = load_interfile_array(header_path)
            except Exception as exc:
                raise RuntimeError(
                    f"SIMIND output {header_path} could not be read: {exc}"
                ) from exc

            key = self._extract_output_key(header_path)
            outputs[key] = ProjectionResult(
                projection=interfile.array,
                header_path=interfile.header_path,
                data_path=interfile.data_path,
                metadata=interfile.metadata,
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

    def _require_output_dir(self) -> None:
        if not self.output_dir.is_dir():
            raise OutputError(
                f"Output directory {self.output_dir} does not exist or is not a "
                "directory; it must be present before writing inputs or running"
            )

    @staticmethod
    def _write_array_file(array: np.ndarray, path: Path, description: str) -> None:
        try:
            array.tofile(path)
        except OSError as exc:
            raise OutputError(
                f"Failed to write {description} to {path}: {exc}"
            ) from exc

    @staticmethod
    def _coerce_scoring_routine(value: Any, source: str) -> ScoringRoutine:
        """Return a ScoringRoutine for a config value without rounding it."""
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"scoring_routine {value!r} from {source} is not a valid "
                "ScoringRoutine value"
            ) from exc
        if not math.isfinite(numeric) or numeric != int(numeric):
            raise ValueError(
                f"scoring_routine {value!r} from {source} is not an integral "
                "ScoringRoutine value"
            )
        try:
            return ScoringRoutine(int(numeric))
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


NumpyConnector = SimindPythonConnector


__all__ = [
    "NumpyConnector",
    "ProjectionResult",
    "RuntimeOperator",
    "SimindPythonConnector",
]
