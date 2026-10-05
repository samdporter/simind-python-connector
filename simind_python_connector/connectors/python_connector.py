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
from simind_python_connector.core.executor import SimindExecutor
from simind_python_connector.core.types import (
    MAX_SOURCE,
    SIMIND_VOXEL_UNIT_CONVERSION,
    PenetrateOutputType,
    ScoringRoutine,
    SimulationError,
)
from simind_python_connector.utils.interfile import (
    InterfileHeader,
    load_interfile_array,
)
from simind_python_connector.utils.simind_utils import create_window_file


ConfigSource = Union[str, os.PathLike[str], SimulationConfig]
PathLike = Union[str, os.PathLike[str]]

ZERO_HISTORY_WARNING_FRACTION = 0.01


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
        quantization_scale: float = 1.0,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self._validate_output_prefix(output_prefix)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_prefix = output_prefix
        self.quantization_scale = float(quantization_scale)
        if (
            not math.isfinite(self.quantization_scale)
            or self.quantization_scale <= 0
            or self.quantization_scale > 1
        ):
            raise ValueError(
                "quantization_scale must be in (0, 1]: the source maximum is "
                f"scaled to {MAX_SOURCE} x quantization_scale integer photon "
                "histories. Use the NN runtime switch for more statistics."
            )

        self.config = self._initialize_config(config_source)
        self.runtime_switches = RuntimeSwitches()
        self.executor = SimindExecutor()
        self.converter = SimindToStirConverter()

        self._outputs: Optional[dict[str, ProjectionResult]] = None
        self._window_file_path: Optional[Path] = None
        self._mpi: Optional[tuple[int, bool]] = None
        self._derived_keys: set[str] = set()

    @staticmethod
    def _validate_output_prefix(prefix: str) -> None:
        """Reject prefixes that could escape the output directory."""
        if (
            not isinstance(prefix, str)
            or not prefix
            or prefix in {".", ".."}
            or "/" in prefix
            or "\\" in prefix
            or Path(prefix).is_absolute()
        ):
            raise ValueError(f"output_prefix must be a plain filename, got {prefix!r}")
        if prefix != prefix.lower():
            raise ValueError(
                f"output_prefix must be lower case, got {prefix!r}: SIMIND "
                "lowercases file names on Unix, so its outputs would not be found"
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

    def configure_voxel_phantom(
        self,
        source: np.ndarray,
        mu_map: np.ndarray,
        voxel_size_mm: float = 4.0,
        scoring_routine: Union[ScoringRoutine, int] = ScoringRoutine.SCATTWIN,
    ) -> tuple[Path, Path]:
        """
        Configure voxel geometry and write source/density input files.

        Returns:
            Tuple of (source_file_path, density_file_path).
        """
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

        if isinstance(scoring_routine, ScoringRoutine):
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
                "scoring_routine must be a ScoringRoutine or int, got "
                f"{type(scoring_routine).__name__}"
            )
        dim_z, dim_y, dim_x = (int(v) for v in source_array.shape)

        cfg = self.config
        cfg.set_flag(5, True)
        cfg.set_value(15, -1)
        cfg.set_value(14, -1)
        cfg.set_flag(14, True)
        cfg.set_value(84, routine.value)

        # Source geometry
        cfg.set_value(2, dim_z * vox_cm / 2.0)
        cfg.set_value(3, dim_x * vox_cm / 2.0)
        cfg.set_value(4, dim_y * vox_cm / 2.0)
        cfg.set_value(28, vox_cm)
        # Projection rows (Index 77) run along the axis of rotation, which is
        # the slice direction of the maps (checked against SIMIND 8.0).
        cfg.set_value(76, max(dim_x, dim_y))
        cfg.set_value(77, dim_z)

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

        source_max = float(source_array.max())
        if source_max > 0:
            source_scaled = (
                source_array / source_max * (MAX_SOURCE * self.quantization_scale)
            )
        else:
            source_scaled = np.zeros_like(source_array)
        source_u16 = np.clip(np.round(source_scaled), 0, MAX_SOURCE).astype(np.uint16)

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
        density_u16.tofile(density_path)
        cfg.set_data_file(5, dns_prefix)

        return source_path, density_path

    def set_energy_windows(
        self,
        lower_bounds: Union[float, list[float]],
        upper_bounds: Union[float, list[float]],
        scatter_orders: Union[int, list[int]],
    ) -> None:
        """Write a SIMIND window file for this connector run."""
        window_path = self.output_dir / f"{self.output_prefix}.win"
        create_window_file(
            lower_bounds,
            upper_bounds,
            scatter_orders,
            output_filename=str(window_path),
        )
        self._window_file_path = window_path

    def run(
        self, runtime_operator: Optional[RuntimeOperator] = None
    ) -> dict[str, ProjectionResult]:
        """Run SIMIND and return projection outputs as NumPy arrays."""
        self._outputs = None

        mpi_kwargs = {}
        if self._mpi is not None:
            processes, split = self._mpi
            if split:
                num_projections = int(round(float(self.config.get_value(29))))
                if num_projections % processes != 0:
                    raise ValueError(
                        f"split_projections needs the number of projections "
                        f"({num_projections}) to be a multiple of the MPI "
                        f"processes ({processes})"
                    )
            mpi_kwargs = {"mpi_processes": processes, "split_projections": split}

        # Runtime-operator switches apply to this run only; merge them into
        # a throwaway switch set instead of persistent connector state.
        run_switches_holder = RuntimeSwitches()
        for key, value in self.runtime_switches.switches.items():
            run_switches_holder.set_switch(key, value)
        orbit_file = None
        if runtime_operator is not None:
            for key, value in runtime_operator.switches.items():
                run_switches_holder.set_switch(key, value)
            orbit_file = self._prepare_orbit_file(runtime_operator.orbit_file)

        # SIMIND writes air, tot and sca with /CA:1 (manual, scattwin
        # routine); primary is then tot - sca.
        if self._scoring_routine_value() == ScoringRoutine.SCATTWIN.value and (
            "CA" not in run_switches_holder.switches
        ):
            run_switches_holder.set_switch("CA", 1)

        window_file = self._window_file_for_run(run_switches_holder.switches)

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
        are protected because SIMIND still needs them on disk.
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
        for path in sorted(self.output_dir.iterdir()):
            if not path.is_file() or path.name in protected:
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

        for suffix in (".res", ".bis", ".spe", ".cor"):
            stale = self.output_dir / f"{self.output_prefix}{suffix}"
            if stale.is_file() and stale.name not in protected:
                stale.unlink()

    def _prepare_orbit_file(self, orbit_file: Optional[PathLike]) -> Optional[Path]:
        if orbit_file is None:
            return None

        orbit_path = Path(orbit_file).expanduser().resolve()
        if not orbit_path.exists():
            raise FileNotFoundError(f"Orbit file not found: {orbit_path}")

        # SIMIND lowercases file names, so pass it a lower-case copy.
        target = self.output_dir / f"{self.output_prefix}_orbit.cor"
        if orbit_path != target:
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

    def _scoring_routine_value(self) -> Optional[int]:
        try:
            return int(round(float(self.config.get_value("scoring_routine"))))
        except Exception:
            return None

    def _is_penetrate_routine(self) -> bool:
        return self._scoring_routine_value() == ScoringRoutine.PENETRATE.value


__all__ = [
    "ProjectionResult",
    "RuntimeOperator",
    "SimindPythonConnector",
]
