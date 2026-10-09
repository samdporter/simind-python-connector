"""Shared implementation of the SIRF and STIR adaptors."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from simind_python_connector.connectors.base import BaseConnector
from simind_python_connector.connectors.python_connector import (
    ConfigSource,
    RuntimeOperator,
    SimindPythonConnector,
)
from simind_python_connector.converters.attenuation import get_attenuation_coefficient
from simind_python_connector.core.config import SimulationConfig
from simind_python_connector.core.types import (
    PenetrateOutputType,
    ScoringRoutine,
    SimulationError,
)
from simind_python_connector.phantoms import (
    LibraryPhantom,
    VoxelPhantom,
    voxelise,
)
from simind_python_connector.utils import get_array
from simind_python_connector.utils.interfile import (
    InterfileHeader,
    ProjectionGeometry,
    check_geometry_match,
    read_header,
    read_projection_geometry,
    read_simind_density_image,
    write_in_template_geometry,
)


class _NativeSimindAdaptor(BaseConnector):
    """Adaptor taking and returning backend-native image and projection objects."""

    def __init__(
        self,
        config_source: ConfigSource,
        output_dir: str,
        output_prefix: str = "output",
        photon_multiplier: int = 1,
        quantization_scale: float = 1.0,
        scoring_routine: Union[ScoringRoutine, int] = ScoringRoutine.SCATTWIN,
        mu_map_type: str = "attenuation",
        mu_map_energy_kev: Optional[float] = None,
    ) -> None:
        self._require_backend()
        self.python_connector = SimindPythonConnector(
            config_source=config_source,
            output_dir=output_dir,
            output_prefix=output_prefix,
            quantization_scale=quantization_scale,
        )
        self._scoring_routine = (
            ScoringRoutine(scoring_routine)
            if isinstance(scoring_routine, int)
            else scoring_routine
        )
        self._mu_map_type = mu_map_type
        self._mu_map_energy_kev = mu_map_energy_kev
        self._source: Any = None
        self._mu_map: Any = None
        self._phantom: Any = None
        self._time_per_projection_s = 1.0
        self._outputs: Optional[dict[str, Any]] = None
        self._template_header: Optional[InterfileHeader] = None
        self._template_geometry: Optional[ProjectionGeometry] = None

        self.add_runtime_switch("NN", photon_multiplier)

    @abstractmethod
    def _require_backend(self) -> None:
        """Raise ImportError when the backend package is not installed."""

    @abstractmethod
    def _load_output(self, header_path: Path) -> Any:
        """Load one projection header as a native backend object."""

    @abstractmethod
    def _voxel_sizes_mm(self, image: Any) -> tuple[float, float, float]:
        """Return the image voxel sizes in mm, in (z, y, x) order."""

    @abstractmethod
    def _origin_mm(self, image: Any) -> tuple[float, float, float]:
        """Return the image origin in mm, in (z, y, x) order."""

    @abstractmethod
    def _image_like(self, template: Any, array: np.ndarray) -> Any:
        """Return a native image with template's geometry holding array (z, y, x)."""

    def _build_image(self, array: np.ndarray, voxel_sizes_mm: tuple) -> Any:
        """Return a native image on its own grid, with voxel sizes in (z, y, x)."""
        from simind_python_connector.builders import STIRSPECTImageDataBuilder

        nz, ny, nx = array.shape
        builder = STIRSPECTImageDataBuilder(
            {
                "!matrix size [1]": str(nx),
                "!matrix size [2]": str(ny),
                "!matrix size [3]": str(nz),
                "scaling factor (mm/pixel) [1]": str(voxel_sizes_mm[2]),
                "scaling factor (mm/pixel) [2]": str(voxel_sizes_mm[1]),
                "scaling factor (mm/pixel) [3]": str(voxel_sizes_mm[0]),
            },
            backend=self._backend,
        )
        builder.set_pixel_array(array.astype(np.float32))
        return builder.build()

    def set_source(self, source: Any) -> None:
        self._source = source

    def set_mu_map(self, mu_map: Any) -> None:
        self._mu_map = mu_map

    def set_phantom(self, phantom: Any, *, time_per_projection_s: float = 1.0) -> None:
        """Simulate an AnalyticPhantom, LibraryPhantom or VoxelPhantom.

        An alternative to set_source/set_mu_map; time_per_projection_s is
        used for VoxelPhantom (known activity, Index 25).
        The scoring routine comes from the configuration (Index 84) for
        analytic and library phantoms.
        """
        self._phantom = phantom
        self._time_per_projection_s = time_per_projection_s
        self._outputs = None

    def get_ground_truth(self, image_template: Any) -> tuple[Any, Any]:
        """Return (activity, mu) native images; mu in cm^-1 at abs(Index 1).

        VoxelPhantom and AnalyticPhantom give images on image_template's grid.
        A LibraryPhantom gives (None, mu) on SIMIND's own density grid, after
        run(); its activity ground truth is not available.
        """
        if self._phantom is None:
            raise RuntimeError(
                "get_ground_truth needs a phantom set with set_phantom()"
            )
        energy = abs(float(self.get_config().get_value("photon_energy")))
        mu_water = float(get_attenuation_coefficient("water", energy))

        if isinstance(self._phantom, LibraryPhantom):
            if self._outputs is None:
                raise RuntimeError(
                    "Run the adaptor first: library phantom densities come from "
                    "SIMIND's aligned density output"
                )
            connector = self.python_connector
            header = connector.output_dir / f"{connector.output_prefix}.hct"
            density, voxel_sizes = read_simind_density_image(header)
            return None, self._build_image(density * mu_water, voxel_sizes)

        shape = tuple(np.asarray(get_array(image_template)).shape)
        voxel_sizes = self._voxel_sizes_mm(image_template)
        if isinstance(self._phantom, VoxelPhantom):
            if shape != self._phantom.activity_mbq.shape or not np.allclose(
                voxel_sizes, self._phantom.voxel_size_mm, atol=1e-3
            ):
                raise ValueError(
                    f"The phantom grid {self._phantom.activity_mbq.shape} at "
                    f"{self._phantom.voxel_size_mm} mm differs from the template "
                    f"grid {shape} at {voxel_sizes} mm"
                )
            activity = self._phantom.activity_mbq
            mu = self._phantom.density_g_cm3 * mu_water
        else:
            activity, attenuator = voxelise(self._phantom, shape, voxel_sizes)
            mu = np.zeros(shape) if attenuator is None else attenuator * mu_water
        return (
            self._image_like(image_template, np.asarray(activity, dtype=np.float32)),
            self._image_like(image_template, np.asarray(mu, dtype=np.float32)),
        )

    def set_template(self, template: Any) -> None:
        """Simulate in the geometry of a measured acquisition.

        template is a backend projection object or a path to its .hs header.
        Outputs are then returned in exactly that geometry.
        """
        # Both are parsed before either is stored, so a rejected replacement
        # leaves the previous template usable.
        header = read_header(template)
        geometry = read_projection_geometry(header)
        self._template_header = header
        self._template_geometry = geometry

    def set_energy_windows(
        self,
        lower_bounds: Union[float, list[float]],
        upper_bounds: Union[float, list[float]],
        scatter_orders: Union[int, list[int]],
    ) -> None:
        self.python_connector.set_energy_windows(
            lower_bounds, upper_bounds, scatter_orders
        )

    def add_config_value(self, index: int, value: Any) -> None:
        self.python_connector.add_config_value(index, value)

    def add_runtime_switch(self, switch: str, value: Any) -> None:
        self.python_connector.add_runtime_switch(switch, value)

    def set_mpi(
        self, processes: Optional[int], split_projections: bool = False
    ) -> None:
        self.python_connector.set_mpi(processes, split_projections)

    def set_activity(
        self, activity_mbq: float, time_per_projection_s: float = 1.0
    ) -> None:
        self.python_connector.set_activity(activity_mbq, time_per_projection_s)

    def run(self, runtime_operator: Optional[RuntimeOperator] = None) -> dict[str, Any]:
        # Drop cached outputs before validation so a failed rerun can never
        # expose results from a previous successful run.
        self._outputs = None
        if self._phantom is not None:
            if self._source is not None or self._mu_map is not None:
                raise ValueError(
                    "Set either a phantom (set_phantom) or a source and mu_map, "
                    "not both"
                )
            if self._template_geometry is not None:
                self.python_connector.configure_acquisition(self._template_geometry)
            self.python_connector.configure_phantom(
                self._phantom, time_per_projection_s=self._time_per_projection_s
            )
        else:
            self._validate_inputs()
            source_arr = np.asarray(get_array(self._source), dtype=np.float32)
            mu_arr = np.asarray(get_array(self._mu_map), dtype=np.float32)
            if self._template_geometry is not None:
                self.python_connector.configure_acquisition(self._template_geometry)
            self.python_connector.configure_voxel_phantom(
                source=source_arr,
                mu_map=mu_arr,
                voxel_size_mm=self._voxel_sizes_mm(self._source),
                scoring_routine=self._scoring_routine,
                mu_map_type=self._mu_map_type,
                mu_map_energy_kev=self._mu_map_energy_kev,
            )
        raw_outputs = self.python_connector.run(runtime_operator=runtime_operator)
        if self._template_geometry is None:
            self._outputs = {
                key: self._load_output(result.header_path)
                for key, result in raw_outputs.items()
            }
            return self._outputs

        first = next(iter(raw_outputs.values()))
        produced = read_projection_geometry(
            InterfileHeader.from_file(first.header_path)
        )
        differences = check_geometry_match(produced, self._template_geometry)
        if differences:
            raise SimulationError(
                "SIMIND output geometry differs from the template: "
                + "; ".join(differences)
            )
        prefix = self.python_connector.output_prefix
        outputs = {}
        for key, result in raw_outputs.items():
            path = result.header_path.with_name(f"{prefix}_{key}_tmpl.hs")
            write_in_template_geometry(result.projection, self._template_header, path)
            outputs[key] = self._load_output(path)
        self._outputs = outputs
        return self._outputs

    def get_outputs(self) -> Dict[str, Any]:
        if self._outputs is None:
            raise RuntimeError("Run the adaptor first to produce outputs.")
        return self._outputs

    def get_total_output(self, window: int = 1) -> Any:
        return self._get_component("tot", window)

    def get_scatter_output(self, window: int = 1) -> Any:
        return self._get_component("sca", window)

    def get_primary_output(self, window: int = 1) -> Any:
        return self._get_component("pri", window)

    def get_air_output(self, window: int = 1) -> Any:
        return self._get_component("air", window)

    def get_penetrate_output(self, component: Union[PenetrateOutputType, str]) -> Any:
        outputs = self.get_outputs()
        key = (
            component.slug if isinstance(component, PenetrateOutputType) else component
        )
        if key not in outputs:
            available = ", ".join(sorted(outputs))
            raise KeyError(f"Output {key!r} not available. Available: {available}")
        return outputs[key]

    def list_available_outputs(self) -> list[str]:
        return sorted(self.get_outputs().keys())

    def get_scoring_routine(self) -> ScoringRoutine:
        return self._scoring_routine

    def get_config(self) -> SimulationConfig:
        return self.python_connector.get_config()

    def _get_component(self, prefix: str, window: int) -> Any:
        outputs = self.get_outputs()
        key = f"{prefix}_w{window}"
        if key not in outputs:
            available = ", ".join(sorted(outputs))
            raise KeyError(f"Output {key!r} not available. Available: {available}")
        return outputs[key]

    def _validate_inputs(self) -> None:
        if self._source is None or self._mu_map is None:
            raise ValueError("Both source and mu_map must be set before run().")

        source_shape = np.asarray(get_array(self._source)).shape
        mu_shape = np.asarray(get_array(self._mu_map)).shape
        if source_shape != mu_shape:
            raise ValueError(
                f"source and mu_map must have matching shapes, got "
                f"{source_shape} and {mu_shape}"
            )

        source_sizes = np.asarray(self._voxel_sizes_mm(self._source), dtype=float)
        mu_sizes = np.asarray(self._voxel_sizes_mm(self._mu_map), dtype=float)
        if not np.allclose(source_sizes, mu_sizes, atol=1e-3):
            raise ValueError(
                f"source and mu_map voxel sizes differ: {tuple(source_sizes)} "
                f"vs {tuple(mu_sizes)} mm"
            )

        source_origin = np.asarray(self._origin_mm(self._source), dtype=float)
        mu_origin = np.asarray(self._origin_mm(self._mu_map), dtype=float)
        # rtol=0 makes atol=1e-3 mm the whole tolerance, like the size checks.
        if not np.allclose(source_origin, mu_origin, atol=1e-3, rtol=0):
            raise ValueError(
                f"source and mu_map origins differ: {tuple(source_origin)} vs "
                f"{tuple(mu_origin)} mm"
            )
