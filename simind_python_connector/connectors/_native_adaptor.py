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
from simind_python_connector.core.config import SimulationConfig
from simind_python_connector.core.types import PenetrateOutputType, ScoringRoutine
from simind_python_connector.utils import get_array


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
        self._source: Any = None
        self._mu_map: Any = None
        self._outputs: Optional[dict[str, Any]] = None

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

    def set_source(self, source: Any) -> None:
        self._source = source

    def set_mu_map(self, mu_map: Any) -> None:
        self._mu_map = mu_map

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

    def run(self, runtime_operator: Optional[RuntimeOperator] = None) -> dict[str, Any]:
        # Drop cached outputs before validation so a failed rerun can never
        # expose results from a previous successful run.
        self._outputs = None
        self._validate_inputs()

        source_arr = np.asarray(get_array(self._source), dtype=np.float32)
        mu_arr = np.asarray(get_array(self._mu_map), dtype=np.float32)
        voxel_size_mm = self._voxel_sizes_mm(self._source)[0]

        self.python_connector.configure_voxel_phantom(
            source=source_arr,
            mu_map=mu_arr,
            voxel_size_mm=voxel_size_mm,
            scoring_routine=self._scoring_routine,
        )
        raw_outputs = self.python_connector.run(runtime_operator=runtime_operator)
        self._outputs = {
            key: self._load_output(result.header_path)
            for key, result in raw_outputs.items()
        }
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
