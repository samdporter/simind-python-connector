"""Abstract connector contract shared by all connector/adaptor implementations."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from simind_python_connector.core.config import SimulationConfig


class BaseConnector(ABC):
    """Minimal interface for all SIMIND connector styles."""

    @abstractmethod
    def add_config_value(self, index: int, value: Any) -> None:
        """Set a SIMIND config value by index."""

    @abstractmethod
    def add_runtime_switch(self, switch: str, value: Any) -> None:
        """Set one runtime switch."""

    def set_runtime_switches(self, switches: Mapping[str, Any]) -> None:
        """Set multiple runtime switches."""
        for switch, value in switches.items():
            self.add_runtime_switch(switch, value)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute simulation and return connector-native outputs."""

    @abstractmethod
    def get_outputs(self) -> dict[str, Any]:
        """Return outputs from the latest successful run."""

    @abstractmethod
    def get_config(self) -> SimulationConfig:
        """Return active simulation configuration."""

    def _get_window_output(self, prefix: str, complement: str, window: int) -> Any:
        """Return ``{prefix}_w{window}``, deriving it from the total if needed.

        SCATTWIN totals satisfy ``tot = pri + sca``, so a missing scatter or
        primary projection is the total minus the other one. When neither is
        available this warns and returns None: SIMIND writes them only when the
        CA runtime switch is set (1 for sca and air, 2 for pri and air).
        """
        outputs = self.get_outputs()
        key = f"{prefix}_w{window}"
        if key in outputs:
            return outputs[key]
        total_key = f"tot_w{window}"
        if total_key not in outputs:
            available = ", ".join(sorted(outputs))
            raise KeyError(f"Output {key!r} not available. Available: {available}")
        complement_key = f"{complement}_w{window}"
        if complement_key in outputs:
            return self._subtract_outputs(outputs[total_key], outputs[complement_key])
        warnings.warn(
            f"Output {key!r} is not available, returning None. Set "
            "add_runtime_switch('CA', 1) or add_runtime_switch('CA', 2) so "
            "SIMIND writes scatter or primary projections.",
            stacklevel=3,
        )
        return None

    def _subtract_outputs(self, minuend: Any, subtrahend: Any) -> Any:
        """Return ``minuend - subtrahend`` as a backend-native output."""
        return minuend - subtrahend


__all__ = ["BaseConnector"]
