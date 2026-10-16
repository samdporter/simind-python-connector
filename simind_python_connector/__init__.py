"""
simind-python-connector connector/adaptor API.
"""

import importlib
from typing import Any

from ._version import __version__ as __version__


def __getattr__(name: str) -> Any:
    if name in {
        "builders",
        "configs",
        "connectors",
        "converters",
        "core",
        "data",
        "utils",
    }:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    elif name in {"SimulationConfig"}:
        core = importlib.import_module(".core", __name__)
        obj = getattr(core, name)
        globals()[name] = obj
        return obj
    elif name in {
        "BaseConnector",
        "NumpyConnector",
        "PyTomographySimindAdaptor",
        "ProjectionResult",
        "RuntimeOperator",
        "SirfSimindAdaptor",
        "SimindPythonConnector",
        "StirSimindAdaptor",
    }:
        connectors = importlib.import_module(".connectors", __name__)
        obj = getattr(connectors, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseConnector",
    "NumpyConnector",
    "ProjectionResult",
    "PyTomographySimindAdaptor",
    "RuntimeOperator",
    "SimindPythonConnector",
    "SimulationConfig",
    "SirfSimindAdaptor",
    "StirSimindAdaptor",
    "builders",
    "configs",
    "connectors",
    "converters",
    "core",
    "data",
    "utils",
]
