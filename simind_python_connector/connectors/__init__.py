"""Connector/adaptor APIs.

Backend-specific adaptors are exposed lazily so that importing this package
never imports SIRF, STIR, or PyTomography.
"""

import importlib

from .base import BaseConnector
from .python_connector import (
    NumpyConnector,
    ProjectionResult,
    RuntimeOperator,
    SimindPythonConnector,
)


_LAZY_ADAPTORS = {
    "PyTomographySimindAdaptor": "pytomography_adaptor",
    "SirfSimindAdaptor": "sirf_adaptor",
    "StirSimindAdaptor": "stir_adaptor",
}


def __getattr__(name):
    if name in _LAZY_ADAPTORS:
        module = importlib.import_module(f".{_LAZY_ADAPTORS[name]}", __name__)
        obj = getattr(module, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseConnector",
    "NumpyConnector",
    "PyTomographySimindAdaptor",
    "ProjectionResult",
    "RuntimeOperator",
    "SirfSimindAdaptor",
    "SimindPythonConnector",
    "StirSimindAdaptor",
]
