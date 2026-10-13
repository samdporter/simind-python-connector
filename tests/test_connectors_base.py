import inspect

import pytest

from simind_python_connector.connectors import (
    BaseConnector,
    PyTomographySimindAdaptor,
    SimindPythonConnector,
    SirfSimindAdaptor,
    StirSimindAdaptor,
)
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    MAX_SOURCE,
)


@pytest.mark.unit
def test_connector_inheritance_hierarchy():
    assert issubclass(SimindPythonConnector, BaseConnector)
    assert issubclass(SirfSimindAdaptor, BaseConnector)
    assert issubclass(StirSimindAdaptor, BaseConnector)
    assert issubclass(PyTomographySimindAdaptor, BaseConnector)


@pytest.mark.unit
def test_native_adaptors_expose_quantization_scale_parameter():
    sirf_params = inspect.signature(SirfSimindAdaptor.__init__).parameters
    stir_params = inspect.signature(StirSimindAdaptor.__init__).parameters
    assert "quantization_scale" in sirf_params
    assert "quantization_scale" in stir_params


@pytest.mark.unit
def test_connector_default_quantization_scale_preserves_legacy_budget():
    assert MAX_SOURCE == 32767
    assert DEFAULT_QUANTIZATION_SCALE == pytest.approx(500 / MAX_SOURCE)
    for connector in (
        SimindPythonConnector,
        SirfSimindAdaptor,
        StirSimindAdaptor,
        PyTomographySimindAdaptor,
    ):
        parameters = inspect.signature(connector.__init__).parameters
        assert parameters["quantization_scale"].default == pytest.approx(
            DEFAULT_QUANTIZATION_SCALE
        )
