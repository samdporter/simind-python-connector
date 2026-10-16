"""
Tests for the backend abstraction layer.

These tests verify that the backend system correctly detects and switches
between SIRF and STIR Python backends.
"""

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.unit
def test_invalid_backend():
    from simind_python_connector.backends import set_backend

    with pytest.raises(ValueError, match="Invalid backend"):
        set_backend("invalid")


@pytest.mark.unit
@pytest.mark.parametrize("method_name", ["as_array", "asarray"])
def test_get_array_uses_object_conversion(method_name):
    from simind_python_connector.utils import get_array

    class MockImage:
        pass

    setattr(MockImage, method_name, lambda self: np.array([1, 2, 3]))

    obj = MockImage()
    arr = get_array(obj)
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [1, 2, 3]


@pytest.mark.unit
def test_get_array_preserves_conversion_errors():
    from simind_python_connector.utils import get_array

    class UnreadableImage:
        def asarray(self):
            raise RuntimeError("Image data could not be read")

    with pytest.raises(RuntimeError, match="Image data could not be read"):
        get_array(UnreadableImage())


@pytest.mark.unit
def test_load_functions_invalid_backend():
    from simind_python_connector.backends import (
        load_acquisition_data,
        load_image_data,
    )

    with pytest.raises(ValueError, match="Invalid backend"):
        load_image_data("phantom.hv", backend="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid backend"):
        load_acquisition_data("scan.hs", backend="invalid")  # type: ignore[arg-type]


@pytest.mark.unit
def test_unwrap_function():
    from simind_python_connector.backends import unwrap

    # Test with mock object
    class MockWrapper:
        @property
        def native_object(self):
            return "native"

    wrapper = MockWrapper()
    assert unwrap(wrapper) == "native"

    # Test with non-wrapper (should return as-is)
    obj = "regular_object"
    assert unwrap(obj) == "regular_object"


@pytest.mark.unit
def test_detect_image_backend_with_mock(monkeypatch):
    """detect_image_backend should recognise mocked SIRF/STIR types."""
    import simind_python_connector.backends as backends

    class FakeSirfImage:
        pass

    class FakeStirImage:
        pass

    monkeypatch.setattr(backends, "_import_sirf_types", lambda: (FakeSirfImage, None))
    monkeypatch.setattr(backends, "_import_stir_types", lambda: (FakeStirImage, None))

    assert backends.detect_image_backend(FakeSirfImage()) == "sirf"
    assert backends.detect_image_backend(FakeStirImage()) == "stir"
    assert backends.detect_image_backend(object()) is None


@pytest.mark.unit
def test_detect_acquisition_backend_with_mock(monkeypatch):
    """detect_acquisition_backend should recognise mocked native classes."""
    import simind_python_connector.backends as backends

    class FakeSirfAcq:
        pass

    class FakeStirProj:
        pass

    monkeypatch.setattr(backends, "_import_sirf_types", lambda: (None, FakeSirfAcq))
    monkeypatch.setattr(backends, "_import_stir_types", lambda: (None, FakeStirProj))

    assert backends.detect_acquisition_backend(FakeSirfAcq()) == "sirf"
    assert backends.detect_acquisition_backend(FakeStirProj()) == "stir"
    assert backends.detect_acquisition_backend(object()) is None


@pytest.mark.unit
def test_detect_backend_from_interface(monkeypatch):
    """detect_backend_from_interface should unwrap and reuse detection helpers."""
    import simind_python_connector.backends as backends

    class FakeSirfAcq:
        pass

    class MockWrapper:
        def __init__(self, native):
            self._native = native

        @property
        def native_object(self):
            return self._native

    monkeypatch.setattr(backends, "_import_sirf_types", lambda: (None, FakeSirfAcq))
    monkeypatch.setattr(backends, "_import_stir_types", lambda: (None, None))

    wrapper = MockWrapper(FakeSirfAcq())
    assert backends.detect_backend_from_interface(wrapper) == "sirf"


def _install_fake_stir(monkeypatch, modern=True):
    """Install a complete fake 'stir' backend for every lookup path.

    ``modern=True`` fakes the ``stir.extra`` layout; ``modern=False`` fakes the
    legacy standalone ``stirextra`` module.
    """
    import sys
    import types

    import simind_python_connector.backends as backends
    import simind_python_connector.backends.stir_backend as stir_backend

    class _FakeVoxels:
        @staticmethod
        def read_from_file(filepath):
            return ("image", filepath)

    class _FakeProjData:
        @staticmethod
        def read_from_file(filepath):
            return ("acquisition", filepath)

    class _FakeProjDataInMemory:
        def __init__(self, obj):
            self.obj = obj

    fake_stir = types.ModuleType("stir")
    fake_stir.FloatVoxelsOnCartesianGrid = _FakeVoxels
    fake_stir.ProjData = _FakeProjData
    fake_stir.ProjDataInMemory = _FakeProjDataInMemory
    fake_extra = types.ModuleType("stir.extra" if modern else "stirextra")

    monkeypatch.setitem(sys.modules, "stir", fake_stir)
    if modern:
        monkeypatch.setitem(sys.modules, "stir.extra", fake_extra)
        monkeypatch.delitem(sys.modules, "stirextra", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "stirextra", fake_extra)
        monkeypatch.delitem(sys.modules, "stir.extra", raising=False)

    # The backend factory resolves modules through sys.modules, while the
    # cached wrapper holds direct references; patch both.
    monkeypatch.setattr(stir_backend, "stir", fake_stir, raising=False)
    monkeypatch.setattr(stir_backend, "STIR_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        stir_backend, "_import_stir_extra", lambda: fake_extra, raising=False
    )
    monkeypatch.setattr(backends, "_import_stir_extra", lambda: fake_extra)


@pytest.mark.unit
def test_create_image_data_accepts_path_objects(monkeypatch):
    """os.PathLike inputs are normalised before backend dispatch."""
    from simind_python_connector.backends import (
        create_image_data,
        reset_backend,
        set_backend,
    )

    _install_fake_stir(monkeypatch)

    reset_backend()
    try:
        set_backend("stir")
        wrapped = create_image_data(Path("/tmp/phantom.hv"))
        assert wrapped.native_object == ("image", "/tmp/phantom.hv")
    finally:
        reset_backend()


@pytest.mark.unit
def test_create_acquisition_data_accepts_path_objects(monkeypatch):
    from simind_python_connector.backends import (
        create_acquisition_data,
        reset_backend,
        set_backend,
    )

    _install_fake_stir(monkeypatch)

    reset_backend()
    try:
        set_backend("stir")
        wrapped = create_acquisition_data(Path("/tmp/proj.hs"))
        assert wrapped.native_object.obj == ("acquisition", "/tmp/proj.hs")
    finally:
        reset_backend()


@pytest.mark.unit
def test_fake_stir_legacy_layout_activates(monkeypatch):
    """The legacy-only fake still activates the STIR backend."""
    from simind_python_connector.backends import (
        create_image_data,
        reset_backend,
        set_backend,
    )

    _install_fake_stir(monkeypatch, modern=False)

    reset_backend()
    try:
        set_backend("stir")
        wrapped = create_image_data(Path("/tmp/legacy.hv"))
        assert wrapped.native_object == ("image", "/tmp/legacy.hv")
    finally:
        reset_backend()


@pytest.mark.requires_sirf
def test_sirf_wrappers_update_in_place(tmp_path):
    """SIRF's maximum() and sapyb() return new objects unless given out=."""
    import sirf.STIR as sirf

    from simind_python_connector.backends.sirf_backend import (
        SirfAcquisitionData,
        SirfImageData,
    )
    from simind_python_connector.utils.stir_utils import create_stir_acqdata

    native = sirf.ImageData()
    native.initialise((2, 3, 4), (1.0, 1.0, 1.0))
    native.fill(-1.0)
    image = SirfImageData(native)
    image.maximum(0)
    assert np.all(image.as_array() == 0)

    x = SirfImageData(native.get_uniform_copy(1.0))
    y = SirfImageData(native.get_uniform_copy(2.0))
    x.sapyb(2, y, 3)
    np.testing.assert_allclose(x.as_array(), 8.0)

    template = create_stir_acqdata([4, 3], 2, [1.0, 1.0], backend="sirf")
    acq_x = SirfAcquisitionData(template.get_uniform_copy(1.0))
    acq_y = SirfAcquisitionData(template.get_uniform_copy(2.0))
    acq_x.sapyb(2, acq_y, 3)
    np.testing.assert_allclose(acq_x.as_array(), 8.0)
