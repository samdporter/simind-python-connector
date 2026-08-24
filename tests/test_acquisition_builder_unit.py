import sys
import types


try:
    import pydicom  # noqa: F401
except ImportError:
    # Only stub when genuinely unavailable so real-pydicom tests can coexist.
    sys.modules["pydicom"] = types.ModuleType("pydicom")

import numpy as np
import pytest

from simind_python_connector.builders.acquisition_builder import (
    STIRSPECTAcquisitionDataBuilder,
)


class DummyAcquisitionData:
    """Minimal stand-in for backend acquisition data used by the builder."""

    def __init__(self, header_path):
        self.header_path = header_path
        self.fill_data = None
        self.write_calls = []

    def clone(self):
        return self

    def fill(self, data):
        self.fill_data = np.asarray(data)
        return self

    def write(self, path):
        self.write_calls.append(path)

    def get_uniform_copy(self, value):
        if self.fill_data is None:
            arr = np.asarray(value, dtype=float)
        else:
            arr = np.full_like(self.fill_data, value, dtype=float)
        copy = DummyAcquisitionData(self.header_path)
        copy.fill(arr)
        return copy


class DummyWrappedAcquisition(DummyAcquisitionData):
    """Wrapper-like stub exposing a native_object attribute."""

    def __init__(self, header_path):
        super().__init__(header_path)
        self._native = DummyAcquisitionData(header_path)

    @property
    def native_object(self):
        return self._native


@pytest.fixture
def fake_create_acquisition(monkeypatch):
    """Patch create_acquisition_data to avoid SIRF/STIR dependency."""
    created_objects = []

    def _factory(header_path):
        obj = DummyAcquisitionData(header_path)
        created_objects.append(obj)
        return obj

    monkeypatch.setattr(
        "simind_python_connector.builders.acquisition_builder.create_acquisition_data",
        _factory,
    )
    return created_objects


@pytest.mark.unit
def test_build_writes_header_and_data(tmp_path, fake_create_acquisition):
    builder = STIRSPECTAcquisitionDataBuilder()

    output_prefix = tmp_path / "acq"
    acq = builder.build(output_path=str(output_prefix))

    header_path = output_prefix.with_suffix(".hs")
    raw_path = output_prefix.with_suffix(".s")

    # Files should exist and contain the terminating key
    assert header_path.exists()
    assert raw_path.exists()
    header_text = header_path.read_text()
    assert "!END OF INTERFILE :=" in header_text

    # Builder should return the stub object and write via it
    assert acq is fake_create_acquisition[0]
    assert acq.write_calls == [str(header_path)]
    assert acq.fill_data is not None
    assert acq.fill_data.shape[0] == 1  # segments dimension


@pytest.mark.unit
def test_build_multi_energy_splits_windows(tmp_path, fake_create_acquisition):
    builder = STIRSPECTAcquisitionDataBuilder()

    # Use small synthetic geometry to keep files light
    builder.header["!matrix size [1]"] = "4"
    builder.header["!matrix size [2]"] = "2"
    builder.header["!number of projections"] = "4"

    # Provide two windows and matching pixel data (4 projections total)
    builder.energy_windows = [
        {"lower": 110.0, "upper": 130.0},
        {"lower": 130.0, "upper": 150.0},
    ]
    builder.pixel_array = np.arange(1 * 4 * 4 * 2, dtype=np.float32).reshape(1, 4, 4, 2)

    outputs = builder.build_multi_energy(output_path_base=str(tmp_path / "multi"))

    # Builder state must be reusable after a multi-energy build
    assert builder.header["!number of projections"] == "4"
    assert builder.pixel_array.shape == (1, 4, 4, 2)
    assert len(outputs) == 2
    # Each stub should see half the projections
    for idx, stub in enumerate(outputs, start=1):
        assert stub.fill_data.shape == (1, 4, 2, 2)
        assert stub.write_calls == [str(tmp_path / f"multi_ew{idx}.hs")]


@pytest.mark.unit
def test_build_with_explicit_backend_restores_global_backend(monkeypatch, tmp_path):
    from simind_python_connector.builders import acquisition_builder as builder_mod

    wrapped_objects = []
    set_backend_calls = []

    def _factory(header_path):
        obj = DummyWrappedAcquisition(header_path)
        wrapped_objects.append(obj)
        return obj

    monkeypatch.setattr(builder_mod, "create_acquisition_data", _factory)
    monkeypatch.setattr(builder_mod.BACKENDS.detection, "get_backend", lambda: "sirf")
    monkeypatch.setattr(
        builder_mod.BACKENDS.detection,
        "set_backend",
        lambda backend: set_backend_calls.append(backend),
    )

    builder = STIRSPECTAcquisitionDataBuilder(backend="stir")
    output = builder.build(output_path=tmp_path / "acq")

    assert set_backend_calls == ["stir", "sirf"]
    assert output is wrapped_objects[0].native_object


@pytest.mark.unit
def test_acquisition_builder_rejects_invalid_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        STIRSPECTAcquisitionDataBuilder(backend="invalid")  # type: ignore[arg-type]


@pytest.mark.unit
def test_acquisition_builder_rejects_pixel_array_shape_mismatch(
    tmp_path, fake_create_acquisition
):
    builder = STIRSPECTAcquisitionDataBuilder(
        header_overrides={
            "!matrix size [1]": "4",
            "!matrix size [2]": "2",
            "!number of projections": "3",
        }
    )
    builder.pixel_array = np.zeros((1, 4, 3, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        builder.build(output_path=tmp_path / "acq")


@pytest.mark.unit
def test_build_multi_energy_requires_pixel_array(tmp_path, fake_create_acquisition):
    builder = STIRSPECTAcquisitionDataBuilder()
    builder.header["!number of projections"] = "4"
    builder.energy_windows = [
        {"lower": 110.0, "upper": 130.0},
        {"lower": 130.0, "upper": 150.0},
    ]

    with pytest.raises(ValueError, match="pixel"):
        builder.build_multi_energy(output_path_base=str(tmp_path / "multi"))


@pytest.mark.unit
def test_build_multi_energy_rejects_non_divisible_projections(
    tmp_path, fake_create_acquisition
):
    builder = STIRSPECTAcquisitionDataBuilder()
    builder.header["!matrix size [1]"] = "4"
    builder.header["!matrix size [2]"] = "2"
    builder.header["!number of projections"] = "5"
    builder.energy_windows = [
        {"lower": 110.0, "upper": 130.0},
        {"lower": 130.0, "upper": 150.0},
    ]
    builder.pixel_array = np.zeros((1, 4, 5, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="divis"):
        builder.build_multi_energy(output_path_base=str(tmp_path / "multi"))

    assert not list(tmp_path.glob("multi_ew*.hs"))


@pytest.mark.unit
def test_acquisition_written_payload_matches_input(tmp_path, fake_create_acquisition):
    """The raw file and the returned object must hold the input orientation."""
    builder = STIRSPECTAcquisitionDataBuilder()
    builder.header["!matrix size [1]"] = "4"
    builder.header["!matrix size [2]"] = "2"
    builder.header["!number of projections"] = "3"

    one_hot = np.zeros((1, 4, 3, 2), dtype=np.float32)
    one_hot[0, 3, 2, 1] = 7.0
    builder.pixel_array = one_hot

    output_prefix = tmp_path / "acq"
    acq = builder.build(output_path=output_prefix)

    raw = np.fromfile(output_prefix.with_suffix(".s"), dtype=np.float32)
    assert np.array_equal(raw.reshape(one_hot.shape), one_hot)
    assert np.array_equal(acq.fill_data, one_hot)
