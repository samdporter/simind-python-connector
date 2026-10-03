import numpy as np
import pytest

from simind_python_connector.builders import image_builder as builder_mod
from simind_python_connector.builders.image_builder import STIRSPECTImageDataBuilder


pytestmark = pytest.mark.unit

_SMALL_HEADER = {
    "!matrix size [1]": "4",
    "!matrix size [2]": "3",
    "!matrix size [3]": "2",
}


@pytest.fixture
def loaded(monkeypatch):
    calls = []

    def fake_load_image(header_path, backend):
        calls.append((header_path, backend))
        return ("native-image", header_path)

    monkeypatch.setattr(builder_mod._native, "load_image", fake_load_image)
    return calls


def test_image_builder_passes_explicit_backend(loaded, tmp_path):
    builder = STIRSPECTImageDataBuilder(header_overrides=_SMALL_HEADER, backend="stir")
    builder.set_pixel_array(np.ones((2, 3, 4), dtype=np.float32))

    output = builder.build(output_path=tmp_path / "img")

    header_path = str(tmp_path / "img.hv")
    assert loaded == [(header_path, "stir")]
    assert output == ("native-image", header_path)


def test_image_builder_passes_none_when_backend_not_specified(loaded, tmp_path):
    builder = STIRSPECTImageDataBuilder(header_overrides=_SMALL_HEADER)
    builder.set_pixel_array(np.zeros((2, 3, 4), dtype=np.float32))

    builder.build(output_path=tmp_path / "img")

    assert loaded == [(str(tmp_path / "img.hv"), None)]


def test_image_builder_writes_header_and_raw_data(loaded, tmp_path):
    builder = STIRSPECTImageDataBuilder(header_overrides=_SMALL_HEADER)
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    builder.set_pixel_array(data)

    builder.build(output_path=tmp_path / "img")

    assert "!name of data file := img.v" in (tmp_path / "img.hv").read_text()
    raw = np.fromfile(tmp_path / "img.v", dtype=np.float32)
    assert np.array_equal(raw.reshape(2, 3, 4), data)


def test_image_builder_rejects_invalid_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        STIRSPECTImageDataBuilder(backend="invalid")  # type: ignore[arg-type]


def test_image_builder_rejects_pixel_array_shape_mismatch(loaded, tmp_path):
    builder = STIRSPECTImageDataBuilder(header_overrides=_SMALL_HEADER)
    builder.set_pixel_array(np.ones((2, 3, 5), dtype=np.float32))

    with pytest.raises(ValueError, match="shape"):
        builder.build(output_path=tmp_path / "img")
