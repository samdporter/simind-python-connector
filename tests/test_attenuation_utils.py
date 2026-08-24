"""Unit tests for attenuation conversion helpers."""

from pathlib import Path

import numpy as np
import pytest

import simind_python_connector.utils.stir_utils as stir_utils_mod
from simind_python_connector.converters.attenuation import (
    get_attenuation_coefficient,
)
from simind_python_connector.utils.stir_utils import (
    get_sirf_attenuation_from_simind,
)


pytestmark = pytest.mark.unit


_ATTN_TABLE_HEADER = "\n" * 12
_WATER_TABLE = _ATTN_TABLE_HEADER + "0.010 0.050\n0.500 0.200\n"
_BONE_TABLE = _ATTN_TABLE_HEADER + "0.010 0.090\n0.500 0.360\n"


@pytest.fixture
def attenuation_tables(tmp_path: Path) -> Path:
    (tmp_path / "h2o.atn").write_text(_WATER_TABLE)
    (tmp_path / "bone.atn").write_text(_BONE_TABLE)
    return tmp_path


def test_get_attenuation_coefficient_reads_override_directory(
    attenuation_tables: Path,
) -> None:
    water = get_attenuation_coefficient(
        "water", 0.10, file_path=str(attenuation_tables)
    )
    bone = get_attenuation_coefficient("bone", 0.10, file_path=str(attenuation_tables))

    expected_water = float(np.interp(0.10, [0.010, 0.500], [0.050, 0.200])) * 1.0
    expected_bone = float(np.interp(0.10, [0.010, 0.500], [0.090, 0.360])) * 1.85

    assert water == pytest.approx(expected_water)
    assert bone == pytest.approx(expected_bone)


def test_get_attenuation_coefficient_rejects_unknown_material(
    attenuation_tables: Path,
) -> None:
    with pytest.raises(ValueError, match="material"):
        get_attenuation_coefficient("lead", 0.10, file_path=str(attenuation_tables))


class _FakeImageData:
    instances: list["_FakeImageData"] = []

    def __init__(self) -> None:
        self.init_args = None
        self.filled = None
        _FakeImageData.instances.append(self)

    def initialise(self, dim=None, vsize=None, origin=None):
        self.init_args = {"dim": dim, "vsize": vsize, "origin": origin}

    def fill(self, array):
        self.filled = np.asarray(array)


@pytest.fixture
def fake_image_data(monkeypatch: pytest.MonkeyPatch) -> list[_FakeImageData]:
    _FakeImageData.instances = []
    monkeypatch.setattr(stir_utils_mod, "ImageData", _FakeImageData)
    return _FakeImageData.instances


def _write_attenuation_files(tmp_path: Path, payload: bytes) -> Path:
    header = tmp_path / "map.hct"
    header.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!matrix size [1] := 2",
                "!matrix size [2] := 3",
                "!matrix size [3] := 4",
                "scaling factor (mm/pixel) [1] := 1.5",
                "scaling factor (mm/pixel) [2] := 2.5",
                "# scaling factor (mm/pixel) [3] := 3.5",
                "# Image Position First image := -128.0000 -128.0000 128.000000",
            ]
        )
    )
    (tmp_path / "map.ict").write_bytes(payload)
    return tmp_path / "map.hct"


def test_get_sirf_attenuation_rho_times_1000_fills_image(
    tmp_path: Path, fake_image_data
) -> None:
    values = np.arange(24, dtype=np.uint16)
    header_path = _write_attenuation_files(tmp_path, values.tobytes())

    image = get_sirf_attenuation_from_simind(str(header_path), attn_type="rho*1000")

    filled = np.asarray(image.filled, dtype=np.float64)
    assert filled.shape == (4, 3, 2)
    assert filled.sum() > 0
    assert image.init_args["vsize"] == pytest.approx((1.5, 2.5, 3.5))


def test_get_sirf_attenuation_rejects_unknown_attn_type(
    tmp_path: Path, fake_image_data
) -> None:
    header_path = _write_attenuation_files(tmp_path, b"\x00" * 48)

    with pytest.raises(ValueError, match="attn_type"):
        get_sirf_attenuation_from_simind(str(header_path), attn_type="rho*100")
