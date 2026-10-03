"""Unit tests for attenuation conversion helpers."""

from pathlib import Path

import numpy as np
import pytest

from simind_python_connector.converters.attenuation import (
    get_attenuation_coefficient,
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
