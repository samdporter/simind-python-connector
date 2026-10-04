"""Unit tests for attenuation conversion helpers."""

from pathlib import Path

import numpy as np
import pytest

from simind_python_connector.converters.attenuation import (
    attenuation_to_density,
    density_to_attenuation,
    get_attenuation_coefficient,
)
from simind_python_connector.data import load_table


pytestmark = pytest.mark.unit


_WATER_TABLE = "# energy [MeV], mu/rho [cm^2/g]\n0.010 0.050\n0.500 0.200\n"
_BONE_TABLE = "# energy [MeV], mu/rho [cm^2/g]\n0.010 0.090\n0.500 0.360\n"


@pytest.fixture
def attenuation_tables(tmp_path: Path) -> Path:
    (tmp_path / "h2o.atn").write_text(_WATER_TABLE)
    (tmp_path / "bone.atn").write_text(_BONE_TABLE)
    return tmp_path


def test_get_attenuation_coefficient_takes_kev(attenuation_tables: Path) -> None:
    water = get_attenuation_coefficient(
        "water", 100.0, file_path=str(attenuation_tables)
    )
    bone = get_attenuation_coefficient("bone", 100.0, file_path=str(attenuation_tables))

    expected_water = float(np.interp(0.10, [0.010, 0.500], [0.050, 0.200])) * 1.0
    expected_bone = float(np.interp(0.10, [0.010, 0.500], [0.090, 0.360])) * 1.85

    assert water == pytest.approx(expected_water)
    assert bone == pytest.approx(expected_bone)


def test_get_attenuation_coefficient_rejects_unknown_material(
    attenuation_tables: Path,
) -> None:
    with pytest.raises(ValueError, match="material"):
        get_attenuation_coefficient("lead", 100.0, file_path=str(attenuation_tables))


def test_packaged_tables_load_and_give_known_values() -> None:
    water_table = load_table("h2o.atn")
    bone_table = load_table("bone.atn")
    assert water_table.shape == (80, 2)
    assert bone_table.shape == (174, 2)
    assert tuple(water_table[0]) == pytest.approx((1.0e-3, 4.077e3))

    assert get_attenuation_coefficient("water", 140.0) == pytest.approx(
        0.1545, rel=0.01
    )
    bone_mass = float(np.interp(0.14, bone_table[:, 0], bone_table[:, 1]))
    assert get_attenuation_coefficient("bone", 140.0) == pytest.approx(1.85 * bone_mass)


@pytest.mark.parametrize("energy_kev", [70.0, 140.0, 208.0])
def test_density_round_trip(energy_kev: float) -> None:
    density = np.array([0.0, 0.5, 1.0, 1.5, 1.85])
    mu = density_to_attenuation(density, energy_kev)
    assert attenuation_to_density(mu, energy_kev) == pytest.approx(density)
