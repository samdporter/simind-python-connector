import numpy as np
import pytest

from simind_python_connector import SimindPythonConnector
from simind_python_connector.configs import get
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
)


pytestmark = [pytest.mark.validation, pytest.mark.requires_simind]

SEED = 7
MU_ENERGY_KEV = 140.0
MU_WATER_PER_CM = 0.15454
MU_BONE_PER_CM = 0.287342
BONE_DENSITY = 1.85
BONE_HU = 1000.0
PATH_LENGTH_CM = 6.0
TRANSMISSION_NN = 100


def _connector(
    directory, prefix: str, quantization_scale: float = DEFAULT_QUANTIZATION_SCALE
) -> SimindPythonConnector:
    connector = SimindPythonConnector(
        get("Example.yaml"), directory, prefix, quantization_scale=quantization_scale
    )
    connector.get_config().set_flag(11, True)
    return connector


def _written_density(tmp_path, prefix: str, mu_map: np.ndarray, mu_map_type: str):
    connector = _connector(tmp_path, prefix)
    source = np.zeros(mu_map.shape, np.float32)
    source[tuple(n // 2 for n in mu_map.shape)] = 1.0
    _, density_path = connector.configure_voxel_phantom(
        source,
        mu_map,
        4.0,
        mu_map_type=mu_map_type,
        mu_map_energy_kev=MU_ENERGY_KEV,
    )
    return np.fromfile(density_path, np.uint16).astype(np.int32)


def test_mu_map_types_write_the_same_density_map(validate, tmp_path):
    bone_region = np.zeros((32, 32, 32), bool)
    bone_region[:, :, 16:] = True
    maps = {
        "attenuation": np.where(bone_region, MU_BONE_PER_CM, MU_WATER_PER_CM),
        "density": np.where(bone_region, BONE_DENSITY, 1.0),
        "hu": np.where(bone_region, BONE_HU, 0.0),
    }
    written = {
        mu_map_type: _written_density(
            tmp_path, mu_map_type, mu_map.astype(np.float32), mu_map_type
        )
        for mu_map_type, mu_map in maps.items()
    }

    validate(
        "mu_map_types_write_the_same_density_map",
        float(
            max(
                np.max(np.abs(written["attenuation"] - written["density"])),
                np.max(np.abs(written["hu"] - written["density"])),
                np.max(np.abs(written["attenuation"] - written["hu"])),
            )
        ),
        expected=0,
        abs_tol=1,
    )


def _primary_sum(
    tmp_path, prefix: str, density: float, nn: int = TRANSMISSION_NN
) -> float:
    grid, voxel_mm, radius_cm = 64, 2.0, PATH_LENGTH_CM
    connector = _connector(tmp_path, prefix, quantization_scale=1.0)
    connector.add_config_value(29, 4)
    connector.add_config_value(53, 0)
    connector.add_config_value(28, voxel_mm / 10)
    connector.add_config_value(76, grid)
    connector.add_config_value(77, grid)
    source = np.zeros((grid,) * 3, np.float32)
    source[(grid // 2,) * 3] = 1.0
    offset = np.indices((grid,) * 3) - (grid - 1) / 2
    inside = (offset**2).sum(axis=0) <= (radius_cm * 10 / voxel_mm) ** 2
    connector.configure_voxel_phantom(
        source,
        (inside * density).astype(np.float32),
        voxel_mm,
        scoring_routine=ScoringRoutine.PENETRATE,
        mu_map_type="density",
    )
    connector.set_energy_windows([126], [154], [0])
    for switch, value in (("FI", "tc99m"), ("CC", "ma-lehr"), ("RR", SEED), ("NN", nn)):
        connector.add_runtime_switch(switch, value)
    return float(connector.run(timeout=1800)["geom_coll_primary"].projection.sum())


@pytest.fixture(scope="module")
def air_primary_sum(tmp_path_factory) -> float:
    return _primary_sum(tmp_path_factory.mktemp("air"), "air", 0.0)


@pytest.mark.parametrize(
    "case, density, mu_per_cm, rel",
    [
        ("water_transmission", 1.0, MU_WATER_PER_CM, 0.025),
        ("bone_transmission", BONE_DENSITY, MU_BONE_PER_CM, 0.04),
    ],
)
def test_beer_lambert_transmission(
    validate, tmp_path, air_primary_sum, case, density, mu_per_cm, rel
):
    attenuated = _primary_sum(tmp_path, case[:5], density)

    validate(
        case,
        attenuated / air_primary_sum,
        expected=float(np.exp(-mu_per_cm * PATH_LENGTH_CM)),
        rel=rel,
        note=f"histories raised: quantization_scale 1.0 and NN {TRANSMISSION_NN}",
    )
