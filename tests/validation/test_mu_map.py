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


def _bone_region_maps(water_value: float, bone_value: float) -> np.ndarray:
    mu_map = np.full((32, 32, 32), water_value, np.float32)
    mu_map[:, :, 16:] = bone_value
    return mu_map


def test_attenuation_and_density_inputs_agree(validate, tmp_path):
    from_attenuation = _written_density(
        tmp_path,
        "attn",
        _bone_region_maps(MU_WATER_PER_CM, MU_BONE_PER_CM),
        "attenuation",
    )
    from_density = _written_density(
        tmp_path, "dens", _bone_region_maps(1.0, BONE_DENSITY), "density"
    )

    validate(
        "attenuation_and_density_inputs_agree",
        float(np.max(np.abs(from_attenuation - from_density))),
        expected=0,
        abs_tol=1,
    )


def test_hu_input_uses_schneider(validate, tmp_path):
    """The HU path is the Schneider et al. 2000 (Phys. Med. Biol. 45:459,
    doi:10.1088/0031-9155/45/2/314) conversion, not the bilinear water/bone model.

    The reference is the piecewise-linear calibration OpenGATE distributes as
    opengate/data/Schneider2000DensitiesTable.txt: (-98 HU, 0.93), (14, 1.03),
    (23, 1.031), (100, 1.1199), (101, 1.0762), (1600, 1.9642) g/cm3. Linear
    interpolation gives 1.01745 at 0 HU and 1.60876 at 1000 HU.
    """
    written = _written_density(
        tmp_path, "hu", _bone_region_maps(0.0, BONE_HU), "hu"
    ).astype(float)
    water_g_cm3, bone_g_cm3 = written[0] / 1000, written[-1] / 1000
    expected_water = 0.93 + (1.03 - 0.93) * (0 + 98) / (14 + 98)
    expected_bone = 1.0762 + (1.9642 - 1.0762) * (1000 - 101) / (1600 - 101)

    validate(
        "hu_input_uses_schneider",
        float(max(abs(water_g_cm3 - expected_water), abs(bone_g_cm3 - expected_bone))),
        expected=0,
        abs_tol=0.01,
        note=f"HU 0 -> {water_g_cm3:.3f} (reference {expected_water:.5f}), "
        f"HU 1000 -> {bone_g_cm3:.3f} (reference {expected_bone:.5f}) g/cm3; "
        "reference: OpenGATE Schneider2000DensitiesTable.txt",
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
    "case, density, mu_per_cm, rel, bias_note",
    [
        (
            "water_transmission",
            1.0,
            MU_WATER_PER_CM,
            0.025,
            "measured is 1.0-1.6% above Beer-Lambert",
        ),
        (
            "bone_transmission",
            BONE_DENSITY,
            MU_BONE_PER_CM,
            0.04,
            "measured is 3.6-3.9% above Beer-Lambert",
        ),
    ],
)
def test_beer_lambert_transmission(
    validate, tmp_path, air_primary_sum, case, density, mu_per_cm, rel, bias_note
):
    attenuated = _primary_sum(tmp_path, case[:5], density)

    validate(
        case,
        attenuated / air_primary_sum,
        expected=float(np.exp(-mu_per_cm * PATH_LENGTH_CM)),
        rel=rel,
        note=(
            f"histories raised: quantization_scale 1.0 and NN {TRANSMISSION_NN}; "
            f"{bias_note}, systematic at high statistics (NN 100 and 300 agree), "
            "likely SIMIND's cross-sections vs the connector's attenuation table"
        ),
    )
