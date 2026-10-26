import numpy as np
import pytest

from simind_python_connector import SimindPythonConnector
from simind_python_connector.configs import get
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
)


pytestmark = [pytest.mark.validation, pytest.mark.requires_simind]

SLICES = 24
IN_PLANE = 32
ROWS = 48
SLICE_MM = 6.0
PIXEL_MM = 3.0
VIEWS = 4
SEED = 7
CYLINDER_RADIUS_MM = 30.0
POINT_NN = 20


def _projection(
    tmp_path,
    prefix: str,
    source: np.ndarray,
    quantization_scale: float = 1.0,
    nn: int = POINT_NN,
):
    connector = SimindPythonConnector(
        get("Example.yaml"), tmp_path, prefix, quantization_scale=quantization_scale
    )
    connector.get_config().set_flag(11, True)
    connector.add_config_value(29, VIEWS)
    connector.add_config_value(53, 0)
    connector.add_config_value(28, PIXEL_MM / 10)
    connector.add_config_value(76, IN_PLANE)
    connector.add_config_value(77, ROWS)
    connector.configure_voxel_phantom(
        source,
        np.zeros(source.shape, np.float32),
        (SLICE_MM, PIXEL_MM, PIXEL_MM),
        scoring_routine=ScoringRoutine.PENETRATE,
        mu_map_type="density",
    )
    connector.set_energy_windows([126], [154], [0])
    for switch, value in (("FI", "tc99m"), ("CC", "ma-lehr"), ("RR", SEED), ("NN", nn)):
        connector.add_runtime_switch(switch, value)
    return connector.run(timeout=600)["geom_coll_primary"].projection


def _row_profile(projection: np.ndarray) -> np.ndarray:
    return projection.sum(axis=(0, 2))


def _row_centroid(projection: np.ndarray) -> float:
    profile = _row_profile(projection)
    return float((profile * np.arange(ROWS)).sum() / profile.sum())


@pytest.fixture(scope="module")
def point_rows(tmp_path_factory) -> dict[int, float]:
    rows = {}
    for k in (4, 12, 19, 20):
        source = np.zeros((SLICES, IN_PLANE, IN_PLANE), np.float32)
        source[k, IN_PLANE // 2, IN_PLANE // 2] = 1.0
        directory = tmp_path_factory.mktemp(f"k{k}")
        rows[k] = _row_centroid(_projection(directory, f"k{k}", source))
    return rows


@pytest.mark.parametrize("first, second", [(4, 12), (12, 20)])
def test_axial_point_spacing(validate, point_rows, first, second):
    validate(
        f"axial_point_spacing_{first}_{second}",
        abs(point_rows[second] - point_rows[first]),
        expected=8 * SLICE_MM / PIXEL_MM,
        abs_tol=0.25,
        note=f"row centroids {point_rows[first]:.3f}, {point_rows[second]:.3f}",
    )


def test_axial_point_centring(validate, point_rows):
    """Slice 4 and slice 19 are mirror images about the slice centre 11.5, so
    their row centroids sum to ROWS - 1."""
    validate(
        "axial_point_centring",
        abs(point_rows[4] + point_rows[19] - (ROWS - 1)),
        upper=0.25,
        note=f"row centroids {point_rows[4]:.3f}, {point_rows[19]:.3f}",
    )


def test_axial_extent(validate, tmp_path):
    source = np.zeros((SLICES, IN_PLANE, IN_PLANE), np.float32)
    offset = np.indices((IN_PLANE, IN_PLANE)) - (IN_PLANE - 1) / 2
    in_cylinder = (offset**2).sum(axis=0) <= (CYLINDER_RADIUS_MM / PIXEL_MM) ** 2
    source[:, in_cylinder] = 1.0
    projection = _projection(tmp_path, "cyl", source, DEFAULT_QUANTIZATION_SCALE, nn=1)
    profile = _row_profile(projection)

    validate(
        "axial_extent",
        float((profile > 0.5 * profile[ROWS // 2 - 1 : ROWS // 2 + 1].max()).sum()),
        expected=48,
        abs_tol=1,
        smoke=True,
    )
