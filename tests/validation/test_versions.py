import time

import numpy as np
import pytest

from simind_python_connector.core.types import ScoringRoutine, SimulationError
from tests.test_release_regressions_simind import GRID, VOXEL_MM, _connector


pytestmark = [pytest.mark.validation, pytest.mark.requires_simind]

PAIR_VIEWS = 8
PAIR_NN = 400


def _centroid(projection: np.ndarray, axis: int) -> np.ndarray:
    """Per-view intensity-weighted mean index along axis (1 rows, 2 bins)."""
    index = np.arange(projection.shape[axis])
    shape = [1, 1, 1]
    shape[axis] = -1
    weighted = (projection * index.reshape(shape)).sum(axis=(1, 2))
    return weighted / projection.sum(axis=(1, 2))


def _point_projection(tmp_path, prefix: str, voxel: tuple[int, int, int]):
    connector = _connector(tmp_path, prefix)
    # configure_voxel_phantom overrides Index 26; NN scales the histories
    connector.add_runtime_switch("NN", PAIR_NN)
    connector.add_config_value(29, PAIR_VIEWS)
    source = np.zeros((GRID, GRID, GRID), np.float32)
    source[voxel] = 1.0
    mu_map = np.full((GRID, GRID, GRID), 0.15, np.float32)
    connector.configure_voxel_phantom(
        source, mu_map, VOXEL_MM, scoring_routine=ScoringRoutine.PENETRATE
    )
    return connector.run()["geom_coll_primary"].projection


@pytest.mark.parametrize("axis, name", [(1, "row"), (2, "bin")])
def test_point_pair_symmetric_about_centre(validate, tmp_path, axis, name):
    first = (5, 4, 11)
    mirror = tuple(GRID - 1 - c for c in first)
    centre = (GRID - 1) / 2
    c1 = _centroid(_point_projection(tmp_path, "pa", first), axis)
    c2 = _centroid(_point_projection(tmp_path, "pb", mirror), axis)

    validate(
        f"point_pair_symmetric_about_centre_{name}",
        float(np.max(np.abs(c1 + c2 - 2 * centre))),
        upper=0.25,
    )


def test_v9_rejects_unvalidated_orientation(validate, not_run, engine, tmp_path):
    if engine.version != 9:
        not_run("v9_rejects_unvalidated_orientation", "v9-only centring guard")
    connector = _connector(tmp_path, "ori")
    connector.add_runtime_switch("DI", 1)

    try:
        connector.run()
        rejected = 0.0
    except ValueError:
        rejected = 1.0

    validate("v9_rejects_unvalidated_orientation", rejected, expected=1.0, abs_tol=0)


def test_timeout_enforced(validate, tmp_path):
    size = 64
    connector = _connector(tmp_path, "slow")
    for index, value in ((76, size), (77, size), (29, 60)):
        connector.add_config_value(index, value)
    connector.configure_voxel_phantom(
        np.ones((size,) * 3, np.float32),
        np.full((size,) * 3, 0.15, np.float32),
        VOXEL_MM,
    )
    connector.add_runtime_switch("NN", 50)

    start = time.perf_counter()
    with pytest.raises(SimulationError, match="timed out"):
        connector.run(timeout=1.0)

    validate("timeout_enforced", time.perf_counter() - start, upper=10.0)
