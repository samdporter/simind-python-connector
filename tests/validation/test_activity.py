import numpy as np
import pytest

from simind_python_connector.core.types import ScoringRoutine
from tests.test_release_regressions_simind import GRID, VOXEL_MM, _connector


pytestmark = [pytest.mark.validation, pytest.mark.requires_simind]


def _long_path(tmp_path):
    # v9 Linux SCATTWIN aborts (heap corruption) for output paths of 35-48 chars
    directory = tmp_path / ("x" * max(1, 60 - len(str(tmp_path)) - 1))
    directory.mkdir(exist_ok=True)
    return directory


def _total(
    tmp_path,
    prefix: str,
    *,
    activity_mbq: float = 10,
    time_s: float = 1,
    seed: int = 7,
    nn: int = 1,
    source_scale: float = 1.0,
) -> np.ndarray:
    connector = _connector(
        _long_path(tmp_path), prefix, scoring_routine=ScoringRoutine.SCATTWIN
    )
    source = np.zeros((GRID, GRID, GRID), np.float32)
    source[GRID // 2, GRID // 2, GRID // 2] = source_scale
    mu_map = np.full((GRID, GRID, GRID), 0.15, np.float32)
    connector.configure_voxel_phantom(
        source, mu_map, VOXEL_MM, scoring_routine=ScoringRoutine.SCATTWIN
    )
    connector.add_runtime_switch("CA", 0)
    connector.add_runtime_switch("RR", seed)
    connector.add_runtime_switch("NN", nn)
    connector.set_activity(activity_mbq, time_s)
    return connector.run()["tot_w1"].projection


def _max_relative_deviation_from(ratio: float, first: np.ndarray, second: np.ndarray):
    positive = first > 0
    return float(np.max(np.abs(second[positive] / first[positive] - ratio) / ratio))


def test_activity_doubling(validate, tmp_path):
    tot1 = _total(tmp_path, "a10", activity_mbq=10)
    tot2 = _total(tmp_path, "a20", activity_mbq=20)

    validate(
        "activity_doubling", float(tot2.sum() / tot1.sum()), expected=2.0, rel=1e-5
    )
    validate(
        "activity_doubling_binwise",
        _max_relative_deviation_from(2.0, tot1, tot2),
        upper=1e-5,
    )


def test_time_doubling(validate, tmp_path):
    tot1 = _total(tmp_path, "t1", time_s=1)
    tot2 = _total(tmp_path, "t2", time_s=2)

    validate("time_doubling", float(tot2.sum() / tot1.sum()), expected=2.0, rel=1e-5)


def test_source_magnitude_does_not_change_counts(validate, tmp_path):
    base = _total(tmp_path, "s1")
    scaled = _total(tmp_path, "s10", source_scale=10.0)

    validate(
        "source_magnitude_does_not_change_counts",
        float(np.max(np.abs(scaled - base))),
        expected=0.0,
        abs_tol=0.0,
    )


def test_histories_do_not_change_counts(validate, tmp_path):
    t7 = float(_total(tmp_path, "h7", seed=7).sum())
    t8 = float(_total(tmp_path, "h8", seed=8).sum())
    d0 = abs(1 - t7 / t8)
    bound = max(3 * d0, 0.02)
    t9 = float(_total(tmp_path, "h9", seed=9, nn=4).sum())

    validate(
        "histories_do_not_change_counts",
        float(abs(1 - t9 / np.mean([t7, t8]))),
        upper=bound,
        smoke=True,
        note=f"d0={d0:.4g}",
    )
