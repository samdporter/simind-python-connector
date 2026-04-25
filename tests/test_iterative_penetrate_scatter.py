import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "iterative_penetrate_scatter.py"
)
SPEC = importlib.util.spec_from_file_location(
    "iterative_penetrate_scatter",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeAcquisitionData:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def asarray(self):
        return self._values

    def sum(self):
        return float(np.sum(self._values))


@pytest.mark.unit
def test_compute_scale_trimmed_mean_uses_retained_ratio_of_sums():
    linear = FakeAcquisitionData([10.0, 20.0, 30.0, 1000.0])
    b02 = FakeAcquisitionData([5.0, 10.0, 15.0, 1.0])

    scale, b02_counts, samples = MODULE.compute_scale(
        linear,
        b02,
        method="trimmed_mean",
        trim_frac=0.25,
    )

    assert scale == pytest.approx(2.0)
    assert b02_counts == pytest.approx(31.0)
    assert samples == 4


@pytest.mark.unit
def test_compute_scale_trimmed_mean_empty_mask_falls_back_to_sum():
    linear = FakeAcquisitionData([0.0, 0.0, 6.0])
    b02 = FakeAcquisitionData([2.0, 3.0, 5.0])

    scale, b02_counts, samples = MODULE.compute_scale(
        linear,
        b02,
        method="trimmed_mean",
        min_linear=10.0,
        fallback_if_empty="sum",
    )

    assert scale == pytest.approx(0.6)
    assert b02_counts == pytest.approx(10.0)
    assert samples == 0


@pytest.mark.unit
def test_compute_scale_trimmed_mean_empty_mask_can_fail_fast():
    linear = FakeAcquisitionData([0.0, 0.0, 6.0])
    b02 = FakeAcquisitionData([2.0, 3.0, 5.0])

    with pytest.raises(ValueError, match="mask empty"):
        MODULE.compute_scale(
            linear,
            b02,
            method="trimmed_mean",
            min_linear=10.0,
            fallback_if_empty="raise",
        )
