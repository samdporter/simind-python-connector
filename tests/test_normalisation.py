import numpy as np
import pytest

from simind_python_connector.normalisation import scale_to_reference


pytestmark = pytest.mark.unit


def test_sum_method_is_the_ratio_of_sums():
    component = np.ones((2, 3, 4))
    reference = np.full((2, 3, 4), 2.5)
    assert scale_to_reference(component, reference) == pytest.approx(2.5)


def test_works_for_any_matching_shape():
    component = np.ones((1, 4, 3, 2))  # SIRF arrays carry a TOF axis
    assert scale_to_reference(component, 3 * component) == pytest.approx(3.0)


def test_mask_limits_the_bins_used():
    component = np.ones(10)
    reference = np.concatenate([np.full(5, 2.0), np.full(5, 100.0)])
    mask = np.arange(10) < 5
    assert scale_to_reference(component, reference, mask=mask) == pytest.approx(2.0)


def test_non_finite_bins_are_ignored():
    component = np.array([1.0, 1.0, np.nan, 1.0])
    reference = np.array([2.0, np.inf, 2.0, 2.0])
    assert scale_to_reference(component, reference) == pytest.approx(2.0)


def test_trimmed_method_ignores_outlier_ratios():
    component = np.ones(100)
    reference = np.full(100, 3.0)
    reference[:5] = 1000.0
    reference[5:10] = 0.001
    assert scale_to_reference(component, reference) != pytest.approx(3.0, rel=0.05)
    assert scale_to_reference(
        component, reference, method="trimmed", trim_fraction=0.1
    ) == pytest.approx(3.0)


def test_trimmed_method_applies_count_thresholds():
    component = np.array([0.0, 1.0, 1.0, 1.0])
    reference = np.array([5.0, 2.0, 2.0, 0.0])
    factor = scale_to_reference(
        component,
        reference,
        method="trimmed",
        trim_fraction=0.0,
        min_reference=0.5,
        min_component=0.5,
    )
    assert factor == pytest.approx(2.0)


def test_trimmed_method_uses_a_ratio_of_sums_not_a_mean_of_ratios():
    component = np.array([1.0, 1.0, 10.0, 1.0])
    reference = np.array([0.01, 1.0, 20.0, 100.0])
    factor = scale_to_reference(
        component, reference, method="trimmed", trim_fraction=0.25
    )
    assert factor == pytest.approx(21.0 / 11.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "median"}, "unknown method"),
        ({"method": "trimmed", "trim_fraction": 0.5}, "trim_fraction"),
        ({"mask": np.zeros(4, dtype=bool)}, "no bins left"),
        ({"mask": np.ones(3, dtype=bool)}, "mask shape"),
    ],
)
def test_invalid_arguments_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        scale_to_reference(np.ones(4), np.ones(4), **kwargs)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shapes differ"):
        scale_to_reference(np.ones(4), np.ones(5))


def test_zero_component_sum_raises():
    with pytest.raises(ValueError, match="must be > 0"):
        scale_to_reference(np.zeros(4), np.ones(4))


def test_thresholds_that_exclude_every_bin_raise():
    with pytest.raises(ValueError, match="no bins left"):
        scale_to_reference(np.ones(3), np.zeros(3), method="trimmed")


def test_sum_with_zero_reference_returns_zero():
    assert scale_to_reference(np.ones(4), np.zeros(4)) == pytest.approx(0.0)
