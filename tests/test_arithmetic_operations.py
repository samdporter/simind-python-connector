"""Element-wise arithmetic on SIRF wrappers, using real backend data."""

import operator

import numpy as np
import pytest

from simind_python_connector.backends.sirf_backend import (
    SirfAcquisitionData,
    SirfImageData,
)
from simind_python_connector.utils.stir_utils import (
    create_stir_acqdata,
    create_stir_image,
)


pytestmark = [pytest.mark.integration, pytest.mark.requires_sirf]


def _make_operands(kind):
    if kind == "image":
        left = SirfImageData(create_stir_image([2, 3, 4], [4.0] * 3, backend="sirf"))
    else:
        left = SirfAcquisitionData(
            create_stir_acqdata([4, 3], 2, [4.42, 4.42], backend="sirf")
        )
    right = left.clone()
    shape = left.as_array().shape
    left.fill(np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape))
    right.fill(np.linspace(2, 3, np.prod(shape), dtype=np.float32).reshape(shape))
    return left, right


@pytest.fixture(params=["image", "acquisition"])
def operands(request):
    return _make_operands(request.param)


@pytest.mark.parametrize(
    "operation", [operator.add, operator.sub, operator.mul, operator.truediv]
)
@pytest.mark.parametrize("operand", ["wrapper", "scalar", "scalar_left"])
def test_wrapper_arithmetic(operands, operation, operand):
    left, right = operands
    left_values = left.as_array().copy()
    right_values = right.as_array().copy()
    if operand == "wrapper":
        result = operation(left, right)
        expected = operation(left_values, right_values)
    elif operand == "scalar":
        result = operation(left, 2.0)
        expected = operation(left_values, 2.0)
    else:
        result = operation(2.0, left)
        expected = operation(2.0, left_values)
    assert type(result) is type(left)
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-5)
    np.testing.assert_array_equal(left.as_array(), left_values)
    np.testing.assert_array_equal(right.as_array(), right_values)


def test_wrapper_negation(operands):
    left, _ = operands
    before = left.as_array().copy()
    result = -left
    assert type(result) is type(left)
    np.testing.assert_allclose(result.as_array(), -before)
    np.testing.assert_array_equal(left.as_array(), before)


def test_triple_energy_window_scatter_estimate():
    lower, upper = _make_operands("acquisition")
    lower_values = lower.as_array().copy()
    upper_values = upper.as_array().copy()
    # A 28 keV photopeak with 6 and 8 keV side windows.
    result = lower * (28 / (2 * 6)) + upper * (28 / (2 * 8))
    np.testing.assert_allclose(
        result.as_array(),
        lower_values * (28 / 12) + upper_values * (28 / 16),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(lower.as_array(), lower_values)
    np.testing.assert_array_equal(upper.as_array(), upper_values)
