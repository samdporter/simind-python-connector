"""Real-STIR arithmetic tests for the reflected operators.

These exercise the scalar-on-left and unary operators that STIR's native
bindings do not implement, on both wrapper classes.
"""

import numpy as np
import pytest

from simind_python_connector.backends.stir_backend import (
    StirAcquisitionData,
    StirImageData,
)
from simind_python_connector.utils.stir_utils import (
    create_stir_acqdata,
    create_stir_image,
)


pytestmark = [pytest.mark.integration, pytest.mark.requires_stir]

OPERATORS = ("radd", "rsub", "rmul", "rtruediv", "neg")


def _make_image() -> StirImageData:
    native = create_stir_image([4, 5, 6], [2.0, 2.0, 2.0], backend="stir")
    image = StirImageData(native)
    shape = image.as_array().shape
    seed = np.linspace(1.0, 2.0, int(np.prod(shape))).reshape(shape)
    image.fill(seed.astype(np.float32))
    return image


def _make_acquisition() -> StirAcquisitionData:
    native = create_stir_acqdata([4, 5], 3, [4.0, 4.0], backend="stir")
    acquisition = StirAcquisitionData(native)
    shape = acquisition.as_array().shape
    seed = np.linspace(1.0, 2.0, int(np.prod(shape))).reshape(shape)
    acquisition.fill(seed.astype(np.float32))
    return acquisition


def _apply(operator: str, obj):
    return {
        "radd": lambda: 2.0 + obj,
        "rsub": lambda: 2.0 - obj,
        "rmul": lambda: 2.0 * obj,
        "rtruediv": lambda: 2.0 / obj,
        "neg": lambda: -obj,
    }[operator]()


def _expected(operator: str, array: np.ndarray) -> np.ndarray:
    return {
        "radd": 2.0 + array,
        "rsub": 2.0 - array,
        "rmul": 2.0 * array,
        "rtruediv": 2.0 / array,
        "neg": -array,
    }[operator]


@pytest.mark.parametrize("operator", OPERATORS)
def test_image_reflected_operators(operator):
    image = _make_image()
    before = image.as_array().copy()

    result = _apply(operator, image)

    assert isinstance(result, StirImageData)
    np.testing.assert_allclose(
        result.as_array(), _expected(operator, before), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_equal(image.as_array(), before)
    assert result.dimensions() == image.dimensions()
    assert result.voxel_sizes() == image.voxel_sizes()


@pytest.mark.parametrize("operator", OPERATORS)
def test_acquisition_reflected_operators(operator):
    acquisition = _make_acquisition()
    before = acquisition.as_array().copy()

    result = _apply(operator, acquisition)

    assert isinstance(result, StirAcquisitionData)
    np.testing.assert_allclose(
        result.as_array(), _expected(operator, before), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_equal(acquisition.as_array(), before)
    assert result.dimensions() == acquisition.dimensions()
    assert result.get_info() == acquisition.get_info()


def test_image_native_left_operands():
    image = _make_image()
    native = image.native_object
    before = image.as_array().copy()

    np.testing.assert_allclose(
        (native + image).as_array(), before + before, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        (native - image).as_array(), before - before, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        (native * image).as_array(), before * before, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        (native / image).as_array(), before / before, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_equal(image.as_array(), before)


def test_acquisition_native_left_operands():
    acquisition = _make_acquisition()
    native = acquisition.native_object
    before = acquisition.as_array().copy()

    np.testing.assert_allclose(
        (native + acquisition).as_array(), before + before, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        (native - acquisition).as_array(), before - before, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        (native * acquisition).as_array(), before * before, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        (native / acquisition).as_array(), before / before, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_equal(acquisition.as_array(), before)


def test_image_forward_operators_agree():
    image = _make_image()
    before = image.as_array().copy()
    other = image.clone()

    np.testing.assert_allclose((image + 2.0).as_array(), before + 2.0, rtol=1e-5)
    np.testing.assert_allclose((image - 2.0).as_array(), before - 2.0, rtol=1e-5)
    np.testing.assert_allclose((image * 2.0).as_array(), before * 2.0, rtol=1e-5)
    np.testing.assert_allclose((image / 2.0).as_array(), before / 2.0, rtol=1e-5)
    np.testing.assert_allclose((image + other).as_array(), before + before, rtol=1e-5)


def test_acquisition_forward_operators_agree():
    acquisition = _make_acquisition()
    before = acquisition.as_array().copy()
    other = acquisition.clone()

    np.testing.assert_allclose((acquisition + 2.0).as_array(), before + 2.0, rtol=1e-5)
    np.testing.assert_allclose((acquisition - 2.0).as_array(), before - 2.0, rtol=1e-5)
    np.testing.assert_allclose((acquisition * 2.0).as_array(), before * 2.0, rtol=1e-5)
    np.testing.assert_allclose((acquisition / 2.0).as_array(), before / 2.0, rtol=1e-5)
    np.testing.assert_allclose(
        (acquisition + other).as_array(), before + before, rtol=1e-5
    )


def test_get_array_matches_stir_extra_conversion():
    from simind_python_connector.utils import get_array
    from simind_python_connector.utils.import_helpers import _import_stir_extra

    extra = _import_stir_extra()

    image = _make_image()
    np.testing.assert_array_equal(
        get_array(image.native_object), extra.to_numpy(image.native_object)
    )

    acquisition = _make_acquisition()
    np.testing.assert_array_equal(
        get_array(acquisition.native_object), extra.to_numpy(acquisition.native_object)
    )
