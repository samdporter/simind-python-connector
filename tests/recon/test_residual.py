import numpy as np
import pytest

from simind_python_connector.recon.corrections import ResidualCorrection
from simind_python_connector.recon.projector import SimindComponent, SimindProjector
from tests.recon.fakes import FakeAdaptor, FakeData


pytestmark = pytest.mark.unit


class _Model:
    """Projects by multiplying by a constant and records what it was given."""

    def __init__(self, factor):
        self.factor = factor
        self.images = []
        self.last_scale = 2.0

    def direct(self, image):
        self.images.append(image.as_array().copy())
        return FakeData(image.as_array() * self.factor)

    __call__ = direct


def test_estimate_is_base_plus_accurate_minus_fast():
    accurate, fast = _Model(3.0), _Model(1.0)
    correction = ResidualCorrection(
        accurate, fast, base_additive=FakeData([[[0.5, 0.5]]])
    )

    additive = correction.estimate(FakeData([[[1.0, 2.0]]]))

    np.testing.assert_allclose(additive.as_array(), [[[2.5, 4.5]]])  # 0.5 + 3x - x
    np.testing.assert_allclose(correction.last_accurate.as_array(), [[[3.0, 6.0]]])
    assert correction.last_scale == 2.0


def test_both_projections_see_the_clipped_masked_image():
    accurate, fast = _Model(3.0), _Model(1.0)
    correction = ResidualCorrection(
        accurate,
        fast,
        base_additive=FakeData(np.zeros((1, 1, 3))),
        mask=FakeData([[[1.0, 1.0, 0.0]]]),
    )
    correction.estimate(FakeData([[[-4.0, 2.0, 5.0]]]))

    np.testing.assert_array_equal(accurate.images[0], [[[0.0, 2.0, 0.0]]])
    np.testing.assert_array_equal(fast.images[0], [[[0.0, 2.0, 0.0]]])


def test_last_scale_is_none_when_the_accurate_model_has_none():
    correction = ResidualCorrection(
        lambda image: image, _Model(1.0), base_additive=FakeData([[[0.0]]])
    )
    assert correction.last_scale is None
    assert correction.last_accurate is None


def test_simind_component_returns_one_scaled_output():
    adaptor = FakeAdaptor(
        {
            "all_interactions": FakeData(np.full((1, 2, 2), 5.0)),
            "geom_coll_primary": FakeData(np.full((1, 2, 2), 2.0)),
        }
    )
    projector = SimindProjector(
        adaptor, "template", "mu", normalise=lambda outputs, image: 3.0
    )
    component = SimindComponent(projector, "all_interactions")

    result = component(FakeData(np.ones((1, 2, 2))))

    np.testing.assert_allclose(result.as_array(), 15.0)
    assert component.last_scale == 3.0
