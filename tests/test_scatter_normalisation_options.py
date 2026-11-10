"""The scatter-normalisation tutorial exposes five usable strategies."""

import numpy as np
import pytest

from examples._scatter_normalisation import STRATEGIES, strategy_options


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def as_array(self):
        return self.values


class _Model:
    def direct(self, image):
        return _Array(np.ones((1, 2, 2)))


class _Measured:
    def as_array(self):
        return np.ones((1, 2, 2))


@pytest.mark.unit
def test_every_scatter_normalisation_strategy_is_constructible():
    options = strategy_options(_Model(), _Measured(), 1.0, 2.0)
    assert set(options) == set(STRATEGIES)
    for settings in options.values():
        assert "projector" in settings
        assert "correction" in settings

    outputs = {"geom_coll_primary": _Array(np.ones((1, 2, 2)))}
    for title in ("Reference, sum", "Reference, trimmed"):
        normalise = options[title]["projector"]["normalise"]
        assert normalise(outputs, None) == pytest.approx(1.0)
