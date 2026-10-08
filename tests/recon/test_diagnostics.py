import math
from types import SimpleNamespace

import numpy as np
import pytest

from simind_python_connector.recon.diagnostics import effective_objective, poisson_nll
from tests.recon.fakes import FakeData


pytestmark = pytest.mark.unit


def test_poisson_nll_matches_a_hand_computed_value():
    # (1 - 2 ln 1) + (0.5 - 0) + (2 - 3 ln 2)
    expected = 3.5 - 3 * math.log(2.0)
    assert poisson_nll([2.0, 0.0, 3.0], [1.0, 0.5, 2.0]) == pytest.approx(expected)


def test_poisson_nll_skips_empty_bins_with_zero_mean():
    assert poisson_nll([0.0, 1.0], [0.0, 1.0]) == pytest.approx(1.0)


@pytest.mark.parametrize("mean", [[0.0, 1.0], [-0.5, 1.0]])
def test_poisson_nll_is_infinite_when_counts_meet_a_non_positive_mean(mean):
    assert poisson_nll([1.0, 1.0], mean) == math.inf


def test_effective_objective_uses_the_last_accurate_projection_plus_base():
    correction = SimpleNamespace(
        last_accurate=FakeData([[[1.0, 1.5]]]), base_additive=FakeData([[[0.0, 0.5]]])
    )
    measured = FakeData([[[2.0, 3.0]]])
    expected = poisson_nll([2.0, 3.0], [1.0, 2.0])
    assert effective_objective(measured, correction) == pytest.approx(expected)


def test_effective_objective_accepts_numpy_measurements():
    correction = SimpleNamespace(
        last_accurate=FakeData([[[1.0]]]), base_additive=FakeData([[[1.0]]])
    )
    assert effective_objective(np.array([[[2.0]]]), correction) == pytest.approx(
        2.0 - 2.0 * math.log(2.0)
    )


def test_effective_objective_needs_an_estimate_first():
    correction = SimpleNamespace(last_accurate=None, base_additive=FakeData([[[0.0]]]))
    with pytest.raises(ValueError, match="after the first estimate"):
        effective_objective(FakeData([[[1.0]]]), correction)
