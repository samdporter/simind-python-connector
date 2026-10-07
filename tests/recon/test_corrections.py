from types import SimpleNamespace

import numpy as np
import pytest

from simind_python_connector.core.types import ScoringRoutine
from simind_python_connector.recon.corrections import ScatterCorrection
from tests.recon.fakes import FakeData


pytestmark = pytest.mark.unit


class _Projector:
    def __init__(self, outputs, routine):
        self.outputs = outputs
        self.adaptor = SimpleNamespace(get_scoring_routine=lambda: routine)
        self.last_scale = 4.0
        self.images = []

    def project(self, image):
        self.images.append(image)
        return self.outputs


def test_penetrate_scatter_is_all_events_minus_the_geometric_primary():
    projector = _Projector(
        {
            "all_interactions": FakeData(np.full((2, 3, 4), 5.0)),
            "geom_coll_primary": FakeData(np.full((2, 3, 4), 3.0)),
        },
        ScoringRoutine.PENETRATE,
    )
    correction = ScatterCorrection(projector)

    np.testing.assert_allclose(correction.estimate("image").as_array(), 2.0)
    assert projector.images == ["image"]
    assert correction.last_scale == 4.0


def test_scattwin_scatter_uses_the_window_and_adds_the_background():
    projector = _Projector(
        {
            "sca_w1": FakeData(np.full((2, 3, 4), 1.0)),
            "sca_w2": FakeData(np.full((2, 3, 4), 7.0)),
        },
        ScoringRoutine.SCATTWIN,
    )
    correction = ScatterCorrection(
        projector, background=FakeData(np.full((2, 3, 4), 0.5)), window=2
    )
    np.testing.assert_allclose(correction.estimate("image").as_array(), 7.5)


def test_smoothing_keeps_the_total_and_stays_within_each_view():
    array = np.zeros((9, 3, 9))  # (axial, view, bin)
    array[4, 1, 4] = 90.0
    projector = _Projector({"sca_w1": FakeData(array)}, ScoringRoutine.SCATTWIN)

    scatter = (
        ScatterCorrection(projector, smoothing_fwhm_bins=2.0)
        .estimate("image")
        .as_array()
    )

    assert scatter.sum() == pytest.approx(90.0)
    assert scatter[:, 0, :].sum() == 0.0
    assert scatter[:, 2, :].sum() == 0.0
    assert scatter[4, 1, 4] < 90.0
    assert scatter[4, 1, 5] > 0.0  # blurred along the bins
    assert scatter[5, 1, 4] > 0.0  # and along the axial direction
