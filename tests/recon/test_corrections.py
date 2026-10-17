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


def _penetrate(total, primary):
    return {
        "all_interactions": FakeData(np.asarray(total, float)),
        "geom_coll_primary": FakeData(np.asarray(primary, float)),
    }


def test_pixelwise_scatter_is_the_measured_data_times_the_simind_scatter_fraction():
    projector = _Projector(
        _penetrate([[[4.0, 10.0]]], [[[3.0, 5.0]]]), ScoringRoutine.PENETRATE
    )
    measured = FakeData([[[8.0, 20.0]]])
    scatter = ScatterCorrection(projector, measured=measured).estimate("image")
    np.testing.assert_allclose(scatter.as_array(), [[[2.0, 10.0]]])


def test_pixelwise_scatter_uses_the_scattwin_window_total():
    projector = _Projector(
        {"sca_w2": FakeData([[[1.0]]]), "tot_w2": FakeData([[[4.0]]])},
        ScoringRoutine.SCATTWIN,
    )
    scatter = ScatterCorrection(projector, window=2, measured=FakeData([[[12.0]]]))
    np.testing.assert_allclose(scatter.estimate("image").as_array(), [[[3.0]]])


def test_pixelwise_scatter_is_zero_where_the_simind_total_is_not_positive():
    projector = _Projector(
        _penetrate([[[0.0, 2.0, 4.0]]], [[[0.0, 3.0, 2.0]]]), ScoringRoutine.PENETRATE
    )
    scatter = ScatterCorrection(projector, measured=FakeData([[[5.0, 4.0, 8.0]]]))
    result = scatter.estimate("image").as_array()
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, [[[0.0, -2.0, 4.0]]])


def test_pixelwise_scatter_does_not_depend_on_the_projector_scale():
    measured = FakeData([[[8.0, 20.0]]])
    results = []
    for scale in (1.0, 7.0):
        outputs = _penetrate(
            [[[4.0 * scale, 10.0 * scale]]], [[[3.0 * scale, 5.0 * scale]]]
        )
        projector = _Projector(outputs, ScoringRoutine.PENETRATE)
        results.append(
            ScatterCorrection(projector, measured=measured).estimate("x").as_array()
        )
    np.testing.assert_allclose(results[0], results[1])


def test_pixelwise_scatter_rejects_measured_data_of_another_shape():
    projector = _Projector(
        _penetrate([[[4.0, 4.0]]], [[[3.0, 3.0]]]), ScoringRoutine.PENETRATE
    )
    scatter = ScatterCorrection(projector, measured=FakeData(np.ones((1, 1, 1, 2))))
    with pytest.raises(
        ValueError, match=r"\(1, 1, 1, 2\).*\(1, 1, 2\)|\(1, 1, 2\).*\(1, 1, 1, 2\)"
    ):
        scatter.estimate("image")


def test_global_and_pixelwise_agree_when_the_data_are_a_multiple_of_the_simind_total():
    from simind_python_connector.recon.projector import (
        SimindProjector,
        measured_normaliser,
    )
    from tests.recon.fakes import FakeAdaptor

    total = np.array([[[4.0, 10.0, 6.0]]])
    primary = np.array([[[3.0, 5.0, 6.0]]])
    measured = FakeData(2.5 * total)
    adaptor = FakeAdaptor(
        {"all_interactions": FakeData(total), "geom_coll_primary": FakeData(primary)}
    )
    projector = SimindProjector(
        adaptor, "template", "mu", normalise=measured_normaliser(measured)
    )
    expected = 2.5 * (total - primary)
    image = FakeData(np.ones((1, 1, 2)))

    global_scatter = ScatterCorrection(projector).estimate(image).as_array()
    pixelwise_scatter = (
        ScatterCorrection(projector, measured=measured).estimate(image).as_array()
    )
    np.testing.assert_allclose(global_scatter, expected)
    np.testing.assert_allclose(pixelwise_scatter, expected)


def test_smoothing_across_views_keeps_the_total_and_wraps_round():
    array = np.zeros((9, 6, 9))  # (axial, view, bin)
    array[4, 0, 4] = 60.0
    projector = _Projector({"sca_w1": FakeData(array)}, ScoringRoutine.SCATTWIN)
    scatter = (
        ScatterCorrection(projector, smoothing_fwhm_bins=2.0, smooth_across_views=True)
        .estimate("image")
        .as_array()
    )
    assert scatter.sum() == pytest.approx(60.0)
    assert scatter[:, 1, :].sum() > 0.0
    assert scatter[:, 5, :].sum() == pytest.approx(scatter[:, 1, :].sum())
