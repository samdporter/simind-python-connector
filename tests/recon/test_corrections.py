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


def test_pixelwise_scatter_names_a_missing_scattwin_total():
    projector = _Projector({"sca_w1": FakeData([[[1.0]]])}, ScoringRoutine.SCATTWIN)
    correction = ScatterCorrection(projector, measured=FakeData([[[12.0]]]))
    with pytest.raises(KeyError, match="tot_w1"):
        correction.estimate("image")


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


def test_smoothing_keeps_a_zero_sum_signed_field():
    """A cancelling signed field must not be erased by its smoothed residual."""
    array = np.zeros((5, 3, 7))  # (axial, view, bin)
    array[1, 1, 1] = 6.0
    array[3, 1, 5] = -6.0
    assert array.sum() == 0.0
    projector = _Projector({"sca_w1": FakeData(array)}, ScoringRoutine.SCATTWIN)

    result = (
        ScatterCorrection(projector, smoothing_fwhm_bins=2.0, smooth_across_views=True)
        .estimate("image")
        .as_array()
    )

    np.testing.assert_allclose(
        result, _gaussian_smooth(array, 2.0, True), rtol=1e-5, atol=1e-6
    )
    assert np.isfinite(result).all()
    assert result.sum() == pytest.approx(0.0, abs=1e-5)
    assert result[1, 1, 1] > 0.0
    assert result[3, 1, 5] < 0.0


def _gaussian_smooth(array, fwhm_bins, across_views=False):
    """Independent NumPy/SciPy reference for the correction's smoothing."""
    from scipy.ndimage import gaussian_filter

    values = np.asarray(array, dtype=np.float64)
    sigma = [0.0] * values.ndim
    mode = ["reflect"] * values.ndim
    sigma[-3] = sigma[-1] = fwhm_bins / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if across_views:
        sigma[-2] = sigma[-1]
        mode[-2] = "wrap"
    smoothed = gaussian_filter(values, sigma, mode=mode)
    # Rescale only for a sum that stands clear of its rounding floor: a
    # cancelling signed field would otherwise be scaled by an arbitrary factor
    # (often zero) drawn from two floor-level residuals.
    original_sum = values.sum()
    if abs(original_sum) > 1e-9 * np.abs(values).sum():
        total = smoothed.sum()
        if total != 0.0:
            smoothed *= original_sum / total
    return smoothed.astype(np.float32)


def _smoothed_pixelwise(scatter, total, measured, fwhm_bins, across_views=False):
    """Y * (G(S) / G(T)), then G, with zero where the smoothed total is zero."""
    smoothed_scatter = _gaussian_smooth(scatter, fwhm_bins, across_views)
    smoothed_total = _gaussian_smooth(total, fwhm_bins, across_views)
    fraction = np.divide(
        smoothed_scatter,
        smoothed_total,
        out=np.zeros_like(smoothed_scatter),
        where=smoothed_total > 0.0,
    )
    return _gaussian_smooth(measured * fraction, fwhm_bins, across_views)


@pytest.mark.parametrize("across_views", [False, True])
def test_pixelwise_smoothing_pre_smooths_the_mc_fields_before_dividing(across_views):
    rng = np.random.default_rng(7)
    shape = (1, 5, 6, 7)
    total = rng.uniform(20.0, 40.0, shape)
    primary = rng.uniform(10.0, 30.0, shape)
    scatter = total - primary
    measured = rng.uniform(100.0, 200.0, shape)
    fwhm_bins = 2.5
    projector = _Projector(_penetrate(total, primary), ScoringRoutine.PENETRATE)

    result = (
        ScatterCorrection(
            projector,
            measured=FakeData(measured),
            smoothing_fwhm_bins=fwhm_bins,
            smooth_across_views=across_views,
        )
        .estimate("image")
        .as_array()
    )

    smoothed_scatter = _gaussian_smooth(scatter, fwhm_bins, across_views)
    smoothed_total = _gaussian_smooth(total, fwhm_bins, across_views)
    fraction = np.divide(
        smoothed_scatter,
        smoothed_total,
        out=np.zeros_like(smoothed_scatter),
        where=smoothed_total > 0.0,
    )
    expected = _gaussian_smooth(measured * fraction, fwhm_bins, across_views)
    legacy = _gaussian_smooth(measured * (scatter / total), fwhm_bins, across_views)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    assert not np.allclose(result, legacy, rtol=1e-4, atol=1e-4)
    # The final smoothing is still applied after the measured scaling.
    assert not np.allclose(result, measured * fraction, rtol=1e-4, atol=1e-4)


def test_pixelwise_smoothing_agrees_between_penetrate_and_scattwin():
    rng = np.random.default_rng(11)
    shape = (1, 4, 5, 6)
    total = rng.uniform(20.0, 40.0, shape)
    primary = rng.uniform(10.0, 30.0, shape)
    scatter = total - primary
    measured = FakeData(rng.uniform(100.0, 200.0, shape))
    options = {
        "measured": measured,
        "smoothing_fwhm_bins": 3.0,
        "smooth_across_views": True,
    }

    penetrate = _Projector(_penetrate(total, primary), ScoringRoutine.PENETRATE)
    scattwin = _Projector(
        {
            "sca_w1": FakeData(scatter),
            "tot_w1": FakeData(total),
        },
        ScoringRoutine.SCATTWIN,
    )

    np.testing.assert_allclose(
        ScatterCorrection(penetrate, **options).estimate("image").as_array(),
        ScatterCorrection(scattwin, **options).estimate("image").as_array(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_pixelwise_smoothing_does_not_mix_the_leading_axis():
    rng = np.random.default_rng(13)
    shape = (2, 5, 6, 7)
    total = rng.uniform(20.0, 40.0, shape)
    primary = rng.uniform(10.0, 30.0, shape)
    measured = rng.uniform(100.0, 200.0, shape)

    full = (
        ScatterCorrection(
            _Projector(_penetrate(total, primary), ScoringRoutine.PENETRATE),
            measured=FakeData(measured),
            smoothing_fwhm_bins=2.5,
            smooth_across_views=True,
        )
        .estimate("image")
        .as_array()
    )

    for index in range(shape[0]):
        single = (
            ScatterCorrection(
                _Projector(
                    _penetrate(total[index], primary[index]),
                    ScoringRoutine.PENETRATE,
                ),
                measured=FakeData(measured[index]),
                smoothing_fwhm_bins=2.5,
                smooth_across_views=True,
            )
            .estimate("image")
            .as_array()
        )
        np.testing.assert_allclose(full[index], single, rtol=1e-6, atol=1e-6)


def test_pixelwise_smoothing_wraps_views_only_when_requested():
    total = np.zeros((3, 6, 3))
    total[1, 0, 1] = 10.0
    primary = np.zeros_like(total)
    measured = FakeData(np.ones_like(total))

    def estimate(across_views):
        projector = _Projector(_penetrate(total, primary), ScoringRoutine.PENETRATE)
        return (
            ScatterCorrection(
                projector,
                measured=measured,
                smoothing_fwhm_bins=2.0,
                smooth_across_views=across_views,
            )
            .estimate("image")
            .as_array()
        )

    reflected = estimate(False)
    wrapped = estimate(True)

    assert wrapped[:, 5, :].sum() == pytest.approx(wrapped[:, 1, :].sum())
    assert not np.allclose(reflected, wrapped)
    np.testing.assert_allclose(
        reflected,
        _smoothed_pixelwise(total, total, measured.as_array(), 2.0, False),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        wrapped,
        _smoothed_pixelwise(total, total, measured.as_array(), 2.0, True),
        rtol=1e-6,
        atol=1e-6,
    )


def test_pixelwise_scatter_rejects_a_total_of_another_shape():
    projector = _Projector(
        {
            "sca_w1": FakeData(np.ones((1, 1, 2))),
            "tot_w1": FakeData(np.ones((1, 1, 3))),
        },
        ScoringRoutine.SCATTWIN,
    )
    correction = ScatterCorrection(
        projector, measured=FakeData(np.ones((1, 1, 2))), smoothing_fwhm_bins=2.0
    )
    with pytest.raises(ValueError, match=r"total shape.*scatter shape"):
        correction.estimate("image")


def test_pixelwise_smoothing_returns_zero_for_an_all_zero_denominator():
    total = np.zeros((1, 3, 4))
    projector = _Projector(_penetrate(total, total), ScoringRoutine.PENETRATE)
    result = (
        ScatterCorrection(
            projector,
            measured=FakeData(np.ones_like(total)),
            smoothing_fwhm_bins=2.0,
        )
        .estimate("image")
        .as_array()
    )
    np.testing.assert_array_equal(result, 0.0)


def test_pixelwise_smoothing_rescues_a_raw_zero_bin_from_its_neighbours():
    total = np.ones((1, 5, 7))
    total[0, 2, 3] = 0.0
    primary = np.zeros_like(total)
    projector = _Projector(_penetrate(total, primary), ScoringRoutine.PENETRATE)
    result = (
        ScatterCorrection(
            projector,
            measured=FakeData(np.ones_like(total)),
            smoothing_fwhm_bins=2.0,
        )
        .estimate("image")
        .as_array()
    )
    assert result[0, 2, 3] > 0.0


def test_pixelwise_smoothing_keeps_a_zero_sum_signed_scatter():
    """A zero-sum signed scatter survives pre-smoothing, division and the
    final smoothing, with the smoothed denominator deciding the division."""
    shape = (5, 3, 7)  # (axial, view, bin)
    scatter = np.zeros(shape)
    scatter[1, 1, 1] = 6.0
    scatter[3, 1, 5] = -6.0
    assert scatter.sum() == 0.0
    total = np.full(shape, 10.0)
    total[2, 1, 2] = 0.0  # raw zero bin, positive once smoothed
    primary = total - scatter
    measured = np.full(shape, 100.0)
    projector = _Projector(_penetrate(total, primary), ScoringRoutine.PENETRATE)

    result = (
        ScatterCorrection(
            projector,
            measured=FakeData(measured),
            smoothing_fwhm_bins=2.0,
            smooth_across_views=True,
        )
        .estimate("image")
        .as_array()
    )

    np.testing.assert_allclose(
        result,
        _smoothed_pixelwise(scatter, total, measured, 2.0, across_views=True),
        rtol=1e-4,
        atol=1e-5,
    )
    assert np.isfinite(result).all()
    # Signed scatter is kept, not clipped and not scaled to zero.
    assert result[1, 1, 1] > 0.0
    assert result[3, 1, 5] < 0.0
    # Division uses the smoothed total: the raw-zero bin is rescued.
    assert result[2, 1, 2] > 0.0


def test_pixelwise_smoothing_accepts_tiny_positive_denominators():
    scatter = FakeData(np.full((1, 1, 2), 2e-30))
    total = FakeData(np.full((1, 1, 2), 1e-30))
    projector = _Projector(
        {"sca_w1": scatter, "tot_w1": total}, ScoringRoutine.SCATTWIN
    )
    result = (
        ScatterCorrection(
            projector,
            measured=FakeData(np.full((1, 1, 2), 7.0)),
            smoothing_fwhm_bins=2.0,
        )
        .estimate("image")
        .as_array()
    )
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, 14.0, rtol=1e-5)


def test_pixelwise_smoothing_is_invariant_to_a_common_mc_scale():
    rng = np.random.default_rng(17)
    shape = (1, 4, 5, 6)
    total = rng.uniform(20.0, 40.0, shape)
    primary = rng.uniform(10.0, 30.0, shape)
    measured = FakeData(rng.uniform(100.0, 200.0, shape))
    results = []
    for scale in (1.0, 1e-9, 1e6):
        projector = _Projector(
            _penetrate(total * scale, primary * scale), ScoringRoutine.PENETRATE
        )
        results.append(
            ScatterCorrection(
                projector,
                measured=measured,
                smoothing_fwhm_bins=2.5,
                smooth_across_views=True,
            )
            .estimate("image")
            .as_array()
        )
    np.testing.assert_allclose(results[0], results[1], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(results[0], results[2], rtol=1e-6, atol=1e-8)


def test_pixelwise_smoothing_does_not_mutate_its_inputs():
    rng = np.random.default_rng(19)
    shape = (1, 4, 5, 6)
    total = rng.uniform(20.0, 40.0, shape)
    primary = rng.uniform(10.0, 30.0, shape)
    measured = rng.uniform(100.0, 200.0, shape)
    total_before, primary_before, measured_before = (
        total.copy(),
        primary.copy(),
        measured.copy(),
    )
    projector = _Projector(_penetrate(total, primary), ScoringRoutine.PENETRATE)
    ScatterCorrection(
        projector,
        measured=FakeData(measured),
        smoothing_fwhm_bins=2.5,
        smooth_across_views=True,
    ).estimate("image")
    np.testing.assert_array_equal(total, total_before)
    np.testing.assert_array_equal(primary, primary_before)
    np.testing.assert_array_equal(measured, measured_before)


def test_pixelwise_background_is_added_after_the_final_smoothing():
    rng = np.random.default_rng(23)
    shape = (1, 4, 5, 6)
    total = rng.uniform(20.0, 40.0, shape)
    primary = rng.uniform(10.0, 30.0, shape)
    measured = rng.uniform(100.0, 200.0, shape)
    background = rng.uniform(1.0, 3.0, shape)
    projector = _Projector(_penetrate(total, primary), ScoringRoutine.PENETRATE)

    result = (
        ScatterCorrection(
            projector,
            background=FakeData(background),
            measured=FakeData(measured),
            smoothing_fwhm_bins=2.5,
            smooth_across_views=True,
        )
        .estimate("image")
        .as_array()
    )
    additive = _smoothed_pixelwise(
        total - primary, total, measured, 2.5, across_views=True
    )

    np.testing.assert_allclose(result, background + additive, rtol=1e-5, atol=1e-5)
    assert not np.allclose(
        result,
        _gaussian_smooth(additive + background, 2.5, across_views=True),
        rtol=1e-4,
        atol=1e-4,
    )


def test_unsmoothed_pixelwise_scatter_keeps_a_signed_fraction():
    projector = _Projector(
        _penetrate([[[10.0, 10.0]]], [[[12.0, 10.0]]]),
        ScoringRoutine.PENETRATE,
    )
    result = (
        ScatterCorrection(projector, measured=FakeData([[[5.0, 5.0]]]))
        .estimate("image")
        .as_array()
    )
    np.testing.assert_allclose(result, [[[-1.0, 0.0]]])


def test_global_scatter_output_is_unchanged_by_the_pixelwise_estimator_change():
    rng = np.random.default_rng(29)
    shape = (1, 4, 5, 6)
    total = rng.uniform(20.0, 40.0, shape)
    primary = rng.uniform(10.0, 30.0, shape)
    projector = _Projector(_penetrate(total, primary), ScoringRoutine.PENETRATE)

    result = (
        ScatterCorrection(projector, smoothing_fwhm_bins=3.0, smooth_across_views=True)
        .estimate("image")
        .as_array()
    )
    np.testing.assert_allclose(
        result,
        _gaussian_smooth(total - primary, 3.0, across_views=True),
        rtol=1e-5,
        atol=1e-5,
    )


def test_pixelwise_pre_smoothing_reduces_the_ratio_bias_in_this_fixture():
    """A fixed-primary, noisy-scatter fixture only; not a general claim."""
    shape = (1, 1, 64, 64)
    true_scatter = np.full(shape, 2.0)
    primary = np.full(shape, 2.0)
    measured = np.full(shape, 100.0)
    truth = measured * true_scatter / (true_scatter + primary)
    fwhm_bins = 3.0

    legacy_sums = []
    smoothed_sums = []
    for seed in range(20):
        noisy_scatter = np.random.default_rng(seed).poisson(true_scatter * 4.0) / 4.0
        total = noisy_scatter + primary
        legacy_sums.append(
            _gaussian_smooth(measured * (noisy_scatter / total), fwhm_bins, True).sum()
        )
        projector = _Projector(_penetrate(total, primary), ScoringRoutine.PENETRATE)
        smoothed_sums.append(
            ScatterCorrection(
                projector,
                measured=FakeData(measured),
                smoothing_fwhm_bins=fwhm_bins,
                smooth_across_views=True,
            )
            .estimate("image")
            .as_array()
            .sum()
        )

    legacy_bias = (np.mean(legacy_sums) - truth.sum()) / truth.sum()
    smoothed_bias = (np.mean(smoothed_sums) - truth.sum()) / truth.sum()

    assert legacy_bias < -0.01
    assert abs(smoothed_bias) < abs(legacy_bias) / 5.0
