"""Projection-metric helpers used by the worked examples."""

from types import SimpleNamespace

import numpy as np
import pytest

from simind_python_connector.utils import projection_metrics as helpers


@pytest.mark.unit
def test_require_output_keys_names_the_missing_outputs():
    outputs = {"tot_w1": object()}
    assert helpers.require_output_keys(outputs, "tot_w1") == outputs
    with pytest.raises(KeyError, match="sca_w1"):
        helpers.require_output_keys(outputs, "tot_w1", "sca_w1")


@pytest.mark.unit
def test_require_output_keys_lists_what_was_written():
    with pytest.raises(KeyError, match="tot_w1"):
        helpers.require_output_keys({"pri_w1": object()}, "tot_w1")


@pytest.mark.unit
def test_symmetric_limits_are_zero_centred_on_the_largest_magnitude():
    vmin, vmax = helpers.symmetric_limits(np.array([-2.0, 1.0]), np.array([0.5, 3.0]))
    assert (vmin, vmax) == (-3.0, 3.0)


@pytest.mark.unit
def test_symmetric_limits_of_zeros_stay_zero():
    assert helpers.symmetric_limits(np.zeros((2, 2))) == (0.0, 0.0)


@pytest.mark.unit
def test_symmetric_limits_need_an_image():
    with pytest.raises(ValueError, match="at least one"):
        helpers.symmetric_limits()


@pytest.mark.unit
def test_normalised_rms_difference_is_zero_for_identical_images():
    image = np.arange(12.0).reshape(3, 4)
    assert helpers.normalised_rms_difference(image, image) == 0.0


@pytest.mark.unit
def test_normalised_rms_difference_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        helpers.normalised_rms_difference(np.zeros(3), np.zeros(4))


@pytest.mark.unit
def test_normalised_rms_difference_reports_a_known_offset():
    # The offset is 2 and the mean magnitude of the pair is 11.
    assert helpers.normalised_rms_difference(
        np.full((2, 2), 10.0), np.full((2, 2), 12.0)
    ) == pytest.approx(2.0 / 11.0)


@pytest.mark.unit
def test_projection_view0_accepts_three_and_four_dimensions():
    three = np.arange(2 * 3 * 4.0).reshape(2, 3, 4)
    assert np.array_equal(helpers.projection_view0(three), three[0])
    four = np.arange(2 * 3 * 4 * 5.0).reshape(2, 3, 4, 5)
    assert np.array_equal(helpers.projection_view0(four), four[0, :, 0, :])


@pytest.mark.unit
def test_projection_view0_rejects_other_dimensions():
    with pytest.raises(ValueError, match="3 or 4 dimensions"):
        helpers.projection_view0(np.zeros((2, 3)))


@pytest.mark.unit
def test_tew_estimate_scales_each_side_window_to_the_peak_width():
    lower = np.array([10.0, 20.0])
    upper = np.array([2.0, 4.0])
    estimate = helpers.tew_scatter_estimate(
        lower,
        upper,
        lower_width_kev=20.0,
        photopeak_width_kev=40.0,
        upper_width_kev=10.0,
    )
    # lower * 40/(2*20) + upper * 40/(2*10) = lower + 2 * upper.
    assert estimate.tolist() == [14.0, 28.0]


@pytest.mark.unit
def test_tew_rejects_non_positive_widths():
    with pytest.raises(ValueError, match="positive"):
        helpers.tew_scatter_estimate(
            np.zeros(1),
            np.zeros(1),
            lower_width_kev=0.0,
            photopeak_width_kev=20.0,
            upper_width_kev=20.0,
        )


@pytest.mark.unit
def test_tew_correction_subtracts_the_estimate_and_clips_at_zero():
    photopeak = np.array([10.0, 2.0])
    lower = np.array([2.0, 3.0])
    upper = np.array([0.0, 0.0])
    estimate, corrected, count_fraction, pixel_fraction = helpers.tew_correction(
        photopeak,
        lower,
        upper,
        lower_width_kev=20.0,
        photopeak_width_kev=40.0,
        upper_width_kev=20.0,
    )
    assert estimate.tolist() == [2.0, 3.0]
    assert corrected.tolist() == [8.0, 0.0]
    # The second pixel loses one count to clipping.
    assert count_fraction == pytest.approx(1.0 / 12.0)
    assert pixel_fraction == pytest.approx(0.5)


@pytest.mark.unit
def test_tew_correction_without_clipping_keeps_every_count():
    photopeak = np.full((3, 3), 10.0)
    estimate, corrected, count_fraction, pixel_fraction = helpers.tew_correction(
        photopeak,
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        lower_width_kev=20.0,
        photopeak_width_kev=20.0,
        upper_width_kev=20.0,
    )
    assert np.all(estimate == 0.0)
    assert np.array_equal(corrected, photopeak)
    assert count_fraction == 0.0
    assert pixel_fraction == 0.0


@pytest.mark.unit
def test_source_weight_sum_matches_the_written_integer_weights():
    source = np.array([0.8, 0.8, 2.0])
    # The maximum maps to 500, so 0.8 maps to 200 and the sum is 600.
    assert helpers.source_weight_sum(source, 500 / 32767) == 900


@pytest.mark.unit
def test_source_weight_sum_of_an_empty_source_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        helpers.source_weight_sum(np.array([]), 1.0)


@pytest.mark.unit
def test_zero_history_fraction_counts_the_weight_sent_to_zero():
    source = np.array([1.0, 1.0, 1.0, 1.0])
    assert helpers.zero_history_fraction(source, 1.0) == 0.0
    # The default scale gives a maximum weight of 500; the small voxel rounds
    # to zero and loses its share of the source weight.
    source = np.array([1.0, 1.0, 1.0, 1e-4])
    assert helpers.zero_history_fraction(source, 500 / 32767) == pytest.approx(
        1e-4 / 3.0001
    )


@pytest.mark.unit
def test_zero_history_fraction_rejects_an_out_of_range_scale():
    with pytest.raises(ValueError, match="quantization_scale"):
        helpers.zero_history_fraction(np.ones(2), 0.0)


@pytest.mark.unit
def test_count_preservation_error_compares_components_with_the_total():
    def result(value):
        return SimpleNamespace(projection=np.array([[[value]]]))

    outputs = {
        "tot_w1": result(10.0),
        "pri_w1": result(7.0),
        "sca_w1": result(3.0),
    }
    assert (
        helpers.count_preservation_error(outputs, "tot_w1", "pri_w1", "sca_w1") == 0.0
    )
    assert helpers.count_preservation_error(
        outputs, "tot_w1", "pri_w1"
    ) == pytest.approx(-0.3)
