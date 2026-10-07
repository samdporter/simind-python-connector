import numpy as np
import pytest

from simind_python_connector.recon.updates import (
    AdditiveUpdater,
    UpdateRecord,
    UpdateSchedule,
)
from tests.recon.fakes import FakeData


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "schedule, due",
    [
        (UpdateSchedule(every=3), [3, 6, 9]),
        (UpdateSchedule(every=3, first=1), [1, 4, 7, 10]),
        (UpdateSchedule(every=3, first=0), [0, 3, 6, 9]),
        (UpdateSchedule(every=2, first=4, stop_at=9), [4, 6, 8]),
        (UpdateSchedule(every=1, stop_at=3), [1, 2]),
        (UpdateSchedule(every=5, first=6, stop_at=6), []),
    ],
)
def test_schedule_is_due(schedule, due):
    assert [i for i in range(11) if schedule.is_due(i)] == due


def test_schedule_rejects_every_below_one():
    with pytest.raises(ValueError, match="every must be >= 1"):
        UpdateSchedule(every=0)


class _SequenceCorrection:
    """Returns the given estimates in turn; last_scale counts calls times 10."""

    def __init__(self, *estimates):
        self.estimates = list(estimates)
        self.images = []
        self.last_scale = None

    def estimate(self, image):
        self.images.append(image)
        self.last_scale = 10.0 * len(self.images)
        return FakeData(self.estimates.pop(0))


def test_updater_starts_from_a_floored_copy_of_the_initial_additive():
    initial = FakeData([[[-1.0, 0.0, 2.0]]])
    updater = AdditiveUpdater(
        _SequenceCorrection(), UpdateSchedule(every=1), initial, floor=0.5
    )
    np.testing.assert_array_equal(updater.raw.as_array(), [[[-1.0, 0.0, 2.0]]])
    np.testing.assert_array_equal(updater.current.as_array(), [[[0.5, 0.5, 2.0]]])
    assert updater.history == []
    initial.array[:] = 7.0
    assert updater.raw.as_array()[0, 0, 0] == -1.0


def test_updater_damps_raw_and_floors_only_current():
    correction = _SequenceCorrection([[[4.0, -2.0]]], [[[0.0, 8.0]]])
    updater = AdditiveUpdater(
        correction,
        UpdateSchedule(every=1),
        FakeData([[[2.0, 2.0]]]),
        damping=0.5,
        floor=0.1,
    )

    updater.update_now(3, "image-3")
    np.testing.assert_allclose(updater.raw.as_array(), [[[3.0, 0.0]]])
    np.testing.assert_allclose(updater.current.as_array(), [[[3.0, 0.1]]])

    updater.update_now(6, "image-6")
    np.testing.assert_allclose(updater.raw.as_array(), [[[1.5, 4.0]]])
    np.testing.assert_allclose(updater.current.as_array(), [[[1.5, 4.0]]])
    assert correction.images == ["image-3", "image-6"]


def test_updater_without_damping_takes_the_estimate():
    updater = AdditiveUpdater(
        _SequenceCorrection([[[5.0, -1.0]]]),
        UpdateSchedule(every=1),
        FakeData([[[100.0, 100.0]]]),
    )
    updater.update_now(1, "image")
    np.testing.assert_allclose(updater.raw.as_array(), [[[5.0, -1.0]]])
    np.testing.assert_allclose(updater.current.as_array(), [[[5.0, 1e-5]]])


def test_updater_records_history():
    updater = AdditiveUpdater(
        _SequenceCorrection([[[1.0, -3.0, 2.0]]]),
        UpdateSchedule(every=1),
        FakeData(np.zeros((1, 1, 3))),
        floor=0.5,
    )
    updater.update_now(4, "image")
    (record,) = updater.history
    assert isinstance(record, UpdateRecord)
    assert record.iteration == 4
    assert record.scale == 10.0
    assert record.additive_sum == pytest.approx(3.5)  # sum of max(D, 0.5)
    assert record.additive_min == pytest.approx(-3.0)  # min of unfloored D
    assert record.seconds >= 0.0


def test_history_scale_is_none_for_corrections_without_last_scale():
    class _Plain:
        def estimate(self, image):
            return FakeData([[[1.0]]])

    updater = AdditiveUpdater(_Plain(), UpdateSchedule(every=1), FakeData([[[0.0]]]))
    updater.update_now(1, "image")
    assert updater.history[0].scale is None


def test_maybe_update_follows_the_schedule():
    correction = _SequenceCorrection([[[1.0]]], [[[2.0]]])
    updater = AdditiveUpdater(
        correction, UpdateSchedule(every=2, stop_at=5), FakeData([[[0.0]]])
    )
    done = [i for i in range(7) if updater.maybe_update(i, f"image-{i}")]
    assert done == [2, 4]
    assert [record.iteration for record in updater.history] == [2, 4]
    assert correction.images == ["image-2", "image-4"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"damping": 0.0}, "damping must be in"),
        ({"damping": 1.5}, "damping must be in"),
        ({"floor": 0.0}, "floor must be > 0"),
        ({"floor": -1.0}, "floor must be > 0"),
    ],
)
def test_updater_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AdditiveUpdater(
            _SequenceCorrection(),
            UpdateSchedule(every=1),
            FakeData([[[0.0]]]),
            **kwargs,
        )
