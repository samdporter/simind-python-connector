"""The shared protocol of the NEMA reconstruction tutorials (examples 10-15)."""

import json

import numpy as np
import pytest

from examples import _nema


pytestmark = pytest.mark.unit

_REAL_DETECT = _nema.detect_engine


@pytest.fixture(autouse=True)
def _fixed_engine(monkeypatch):
    monkeypatch.setattr(
        _nema, "detect_engine", lambda *args, **kwargs: "SIMIND V8.0 [sha256:test]"
    )


def test_quantization_loss_is_below_one_percent_at_the_default_budget():
    activity = np.full((4, 4, 4), 1.0)
    activity[1:2, 1:2, 1:2] = 8.0

    loss = _nema.quantization_loss(activity, _nema.DEFAULT_QUANTIZATION_SCALE)
    loss_with_four_times_the_histories = _nema.quantization_loss(
        activity, 4 * _nema.DEFAULT_QUANTIZATION_SCALE
    )

    assert loss < 0.01
    assert loss_with_four_times_the_histories < loss / 2


def test_source_map_never_exceeds_the_signed_range():
    activity = np.array([1.0, 2.0, 8.0])
    source = _nema.source_map(activity, 1.0)

    assert source.max() == 32767
    assert source.dtype == np.float64
    assert source.argmax() == activity.argmax()


def test_scaled_counts_is_linear_in_acquisition_time_and_non_negative():
    counts = np.array([0.0, 10.0, 100.0])

    np.testing.assert_allclose(_nema.scaled_counts(counts, 2.0), counts)
    np.testing.assert_allclose(_nema.scaled_counts(counts, 20.0), 10.0 * counts)
    np.testing.assert_allclose(
        _nema.scaled_counts(np.array([-5.0, 0.0]), 10.0), [0.0, 0.0]
    )


class _FakeExpectation:
    def __init__(self):
        self.calls = []

    def measured(self, template, seed, time_per_view_s):
        self.calls.append((template, seed, time_per_view_s))
        return {"seed": seed, "time": time_per_view_s}


def test_paired_realisations_samples_each_seed_once_at_the_same_budget():
    expectation = _FakeExpectation()

    realisations = list(
        _nema.paired_realisations(
            expectation, "template", seeds=(1, 2, 3), time_per_view_s=10.0
        )
    )

    assert [seed for seed, _ in realisations] == [1, 2, 3]
    assert expectation.calls == [
        ("template", 1, 10.0),
        ("template", 2, 10.0),
        ("template", 3, 10.0),
    ]
    assert {data["time"] for _, data in realisations} == {10.0}


class _Array:
    def __init__(self, array):
        self.array = array

    def as_array(self):
        return self.array


class _Template:
    def __init__(self):
        self.array = None

    def clone(self):
        return _Template()

    def fill(self, array):
        self.array = array


def test_measured_is_one_seeded_poisson_draw_and_preserves_the_time_scale():
    rng = np.random.default_rng(0)
    counts = rng.uniform(10.0, 100.0, size=(1, 4, 3, 5))
    expectation = _nema.Expectation(
        counts=counts,
        primary=np.zeros_like(counts),
        histories=1.0,
        quantization_loss=0.0,
        seconds=0.0,
        reused=False,
    )

    first = expectation.measured(_Template(), seed=7, time_per_view_s=20.0)
    second = expectation.measured(_Template(), seed=7, time_per_view_s=20.0)
    different = expectation.measured(_Template(), seed=8, time_per_view_s=20.0)

    np.testing.assert_array_equal(first.array, second.array)
    assert not np.array_equal(first.array, different.array)
    assert (first.array >= 0).all()
    assert first.array.sum() == pytest.approx(
        _nema.scaled_counts(counts, 20.0).sum(), rel=0.02
    )


class _FakeAdaptor:
    def __init__(self, calls):
        self.calls = calls
        self.activities = []

    def set_template(self, template):
        pass

    def set_source(self, image):
        pass

    def set_mu_map(self, image):
        pass

    def set_activity(self, activity, time):
        self.activities.append((activity, time))

    def run(self):
        self.calls.append("run")
        return {
            "all_interactions": _Array(np.array([[[[6.0, 4.0]]]])),
            "geom_coll_primary": _Array(np.array([[[[3.0, 1.0]]]])),
        }


class _FakePhantom:
    activity = np.array([[1.0, 2.0], [3.0, 8.0]])
    mu = np.zeros_like(activity)

    @property
    def total_mbq(self):
        return float(self.activity.sum())


def test_expectation_is_simulated_once_and_then_reused(tmp_path, monkeypatch):
    calls = []

    def make_adaptor(*args, **kwargs):
        return _FakeAdaptor(calls)

    monkeypatch.setattr(_nema, "make_adaptor", make_adaptor)
    monkeypatch.setattr(_nema, "to_sirf_image", lambda array: array)

    first = _nema.simulate_expectation(
        _FakePhantom(), template=None, cache_dir=tmp_path, histories=4
    )
    second = _nema.simulate_expectation(
        _FakePhantom(), template=None, cache_dir=tmp_path, histories=4
    )

    assert calls == ["run"]
    assert first.reused is False
    assert second.reused is True
    assert first.counts.sum() == second.counts.sum() == 10.0
    assert first.true_scatter.sum() == 6.0
    cache = _nema.expectation_cache_file(
        _FakePhantom(), cache_dir=tmp_path, histories=4
    )
    assert json.loads(str(np.load(cache)["meta"]))["settings"]["histories"] == 4


def test_expectation_preserves_the_total_activity_and_reference_time(
    tmp_path, monkeypatch
):
    created = []

    def make_adaptor(*args, **kwargs):
        adaptor = _FakeAdaptor([])
        created.append(adaptor)
        return adaptor

    monkeypatch.setattr(_nema, "make_adaptor", make_adaptor)
    monkeypatch.setattr(_nema, "to_sirf_image", lambda array: array)

    _nema.simulate_expectation(
        _FakePhantom(), template=None, cache_dir=tmp_path, histories=4
    )

    assert created[0].activities == [(14.0, _nema.REFERENCE_TIME_S)]


def test_mean_and_sem_is_over_independent_realisations():
    values = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]

    mean, sem = _nema.mean_and_sem(values)

    np.testing.assert_allclose(mean, [2.0, 4.0])
    np.testing.assert_allclose(sem, np.std(values, axis=0, ddof=1) / np.sqrt(3))


def test_mean_and_std_is_the_sample_spread_over_realisations():
    values = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]

    mean, spread = _nema.mean_and_std(values)

    np.testing.assert_allclose(mean, [2.0, 4.0])
    np.testing.assert_allclose(spread, np.std(values, axis=0, ddof=1))


def test_detect_engine_reads_the_change_banner_and_hashes_the_binary(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_nema, "detect_engine", _REAL_DETECT)
    binary = tmp_path / "simind"
    binary.write_bytes(b"fake simind binary")
    binary.chmod(0o755)
    change = tmp_path / "change"
    change.write_text(
        "#!/bin/sh\necho ' C H A N G E: Main page for SIMIND version V9.1 '\n"
    )
    change.chmod(0o755)

    engine = _nema.detect_engine(str(binary))

    assert engine.startswith("SIMIND V9.1 [sha256:")
    assert _nema.engine_label(engine) == "SIMIND v9"


def test_cache_key_includes_the_engine(tmp_path):
    first = _nema.expectation_cache_file(
        _FakePhantom(), cache_dir=tmp_path, engine="SIMIND V8.0 [sha256:a]"
    )
    second = _nema.expectation_cache_file(
        _FakePhantom(), cache_dir=tmp_path, engine="SIMIND V8.0 [sha256:b]"
    )

    assert first != second


def test_reused_cache_rejects_metadata_from_another_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(_nema, "make_adaptor", lambda *a, **k: _FakeAdaptor([]))
    monkeypatch.setattr(_nema, "to_sirf_image", lambda array: array)
    engine = "SIMIND V8.0 [sha256:a]"
    kwargs = dict(template=None, cache_dir=tmp_path, histories=4, engine=engine)
    _nema.simulate_expectation(_FakePhantom(), **kwargs)
    cache = _nema.expectation_cache_file(
        _FakePhantom(), cache_dir=tmp_path, histories=4, engine=engine
    )
    data = np.load(cache)
    meta = json.loads(str(data["meta"]))
    meta["engine"] = "SIMIND V8.0 [sha256:b]"
    np.savez_compressed(
        cache, counts=data["counts"], primary=data["primary"], meta=json.dumps(meta)
    )

    with pytest.raises(RuntimeError, match="produced by"):
        _nema.simulate_expectation(_FakePhantom(), **kwargs)


def test_reused_cache_rejects_metadata_with_other_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(_nema, "make_adaptor", lambda *a, **k: _FakeAdaptor([]))
    monkeypatch.setattr(_nema, "to_sirf_image", lambda array: array)
    engine = "SIMIND V8.0 [sha256:a]"
    kwargs = dict(template=None, cache_dir=tmp_path, histories=4, engine=engine)
    _nema.simulate_expectation(_FakePhantom(), **kwargs)
    cache = _nema.expectation_cache_file(
        _FakePhantom(), cache_dir=tmp_path, histories=4, engine=engine
    )
    data = np.load(cache)
    meta = json.loads(str(data["meta"]))
    meta["settings"]["histories"] = 99
    np.savez_compressed(
        cache, counts=data["counts"], primary=data["primary"], meta=json.dumps(meta)
    )

    with pytest.raises(RuntimeError, match="settings do not match"):
        _nema.simulate_expectation(_FakePhantom(), **kwargs)
