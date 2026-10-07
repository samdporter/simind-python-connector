import numpy as np
import pytest

from tests.recon.fakes import FixedCorrection


pytestmark = [pytest.mark.requires_sirf, pytest.mark.requires_cil]


def _objectives(sirf_scene, num_subsets, mode="staggered"):
    from simind_python_connector.recon.cil import build_subset_objectives

    measured, image, make_model = sirf_scene
    return build_subset_objectives(
        measured,
        measured.get_uniform_copy(0),
        make_model,
        num_subsets,
        image,
        mode=mode,
    )


def _fresh(objectives, full_additive):
    from cil.optimisation.functions import KullbackLeibler, OperatorCompositionFunction

    array = full_additive.as_array()
    fresh = []
    for b, views, operator in zip(
        objectives.data, objectives.view_indices, objectives.linear_models
    ):
        eta = b.clone()
        eta.fill(np.take(array, list(views), axis=-2))
        fresh.append(
            OperatorCompositionFunction(KullbackLeibler(b=b, eta=eta), operator)
        )
    return fresh


@pytest.mark.parametrize("mode", ["staggered", "sequential"])
@pytest.mark.parametrize("num_subsets", [1, 3])
def test_set_subset_additive_matches_freshly_built_objectives(
    sirf_scene, num_subsets, mode, monkeypatch
):
    import sirf.STIR as sirf
    from sirf.contrib.partitioner import partitioner

    from simind_python_connector.recon.cil import (
        build_subset_objectives,
        set_subset_additive,
    )

    measured, image, make_model = sirf_scene
    num_views = int(measured.as_array().shape[-2])
    partition_indices = partitioner.partition_indices
    seen_view_counts = []

    def recording_partition_indices(num_batches, indices, *args, **kwargs):
        if isinstance(indices, int):
            assert type(indices) is int
            assert indices == num_views
            seen_view_counts.append(indices)
        return partition_indices(num_batches, indices, *args, **kwargs)

    monkeypatch.setattr(partitioner, "partition_indices", recording_partition_indices)
    sirf.AcquisitionData.set_storage_scheme("file")
    try:
        file_measured = measured.clone()
        file_zero = measured.get_uniform_copy(0)
        file_additive = measured.clone()
        file_additive.fill(measured.as_array() * 0.1 + 0.5)
        objectives = build_subset_objectives(
            file_measured,
            file_zero,
            make_model,
            num_subsets,
            image,
            mode=mode,
        )
    finally:
        sirf.AcquisitionData.set_storage_scheme("memory")

    np.testing.assert_array_equal(file_measured.as_array(), measured.as_array())
    np.testing.assert_array_equal(file_zero.as_array(), 0.0)

    assert seen_view_counts
    assert len(objectives.functions) == num_subsets
    assert len(objectives.data) == num_subsets
    assert len(objectives.view_indices) == num_subsets
    assert len(objectives.linear_models) == num_subsets
    assert sorted(int(v) for views in objectives.view_indices for v in views) == list(
        range(num_views)
    )

    for data, views in zip(objectives.data, objectives.view_indices):
        expected = np.take(file_measured.as_array(), list(views), axis=-2)
        np.testing.assert_array_equal(data.as_array(), expected)

    for function in objectives.functions:
        np.testing.assert_allclose(function.function.eta.as_array(), 1e-5)

    snapshot = file_additive.as_array().copy()
    sirf.AcquisitionData.set_storage_scheme("file")
    try:
        set_subset_additive(objectives, file_additive)
    finally:
        sirf.AcquisitionData.set_storage_scheme("memory")

    np.testing.assert_array_equal(file_additive.as_array(), snapshot)
    for function, views in zip(objectives.functions, objectives.view_indices):
        np.testing.assert_allclose(
            function.function.eta.as_array(),
            np.take(snapshot, list(views), axis=-2),
        )

    x = image * 0.8 + 0.1
    for function, fresh in zip(objectives.functions, _fresh(objectives, file_additive)):
        assert function(x) == pytest.approx(fresh(x), rel=1e-6)
        actual_gradient = function.gradient(x).as_array()
        fresh_gradient = fresh.gradient(x).as_array()
        gradient_scale = max(1.0, float(np.abs(fresh_gradient).max()))
        np.testing.assert_allclose(
            actual_gradient,
            fresh_gradient,
            rtol=1e-4,
            atol=1e-6 * gradient_scale,
        )


@pytest.mark.parametrize("kind", ["svrg", "saga"])
def test_refresh_makes_the_next_gradient_the_full_gradient(sirf_scene, kind):
    from cil.optimisation.functions import SAGAFunction, SumFunction, SVRGFunction
    from cil.optimisation.utilities import Sampler

    from simind_python_connector.recon.cil import (
        refresh_stochastic_state,
        set_subset_additive,
    )

    measured, image, _ = sirf_scene
    objectives = _objectives(sirf_scene, 3)
    sampler = Sampler.sequential(3)
    if kind == "svrg":
        f = SVRGFunction(
            objectives.functions, sampler=sampler, snapshot_update_interval=100
        )
    else:
        f = SAGAFunction(objectives.functions, sampler=sampler)
    x0, x1 = image * 0.5 + 0.1, image * 0.9 + 0.2

    refresh_stochastic_state(f, x0)  # state built with the initial additive
    set_subset_additive(objectives, measured * 0.1 + 0.5)
    expected = SumFunction(*objectives.functions).gradient(x1).as_array()
    stale = f.gradient(x1).as_array()
    refresh_stochastic_state(f, x1)
    fresh = f.gradient(x1).as_array()

    size = np.abs(expected).max()
    assert np.abs(stale - expected).max() > 1e-3 * size  # the refresh matters
    np.testing.assert_allclose(fresh, expected, rtol=1e-4, atol=1e-6 * size)


def test_callback_refreshes_the_objectives_during_ista(sirf_scene):
    from cil.optimisation.algorithms import ISTA
    from cil.optimisation.functions import IndicatorBox, SumFunction

    from simind_python_connector.recon.cil import CorrectionCallback
    from simind_python_connector.recon.updates import AdditiveUpdater, UpdateSchedule

    measured, image, _ = sirf_scene
    objectives = _objectives(sirf_scene, 3)
    fixed = measured * 0.1 + 0.5
    correction = FixedCorrection(fixed)
    updater = AdditiveUpdater(
        correction,
        UpdateSchedule(every=5, stop_at=6),
        measured.get_uniform_copy(0),
        floor=1e-5,
    )
    algorithm = ISTA(
        initial=image.get_uniform_copy(1.0),
        f=SumFunction(*objectives.functions),
        g=IndicatorBox(lower=0),
        step_size=1e-4,
    )
    callback_iterations = []

    class _RecordingCallback(CorrectionCallback):
        def __call__(self, algorithm):
            callback_iterations.append(algorithm.iteration)
            if algorithm.iteration == 0:
                np.testing.assert_array_equal(algorithm.solution.as_array(), 1.0)
            super().__call__(algorithm)

    algorithm.run(8, callbacks=[_RecordingCallback(updater, objectives)])

    assert callback_iterations == list(range(9))

    assert [record.iteration for record in updater.history] == [5]
    assert correction.calls == 1
    floored = fixed.maximum(1e-5)
    for function, views in zip(objectives.functions, objectives.view_indices):
        np.testing.assert_allclose(
            function.function.eta.as_array(),
            np.take(floored.as_array(), list(views), axis=-2),
        )
    x = algorithm.solution
    fresh = SumFunction(*_fresh(objectives, floored))
    assert SumFunction(*objectives.functions)(x) == pytest.approx(fresh(x), rel=1e-6)


def test_prime_installs_an_estimate_before_the_run(sirf_scene):
    from simind_python_connector.recon.cil import CorrectionCallback
    from simind_python_connector.recon.updates import AdditiveUpdater, UpdateSchedule

    measured, image, _ = sirf_scene
    objectives = _objectives(sirf_scene, 3)
    fixed = measured * 0.1 + 0.5
    updater = AdditiveUpdater(
        FixedCorrection(fixed), UpdateSchedule(every=100), measured.get_uniform_copy(0)
    )
    CorrectionCallback(updater, objectives).prime(image)

    assert [record.iteration for record in updater.history] == [0]
    views = objectives.view_indices[1]
    np.testing.assert_allclose(
        objectives.functions[1].function.eta.as_array(),
        np.take(fixed.as_array(), list(views), axis=-2),
    )
