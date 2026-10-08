import numpy as np
import pytest

from tests.recon.fakes import FixedCorrection


pytestmark = pytest.mark.requires_sirf


def test_run_osem_with_corrections_sets_each_additive_term(sirf_scene, monkeypatch):
    import sirf.STIR as sirf

    from simind_python_connector.recon.osem import run_osem_with_corrections
    from simind_python_connector.recon.updates import AdditiveUpdater, UpdateSchedule

    measured, image, make_model = sirf_scene
    fixed = measured * 0.1 + 0.5
    correction = FixedCorrection(fixed)
    updater = AdditiveUpdater(
        correction,
        UpdateSchedule(every=1),
        measured.get_uniform_copy(0),
        floor=1e-3,
    )
    models = []
    seen = []

    def recording_model_factory():
        model = make_model()
        models.append(model)
        set_additive_term = model.set_additive_term

        def spy(term):
            seen.append((model, term.as_array().copy()))
            set_additive_term(term)

        model.set_additive_term = spy
        return model

    initial = image.get_uniform_copy(1.0)
    objectives = []
    objective_models = []
    make_objective = sirf.make_Poisson_loglikelihood

    def recording_make_objective(data):
        objective = make_objective(data)
        objectives.append(objective)
        set_acquisition_model = objective.set_acquisition_model

        def recording_set_acquisition_model(model):
            objective_models.append(model)
            set_acquisition_model(model)

        objective.set_acquisition_model = recording_set_acquisition_model
        return objective

    monkeypatch.setattr(sirf, "make_Poisson_loglikelihood", recording_make_objective)
    num_updates = 1
    result = run_osem_with_corrections(
        measured,
        recording_model_factory,
        initial,
        updater,
        num_subsets=3,
        subiterations_per_update=3,
        num_updates=num_updates,
    )

    expected_passes = num_updates + 1
    assert len(models) == expected_passes
    assert len(objectives) == expected_passes
    assert len(objective_models) == expected_passes
    assert len(seen) == expected_passes
    assert len({id(model) for model in models}) == expected_passes
    assert len({id(objective) for objective in objectives}) == expected_passes
    for model, objective_model, (additive_model, _) in zip(
        models, objective_models, seen
    ):
        assert objective_model is model
        assert additive_model is model
    np.testing.assert_allclose(seen[0][1], 1e-3)
    np.testing.assert_allclose(seen[1][1], fixed.maximum(1e-3).as_array())
    assert correction.calls == num_updates
    assert [record.iteration for record in updater.history] == [3]
    assert np.isfinite(result.as_array()).all() and result.as_array().max() > 0
    np.testing.assert_array_equal(initial.as_array(), 1.0)


def test_run_osem_with_corrections_rejects_non_multiple_subiterations(sirf_scene):
    from simind_python_connector.recon.osem import run_osem_with_corrections
    from simind_python_connector.recon.updates import AdditiveUpdater, UpdateSchedule

    measured, image, _ = sirf_scene
    fixed = measured * 0.1 + 0.5
    updater = AdditiveUpdater(
        FixedCorrection(fixed),
        UpdateSchedule(every=1),
        measured.get_uniform_copy(0),
    )

    def make_model():
        raise AssertionError("model factory called")

    with pytest.raises(ValueError, match="multiple of"):
        run_osem_with_corrections(
            measured,
            make_model,
            image.get_uniform_copy(1.0),
            updater,
            num_subsets=5,
            subiterations_per_update=3,
            num_updates=1,
        )
