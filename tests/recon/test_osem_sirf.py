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
    updater = AdditiveUpdater(
        FixedCorrection(fixed),
        UpdateSchedule(every=1),
        measured.get_uniform_copy(0),
        floor=1e-3,
    )
    model = make_model()
    seen = []
    set_additive_term = model.set_additive_term

    def spy(term):
        seen.append(term.as_array().copy())
        set_additive_term(term)

    model.set_additive_term = spy
    initial = image.get_uniform_copy(1.0)

    objectives = []
    make_objective = sirf.make_Poisson_loglikelihood

    def recording_make_objective(data):
        objective = make_objective(data)
        objectives.append(objective)
        return objective

    monkeypatch.setattr(sirf, "make_Poisson_loglikelihood", recording_make_objective)
    result = run_osem_with_corrections(
        measured,
        model,
        initial,
        updater,
        num_subsets=3,
        subiterations_per_update=3,
        num_updates=1,
    )

    assert len(objectives) == 2
    assert objectives[0] is not objectives[1]
    assert len(seen) == 2
    np.testing.assert_allclose(seen[0], 1e-3)
    np.testing.assert_allclose(seen[1], fixed.maximum(1e-3).as_array())
    assert [record.iteration for record in updater.history] == [3]
    assert np.isfinite(result.as_array()).all() and result.as_array().max() > 0
    np.testing.assert_array_equal(initial.as_array(), 1.0)
