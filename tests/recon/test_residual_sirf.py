import numpy as np
import pytest

from tests.recon.fakes import FixedCorrection


pytestmark = pytest.mark.requires_sirf

ITERATIONS = 60
EVERY = 5


def _mlem(scene, model_factory, updater, every, num_updates):
    from simind_python_connector.recon.osem import run_osem_with_corrections

    return run_osem_with_corrections(
        scene.measured,
        model_factory,
        scene.x_true.get_uniform_copy(1.0),
        updater,
        num_subsets=1,
        subiterations_per_update=every,
        num_updates=num_updates,
    )


def _fixed_background(scene):
    from simind_python_connector.recon.updates import AdditiveUpdater, UpdateSchedule

    return AdditiveUpdater(
        FixedCorrection(scene.background), UpdateSchedule(every=1), scene.background
    )


def test_psf_residual_mlem_agrees_in_data_space(psf_scene):
    from simind_python_connector.recon.corrections import ResidualCorrection
    from simind_python_connector.recon.diagnostics import effective_objective
    from simind_python_connector.recon.updates import AdditiveUpdater, UpdateSchedule

    scene = psf_scene
    x_ref = _mlem(scene, scene.make_accurate, _fixed_background(scene), ITERATIONS, 0)
    x_fast = _mlem(scene, scene.make_fast, _fixed_background(scene), ITERATIONS, 0)

    correction = ResidualCorrection(scene.accurate.direct, scene.fast, scene.background)
    objectives = []
    estimate = correction.estimate

    def estimate_and_record(image):
        additive = estimate(image)
        objectives.append(effective_objective(scene.measured, correction))
        return additive

    correction.estimate = estimate_and_record
    updater = AdditiveUpdater(correction, UpdateSchedule(every=EVERY), scene.background)
    x_rc = _mlem(scene, scene.make_fast, updater, EVERY, ITERATIONS // EVERY - 1)

    ref, fast, rc = x_ref.as_array(), x_fast.as_array(), x_rc.as_array()
    image_gap_rc = np.linalg.norm(rc - ref)
    image_gap_fast = np.linalg.norm(fast - ref)
    proj_ref = scene.accurate.direct(x_ref).as_array()
    proj_rc = scene.accurate.direct(x_rc).as_array()
    data_gap = np.linalg.norm(proj_rc - proj_ref) / np.linalg.norm(proj_ref)
    print(
        f"image gap rc={image_gap_rc:.4g} fast={image_gap_fast:.4g}; "
        f"data gap {data_gap:.4f}; effective objective {objectives[0]:.6g} -> "
        f"{objectives[-1]:.6g} over {len(objectives)} updates"
    )

    assert [record.iteration for record in updater.history] == list(
        range(EVERY, ITERATIONS, EVERY)
    )
    assert len(objectives) == ITERATIONS // EVERY - 1
    assert np.isfinite(objectives).all()
    assert data_gap < 0.05
    assert objectives[-1] < objectives[0]


def test_residual_correction_rejects_a_fast_model_with_an_additive_term(psf_scene):
    from simind_python_connector.recon.corrections import ResidualCorrection

    scene = psf_scene
    model = scene.make_fast()
    model.set_up(scene.measured, scene.x_true)
    model.set_additive_term(scene.background)
    correction = ResidualCorrection(scene.accurate.direct, model, scene.background)
    with pytest.raises(Exception, match="not linear"):
        correction.estimate(scene.x_true)
