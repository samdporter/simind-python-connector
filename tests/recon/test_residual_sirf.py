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


def test_interleaved_spectub_residual_matches_fresh_projection(psf_scene):
    """The shared cache-off model factory keeps OSEM residuals consistent."""
    import sirf.STIR as sirf

    from simind_python_connector.recon.corrections import ResidualCorrection
    from simind_python_connector.recon.osem import run_osem_with_corrections
    from simind_python_connector.recon.updates import AdditiveUpdater, UpdateSchedule

    scene = psf_scene
    mu_image = scene.x_true.get_uniform_copy(0.0)

    def make_fast():
        matrix = sirf.SPECTUBMatrix()
        matrix.set_attenuation_image(mu_image)
        matrix.set_keep_all_views_in_cache(False)
        matrix.set_resolution_model(0.0, 0.0, False)
        return sirf.AcquisitionModelUsingMatrix(matrix)

    fast = make_fast()
    accurate = scene.make_accurate()
    fast.set_up(scene.measured, scene.x_true)
    accurate.set_up(scene.measured, scene.x_true)

    correction = ResidualCorrection(accurate.direct, fast, scene.background)
    residual = correction.estimate
    deltas = []

    def estimate_and_compare(image):
        additive = residual(image)
        fresh = scene.background + accurate.direct(image) - fast.direct(image)
        delta = np.abs(additive.as_array() - fresh.as_array())
        deltas.append(float(delta.max()))
        return additive

    correction.estimate = estimate_and_compare
    updater = AdditiveUpdater(correction, UpdateSchedule(every=2), scene.background)
    run_osem_with_corrections(
        scene.measured,
        make_fast,
        scene.x_true.get_uniform_copy(1.0),
        updater,
        num_subsets=1,
        subiterations_per_update=2,
        num_updates=2,
    )

    assert len(deltas) == 2
    assert all(delta < 2e-6 for delta in deltas)
