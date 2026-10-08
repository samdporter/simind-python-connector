import numpy as np
import pytest


pytestmark = [pytest.mark.requires_sirf, pytest.mark.requires_cil]


def test_residual_correction_through_the_cil_callback(psf_scene):
    from cil.optimisation.algorithms import ISTA
    from cil.optimisation.functions import IndicatorBox, SumFunction
    from cil.optimisation.utilities import AdaptiveSensitivity

    from simind_python_connector.recon.cil import (
        CorrectionCallback,
        build_subset_objectives,
    )
    from simind_python_connector.recon.corrections import ResidualCorrection
    from simind_python_connector.recon.diagnostics import effective_objective
    from simind_python_connector.recon.updates import AdditiveUpdater, UpdateSchedule

    scene = psf_scene
    initial = scene.x_true.get_uniform_copy(1.0)
    objectives = build_subset_objectives(
        scene.measured, scene.background, scene.make_fast, 4, initial
    )
    correction = ResidualCorrection(scene.accurate.direct, scene.fast, scene.background)
    updater = AdditiveUpdater(
        correction, UpdateSchedule(every=5, stop_at=21), scene.background
    )
    recorded = []

    def on_update(algorithm, updater):
        recorded.append(effective_objective(scene.measured, correction))

    algorithm = ISTA(
        initial=initial,
        f=SumFunction(*objectives.functions),
        g=IndicatorBox(lower=0),
        step_size=1.0,
        preconditioner=AdaptiveSensitivity(scene.fast, max_iterations=100),
    )
    algorithm.run(25, callbacks=[CorrectionCallback(updater, objectives, on_update)])
    print(f"effective objective at updates 5, 10, 15, 20: {recorded}")

    assert [record.iteration for record in updater.history] == [5, 10, 15, 20]
    for function, views in zip(objectives.functions, objectives.view_indices):
        np.testing.assert_allclose(
            function.function.eta.as_array(),
            updater.current.get_subset(views).as_array(),
        )
    assert len(recorded) == len(updater.history)
    assert all(later < earlier for earlier, later in zip(recorded, recorded[1:]))
