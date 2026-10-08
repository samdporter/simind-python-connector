"""SIRF OSEM with an additive term that is refreshed between passes."""

from __future__ import annotations

from typing import Any, Callable

import sirf.STIR as sirf


def run_osem_with_corrections(
    acquisition_data: Any,
    acq_model_factory: Callable[[], Any],
    initial_image: Any,
    updater: Any,
    num_subsets: int,
    subiterations_per_update: int,
    num_updates: int,
    prior: Any = None,
) -> Any:
    """Run OSEM in num_updates + 1 passes, refreshing the additive term in between.

    acq_model_factory() must return a fresh, configured, not-yet-set-up SIRF
    acquisition model each time, with independent model/matrix state and
    the same linear operator settings. The current additive is installed
    before reconstructor setup. On SIRF 3.10.1 / STIR 6.4, reusing a model
    for another reconstruction can produce an all-zero image.

    Each pass builds a fresh model, objective and reconstructor, continuing
    from the preceding pass's image. initial_image is cloned and not modified.
    updater.update_now is called between passes at completed subiterations;
    updater.schedule is not used.
    """
    # Each pass rebuilds a reconstructor that restarts at subset 0, so a
    # remainder would use some subsets more often and others never.
    if subiterations_per_update % num_subsets != 0:
        raise ValueError(
            f"subiterations_per_update ({subiterations_per_update}) must be "
            f"a multiple of num_subsets ({num_subsets})"
        )

    image = initial_image.clone()
    for k in range(num_updates + 1):
        acq_model = acq_model_factory()
        acq_model.set_additive_term(updater.current)
        objective = sirf.make_Poisson_loglikelihood(acquisition_data)
        objective.set_acquisition_model(acq_model)
        objective.set_num_subsets(num_subsets)
        if prior is not None:
            objective.set_prior(prior)
        reconstructor = sirf.OSMAPOSLReconstructor()
        reconstructor.set_objective_function(objective)
        reconstructor.set_num_subsets(num_subsets)
        reconstructor.set_num_subiterations(subiterations_per_update)
        reconstructor.set_input(acquisition_data)
        reconstructor.set_up(image)
        reconstructor.reconstruct(image)
        if k < num_updates:
            updater.update_now((k + 1) * subiterations_per_update, image)
    return image
