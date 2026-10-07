"""SIRF OSEM with an additive term that is refreshed between passes."""

from __future__ import annotations

from typing import Any

import sirf.STIR as sirf


def run_osem_with_corrections(
    acquisition_data: Any,
    acq_model: Any,
    initial_image: Any,
    updater: Any,
    num_subsets: int,
    subiterations_per_update: int,
    num_updates: int,
    prior: Any = None,
) -> Any:
    """Run OSEM in num_updates + 1 passes, refreshing the additive term in between.

    STIR reads the additive term when the objective is set up, so each pass
    builds a new objective and reconstructor. updater.schedule is not used.
    """
    image = initial_image.clone()
    for k in range(num_updates + 1):
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
