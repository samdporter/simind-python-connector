#!/usr/bin/env python
"""
MC scatter in the forward model with CIL.

The same data as example 09, reconstructed with CIL's ISTA over subset
Kullback-Leibler objectives. The AdaptiveSensitivity preconditioner with a
unit step makes each ISTA step an MLEM update. A CorrectionCallback refreshes
the SIMIND scatter estimate every 10 iterations and swaps it into the
objectives. Installed CIL calls back at iteration 0 before the first update,
then after later updates. Passing stop_at=ITERATIONS (an exclusive bound)
records refreshes at 10, 20 and 30 only, each consumed by a later iteration.

Tiny on purpose: 32^3 voxels, 30 views, NN = 1. Needs SIRF, CIL and SIMIND.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from _mc_scatter_common import (
    contrast_error,
    fast_model_factory,
    image,
    phantom,
    print_updates,
    scatter_updater,
    simulate_measured,
    template,
)
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import IndicatorBox, SumFunction
from cil.optimisation.utilities import AdaptiveSensitivity

from simind_python_connector.recon.cil import (
    CorrectionCallback,
    build_subset_objectives,
)


NUM_SUBSETS = 5
ITERATIONS = 40
UPDATE_EVERY = 10


def _ista(objectives, initial, full_model, callbacks):
    algorithm = ISTA(
        initial=initial,
        f=SumFunction(*objectives.functions),
        g=IndicatorBox(lower=0),
        step_size=1.0,
        preconditioner=AdaptiveSensitivity(full_model, max_iterations=ITERATIONS + 1),
    )
    algorithm.run(ITERATIONS, callbacks=callbacks)
    return algorithm.solution


def main() -> None:
    if shutil.which("simind") is None:
        raise RuntimeError("SIMIND executable not found in PATH.")
    output_dir = Path("output/mc_scatter_cil")
    output_dir.mkdir(parents=True, exist_ok=True)

    activity, mu = phantom()
    activity_image, mu_image = image(activity), image(mu)
    template_data = template(output_dir)
    measured = simulate_measured(output_dir, template_data, activity_image, mu_image)

    make_model = fast_model_factory(mu_image)
    full_model = make_model()
    full_model.set_up(measured, activity_image)
    initial = activity_image.get_uniform_copy(1.0)
    zero = measured.get_uniform_copy(0)

    baseline_objectives = build_subset_objectives(
        measured, zero, make_model, NUM_SUBSETS, initial
    )
    baseline = _ista(baseline_objectives, initial, full_model, callbacks=[])

    objectives = build_subset_objectives(
        measured, zero, make_model, NUM_SUBSETS, initial
    )
    updater = scatter_updater(
        output_dir,
        template_data,
        mu_image,
        measured,
        full_model,
        UPDATE_EVERY,
        stop_at=ITERATIONS,
    )
    callback = CorrectionCallback(updater, objectives)
    corrected = _ista(objectives, initial, full_model, callbacks=[callback])

    print_updates(updater)
    before = contrast_error(baseline, activity)
    after = contrast_error(corrected, activity)
    print(f"hot-sphere contrast error, no scatter correction: {before:.1%}")
    print(f"hot-sphere contrast error, SIMIND scatter:        {after:.1%}")


if __name__ == "__main__":
    main()
