#!/usr/bin/env python
"""
MC scatter in the forward model with SIRF OSEM.

1. Simulate "measured" data with SIMIND (PENETRATE, known activity) in a
   template geometry and add Poisson noise.
2. Reconstruct it with OSEM twice: without scatter correction, and with a
   SIMIND scatter estimate (all events minus the geometric primary, scaled to
   the fast model) refreshed every 2 epochs.
3. Print the hot-sphere contrast error of both against the truth.

Tiny on purpose: 32^3 voxels, 30 views, NN = 1. Needs SIRF and SIMIND.
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

from simind_python_connector.recon import AdditiveUpdater, UpdateSchedule
from simind_python_connector.recon.osem import run_osem_with_corrections


NUM_SUBSETS = 5
EPOCHS_PER_UPDATE = 2
NUM_UPDATES = 3


def main() -> None:
    if shutil.which("simind") is None:
        raise RuntimeError("SIMIND executable not found in PATH.")
    output_dir = Path("output/mc_scatter_osem")
    output_dir.mkdir(parents=True, exist_ok=True)

    activity, mu = phantom()
    activity_image, mu_image = image(activity), image(mu)
    template_data = template(output_dir)
    measured = simulate_measured(output_dir, template_data, activity_image, mu_image)

    make_model = fast_model_factory(mu_image)
    full_model = make_model()
    full_model.set_up(measured, activity_image)
    initial = activity_image.get_uniform_copy(1.0)
    subiterations = EPOCHS_PER_UPDATE * NUM_SUBSETS

    # One pass with the same number of subiterations and a zero additive term;
    # with num_updates=0 the updater never asks for an estimate.
    baseline = run_osem_with_corrections(
        measured,
        make_model,
        initial,
        AdditiveUpdater(None, UpdateSchedule(every=1), measured.get_uniform_copy(0)),
        NUM_SUBSETS,
        subiterations_per_update=subiterations * (NUM_UPDATES + 1),
        num_updates=0,
    )
    updater = scatter_updater(
        output_dir, template_data, mu_image, measured, full_model, subiterations
    )
    corrected = run_osem_with_corrections(
        measured,
        make_model,
        initial,
        updater,
        NUM_SUBSETS,
        subiterations_per_update=subiterations,
        num_updates=NUM_UPDATES,
    )

    print_updates(updater)
    before = contrast_error(baseline, activity)
    after = contrast_error(corrected, activity)
    print(f"hot-sphere contrast error, no scatter correction: {before:.1%}")
    print(f"hot-sphere contrast error, SIMIND scatter:        {after:.1%}")


if __name__ == "__main__":
    main()
