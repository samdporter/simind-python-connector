#!/usr/bin/env python
"""
Residual correction (Fu & Qi) with SIMIND as the accurate forward model.

1. Simulate "measured" data with a full PENETRATE SIMIND run at 208 keV
   (Lu-177-like, medium-energy collimator), known activity, Poisson noise.
2. Reconstruct it with SIRF OSEM three times:
   - the fast model only (SPECTUB, attenuation, no PSF)
   - the fast model plus the SIMIND scatter estimate of example 09
   - the fast model plus the full residual A_acc(x) - A_fast(x), with A_acc
     all SIMIND events scaled to the fast model
3. Print the seed base, the hot-sphere contrast recovery and background
   CoV of each, and the effective objective (the accurate model's data
   term) at each update, alongside the true-phantom oracle.

Tiny on purpose: 32^3 voxels, 30 views, NN = 1. Needs SIRF and SIMIND.

`SIMIND_SEED_BASE` defaults to 100. It sets the measured-data seed; the
accurate projector uses that base plus 100. Characterise the example
separately with bases 100, 300 and 400, recording the true-phantom oracle
alongside the reconstruction metrics.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from _mc_scatter_common import (
    background_cov,
    contrast_recovery,
    fast_model_factory,
    image,
    phantom,
    print_updates,
    scatter_updater,
    simind_adaptor,
    simulate_measured,
    template,
)

from simind_python_connector.recon import (
    AdditiveUpdater,
    ResidualCorrection,
    SimindComponent,
    SimindProjector,
    UpdateSchedule,
    effective_objective,
    reference_normaliser,
)
from simind_python_connector.recon.osem import run_osem_with_corrections


SEED_BASE = int(os.environ.get("SIMIND_SEED_BASE", "100"))
LU177_LIKE = {
    "photon_energy_kev": 208.0,
    "window_kev": (187.0, 229.0),
    "collimator": "ma-megp",
}
NUM_SUBSETS = 5
EPOCHS_PER_UPDATE = 2
NUM_UPDATES = 3


class LoggedResidualCorrection(ResidualCorrection):
    """Records the effective objective after every estimate."""

    def __init__(self, measured, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measured = measured
        self.objectives = []

    def estimate(self, image):
        additive = super().estimate(image)
        self.objectives.append(effective_objective(self.measured, self))
        return additive


def main() -> None:
    if shutil.which("simind") is None:
        raise RuntimeError("SIMIND executable not found in PATH.")
    output_dir = Path("output/residual_correction")
    output_dir.mkdir(parents=True, exist_ok=True)

    activity, mu = phantom()
    activity_image, mu_image = image(activity), image(mu)
    template_data = template(output_dir)
    print(f"seed base: {SEED_BASE}")
    measured = simulate_measured(
        output_dir,
        template_data,
        activity_image,
        mu_image,
        seed=SEED_BASE,
        **LU177_LIKE,
    )

    make_model = fast_model_factory(mu_image)
    full_model = make_model()
    full_model.set_up(measured, activity_image)
    initial = activity_image.get_uniform_copy(1.0)
    zero = measured.get_uniform_copy(0)
    every = EPOCHS_PER_UPDATE * NUM_SUBSETS

    def osem(updater, num_updates):
        passes = num_updates + 1
        return run_osem_with_corrections(
            measured,
            make_model,
            initial,
            updater,
            NUM_SUBSETS,
            subiterations_per_update=every * (NUM_UPDATES + 1) // passes,
            num_updates=num_updates,
        )

    results = {}
    results["fast only"] = osem(AdditiveUpdater(None, UpdateSchedule(every=1), zero), 0)

    scatter = scatter_updater(
        output_dir, template_data, mu_image, measured, full_model, every, **LU177_LIKE
    )
    results["fast + SIMIND scatter"] = osem(scatter, NUM_UPDATES)

    projector = SimindProjector(
        simind_adaptor(output_dir / "accurate", "accurate", **LU177_LIKE),
        template_data,
        mu_image,
        normalise=reference_normaliser(full_model),
        seed=SEED_BASE + 100,
    )
    residual = LoggedResidualCorrection(
        measured, SimindComponent(projector, "all_interactions"), full_model, zero
    )
    residual_updater = AdditiveUpdater(residual, UpdateSchedule(every=every), zero)
    results["fast + full residual"] = osem(residual_updater, NUM_UPDATES)

    print_updates(residual_updater)
    for name, reconstruction in results.items():
        recovery = contrast_recovery(reconstruction, activity)
        cov = background_cov(reconstruction)
        print(f"{name:24s} contrast recovery {recovery:.3f}  background CoV {cov:.3f}")
    print(
        "true-phantom oracle  "
        f"contrast recovery {contrast_recovery(activity_image, activity):.3f}  "
        f"background CoV {background_cov(activity_image):.3f}"
    )
    objectives = ", ".join(f"{value:.6g}" for value in residual.objectives)
    print(f"effective objective at each update: {objectives}")


if __name__ == "__main__":
    main()
