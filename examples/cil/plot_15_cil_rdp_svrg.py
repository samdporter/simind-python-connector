"""RDP-regularised reconstruction with SVRG.

Reconstruct the simulated NEMA/IEC acquisition with a SIRF relative difference
prior (RDP) and CIL's stochastic variance reduced gradient (SVRG), and compare
it with OSEM. Both use a SIMIND scatter estimate refreshed from the current
image. The fixture is phantomgen's ``pet`` preset with 8:1 sphere-to-background
activity in MBq per voxel, four hot spheres, two cold spheres and named masks.
SIMIND simulates one expectation with the known activity and 2 s per view;
three seeded Poisson draws (12345, 12346 and 12347) share it.

The fast model is SPECTUB with attenuation and no PSF or scatter. Six subset
objectives of ten views are used by both reconstructions. The RDP is smooth, so
it belongs in the gradient-based objective. Each subset carries a sixth of the
prior strength, so the SVRG snapshot gradient still sums to the full prior.
``SVRGFunction`` draws one subset per iteration and retakes the full gradient
every 12 iterations. ``CorrectionCallback`` refreshes the scatter estimate
every three epochs and then the stored gradients.

The recorded demonstration settings are beta 0.20 and 24 epochs. The original
beta/epoch pilot used its own fixed estimate, image, updater and output
directory, and did not modify the final run's data, model, prior constants or
SVRG sampler. Removing it therefore cannot change the final-run state; it only
removes extra estimates and reconstructions. A step size of 0.1 is used: a 0.5
step made edge voxels grow without bound with this prior and refreshed scatter,
so the comparison also changes algorithm and step size, not only the prior.
"""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sirf.STIR as sirf
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import Function, IndicatorBox, SumFunction, SVRGFunction
from cil.optimisation.utilities import AdaptiveSensitivity, Sampler

from examples import _nema
from simind_python_connector.recon import (
    AdditiveUpdater,
    ScatterCorrection,
    SimindProjector,
    UpdateSchedule,
    reference_normaliser,
)
from simind_python_connector.recon.cil import (
    CorrectionCallback,
    build_subset_objectives,
)
from simind_python_connector.recon.osem import run_osem_with_corrections


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError("SIMIND executable not found ($SIMIND_BIN or simind on PATH).")

output_dir = Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "cil_rdp_svrg"
output_dir.mkdir(parents=True, exist_ok=True)
estimate_dir = output_dir / "estimates"
estimate_dir.mkdir(exist_ok=True)

nema_phantom = _nema.phantom()
nema = nema_phantom.nema
slice_z = nema_phantom.slice_z
activity_image = _nema.to_sirf_image(nema.activity_mbq)
mu_image = _nema.to_sirf_image(nema_phantom.mu)
spectub_mu_image = _nema.to_sirf_image(nema_phantom.mu[..., ::-1])
template = _nema.template(output_dir)

expectation = _nema.simulate_expectation(nema_phantom, template)
realisations = dict(_nema.paired_realisations(expectation, template))
print(
    f"Expectation counts: {expectation.total:.4g}, "
    f"histories {expectation.histories:.4g}"
)

NUM_SUBSETS = _nema.NUM_SUBSETS
OSEM_SUBITERATIONS = 12
OSEM_UPDATES = 3
RDP_STRENGTH = 0.20
RDP_EPOCHS = 24
STEP_SIZE = 0.1
UPDATE_EVERY_EPOCHS = 3
ITERATIONS = RDP_EPOCHS * NUM_SUBSETS
UPDATE_EVERY = UPDATE_EVERY_EPOCHS * NUM_SUBSETS
initial_image = activity_image.get_uniform_copy(1.0)
zero_additive = next(iter(realisations.values())).get_uniform_copy(0)
ESTIMATE_QUANTIZATION_SCALE = 0.1 * _nema.QUANTIZATION_SCALE


def make_model():
    matrix = sirf.SPECTUBMatrix()
    matrix.set_attenuation_image(spectub_mu_image)
    # Separately set-up SPECTUB models share this cache in STIR 6.4.
    matrix.set_keep_all_views_in_cache(False)
    matrix.set_resolution_model(0.0, 0.0, False)
    return sirf.AcquisitionModelUsingMatrix(matrix)


full_model = make_model()
full_model.set_up(next(iter(realisations.values())), activity_image)


def make_scatter_updater(name, schedule):
    projector = SimindProjector(
        _nema.make_adaptor(
            estimate_dir,
            name,
            quantization_scale=ESTIMATE_QUANTIZATION_SCALE,
            histories=1,
        ),
        template,
        mu_image,
        normalise=reference_normaliser(full_model),
        seed=None,
    )
    return AdditiveUpdater(
        ScatterCorrection(projector, smoothing_fwhm_bins=3.0, smooth_across_views=True),
        schedule,
        zero_additive,
        damping=0.5,
    )


osem = {}
for seed, measured in realisations.items():
    updater = make_scatter_updater(
        f"osem_estimate_{seed}", UpdateSchedule(every=OSEM_SUBITERATIONS)
    )
    osem[seed] = run_osem_with_corrections(
        measured,
        make_model,
        initial_image,
        updater,
        NUM_SUBSETS,
        OSEM_SUBITERATIONS,
        OSEM_UPDATES,
    )


class PriorFunction(Function):
    """CIL view of a SIRF prior, which provides its own value and gradient."""

    def __init__(self, prior):
        super().__init__()
        self.prior = prior

    def __call__(self, image):
        return self.prior.get_value(image)

    def gradient(self, image, out=None):
        return self.prior.get_gradient(image, out=out)


def rdp_term(strength):
    prior = sirf.RelativeDifferencePrior()
    prior.set_penalisation_factor(strength / NUM_SUBSETS)
    prior.set_up(initial_image)
    return PriorFunction(prior)


def svrg_run(measured, additive, callbacks):
    objectives = build_subset_objectives(
        measured, additive, make_model, NUM_SUBSETS, initial_image
    )
    subset_functions = [
        SumFunction(function, rdp_term(RDP_STRENGTH))
        for function in objectives.functions
    ]
    svrg = SVRGFunction(
        subset_functions,
        sampler=Sampler.sequential(NUM_SUBSETS),
        snapshot_update_interval=2 * NUM_SUBSETS,
    )
    algorithm = ISTA(
        initial=initial_image,
        f=svrg,
        g=IndicatorBox(lower=0),
        step_size=STEP_SIZE,
        preconditioner=AdaptiveSensitivity(full_model, max_iterations=ITERATIONS + 1),
    )
    algorithm.run(ITERATIONS, callbacks=callbacks(objectives))
    return algorithm.solution


svrg_image = {}
for seed in realisations:
    updater = make_scatter_updater(
        f"svrg_estimate_{seed}",
        UpdateSchedule(every=UPDATE_EVERY, stop_at=ITERATIONS),
    )
    svrg_image[seed] = svrg_run(
        realisations[seed],
        zero_additive,
        lambda objectives, updater=updater: [CorrectionCallback(updater, objectives)],
    )


def mean_relative(images):
    array = np.mean([image.as_array() for image in images], axis=0)
    return nema_phantom.relative_to_background(array)


truth_array = nema_phantom.relative_to_background(nema.activity_mbq)
arrays = {
    "OSEM": mean_relative(osem.values()),
    "RDP + SVRG": mean_relative(svrg_image.values()),
}
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for axis, (title, image) in zip(axes, (("Truth", truth_array), *arrays.items())):
    shown = axis.imshow(image[slice_z], cmap="gray_r", vmin=0, vmax=truth_array.max())
    axis.set_title(title)
    axis.axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
plt.show()

truth_recovery = _nema.contrast_recovery(nema.activity_mbq, nema_phantom)
truth_cov = _nema.background_cov(nema.activity_mbq, nema_phantom)
scores = {"Truth": (truth_recovery, [0.0] * 4, truth_cov, 0.0)}
for title, images in (("OSEM", osem), ("RDP + SVRG", svrg_image)):
    recovery, recovery_sem = _nema.mean_and_sem(
        [
            _nema.contrast_recovery(image.as_array(), nema_phantom)
            for image in images.values()
        ]
    )
    cov, cov_sem = _nema.mean_and_sem(
        [
            _nema.background_cov(image.as_array(), nema_phantom)
            for image in images.values()
        ]
    )
    scores[title] = (recovery, recovery_sem, float(cov), float(cov_sem))

diameters = [str(d) for d in _nema.SPHERES.values()]
print(f"{'':24}" + "".join(f"{d:>8} mm" for d in diameters) + "     CoV")
for title, (recovery, _, cov, _) in scores.items():
    print(
        f"{title:<24}" + "".join(f"{value:11.2f}" for value in recovery) + f"{cov:8.2f}"
    )

x = np.arange(len(diameters))
fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
for offset, (title, (recovery, recovery_sem, _, _)) in zip(
    (-0.2, 0, 0.2), scores.items()
):
    axis.bar(x + offset, recovery, 0.2, yerr=recovery_sem, capsize=2, label=title)
axis.set_xticks(x, diameters)
axis.set_xlabel("Sphere diameter (mm)")
axis.set_ylabel("Contrast recovery")
axis.set_ylim(0, max(truth_recovery) * 1.15)
axis.legend()
plt.show()

# Recorded: OSEM 0.36/0.43/0.56/0.61 with CoV 0.69; RDP + SVRG
# 0.24/0.29/0.42/0.50 with CoV 0.35. The comparison changes algorithm and step
# size as well as the prior, so it does not isolate the prior's effect.
