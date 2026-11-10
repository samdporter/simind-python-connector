"""Monte Carlo scatter correction with CIL ISTA.

Reconstruct the simulated NEMA/IEC acquisition with CIL ISTA, without and with
a SIMIND scatter estimate refreshed from the current image. The fixture is
phantomgen's ``pet`` preset: 8:1 sphere-to-background activity in MBq per voxel,
four hot spheres, two cold spheres and named masks. SIMIND simulates one
expectation with the known activity and 2 s per view, so the data are in
counts. Three seeded Poisson draws (12345, 12346 and 12347) share it.

The fast model is SPECTUB with the attenuation map and no PSF or scatter. Six
CIL Kullback-Leibler objectives, one per group of ten views, share one additive
term. ``CorrectionCallback`` swaps in a new SIMIND scatter estimate every 12
iterations; the estimate is all interactions minus geometric primary, scaled to
the fast model by ``reference_normaliser``, smoothed over 3 bins and across
views, and damped by 0.5. The final run uses 48 iterations; the estimate from
the previous callback is used for the last iterations. The estimate uses a
fortieth of the expectation's history budget.
"""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sirf.STIR as sirf
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import IndicatorBox, SumFunction
from cil.optimisation.utilities import AdaptiveSensitivity

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


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError("SIMIND executable not found ($SIMIND_BIN or simind on PATH).")

output_dir = Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "cil_scatter"
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
ITERATIONS = 48
UPDATE_EVERY = 12
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


def ista(objectives, callbacks, iterations=ITERATIONS):
    algorithm = ISTA(
        initial=initial_image,
        f=SumFunction(*objectives.functions),
        g=IndicatorBox(lower=0),
        step_size=1.0,
        preconditioner=AdaptiveSensitivity(full_model, max_iterations=iterations + 1),
    )
    algorithm.run(iterations, callbacks=callbacks)
    return algorithm.solution


def make_updater(seed):
    projector = SimindProjector(
        _nema.make_adaptor(
            estimate_dir,
            f"estimate_{seed}",
            quantization_scale=ESTIMATE_QUANTIZATION_SCALE,
            histories=1,
        ),
        template,
        mu_image,
        normalise=reference_normaliser(full_model),
        seed=None,
    )
    updater = AdditiveUpdater(
        ScatterCorrection(projector, smoothing_fwhm_bins=3.0, smooth_across_views=True),
        UpdateSchedule(every=UPDATE_EVERY, stop_at=ITERATIONS),
        zero_additive,
        damping=0.5,
    )
    objectives = build_subset_objectives(
        realisations[seed], zero_additive, make_model, NUM_SUBSETS, initial_image
    )
    return updater, objectives


uncorrected = {}
for seed, measured in realisations.items():
    objectives = build_subset_objectives(
        measured, zero_additive, make_model, NUM_SUBSETS, initial_image
    )
    uncorrected[seed] = ista(objectives, callbacks=[])

corrected = {}
for seed in realisations:
    updater, objectives = make_updater(seed)
    corrected[seed] = ista(
        objectives, callbacks=[CorrectionCallback(updater, objectives)]
    )


def mean_relative(images):
    array = np.mean([image.as_array() for image in images], axis=0)
    return nema_phantom.relative_to_background(array)


truth_array = nema_phantom.relative_to_background(nema.activity_mbq)
arrays = {
    "No scatter correction": mean_relative(uncorrected.values()),
    "SIMIND scatter": mean_relative(corrected.values()),
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
for title, images in (
    ("No scatter correction", uncorrected),
    ("SIMIND scatter", corrected),
):
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

# Recorded: no correction 0.32/0.39/0.48/0.54 with CoV 0.62; SIMIND scatter
# 0.36/0.43/0.55/0.61 with CoV 0.68. The correction raises recovery and noise.
