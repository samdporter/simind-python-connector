"""Monte Carlo scatter correction in OSEM.

Reconstruct a simulated NEMA/IEC phantom twice with SIRF OSEM: once without
scatter correction and once with a SIMIND scatter estimate refreshed from the
current image between passes.

The phantom is phantomgen's ``pet`` preset: 8:1 sphere-to-background activity
in MBq per voxel, four hot spheres (17, 22, 28 and 37 mm), two cold spheres
(10 and 13 mm), and named masks. SIMIND simulates one shared expectation with
a known total activity and 2 s per view, so the projections are in counts. It
uses four times the default history budget and eight MPI processes with fixed
seeds (no ``RR`` switch). Three independent seeded Poisson draws share that
expectation; both arms use the same three draws.

The reconstruction uses a fast SPECTUB model with attenuation but no PSF or
scatter. ``SimindProjector`` simulates the current image, ``reference_normaliser``
scales its geometric primary to the fast model, ``ScatterCorrection`` smooths
the estimate over 3 bins and views, and ``AdditiveUpdater`` refreshes it with
damping 0.5 every 12 subiterations. Both arms run 6 subsets, 12 subiterations
per pass and 4 passes (8 full iterations). Recovery is the hot-sphere mask mean
over the background mean, divided by the true ratio; the masks include edge
voxels, so the truth's own score is the ceiling. The three realisations give
the mean and standard error of recovery and background CoV.
"""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sirf.STIR as sirf

from examples import _nema
from simind_python_connector.recon import (
    AdditiveUpdater,
    ScatterCorrection,
    SimindProjector,
    UpdateSchedule,
    reference_normaliser,
)
from simind_python_connector.recon.osem import run_osem_with_corrections


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError("SIMIND executable not found ($SIMIND_BIN or simind on PATH).")

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "mc_scatter_osem"
)
output_dir.mkdir(parents=True, exist_ok=True)
estimate_dir = output_dir / "estimates"
estimate_dir.mkdir(exist_ok=True)

nema_phantom = _nema.phantom()
nema = nema_phantom.nema
slice_z = nema_phantom.slice_z
background = nema_phantom.background
activity_image = _nema.to_sirf_image(nema.activity_mbq)
mu_image = _nema.to_sirf_image(nema_phantom.mu)
# STIR SPECTUB mirrors x relative to the mu map (UCL/STIR#452).
spectub_mu_image = _nema.to_sirf_image(nema_phantom.mu[..., ::-1])
template = _nema.template(output_dir)

expectation = _nema.simulate_expectation(nema_phantom, template)
realisations = dict(_nema.paired_realisations(expectation, template))
print(
    f"Expectation: {expectation.total:.4g} counts, "
    f"{expectation.histories:.4g} histories per projection, "
    f"quantization loss {expectation.quantization_loss:.2%}"
)

NUM_SUBSETS = _nema.NUM_SUBSETS
SUBITERATIONS_PER_UPDATE = 12
NUM_UPDATES = 3
initial_image = activity_image.get_uniform_copy(1.0)
zero_additive = next(iter(realisations.values())).get_uniform_copy(0)


def make_acq_model():
    acq_matrix = sirf.SPECTUBMatrix()
    acq_matrix.set_attenuation_image(spectub_mu_image)
    acq_matrix.set_resolution_model(0.0, 0.0, False)
    return sirf.AcquisitionModelUsingMatrix(acq_matrix)


full_model = make_acq_model()
full_model.set_up(next(iter(realisations.values())), activity_image)

uncorrected = {
    seed: run_osem_with_corrections(
        measured,
        make_acq_model,
        initial_image,
        AdditiveUpdater(None, UpdateSchedule(every=1), zero_additive),
        NUM_SUBSETS,
        SUBITERATIONS_PER_UPDATE * (NUM_UPDATES + 1),
        num_updates=0,
    )
    for seed, measured in realisations.items()
}

ESTIMATE_QUANTIZATION_SCALE = 0.1 * _nema.QUANTIZATION_SCALE
corrected = {}
for seed, measured in realisations.items():
    estimate_adaptor = _nema.make_adaptor(
        estimate_dir,
        f"scatter_{seed}",
        quantization_scale=ESTIMATE_QUANTIZATION_SCALE,
        histories=1,
    )
    projector = SimindProjector(
        estimate_adaptor,
        template,
        mu_image,
        normalise=reference_normaliser(full_model, method="sum"),
        seed=None,
    )
    updater = AdditiveUpdater(
        ScatterCorrection(projector, smoothing_fwhm_bins=3.0, smooth_across_views=True),
        UpdateSchedule(every=SUBITERATIONS_PER_UPDATE),
        zero_additive,
        damping=0.5,
    )
    corrected[seed] = run_osem_with_corrections(
        measured,
        make_acq_model,
        initial_image,
        updater,
        NUM_SUBSETS,
        SUBITERATIONS_PER_UPDATE,
        NUM_UPDATES,
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
scores = {"Truth": (truth_recovery, [0.0] * len(truth_recovery), truth_cov, 0.0)}
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
print(f"{'':22}" + "".join(f"{d:>8} mm" for d in diameters) + "     CoV")
for title, (recovery, _, cov, _) in scores.items():
    print(
        f"{title:<22}" + "".join(f"{value:11.2f}" for value in recovery) + f"{cov:8.2f}"
    )

x = np.arange(len(diameters))
fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
for offset, (title, (recovery, recovery_sem, _, _)) in zip(
    (-0.27, 0, 0.27), scores.items()
):
    axis.bar(x + offset, recovery, 0.27, yerr=recovery_sem, capsize=2, label=title)
axis.set_xticks(x, diameters)
axis.set_xlabel("Sphere diameter (mm)")
axis.set_ylabel("Contrast recovery")
axis.set_ylim(0, max(truth_recovery) * 1.15)
axis.legend()
plt.show()

# The recorded recovery rises from 0.33 to 0.36 at 17 mm, 0.38 to 0.43 at
# 22 mm, 0.49 to 0.56 at 28 mm and 0.54 to 0.61 at 37 mm. Background CoV rises
# from 0.64 to 0.69: the correction buys recovery at a cost in noise.
