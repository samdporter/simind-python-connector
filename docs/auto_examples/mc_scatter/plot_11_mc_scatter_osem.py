"""
Monte Carlo scatter correction in OSEM
======================================

Reconstruct a simulated NEMA/IEC phantom acquisition twice with SIRF OSEM: once
without scatter correction, and once with a SIMIND scatter estimate that is
refreshed from the current image between OSEM passes.

The data are simulated by SIMIND with four histories per measured count, so
the Monte Carlo variance of the expectation is a quarter of the Poisson
variance of the acquisition. The reconstruction uses a fast SPECTUB model with
attenuation but no collimator blur and no scatter. SIMIND provides the scatter:
all interactions minus the geometric primary, scaled to the fast model and
added to its forward projection.

Both arms use 6 subsets, three passes of 12 subiterations (8 full iterations)
and the same initial image. The uncorrected arm uses the same schedule with a
zero additive term. Three independent Poisson realisations give the mean and
standard error of every score.

.. warning::
   The first tutorial run simulates the shared expectation and takes about an
   hour with 8 MPI processes. Each scatter estimate uses a fortieth of that
   history budget and takes about a minute; this example makes nine estimates
   in all, so it takes about half an hour. Later tutorials reuse the cached
   expectation.

The phantom needs phantomgen:
``pip install "phantomgen @ git+https://github.com/varzakis/phantomgen"``.
"""

# %%
# Setup
# -----
# Check that SIMIND is installed and choose where the outputs are written.

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
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "mc_scatter_osem"
)
output_dir.mkdir(parents=True, exist_ok=True)
estimate_dir = output_dir / "estimates"
estimate_dir.mkdir(exist_ok=True)

# %%
# NEMA/IEC phantom
# ----------------
# phantomgen's ``pet`` preset has an 8:1 sphere-to-background activity ratio.
# Its two smallest spheres, 10 and 13 mm, are cold; the four hot spheres are
# 17, 22, 28 and 37 mm. The activity is in MBq per voxel. The attenuation map is
# the density times the attenuation coefficient of water at 140 keV.

nema_phantom = _nema.phantom()
nema = nema_phantom.nema
slice_z = nema_phantom.slice_z

fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
for ax, (title, image) in zip(
    axes,
    (
        ("Activity (MBq per voxel)", nema.activity_mbq),
        ("Density (g/cm³)", nema.density_g_cm3),
    ),
):
    ax.imshow(image[slice_z], cmap="gray_r")
    ax.set_title(title)
    ax.axis("off")
plt.show()

activity_image = _nema.to_sirf_image(nema.activity_mbq)
mu_image = _nema.to_sirf_image(nema_phantom.mu)
# STIR SPECTUB mirrors x relative to the mu map (UCL/STIR#452)
spectub_mu_image = _nema.to_sirf_image(nema_phantom.mu[..., ::-1])

# %%
# Template
# --------
# The template sets the projection geometry: 48 x 48 bins of 7 mm, 60 views
# over 360 degrees and a detector radius of 250 mm. Both the simulation and the
# reconstruction take their geometry from it.

template = _nema.template(output_dir)

# %%
# "Measured" data
# ---------------
# SIMIND simulates the phantom for Tc-99m with a 126-154 keV window and the
# ma-lehr collimator. The total activity and the time per view are given, so the
# projections come out in counts (Index 25). The 8 MPI processes each simulate
# all views and SIMIND sums them; without an ``RR`` switch their seeds are fixed.
# Poisson noise from a seeded generator is added to the simulated counts at
# 2 s per view. The three realisations are shared by both arms.

expectation = _nema.simulate_expectation(nema_phantom, template)
realisations = dict(_nema.paired_realisations(expectation, template))
print(
    f"Expectation: {expectation.total:.4g} counts from "
    f"{expectation.histories:.4g} histories, "
    f"quantization loss {expectation.quantization_loss:.2%}"
)
for seed, measured in realisations.items():
    print(f"Seed {seed}: {measured.sum():.4g} measured counts")

fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
counts = next(iter(realisations.values())).as_array()[0]  # axial, view, bin
axes[0].imshow(counts[:, 0, :], cmap="magma")
axes[0].set_title("Projection, view 0")
axes[1].imshow(counts[counts.shape[0] // 2], cmap="magma")
axes[1].set_title("Sinogram, central row")
axes[1].set_xlabel("bin")
axes[1].set_ylabel("views")
plt.show()

# %%
# OSEM without scatter correction
# -------------------------------
# The fast model is SPECTUB with the flipped attenuation map and no PSF. It has
# no collimator blur and no scatter. ``run_osem_with_corrections`` runs OSEM in
# passes and calls the updater between them; with a zero additive term and no
# updates it is plain OSEM.

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

uncorrected = {}
for seed, measured in realisations.items():
    uncorrected[seed] = run_osem_with_corrections(
        measured,
        make_acq_model,
        initial_image,
        AdditiveUpdater(None, UpdateSchedule(every=1), zero_additive),
        NUM_SUBSETS,
        SUBITERATIONS_PER_UPDATE * (NUM_UPDATES + 1),
        num_updates=0,
    )

# %%
# OSEM with SIMIND scatter
# ------------------------
# ``SimindProjector`` simulates the current image with the template geometry.
# ``reference_normaliser`` scales the geometric primary of that simulation to
# the fast model's forward projection of the same image, so that the scatter
# comes out on the scale of the data. ``ScatterCorrection`` smooths the
# estimate over 3 bins, across views as well, because a raw Monte Carlo estimate
# is too noisy to subtract. The additive term starts at zero and is refreshed
# after each pass of 12 subiterations, averaging successive estimates
# (``damping=0.5``). The image at the first update is not corrected for
# scatter, so the first estimate is too high.
#
# The estimate uses a quarter of the expectation's history budget. It is one
# simulation for all three realisations of an update only if the images are the
# same; here each realisation has its own image, so each makes its own estimates.
# ``seed=None`` leaves the ``RR`` switch unset, so the 8 MPI processes use their
# fixed seeds.

ESTIMATE_QUANTIZATION_SCALE = 0.1 * _nema.QUANTIZATION_SCALE
corrected = {}
updaters = {}
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
    updaters[seed] = updater

# %%
# Scatter updates
# ---------------
# For each update, the sum of the additive term and the time of the SIMIND run.

for seed, updater in updaters.items():
    for record in updater.history:
        print(
            f"seed {seed} update at iteration {record.iteration}: "
            f"scatter sum {record.additive_sum:.4g}, {record.seconds:.0f} s"
        )

true_scatter = expectation.true_scatter
print(f"True scatter sum: {true_scatter.sum():.4g}")
for seed, updater in updaters.items():
    print(
        f"Seed {seed} scatter fraction: "
        f"true {true_scatter.sum() / realisations[seed].sum():.0%}, "
        f"estimated {updater.current.sum() / realisations[seed].sum():.0%}"
    )

central_view = next(iter(realisations.values())).as_array()[0].shape[1] // 2
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
first_seed = next(iter(realisations))
for ax, (title, data) in zip(
    axes,
    (
        ("Measured", realisations[first_seed]),
        ("True scatter", _nema.to_acquisition(true_scatter, template)),
        ("Estimated scatter", updaters[first_seed].current),
    ),
):
    ax.imshow(data.as_array()[0][:, central_view, :], cmap="magma")
    ax.set_title(f"{title}, view {central_view}")
    ax.axis("off")
plt.show()

# %%
# Reconstructions
# ---------------
# Both images are shown relative to their mean in the background mask, so the
# arbitrary scale of the reconstructions does not matter. Each image is the
# mean of the three realisations. The difference panel is zero-centred and has a
# symmetric scale, so it shows where the correction moves counts without being
# dominated by either image's overall level.

background = nema_phantom.background


def mean_relative(images):
    array = np.mean([image.as_array() for image in images], axis=0)
    return nema_phantom.relative_to_background(array)


truth_array = nema_phantom.relative_to_background(nema.activity_mbq)
arrays = {
    "No scatter correction": mean_relative(uncorrected.values()),
    "SIMIND scatter": mean_relative(corrected.values()),
}
difference = arrays["SIMIND scatter"] - arrays["No scatter correction"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, (title, image) in zip(axes, (("Truth", truth_array), *arrays.items())):
    shown = ax.imshow(image[slice_z], cmap="gray_r", vmin=0, vmax=truth_array.max())
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
plt.show()

fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
limit = np.abs(difference).max()
shown = ax.imshow(difference[slice_z], cmap="coolwarm", vmin=-limit, vmax=limit)
ax.set_title("SIMIND scatter − no correction")
ax.axis("off")
fig.colorbar(shown, ax=ax, label="difference, relative to background")
plt.show()

# %%
# Contrast recovery
# -----------------
# For each hot sphere, the mean in its mask over the mean in the background
# mask, divided by the true ratio of 8. A value of 1 is full recovery. The
# masks include voxels on the sphere edge, which hold less activity, so even
# the truth scores below 1; its score is the ceiling for the reconstruction.
# The 10 and 13 mm spheres are cold and are not scored. The error bars are the
# standard error over the three independent realisations.

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
for title, (recovery, recovery_sem, cov, cov_sem) in scores.items():
    print(
        f"{title:<22}" + "".join(f"{value:11.2f}" for value in recovery) + f"{cov:8.2f}"
    )

x = np.arange(len(diameters))
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for offset, (title, (recovery, recovery_sem, _, _)) in zip(
    (-0.27, 0, 0.27), scores.items()
):
    ax.bar(x + offset, recovery, 0.27, yerr=recovery_sem, capsize=2, label=title)
ax.set_xticks(x, diameters)
ax.set_xlabel("Sphere diameter (mm)")
ax.set_ylabel("Contrast recovery")
ax.set_ylim(0, max(truth_recovery) * 1.15)
ax.legend()
plt.show()

# %%
# Scatter correction raises the recovery at every hot sphere, by 0.06 at 28 mm
# and 0.08 at 37 mm, and by 0.02 and 0.04 at 17 and 22 mm. Every corrected
# score stays below the truth's own masked score. The background noise rises
# from 0.71 to 0.79 because the scatter estimate is a noisy additive term, so
# the correction buys recovery at a cost in noise rather than dominating both.
