"""
NEMA simulation and OSEM
========================

Simulate "measured" data from the NEMA/IEC body phantom with a known activity,
reconstruct it with SIRF OSEM and measure the contrast recovery of the hot
spheres.

The data are simulated by SIMIND with four histories per measured count, so
the Monte Carlo variance is a quarter of the Poisson variance of the
acquisition. The simulation is run once and cached; every reconstruction
tutorial in this gallery reuses it. The reconstruction uses a SPECTUB model
with attenuation but no collimator blur and no scatter term, so OSEM treats
scattered events as if they came from the primary distribution. Contrast
recovery is therefore reported against the truth's own masked score, not
against an idealised 1.

Each of three independent Poisson realisations is reconstructed, and the bars
show the mean and standard error over those realisations. The displayed image
is the mean of the three.

.. warning::
   The first tutorial run simulates the shared expectation and takes about an
   hour with 8 MPI processes. Later tutorials in this gallery reuse the cached
   expectation and skip the simulation. The reconstruction itself takes
   seconds.

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


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "nema_osem"
output_dir.mkdir(parents=True, exist_ok=True)

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
#
# The expectation uses the default quantization (a 500-history budget at the
# full signed range) multiplied by four. The printed quantization loss is the
# L1 error of the integer source map against the float map; it is below 1%.
#
# Poisson noise from a seeded generator is added to the simulated counts at
# 2 s per view. The three realisations use independent seeds and share the one
# expectation.

expectation = _nema.simulate_expectation(nema_phantom, template)
TIME_PER_VIEW_S = _nema.REFERENCE_TIME_S
realisations = dict(_nema.paired_realisations(expectation, template))

print(f"Expectation counts: {expectation.total:.4g}")
print(f"Photon histories: {expectation.histories:.4g}")
print(
    f"Histories per measured count: {expectation.histories_per_count:.2f} "
    "(so the Monte Carlo variance is that fraction of the Poisson variance)"
)
print(f"Quantization loss: {expectation.quantization_loss:.2%}")
for seed, measured in realisations.items():
    print(f"Seed {seed}: {measured.sum():.4g} measured counts")

fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
counts = next(iter(realisations.values())).as_array()[0]  # axial, view, bin
axes[0].imshow(counts[:, 0, :], cmap="magma")
axes[0].set_title("Projection, view 0")
axes[1].imshow(counts[counts.shape[0] // 2], cmap="magma")
axes[1].set_title("Sinogram, central row")
axes[1].set_xlabel("bin")
axes[1].set_ylabel("view")
plt.show()

# %%
# OSEM
# ----
# The model is SPECTUB with the flipped attenuation map and no PSF. OSEM runs
# with 6 subsets. We reconstruct each realisation for 2, 4 and 8 full
# iterations and compare the contrast recovery and the background noise.
# Recovery rises by at most 0.02 from four to eight iterations, while the
# background noise rises from about 0.55 to 0.71; four iterations is the first
# setting that is within that margin of the longer run without paying its
# noise.

acq_matrix = sirf.SPECTUBMatrix()
acq_matrix.set_attenuation_image(spectub_mu_image)
acq_matrix.set_resolution_model(0.0, 0.0, False)
acq_model = sirf.AcquisitionModelUsingMatrix(acq_matrix)


def osem(data, iterations):
    objective = sirf.make_Poisson_loglikelihood(data)
    objective.set_acquisition_model(acq_model)
    objective.set_num_subsets(_nema.NUM_SUBSETS)
    recon = sirf.OSMAPOSLReconstructor()
    recon.set_objective_function(objective)
    recon.set_num_subsets(_nema.NUM_SUBSETS)
    recon.set_num_subiterations(_nema.NUM_SUBSETS * iterations)
    recon.set_input(data)
    image = activity_image.get_uniform_copy(1.0)
    recon.set_up(image)
    recon.reconstruct(image)
    return image


truth_array = nema_phantom.relative_to_background(nema.activity_mbq)
truth_recovery = _nema.contrast_recovery(nema.activity_mbq, nema_phantom)
truth_cov = _nema.background_cov(nema.activity_mbq, nema_phantom)

print(f"{'':10}{'17 mm':>8}{'22 mm':>8}{'28 mm':>8}{'37 mm':>8}{'CoV':>8}")
print(
    f"{'Truth':<10}"
    + "".join(f"{value:8.2f}" for value in truth_recovery)
    + f"{truth_cov:8.2f}"
)

# The expectation on its own has no acquisition noise. Reconstructing it shows
# what the fast model can recover when the data are exact, so the difference to
# the Poisson arms is the acquisition noise and the reconstruction's response
# to it, not Monte Carlo noise.
expectation_image = osem(
    _nema.to_acquisition(expectation.counts, template), _nema.OSEM_ITERATIONS
)
expectation_recovery = _nema.contrast_recovery(
    expectation_image.as_array(), nema_phantom
)
expectation_cov = _nema.background_cov(expectation_image.as_array(), nema_phantom)
print(
    f"{'Expectation':<10}"
    + "".join(f"{value:8.2f}" for value in expectation_recovery)
    + f"{expectation_cov:8.2f}"
)

stopping = {}
for iterations in (2, 4, 8):
    images = [osem(data, iterations) for data in realisations.values()]
    recovery, recovery_sem = _nema.mean_and_sem(
        [_nema.contrast_recovery(image.as_array(), nema_phantom) for image in images]
    )
    _, cov_sem = _nema.mean_and_sem(
        [_nema.background_cov(image.as_array(), nema_phantom) for image in images]
    )
    cov = np.mean(
        [_nema.background_cov(image.as_array(), nema_phantom) for image in images]
    )
    stopping[iterations] = (images, recovery, recovery_sem, cov, cov_sem)
    print(
        f"{iterations:>2} it     "
        + "".join(f"{value:8.2f}" for value in recovery)
        + f"{cov:8.2f}"
    )

# %%
# Reconstruction and truth
# ------------------------
# Both images are shown relative to their mean in the background mask, so the
# arbitrary scale of the reconstruction does not matter. The reconstruction is
# the mean of the three realisations.

images, recovery, recovery_sem, cov, cov_sem = stopping[_nema.OSEM_ITERATIONS]
recon_array = nema_phantom.relative_to_background(
    np.mean([image.as_array() for image in images], axis=0)
)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
for ax, (title, image) in zip(axes, (("Truth", truth_array), ("OSEM", recon_array))):
    shown = ax.imshow(image[slice_z], cmap="gray_r", vmin=0, vmax=truth_array.max())
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
plt.show()

# %%
# Contrast recovery
# -----------------
# For each hot sphere, the mean in its mask over the mean in the background
# mask, divided by the true ratio of 8. A value of 1 is full recovery. The
# masks include voxels on the sphere edge, which hold less activity, so even
# the truth scores below 1; its score is the ceiling for the reconstruction.
# The 10 and 13 mm spheres are cold and are not scored. The error bars are the
# standard error over the three independent Poisson realisations.

diameters = [str(d) for d in _nema.SPHERES.values()]
x = np.arange(len(diameters))
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
ax.bar(x - 0.2, truth_recovery, 0.4, label="Truth")
ax.bar(
    x + 0.2,
    recovery,
    0.4,
    yerr=recovery_sem,
    capsize=3,
    label=f"OSEM, {_nema.OSEM_ITERATIONS} iterations",
)
ax.set_xticks(x, diameters)
ax.set_xlabel("Sphere diameter (mm)")
ax.set_ylabel("Contrast recovery")
ax.set_ylim(0, max(truth_recovery) * 1.15)
ax.legend()
plt.show()

for diameter, value, error, limit in zip(
    diameters, recovery, recovery_sem, truth_recovery
):
    print(f"{diameter} mm sphere: {value:.2f} ± {error:.2f} (truth {limit:.2f})")
