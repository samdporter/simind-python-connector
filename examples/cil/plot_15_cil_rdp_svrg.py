"""
RDP-regularised reconstruction with SVRG
========================================

Reconstruct a simulated NEMA/IEC phantom acquisition with a SIRF relative
difference prior (RDP) and CIL's stochastic variance reduced gradient (SVRG),
and compare it with OSEM. Both reconstructions use a SIMIND scatter estimate
that is refreshed from the current image during the run.

The data are simulated by SIMIND with four histories per measured count, so
the Monte Carlo variance of the expectation is a quarter of the Poisson
variance of the acquisition. The reconstruction uses a fast SPECTUB model that
has no scatter. SIMIND provides the scatter: all interactions minus the
geometric primary, scaled to the fast model and added to its forward
projection. The RDP is smooth, so it is part of the gradient-based objective
rather than a proximal term.

The prior strength and epoch count are chosen with a paired pilot on one
realisation, using one fixed scatter estimate and the same data: beta 0, 0.05,
0.1 and 0.2 at 12 and 24 epochs. The chosen rule is the strongest prior whose
37 mm recovery is within 0.05 of the unregularised recovery, at the shorter
epoch count whose recovery has changed by less than 0.02 from 24 epochs. The
final comparison then reruns the chosen protocol with a refreshed scatter
estimate on three independent Poisson realisations.

.. warning::
   The first tutorial run simulates the shared expectation and takes about an
   hour with 8 MPI processes. Each scatter estimate uses a fortieth of that
   history budget and takes about a minute; this example makes one estimate
   for the pilot and eighteen in the final run, and the SVRG pilot and final
   runs add repeated subset-gradient set-ups. It takes about two to three
   hours. Later tutorials reuse the cached expectation.

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
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import (
    Function,
    IndicatorBox,
    SumFunction,
    SVRGFunction,
)
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
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "cil_rdp_svrg"
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
# Poisson noise from a seeded generator is added to the expectation at 2 s per
# view; the three realisations are shared by both reconstructions.

expectation = _nema.simulate_expectation(nema_phantom, template)
realisations = dict(_nema.paired_realisations(expectation, template))
print(
    f"Expectation: {expectation.total:.4g} counts from "
    f"{expectation.histories:.4g} histories, "
    f"quantization loss {expectation.quantization_loss:.2%}"
)
for seed, measured in realisations.items():
    print(f"Seed {seed}: {measured.sum():.4g} measured counts")

# %%
# Model and scatter updates
# -------------------------
# The model is SPECTUB with the flipped attenuation map and no PSF. Six subsets
# of 10 views are used by both reconstructions. ``SimindProjector`` simulates
# the current image with the template geometry, and ``reference_normaliser``
# scales the geometric primary of that simulation to the fast model's forward
# projection of the same image. ``ScatterCorrection`` keeps all interactions
# minus the geometric primary and smooths the estimate over 3 bins, across
# views as well. The additive term starts at zero and averages successive
# estimates (``damping=0.5``). ``seed=None`` leaves the ``RR`` switch unset, so
# the 8 MPI processes use their fixed seeds.

NUM_SUBSETS = _nema.NUM_SUBSETS
OSEM_SUBITERATIONS = 12
OSEM_UPDATES = 3
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


# %%
# OSEM
# ----
# The comparison uses the same recipe as the OSEM scatter tutorial: 6 subsets,
# passes of 12 subiterations and three scatter updates, 8 full iterations in
# all.

osem = {}
osem_updaters = {}
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
    osem_updaters[seed] = updater

# %%
# RDP + SVRG
# ----------
# The RDP is smooth, so it belongs in the smooth part of ISTA. Each subset
# objective carries the prior with a sixth of its strength, so the SVRG snapshot
# gradient still sums to the full prior. This is the split used by the MaGeZ
# algorithm submitted to the PETRIC reconstruction challenge, whose late-epoch
# step size is 0.5. With CIL's ``AdaptiveSensitivity``, a 0.5 step made edge
# voxels grow without bound in this setup: the prior pilot returned non-finite
# images and the refreshed scatter estimate fed the growth back into the data
# term. A 0.1 step is stable over both pilot epoch counts and is used
# throughout; it is a numerical choice, not a value tuned for recovery.
#
# ``SVRGFunction`` draws one subset per iteration and retakes the full gradient
# every 12 iterations. ``CorrectionCallback`` refreshes the scatter estimate
# every three epochs and then the stored SVRG gradients.

ETA_PILOT = (0.0, 0.05, 0.1, 0.2)
EPOCH_PILOT = (12, 24)
STEP_SIZE = 0.1
UPDATE_EVERY_EPOCHS = 3


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


def svrg_run(measured, additive, strength, epochs, callbacks):
    objectives = build_subset_objectives(
        measured, additive, make_model, NUM_SUBSETS, initial_image
    )
    subset_functions = [
        SumFunction(f, rdp_term(strength)) for f in objectives.functions
    ]
    svrg = SVRGFunction(
        subset_functions,
        sampler=Sampler.sequential(NUM_SUBSETS),
        snapshot_update_interval=2 * NUM_SUBSETS,
    )
    iterations = epochs * NUM_SUBSETS
    algorithm = ISTA(
        initial=initial_image,
        f=svrg,
        g=IndicatorBox(lower=0),
        step_size=STEP_SIZE,
        preconditioner=AdaptiveSensitivity(full_model, max_iterations=iterations + 1),
    )
    algorithm.run(iterations, callbacks=callbacks(objectives, iterations))
    return algorithm.solution, objectives


# %%
# Prior and epoch pilot
# ---------------------
# The pilot uses one realisation and one scatter estimate, made from the
# unregularised OSEM image and then held fixed. Only beta and the epoch count
# change, so the comparison isolates the prior. For each run we report the
# recovery at 37 mm and the background noise.

pilot_seed = next(iter(realisations))
pilot_projector = SimindProjector(
    _nema.make_adaptor(
        estimate_dir,
        "pilot_estimate",
        quantization_scale=ESTIMATE_QUANTIZATION_SCALE,
        histories=1,
    ),
    template,
    mu_image,
    normalise=reference_normaliser(full_model),
    seed=None,
)
pilot_additive = ScatterCorrection(
    pilot_projector, smoothing_fwhm_bins=3.0, smooth_across_views=True
).estimate(osem[pilot_seed])

pilot = {}
for epochs in EPOCH_PILOT:
    for strength in ETA_PILOT:
        image, _ = svrg_run(
            realisations[pilot_seed],
            pilot_additive,
            strength,
            epochs,
            lambda objectives, iterations: [],
        )
        recovery, cov = _nema.contrast_recovery_scores(image.as_array(), nema_phantom)
        pilot[strength, epochs] = (image, recovery, cov)
        print(
            f"beta {strength:.2f}, {epochs:2d} epochs: "
            f"37 mm {recovery[-1]:.2f}, CoV {cov:.2f}"
        )

# Choose the strongest prior whose 37 mm recovery is within 0.05 of the
# unregularised run, then the shorter epoch count whose 37 mm recovery is within
# 0.02 of the longer one.
candidates = [
    strength
    for strength in ETA_PILOT
    if abs(pilot[strength, EPOCH_PILOT[0]][1][-1] - pilot[0.0, EPOCH_PILOT[0]][1][-1])
    <= 0.05
]
RDP_STRENGTH = max(candidates)
RDP_EPOCHS = min(
    epochs
    for epochs in EPOCH_PILOT
    if abs(
        pilot[RDP_STRENGTH, epochs][1][-1] - pilot[RDP_STRENGTH, EPOCH_PILOT[-1]][1][-1]
    )
    <= 0.02
)
print(f"Chosen protocol: beta {RDP_STRENGTH:.2f}, {RDP_EPOCHS} epochs")

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for strength in ETA_PILOT:
    ax.plot(
        EPOCH_PILOT,
        [pilot[strength, epochs][1][-1] for epochs in EPOCH_PILOT],
        "o-",
        label=f"beta {strength:.2f}",
    )
ax.set_xlabel("Epochs")
ax.set_ylabel("37 mm contrast recovery")
ax.legend()
plt.show()

# %%
# Final RDP + SVRG and OSEM
# -------------------------
# The chosen protocol is rerun on every realisation with a scatter estimate
# refreshed every three epochs, so the SVRG gradients and the additive term are
# refreshed together. The epoch count is the pilot's choice, not a value picked
# by eye.

ITERATIONS = RDP_EPOCHS * NUM_SUBSETS
UPDATE_EVERY = UPDATE_EVERY_EPOCHS * NUM_SUBSETS
svrg_image = {}
svrg_updaters = {}
for seed in realisations:
    updater = make_scatter_updater(
        f"svrg_estimate_{seed}",
        UpdateSchedule(every=UPDATE_EVERY, stop_at=ITERATIONS),
    )

    def callbacks(objectives, iterations, updater=updater):
        return [CorrectionCallback(updater, objectives)]

    svrg_image[seed], _ = svrg_run(
        realisations[seed],
        zero_additive,
        RDP_STRENGTH,
        RDP_EPOCHS,
        callbacks,
    )
    svrg_updaters[seed] = updater

for seed in realisations:
    for name, updater in (
        ("OSEM", osem_updaters[seed]),
        ("SVRG", svrg_updaters[seed]),
    ):
        for record in updater.history:
            print(
                f"seed {seed} {name} update at iteration {record.iteration}: "
                f"scatter sum {record.additive_sum:.4g}, {record.seconds:.0f} s"
            )

true_scatter = expectation.true_scatter
print(f"True scatter sum: {true_scatter.sum():.4g}")
for seed in realisations:
    print(
        f"Seed {seed} scatter fraction: "
        f"true {true_scatter.sum() / realisations[seed].sum():.0%}, "
        f"OSEM {osem_updaters[seed].current.sum() / realisations[seed].sum():.0%}, "
        f"SVRG {svrg_updaters[seed].current.sum() / realisations[seed].sum():.0%}"
    )

# %%
# Reconstructions and profile
# ---------------------------
# The images are shown relative to their mean in the background mask. Each
# image and profile is the mean over the three realisations; the band is the
# standard deviation. The profile runs along x through the centres of the 22
# and 37 mm spheres.

background = nema_phantom.background


def mean_relative(images):
    array = np.mean([image.as_array() for image in images], axis=0)
    return nema_phantom.relative_to_background(array)


truth_array = nema_phantom.relative_to_background(nema.activity_mbq)
arrays = {
    "OSEM": mean_relative(osem.values()),
    "RDP + SVRG": mean_relative(svrg_image.values()),
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, (title, image) in zip(axes, (("Truth", truth_array), *arrays.items())):
    shown = ax.imshow(image[slice_z], cmap="gray_r", vmin=0, vmax=truth_array.max())
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
plt.show()

difference = arrays["RDP + SVRG"] - arrays["OSEM"]
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
limit = np.abs(difference).max()
shown = ax.imshow(difference[slice_z], cmap="coolwarm", vmin=-limit, vmax=limit)
ax.set_title("RDP + SVRG − OSEM")
ax.axis("off")
fig.colorbar(shown, ax=ax, label="difference, relative to background")
plt.show()


def relative_profile(image):
    array = image.as_array()
    return (
        array[nema_phantom.slice_z, nema_phantom.profile_row] / array[background].mean()
    )


profiles = {
    "Truth": np.array([relative_profile(activity_image)]),
    "OSEM": np.array([relative_profile(image) for image in osem.values()]),
    "RDP + SVRG": np.array([relative_profile(image) for image in svrg_image.values()]),
}
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for title, values in profiles.items():
    mean = values.mean(axis=0)
    spread = values.std(axis=0)
    ax.plot(mean, "k--" if title == "Truth" else "-", label=title)
    if len(values) > 1:
        ax.fill_between(np.arange(mean.size), mean - spread, mean + spread, alpha=0.15)
ax.set_xlabel("x (voxel)")
ax.set_ylabel("relative to background")
ax.legend()
plt.show()

# %%
# Contrast recovery and noise
# ---------------------------
# For each hot sphere, the mean in its mask over the mean in the background
# mask, divided by the true ratio of 8. A value of 1 is full recovery. The
# masks include voxels on the sphere edge, which hold less activity, so even
# the truth scores below 1; its score is the ceiling for the reconstructions.
# The 10 and 13 mm spheres are cold and are not scored. The noise is the
# coefficient of variation in the background mask. The error bars are the
# standard error over the three independent realisations.

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
for title, (recovery, recovery_sem, cov, cov_sem) in scores.items():
    print(
        f"{title:<24}" + "".join(f"{value:11.2f}" for value in recovery) + f"{cov:8.2f}"
    )

x = np.arange(len(diameters))
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for offset, (title, (recovery, recovery_sem, _, _)) in zip(
    (-0.2, 0, 0.2), scores.items()
):
    ax.bar(x + offset, recovery, 0.2, yerr=recovery_sem, capsize=2, label=title)
ax.set_xticks(x, diameters)
ax.set_xlabel("Sphere diameter (mm)")
ax.set_ylabel("Contrast recovery")
ax.set_ylim(0, max(truth_recovery) * 1.15)
ax.legend()
plt.show()

# %%
# The printed table and the profile show the trade-off rather than a ranking:
# RDP + SVRG is smoother and its background noise and sphere recovery are
# reported next to OSEM, with the spread over three realisations. The
# comparison also changes the algorithm and step size, not only the prior, so
# it does not isolate the prior's effect. Three realisations do not rank the
# two methods, and the 10 and 13 mm spheres remain cold.
