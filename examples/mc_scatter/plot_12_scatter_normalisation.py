"""
Normalising the Monte Carlo scatter
===================================

A SIMIND run does not know the scale of the data. This example reconstructs one
simulated NEMA/IEC phantom acquisition with SIRF OSEM and a SIMIND scatter
estimate that is refreshed from the current image, and compares five ways of
putting that estimate on the scale of the data:

1. **Known activity.** The phantom's total activity and the time per view go to
   SIMIND (Index 25), so its output is in counts and no factor is applied.
2. **Reference, sum.** The geometric primary of the SIMIND run is scaled to the
   fast model's forward projection of the same image, with the ratio of sums.
3. **Reference, trimmed.** The same, with a trimmed mean of the bin ratios.
4. **Dewaraja, global.** One factor scales SIMIND's total to the measured data
   (Dewaraja et al., Med Phys 44:6364, 2017).
5. **Dewaraja, pixel-wise.** The scatter fraction of SIMIND's run in each bin,
   times the measured data, as in the same paper.

The expectation is simulated once with four histories per measured count, so
its Monte Carlo variance is a quarter of the Poisson variance. Each strategy is
run on the same three independent Poisson realisations of it, with the same
OSEM schedule, smoothing and damping, so the comparison is paired. The scatter
estimates use a fortieth of the expectation's history budget; the smoothing
over 3 bins and across views suppresses their Monte Carlo noise.

.. warning::
   The first tutorial run simulates the shared expectation and takes about an
   hour with 8 MPI processes. Each of the five strategies makes three scatter
   estimates on each of three realisations, at about a minute each; this
   example takes about two hours in all. Later tutorials reuse the cached
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
    measured_normaliser,
    reference_normaliser,
)
from simind_python_connector.recon.osem import run_osem_with_corrections


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError(
        "SIMIND executable not found ($SIMIND_BIN or simind on PATH). "
        "Install SIMIND before running this example."
    )

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "scatter_normalisation"
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
# Poisson noise from a seeded generator is added to the expectation. The true
# scatter is the expectation's all interactions minus its geometric primary.

expectation = _nema.simulate_expectation(nema_phantom, template)
TOTAL_MBQ = nema_phantom.total_mbq
TIME_PER_VIEW_S = _nema.REFERENCE_TIME_S
realisations = dict(_nema.paired_realisations(expectation, template))
true_scatter = expectation.true_scatter
print(
    f"Expectation: {expectation.total:.4g} counts, true scatter "
    f"{true_scatter.sum():.4g}, quantization loss "
    f"{expectation.quantization_loss:.2%}"
)
for seed, measured in realisations.items():
    print(f"Seed {seed}: {measured.sum():.4g} measured counts")

# %%
# The fast model
# --------------
# The reconstruction model is SPECTUB with the flipped attenuation map and no
# PSF. It has no collimator blur and no scatter. ``run_osem_with_corrections``
# runs OSEM in passes and calls the updater between them; with a zero additive
# term and no updates it is plain OSEM. Every arm uses 6 subsets, 12
# subiterations per pass and three updates (8 full iterations).

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

# %%
# Five normalisations
# -------------------
# Each strategy reconstructs the same measured data with a SIMIND scatter
# estimate refreshed after every pass of 12 subiterations. The estimate is
# smoothed over 3 bins, across views as well, and the additive term starts at
# zero and averages successive estimates (``damping=0.5``). With ``seed=None``
# the 8 MPI processes use their fixed seeds in every call, so every strategy
# uses the same rank seeds in every pass.
#
# ``RecordedScatter`` keeps every raw estimate, before the damping, the image it
# was made from, and the total of each SIMIND run before it is scaled.

ESTIMATE_QUANTIZATION_SCALE = 0.1 * _nema.QUANTIZATION_SCALE


class RecordedScatter(ScatterCorrection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimates = []
        self.images = []
        self.simind_totals = []

    def estimate(self, image):
        scatter = super().estimate(image)
        self.estimates.append(scatter)
        self.images.append(image.clone())
        total = self.projector.last_outputs["all_interactions"].as_array()
        self.simind_totals.append(total / self.last_scale)
        return scatter


def strategy_options(title, measured):
    """The projector and correction options of one normalisation strategy."""
    if title == "Known activity":
        return (
            {
                "activity_mbq": lambda image: TOTAL_MBQ,
                "time_per_projection_s": TIME_PER_VIEW_S,
            },
            {},
        )
    if title == "Reference, sum":
        return {"normalise": reference_normaliser(full_model, method="sum")}, {}
    if title == "Reference, trimmed":
        return {"normalise": reference_normaliser(full_model, method="trimmed")}, {}
    if title == "Dewaraja, global":
        return {"normalise": measured_normaliser(measured)}, {}
    if title == "Dewaraja, pixel-wise":
        # The projector still needs a normaliser; its scale cancels in the ratio.
        return (
            {"normalise": measured_normaliser(measured)},
            {"measured": measured},
        )
    raise ValueError(f"Unknown strategy {title!r}")


STRATEGIES = (
    "Known activity",
    "Reference, sum",
    "Reference, trimmed",
    "Dewaraja, global",
    "Dewaraja, pixel-wise",
)


def reconstruct(seed, projector_options, correction_options):
    measured = realisations[seed]
    projector = SimindProjector(
        _nema.make_adaptor(
            estimate_dir,
            f"estimate_{seed}",
            quantization_scale=ESTIMATE_QUANTIZATION_SCALE,
            histories=1,
        ),
        template,
        mu_image,
        seed=None,
        **projector_options,
    )
    correction = RecordedScatter(
        projector,
        smoothing_fwhm_bins=3.0,
        smooth_across_views=True,
        **correction_options,
    )
    updater = AdditiveUpdater(
        correction,
        UpdateSchedule(every=SUBITERATIONS_PER_UPDATE),
        zero_additive,
        damping=0.5,
    )
    image = run_osem_with_corrections(
        measured,
        make_acq_model,
        initial_image,
        updater,
        NUM_SUBSETS,
        SUBITERATIONS_PER_UPDATE,
        NUM_UPDATES,
    )
    return image, updater, correction


runs = {
    title: {
        seed: reconstruct(seed, *strategy_options(title, realisations[seed]))
        for seed in realisations
    }
    for title in STRATEGIES
}

# %%
# The first update is shared by every strategy
# --------------------------------------------
# The first pass has a zero additive term and the same image for every
# strategy, so with the same seeds every strategy hands SIMIND the same source.
# The raw totals of that update therefore differ only by float rounding. The
# table gives the largest difference between the totals, each normalised to
# unit sum, relative to the largest bin.

for seed in realisations:
    first_totals = {
        title: runs[title][seed][2].simind_totals[0]
        / runs[title][seed][2].simind_totals[0].sum()
        for title in STRATEGIES
    }
    reference_total = first_totals["Known activity"]
    differences = [
        np.abs(total - reference_total).max() / reference_total.max()
        for total in first_totals.values()
    ]
    print(f"Seed {seed}: largest first-update difference {max(differences):.1e}")

# %%
# The differences are at the level of float rounding, so every strategy sees
# the same Monte Carlo noise at the first update: only the normalisation
# differs. Later updates use the same seeds but images that differ with the
# strategy, so their noise differs too.

# %%
# The scale of each update
# ------------------------
# Known activity applies no factor. Each of the other strategies scales every
# SIMIND output by one factor per update. The raw estimate of an update is the
# smoothed scatter of its SIMIND run, before the damped average with the
# estimates of the earlier updates. The lines and bands are the mean and
# standard deviation over the three realisations.

updates = np.arange(1, NUM_UPDATES + 1)
true_sum = true_scatter.sum()
print(f"True scatter sum: {true_sum:.4g}")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for color, title in enumerate(STRATEGIES):
    color = f"C{color}"
    scales = np.array(
        [
            [record.scale for record in runs[title][seed][1].history]
            for seed in realisations
        ]
    )
    sums = np.array(
        [
            [estimate.sum() for estimate in runs[title][seed][2].estimates]
            for seed in realisations
        ]
    )
    if title != "Known activity":
        axes[0].plot(updates, scales.mean(axis=0), "o-", color=color, label=title)
    axes[1].plot(updates, sums.mean(axis=0), "o-", color=color, label=title)
axes[1].axhline(true_sum, color="k", linestyle="--", label="True scatter")
axes[0].set_title("Scale factor")
axes[1].set_title("Sum of the raw scatter estimate")
for ax in axes:
    ax.set_xticks(updates)
    ax.set_xlabel("Update")
axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

for title in STRATEGIES:
    scales = ", ".join(
        f"{np.mean([record.scale for record in runs[title][seed][1].history]):.4g}"
        for seed in realisations
    )
    sums = ", ".join(
        f"{np.mean([e.sum() for e in runs[title][seed][2].estimates]):.4g}"
        for seed in realisations
    )
    print(f"{title}\n  mean scale: {scales}\n  mean raw estimate sum: {sums}")

# %%
# The last raw estimates
# ----------------------
# Each strategy's last raw estimate against the true scatter: the sum, and the
# sum of the absolute difference over the true sum. The additive term of the
# last OSEM pass is a damped average of the estimates, so it is not what is
# shown here.

true_array = true_scatter
last_estimates = {
    title: np.mean(
        [runs[title][seed][2].estimates[-1].as_array() for seed in realisations],
        axis=0,
    )
    for title in STRATEGIES
}
for title, estimate in last_estimates.items():
    error = np.abs(estimate - true_array).sum() / true_array.sum()
    print(f"{title}: mean sum {estimate.sum():.4g}, absolute difference {error:.0%}")

expected_array = expectation.counts
central_view = true_array.shape[-2] // 2
maps = {"True scatter": true_array, **last_estimates}
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), constrained_layout=True)
for ax, (title, scatter) in zip(axes.flat, maps.items()):
    fraction = np.divide(
        scatter,
        expected_array,
        out=np.full(scatter.shape, np.nan),
        where=expected_array >= 1,
    )
    shown = ax.imshow(fraction[0][:, central_view, :], cmap="magma", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(shown, ax=axes, label=f"scatter / expected counts, view {central_view}")
plt.show()

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for title, scatter in maps.items():
    ax.plot(
        scatter[0].sum(axis=(0, 2)),
        "k--" if title == "True scatter" else "-",
        label=title,
    )
ax.set_xlabel("View")
ax.set_ylabel("Scatter counts in the view")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %%
# Reconstructions
# ---------------
# The images are shown relative to their mean in the background mask. The
# profile runs along x through the centres of the 22 and 37 mm spheres. Each
# image and profile is the mean over the three realisations; the band is the
# standard deviation.

background = nema_phantom.background


def mean_relative(images):
    array = np.mean([image.as_array() for image in images], axis=0)
    return nema_phantom.relative_to_background(array)


arrays = {
    "Truth": nema_phantom.relative_to_background(nema.activity_mbq),
    "No correction": mean_relative(uncorrected.values()),
    **{
        title: mean_relative([runs[title][seed][0] for seed in realisations])
        for title in STRATEGIES
    },
}

fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), constrained_layout=True)
for ax, (title, image) in zip(axes.flat, arrays.items()):
    shown = ax.imshow(
        image[nema_phantom.slice_z],
        cmap="gray_r",
        vmin=0,
        vmax=arrays["Truth"].max(),
    )
    ax.set_title(title)
    ax.axis("off")
axes.flat[-1].axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
plt.show()

difference = arrays["Known activity"] - arrays["No correction"]
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
limit = np.abs(difference).max()
shown = ax.imshow(
    difference[nema_phantom.slice_z], cmap="coolwarm", vmin=-limit, vmax=limit
)
ax.set_title("Known activity − no correction")
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
    "No correction": np.array(
        [relative_profile(image) for image in uncorrected.values()]
    ),
    **{
        title: np.array(
            [relative_profile(runs[title][seed][0]) for seed in realisations]
        )
        for title in STRATEGIES
    },
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
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %%
# Contrast recovery, noise and total activity
# -------------------------------------------
# Contrast recovery is, for each hot sphere, the mean in its mask over the mean
# in the background mask, divided by the true ratio of 8. A value of 1 is full
# recovery. The masks include voxels on the sphere edge, which hold less
# activity, so even the truth scores below 1: its score is the ceiling. The 10
# and 13 mm spheres are cold and are not scored. The noise is the coefficient
# of variation in the background mask. The error bars are the standard error
# over the three realisations.
#
# The reconstruction has the scale of the fast model, not of the activity. The
# total activity is its image sum over the truth's sum, with the scale fixed by
# the true primary of the data: counts of the primary per unit of the fast
# model's projection of the true activity. A value of 1 is the true total.

truth_recovery = _nema.contrast_recovery(nema.activity_mbq, nema_phantom)
truth_cov = _nema.background_cov(nema.activity_mbq, nema_phantom)
counts_per_model_count = (
    expectation.primary.sum() / full_model.direct(activity_image).sum()
)
true_total = counts_per_model_count * nema.activity_mbq.sum()


def table_row(title, safe, recovery, total):
    cov, cov_sem = _nema.mean_and_sem(safe)
    table.append(
        (
            title,
            recovery.mean(axis=0),
            recovery.std(axis=0, ddof=1) / np.sqrt(len(recovery)),
            float(cov),
            float(cov_sem),
            float(np.mean(total) / true_total),
            float(np.std(total, ddof=1) / np.sqrt(len(total)) / true_total),
        )
    )


table = []
table.append(
    (
        "Truth",
        np.array(truth_recovery),
        np.zeros(4),
        truth_cov,
        0.0,
        1.0,
        0.0,
    )
)
table_row(
    "No correction",
    [
        _nema.background_cov(image.as_array(), nema_phantom)
        for image in uncorrected.values()
    ],
    np.array(
        [
            _nema.contrast_recovery(image.as_array(), nema_phantom)
            for image in uncorrected.values()
        ]
    ),
    np.array([image.sum() for image in uncorrected.values()]),
)
for title in STRATEGIES:
    images = [runs[title][seed][0] for seed in realisations]
    table_row(
        title,
        [_nema.background_cov(image.as_array(), nema_phantom) for image in images],
        np.array(
            [
                _nema.contrast_recovery(image.as_array(), nema_phantom)
                for image in images
            ]
        ),
        np.array([image.sum() for image in images]),
    )

diameters = [str(d) for d in _nema.SPHERES.values()]
print(f"{'':22}" + "".join(f"{d:>7} mm" for d in diameters) + "        CoV       total")
for title, recovery, recovery_sem, cov, cov_sem, total, total_sem in table:
    print(
        f"{title:<22}"
        + "".join(f"{value:10.2f}" for value in recovery)
        + f"{cov:7.2f}±{cov_sem:.2f}{total:7.2f}±{total_sem:.2f}"
    )

x = np.arange(len(diameters))
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for offset, row in zip(np.linspace(-0.32, 0.32, len(table)), table):
    title, recovery, recovery_sem = row[0], row[1], row[2]
    ax.bar(x + offset, recovery, 0.16, yerr=recovery_sem, capsize=2, label=title)
ax.set_xticks(x, diameters)
ax.set_xlabel("Sphere diameter (mm)")
ax.set_ylabel("Contrast recovery")
ax.set_ylim(0, max(truth_recovery) * 1.15)
ax.legend(fontsize=8)
plt.show()

# %%
# The known-activity, Dewaraja global and pixel-wise strategies track the true
# scatter sum at every update. The two reference normalisations sit above it,
# because they inherit the scale of the image before scatter correction, which
# has more activity than the phantom. Three realisations separate those two
# groups but do not rank the strategies within a group. The total activity of
# the reconstruction reflects the same scale difference.
