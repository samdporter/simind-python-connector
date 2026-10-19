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

.. warning::
   This example takes about 40 minutes, nearly all of it in the SIMIND runs:
   the simulation of the data and one scatter estimate per update for each of
   the five strategies. Every run uses 8 MPI processes.

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

from simind_python_connector import SirfSimindAdaptor, configs
from simind_python_connector.builders import (
    STIRSPECTAcquisitionDataBuilder,
    STIRSPECTImageDataBuilder,
)
from simind_python_connector.converters.attenuation import get_attenuation_coefficient
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
)
from simind_python_connector.phantoms import nema_iec_phantom
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

GRID = (48, 48, 48)
VOXEL_MM = 7.0

# %%
# NEMA/IEC phantom
# ----------------
# phantomgen's ``pet`` preset has an 8:1 sphere-to-background activity ratio.
# Its two smallest spheres, 10 and 13 mm, are cold; the four hot spheres are
# 17, 22, 28 and 37 mm. The activity is in MBq per voxel. The attenuation map is
# the density times the attenuation coefficient of water at 140 keV.

nema = nema_iec_phantom(GRID, (VOXEL_MM,) * 3, preset="pet", supersample=4)
mu = nema.density_g_cm3 * get_attenuation_coefficient("water", 140.0)
slice_z, profile_row = (
    np.argwhere(nema.masks["sphere_6"]).mean(axis=0).round().astype(int)[:2]
)


def to_sirf_image(array):
    builder = STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": str(GRID[2]),
            "!matrix size [2]": str(GRID[1]),
            "!matrix size [3]": str(GRID[0]),
            "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [3]": str(VOXEL_MM),
        },
        backend="sirf",
    )
    builder.set_pixel_array(array.astype(np.float32))
    return builder.build()


activity_image = to_sirf_image(nema.activity_mbq)
mu_image = to_sirf_image(mu)
# STIR SPECTUB mirrors x relative to the mu map (UCL/STIR#452)
spectub_mu_image = to_sirf_image(mu[..., ::-1])

# %%
# Template
# --------
# The template sets the projection geometry: 48 x 48 bins of 7 mm, 60 views
# over 360 degrees and a detector radius of 250 mm. Both the simulation and the
# reconstruction take their geometry from it.

sirf.AcquisitionData.set_storage_scheme("memory")
template = STIRSPECTAcquisitionDataBuilder(
    header_overrides={
        "!matrix size [1]": "48",
        "!matrix size [2]": "48",
        "!number of projections": "60",
        "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
        "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
        "!direction of rotation": "CW",
        "start angle": "180",
        "Radius": "250",
    },
    backend="sirf",
).build(output_path=output_dir / "template")

# %%
# "Measured" data
# ---------------
# SIMIND simulates the phantom for Tc-99m with a 126-154 keV window and the
# ma-lehr collimator. The total activity and the time per view are given, so the
# projections come out in counts (Index 25). The 8 MPI processes each simulate
# all views and SIMIND sums them; without an ``RR`` switch their seeds are fixed.
# Poisson noise from a seeded generator is added to the expectation. The true
# scatter is the expectation's all interactions minus its geometric primary.

TOTAL_MBQ = float(nema.activity_mbq.sum())
TIME_PER_VIEW_S = 2.0
QUANTIZATION_SCALE = 0.06 * DEFAULT_QUANTIZATION_SCALE


def make_adaptor(name, quantization_scale=QUANTIZATION_SCALE):
    adaptor = SirfSimindAdaptor(
        configs.get("Example.yaml"),
        str(output_dir),
        name,
        scoring_routine=ScoringRoutine.PENETRATE,
        quantization_scale=quantization_scale,
    )
    adaptor.set_energy_windows([126], [154], [0])
    adaptor.add_runtime_switch("CC", "ma-lehr")
    adaptor.set_mpi(8)
    adaptor.add_config_value(1, 140.0)
    return adaptor


adaptor = make_adaptor("measured")
adaptor.set_template(template)
adaptor.set_source(activity_image)
adaptor.set_mu_map(mu_image)
adaptor.set_activity(TOTAL_MBQ, TIME_PER_VIEW_S)
outputs = adaptor.run()
expected = outputs["all_interactions"]
true_scatter = expected - outputs["geom_coll_primary"]

rng = np.random.default_rng(12345)
measured = expected.clone()
measured.fill(rng.poisson(np.maximum(expected.as_array(), 0)).astype(np.float32))
print(f"Measured counts: {measured.sum():.4g}, true scatter: {true_scatter.sum():.4g}")

# %%
# The fast model
# --------------
# The reconstruction model is SPECTUB with the flipped attenuation map and no
# PSF. It has no collimator blur and no scatter. OSEM runs with 6 subsets of 10
# views each. ``run_osem_with_corrections`` runs OSEM in passes and calls the
# updater between them; with a zero additive term and no updates it is plain
# OSEM. Every reconstruction uses 48 subiterations, 8 full iterations.

NUM_SUBSETS = 6
SUBITERATIONS_PER_UPDATE = 12
NUM_UPDATES = 3
initial_image = activity_image.get_uniform_copy(1.0)
zero_additive = measured.get_uniform_copy(0)


def make_acq_model():
    acq_matrix = sirf.SPECTUBMatrix()
    acq_matrix.set_attenuation_image(spectub_mu_image)
    acq_matrix.set_resolution_model(0.0, 0.0, False)
    return sirf.AcquisitionModelUsingMatrix(acq_matrix)


full_model = make_acq_model()
full_model.set_up(measured, activity_image)

uncorrected = run_osem_with_corrections(
    measured,
    make_acq_model,
    initial_image,
    AdditiveUpdater(None, UpdateSchedule(every=1), zero_additive),
    NUM_SUBSETS,
    SUBITERATIONS_PER_UPDATE * (NUM_UPDATES + 1),
    num_updates=0,
)

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


STRATEGIES = {
    "Known activity": (
        {
            "activity_mbq": lambda image: TOTAL_MBQ,
            "time_per_projection_s": TIME_PER_VIEW_S,
        },
        {},
    ),
    "Reference, sum": (
        {"normalise": reference_normaliser(full_model, method="sum")},
        {},
    ),
    "Reference, trimmed": (
        {"normalise": reference_normaliser(full_model, method="trimmed")},
        {},
    ),
    "Dewaraja, global": ({"normalise": measured_normaliser(measured)}, {}),
    # The projector still needs a normaliser; its scale cancels in the ratio.
    "Dewaraja, pixel-wise": (
        {"normalise": measured_normaliser(measured)},
        {"measured": measured},
    ),
}


def reconstruct(projector_options, correction_options):
    projector = SimindProjector(
        make_adaptor("estimate"), template, mu_image, seed=None, **projector_options
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


runs = {title: reconstruct(*options) for title, options in STRATEGIES.items()}

# %%
# The first update's Monte Carlo noise
# ------------------------------------
# The first pass has a zero additive term, so every strategy hands SIMIND the
# same image at the first update. With the same seeds, the SIMIND totals of that
# update therefore differ only by the scale factor. The table gives the largest
# difference between the totals, each normalised to unit sum, relative to the
# largest bin.

first_totals = {
    title: correction.simind_totals[0] / correction.simind_totals[0].sum()
    for title, (_, _, correction) in runs.items()
}
reference_total = first_totals["Known activity"]
for title, total in first_totals.items():
    difference = np.abs(total - reference_total).max() / reference_total.max()
    print(f"{title}: {difference:.1e}")

# %%
# At the first update the totals differ by at most 2e-7, which is float rounding,
# so every strategy sees the same Monte Carlo noise there: only the
# normalisation differs. Later updates use the same seeds but images that differ
# with the strategy, so their noise differs too.

# %%
# The scale of each update
# ------------------------
# Known activity applies no factor. Each of the other strategies scales every
# SIMIND output by one factor per update. The raw estimate of an update is the
# smoothed scatter of its SIMIND run, before the damped average with the
# estimates of the earlier updates. The first update is made from an image that
# is not corrected for scatter.

true_sum = true_scatter.sum()
print(f"True scatter sum: {true_sum:.4g}")
for title, (_, updater, correction) in runs.items():
    scales = ", ".join(f"{record.scale:.4g}" for record in updater.history)
    sums = ", ".join(f"{estimate.sum():.4g}" for estimate in correction.estimates)
    print(f"{title}\n  scale: {scales}\n  raw estimate sum: {sums}")

updates = np.arange(1, NUM_UPDATES + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for color, (title, (_, updater, correction)) in enumerate(runs.items()):
    color = f"C{color}"
    if title != "Known activity":
        axes[0].plot(
            updates, [record.scale for record in updater.history], "o-", color=color
        )
    axes[1].plot(
        updates,
        [estimate.sum() for estimate in correction.estimates],
        "o-",
        color=color,
        label=title,
    )
axes[1].axhline(true_sum, color="k", linestyle="--", label="True scatter")
axes[0].set_title("Scale factor")
axes[1].set_title("Sum of the raw scatter estimate")
for ax in axes:
    ax.set_xticks(updates)
    ax.set_xlabel("Update")
axes[1].legend()
plt.show()

# %%
# The estimates of known activity and of the Dewaraja global model stay within 1%
# of the true scatter sum (2.327e6) at every update, and the global model's scale
# stays near 1130. The pixel-wise estimate drifts down to about 6% below the
# true sum.
#
# The two reference normalisations estimate about 6.1e6, 1.4e6 and 2.5e6. The
# first is 2.6 times the true scatter, and the reconstruction without correction
# has 2.6 times the true total activity (see the table below). The reference
# scale follows the scale of the current image, which the pass without scatter
# correction overestimates. The scales of sum and trimmed agree within 1% at
# every update.

# %%
# Sum against trimmed at low history
# ----------------------------------
# One run of the first-update image at a hundredth of the histories, scaled both
# ways:

first_update_image = runs["Reference, sum"][2].images[0]
low_history = make_adaptor(
    "low_history", quantization_scale=0.01 * DEFAULT_QUANTIZATION_SCALE
)
low_history.set_template(template)
low_history.set_mu_map(mu_image)
low_history.set_source(first_update_image)
low_history_outputs = low_history.run()
sum_scale = reference_normaliser(full_model, method="sum")(
    low_history_outputs, first_update_image
)
trimmed_scale = reference_normaliser(full_model, method="trimmed")(
    low_history_outputs, first_update_image
)
print(
    f"At 1% of the histories: sum {sum_scale:.4g}, trimmed {trimmed_scale:.4g}, "
    f"ratio {trimmed_scale / sum_scale:.2f}"
)

# %%
# At these history counts ``sum`` and ``trimmed`` agree within 1%. At a
# hundredth of the default history budget, ``trimmed`` scales to about 0.92 of
# ``sum``: low counts bias it low, which is why the other tutorials use ``sum``.

# %%
# The last raw estimates
# ----------------------
# Each strategy's last raw estimate against the true scatter: the sum, and the
# sum of the absolute difference over the true sum. The additive term of the
# last OSEM pass is a damped average of the estimates, so it is not what is
# shown here.

true_array = true_scatter.as_array()
last_estimates = {
    title: correction.estimates[-1].as_array()
    for title, (_, _, correction) in runs.items()
}
for title, estimate in last_estimates.items():
    error = np.abs(estimate - true_array).sum() / true_array.sum()
    print(f"{title}: sum {estimate.sum():.4g}, absolute difference {error:.0%}")

expected_array = expected.as_array()
maps = {"True scatter": true_array, **last_estimates}
central_view = true_array.shape[-2] // 2
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
# Known activity and the Dewaraja global model have a last estimate within 1% of
# the true sum and follow the true scatter over the views. The reference
# normalisations are 8 to 10% above it and the pixel-wise model is 6% below.
# Bin by bin, every strategy differs from the true scatter by about half of the
# true sum, including the strategies whose sums agree.

# %%
# Reconstructions
# ---------------
# The images are shown relative to their mean in the background mask. The
# profile runs along x through the centres of the 22 and 37 mm spheres.

background = nema.masks["background"] > 0


def relative_to_background(array):
    return array / array[background].mean()


recon_arrays = {
    "Truth": relative_to_background(nema.activity_mbq),
    "No correction": relative_to_background(uncorrected.as_array()),
    **{
        title: relative_to_background(image.as_array())
        for title, (image, _, _) in runs.items()
    },
}

fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), constrained_layout=True)
for ax, (title, image) in zip(axes.flat, recon_arrays.items()):
    shown = ax.imshow(
        image[slice_z], cmap="gray_r", vmin=0, vmax=recon_arrays["Truth"].max()
    )
    ax.set_title(title)
    ax.axis("off")
axes.flat[-1].axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
plt.show()

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for title, image in recon_arrays.items():
    ax.plot(
        image[slice_z, profile_row], "k--" if title == "Truth" else "-", label=title
    )
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
# of variation in the background mask.
#
# The reconstruction has the scale of the fast model, not of the activity. The
# total activity is its image sum over the truth's sum, with the scale fixed by
# the true primary of the data: counts of the primary per unit of the fast
# model's projection of the true activity. A value of 1 is the true total.

SPHERES = {"sphere_3": 17, "sphere_4": 22, "sphere_5": 28, "sphere_6": 37}
TRUE_RATIO = 8.0
counts_per_model_count = (
    outputs["geom_coll_primary"].sum() / full_model.direct(activity_image).sum()
)
true_total = counts_per_model_count * nema.activity_mbq.sum()
recon_totals = {"Truth": true_total}
recon_totals["No correction"] = uncorrected.sum()
recon_totals.update({title: image.sum() for title, (image, _, _) in runs.items()})

print(f"{'':22}" + "".join(f"{d:>7} mm" for d in SPHERES.values()) + "   CoV  total")
for title, image in recon_arrays.items():
    recovery = [
        image[nema.masks[name] > 0].mean() / image[background].mean() / TRUE_RATIO
        for name in SPHERES
    ]
    cov = image[background].std() / image[background].mean()
    print(
        f"{title:<22}"
        + "".join(f"{value:10.2f}" for value in recovery)
        + f"{cov:7.2f}{recon_totals[title] / true_total:7.2f}"
    )

# %%
# All five corrections raise the recovery of the four hot spheres above the
# reconstruction without correction (0.24, 0.32, 0.37 and 0.40), and all stay
# below the truth's own score. The differences between strategies, 0.09 at
# most, are small next to the gap to the truth, and there is one noise
# realisation, so the table does not rank them. Trimmed and sum differ by 0.02
# at most.
#
# The total activity separates the strategies more: 1.01 for the reference
# normalisations, 1.23 to 1.28 for the others, and 2.60 without correction. The
# order follows the last scatter sums, with the larger scatter estimate giving
# the smaller total. The background noise of the corrected images, 1.5 to 1.7,
# is higher than without correction, 1.1. The truth's own value is 0.38 because
# the mask includes voxels on the sphere edges.
