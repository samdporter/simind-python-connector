"""Normalising the Monte Carlo scatter.

Reconstruct one simulated NEMA/IEC acquisition with five ways of putting a
SIMIND scatter estimate on the data scale. All five share the same three
Poisson realisations, OSEM schedule, smoothing and damping, so the comparison
is paired.

The fixture is phantomgen's ``pet`` preset: 8:1 sphere-to-background activity in
MBq per voxel, four hot spheres (17, 22, 28 and 37 mm), two cold spheres and
named masks. SIMIND simulates one expectation with the known total activity and
2 s per view, so its output is in counts. The three seeded draws (12345, 12346
and 12347) share that expectation. The estimate uses a fortieth of the
expectation's history budget.

The five methods are:

1. Known activity: pass the phantom's total activity and time per view to
   SIMIND (Index 25), so no scale factor is applied.
2. Reference, sum: scale the geometric primary to the fast model's forward
   projection of the same image by the ratio of sums.
3. Reference, trimmed: the same with a trimmed mean of bin ratios.
4. Dewaraja, global: scale SIMIND's total to the measured data.
5. Dewaraja, pixel-wise: smooth scatter and total with the same Gaussian,
   multiply the scatter-to-total ratio by the measured data, then smooth again.

The fast model is SPECTUB with attenuation and no PSF or scatter. The scatter
estimate is refreshed every 12 subiterations (three updates, 8 full iterations
with 6 subsets), smoothed over 3 bins and across views, and damped by 0.5.
Background CoV is the coefficient of variation in the background mask; recovery
is the hot-sphere mean over the background mean, divided by the true 8:1 ratio.
"""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sirf.STIR as sirf

from examples import _nema
from examples._scatter_normalisation import STRATEGIES, strategy_options
from simind_python_connector.recon import (
    AdditiveUpdater,
    ScatterCorrection,
    SimindProjector,
    UpdateSchedule,
)
from simind_python_connector.recon.osem import run_osem_with_corrections


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError("SIMIND executable not found ($SIMIND_BIN or simind on PATH).")

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "scatter_normalisation"
)
output_dir.mkdir(parents=True, exist_ok=True)
estimate_dir = output_dir / "estimates"
estimate_dir.mkdir(exist_ok=True)

nema_phantom = _nema.phantom()
nema = nema_phantom.nema
activity_image = _nema.to_sirf_image(nema.activity_mbq)
mu_image = _nema.to_sirf_image(nema_phantom.mu)
spectub_mu_image = _nema.to_sirf_image(nema_phantom.mu[..., ::-1])
template = _nema.template(output_dir)

expectation = _nema.simulate_expectation(nema_phantom, template)
realisations = dict(_nema.paired_realisations(expectation, template))
true_scatter = expectation.true_scatter
print(
    f"Expectation counts: {expectation.total:.4g}, "
    f"true scatter {true_scatter.sum():.4g}"
)

NUM_SUBSETS = _nema.NUM_SUBSETS
SUBITERATIONS_PER_UPDATE = 12
NUM_UPDATES = 3
initial_image = activity_image.get_uniform_copy(1.0)
zero_additive = next(iter(realisations.values())).get_uniform_copy(0)
ESTIMATE_QUANTIZATION_SCALE = 0.1 * _nema.QUANTIZATION_SCALE


def make_acq_model():
    matrix = sirf.SPECTUBMatrix()
    matrix.set_attenuation_image(spectub_mu_image)
    matrix.set_resolution_model(0.0, 0.0, False)
    return sirf.AcquisitionModelUsingMatrix(matrix)


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


def reconstruct(seed, title, measured):
    settings = strategy_options(
        full_model,
        measured,
        nema_phantom.total_mbq,
        _nema.REFERENCE_TIME_S,
    )[title]
    projector = SimindProjector(
        _nema.make_adaptor(
            estimate_dir,
            f"{title}_{seed}",
            quantization_scale=ESTIMATE_QUANTIZATION_SCALE,
            histories=1,
        ),
        template,
        mu_image,
        seed=None,
        **settings["projector"],
    )
    correction = ScatterCorrection(
        projector,
        smoothing_fwhm_bins=3.0,
        smooth_across_views=True,
        **settings["correction"],
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
    return image, updater


def raw_estimate_sums(updater):
    """Recover raw estimate sums from the damped additive history.

    With zero start and damping 0.5, E_k = 2 D_k - D_{k-1}; the positivity
    floor contributes at most a few counts and is negligible on this scale.
    """

    raw = []
    previous = 0.0
    for record in updater.history:
        raw.append(2.0 * record.additive_sum - previous)
        previous = record.additive_sum
    return np.array(raw)


runs = {title: {} for title in STRATEGIES}
series = {title: {"scales": [], "raw_sums": []} for title in STRATEGIES}
for title in STRATEGIES:
    for seed, measured in realisations.items():
        image, updater = reconstruct(seed, title, measured)
        runs[title][seed] = image
        series[title]["scales"].append([record.scale for record in updater.history])
        series[title]["raw_sums"].append(raw_estimate_sums(updater))

updates = np.arange(1, NUM_UPDATES + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for color, title in enumerate(STRATEGIES):
    color = f"C{color}"
    scales = np.array(series[title]["scales"])
    sums = np.array(series[title]["raw_sums"])
    if title != "Known activity":
        mean, spread = _nema.mean_and_std(scales)
        axes[0].plot(updates, mean, "o-", color=color, label=title)
        axes[0].fill_between(
            updates, mean - spread, mean + spread, color=color, alpha=0.15
        )
    mean, spread = _nema.mean_and_std(sums)
    axes[1].plot(updates, mean, "o-", color=color, label=title)
    axes[1].fill_between(updates, mean - spread, mean + spread, color=color, alpha=0.15)
axes[1].axhline(true_scatter.sum(), color="k", linestyle="--", label="True scatter")
axes[0].set_title("Scale factor")
axes[1].set_title("Sum of the raw scatter estimate")
for axis in axes:
    axis.set_xticks(updates)
    axis.set_xlabel("Update")
axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.savefig(output_dir / "scale_scatter.png", dpi=140)
plt.show()


def mean_relative(images):
    array = np.mean([image.as_array() for image in images], axis=0)
    return nema_phantom.relative_to_background(array)


truth_recovery = _nema.contrast_recovery(nema.activity_mbq, nema_phantom)
truth_cov = _nema.background_cov(nema.activity_mbq, nema_phantom)
counts_per_model_count = (
    expectation.primary.sum() / full_model.direct(activity_image).sum()
)
true_total = counts_per_model_count * nema.activity_mbq.sum()
table = [("Truth", np.array(truth_recovery), np.zeros(4), truth_cov, 0.0, 1.0, 0.0)]


def add_row(title, images):
    recovery, recovery_sem = _nema.mean_and_sem(
        [_nema.contrast_recovery(image.as_array(), nema_phantom) for image in images]
    )
    cov, cov_sem = _nema.mean_and_sem(
        [_nema.background_cov(image.as_array(), nema_phantom) for image in images]
    )
    total = np.array([image.sum() for image in images])
    table.append(
        (
            title,
            recovery,
            recovery_sem,
            float(cov),
            float(cov_sem),
            float(total.mean() / true_total),
            float(total.std(ddof=1) / np.sqrt(len(total)) / true_total),
        )
    )


add_row("No correction", list(uncorrected.values()))
for title in STRATEGIES:
    add_row(title, [runs[title][seed] for seed in realisations])

diameters = [str(d) for d in _nema.SPHERES.values()]
print(f"{'':22}" + "".join(f"{d:>7} mm" for d in diameters) + "        CoV       total")
for title, recovery, recovery_sem, cov, cov_sem, total, total_sem in table:
    print(
        f"{title:<22}"
        + "".join(f"{value:10.2f}" for value in recovery)
        + f"{cov:7.2f}±{cov_sem:.2f}{total:7.2f}±{total_sem:.2f}"
    )

x = np.arange(len(diameters))
fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
for offset, row in zip(np.linspace(-0.32, 0.32, len(table)), table):
    title, recovery, recovery_sem = row[0], row[1], row[2]
    axis.bar(x + offset, recovery, 0.16, yerr=recovery_sem, capsize=2, label=title)
axis.set_xticks(x, diameters)
axis.set_xlabel("Sphere diameter (mm)")
axis.set_ylabel("Contrast recovery")
axis.set_ylim(0, max(truth_recovery) * 1.15)
axis.legend(fontsize=8)
fig.savefig(output_dir / "recovery.png", dpi=140)
plt.show()

# Recorded: all strategies raise recovery above no correction and stay below
# the truth. Raw scatter sums are 5.686e5, 5.915e5, 5.968e5, 5.730e5 and
# 5.695e5 against a true 5.703e5. Total activity is 1.05 for known activity,
# Dewaraja global and pixel-wise, 1.01 for reference sum and 1.00 for trimmed.
