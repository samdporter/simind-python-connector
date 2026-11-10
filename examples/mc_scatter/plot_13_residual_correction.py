"""Residual correction with SIMIND as the accurate model.

Reconstruct the simulated NEMA/IEC acquisition with three SIRF OSEM arms:
fast SPECTUB only, SIMIND scatter subtraction, and a Fu & Qi residual
correction using SIMIND as the accurate forward model.

The fixture is phantomgen's ``pet`` preset: 8:1 sphere-to-background activity
in MBq per voxel, four hot spheres, two cold spheres and named masks. SIMIND
simulates one expectation with the known activity and 2 s per view, so the data
are in counts. Three seeded Poisson draws (12345, 12346 and 12347) share it.

The fast model has attenuation and no PSF or scatter. The scatter arm subtracts
SIMIND's all interactions minus its geometric primary. The residual correction
adds ``A_acc(x) - A_fast(x)`` to the fast model, where ``A_acc`` is SIMIND's all
interactions scaled to the fast model and ``A_fast`` is the fast projection of
the same image. Only the forward projection is accurate; the update stays with
the fast model. Both corrected arms use the same three updates of 12
subiterations (8 full iterations with 6 subsets) and damping 0.5. Only the
scatter arm smooths its estimate over 3 bins and across views; the residual
correction is unsmoothed. That unequal smoothing matters when comparing the
residual arm's higher background variability. The estimate uses a fortieth of
the expectation's history budget.
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
    ResidualCorrection,
    ScatterCorrection,
    SimindComponent,
    SimindProjector,
    UpdateSchedule,
    effective_objective,
    reference_normaliser,
)
from simind_python_connector.recon.osem import run_osem_with_corrections


if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
    raise RuntimeError("SIMIND executable not found ($SIMIND_BIN or simind on PATH).")

output_dir = (
    Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "residual_correction"
)
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

fast_only = {
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


def make_projector(name):
    return SimindProjector(
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


scatter_corrected = {}
for seed, measured in realisations.items():
    updater = AdditiveUpdater(
        ScatterCorrection(
            make_projector(f"scatter_{seed}"),
            smoothing_fwhm_bins=3.0,
            smooth_across_views=True,
        ),
        UpdateSchedule(every=SUBITERATIONS_PER_UPDATE),
        zero_additive,
        damping=0.5,
    )
    scatter_corrected[seed] = run_osem_with_corrections(
        measured,
        make_acq_model,
        initial_image,
        updater,
        NUM_SUBSETS,
        SUBITERATIONS_PER_UPDATE,
        NUM_UPDATES,
    )


class RecordedResidual(ResidualCorrection):
    """Keep the accurate-model objective next to the residual update."""

    def __init__(self, measured, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measured = measured
        self.objectives = []

    def estimate(self, image):
        additive = super().estimate(image)
        self.objectives.append(effective_objective(self.measured, self))
        return additive


residual_corrected = {}
residual_objectives = {}
for seed, measured in realisations.items():
    residual = RecordedResidual(
        measured,
        SimindComponent(make_projector(f"residual_{seed}"), "all_interactions"),
        full_model,
        zero_additive,
    )
    updater = AdditiveUpdater(
        residual,
        UpdateSchedule(every=SUBITERATIONS_PER_UPDATE),
        zero_additive,
        damping=0.5,
    )
    residual_corrected[seed] = run_osem_with_corrections(
        measured,
        make_acq_model,
        initial_image,
        updater,
        NUM_SUBSETS,
        SUBITERATIONS_PER_UPDATE,
        NUM_UPDATES,
    )
    residual_objectives[seed] = residual.objectives
    print(
        f"Seed {seed} effective objective: "
        + ", ".join(f"{value:.6g}" for value in residual.objectives)
    )


def mean_relative(images):
    array = np.mean([image.as_array() for image in images], axis=0)
    return nema_phantom.relative_to_background(array)


truth_array = nema_phantom.relative_to_background(nema.activity_mbq)
arrays = {
    "Fast only": mean_relative(fast_only.values()),
    "Fast + SIMIND scatter": mean_relative(scatter_corrected.values()),
    "Fast + full residual": mean_relative(residual_corrected.values()),
}
fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
for axis, (title, image) in zip(axes.flat, (("Truth", truth_array), *arrays.items())):
    shown = axis.imshow(image[slice_z], cmap="gray_r", vmin=0, vmax=truth_array.max())
    axis.set_title(title)
    axis.axis("off")
fig.colorbar(shown, ax=axes, label="relative to background")
fig.savefig(output_dir / "reconstructions.png", dpi=140)
plt.show()

truth_recovery = _nema.contrast_recovery(nema.activity_mbq, nema_phantom)
truth_cov = _nema.background_cov(nema.activity_mbq, nema_phantom)
scores = {"Truth": (truth_recovery, [0.0] * 4, truth_cov, 0.0)}
for title, images in (
    ("Fast only", fast_only),
    ("Fast + SIMIND scatter", scatter_corrected),
    ("Fast + full residual", residual_corrected),
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
    (-0.3, -0.1, 0.1, 0.3), scores.items()
):
    axis.bar(x + offset, recovery, 0.2, yerr=recovery_sem, capsize=2, label=title)
axis.set_xticks(x, diameters)
axis.set_xlabel("Sphere diameter (mm)")
axis.set_ylabel("Contrast recovery")
axis.set_ylim(0, max(truth_recovery) * 1.15)
axis.legend()
fig.savefig(output_dir / "recovery.png", dpi=140)
plt.show()

# Recorded: fast only 0.33/0.38/0.49/0.54, scatter 0.36/0.43/0.56/0.61,
# residual 0.37/0.41/0.54/0.60; CoV 0.64/0.69/0.91. The effective objective
# becomes less negative after each update, so the accurate-model fit worsens.
