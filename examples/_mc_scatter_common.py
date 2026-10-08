"""Shared set-up for examples 09 and 10: phantom, template, data and metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sirf.STIR as sirf

from simind_python_connector import SirfSimindAdaptor, configs
from simind_python_connector.builders import (
    STIRSPECTAcquisitionDataBuilder,
    STIRSPECTImageDataBuilder,
)
from simind_python_connector.core.types import ScoringRoutine
from simind_python_connector.recon import (
    AdditiveUpdater,
    ScatterCorrection,
    SimindProjector,
    UpdateSchedule,
    reference_normaliser,
)


N = 32
VOXEL_MM = 4.42
NUM_VIEWS = 30
ACTIVITY_MBQ = 100.0
TIME_PER_VIEW_S = 10.0
HOT_CENTRE_OFFSET = 5  # voxels along x from the axis


def image(array: np.ndarray) -> sirf.ImageData:
    builder = STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": str(N),
            "!matrix size [2]": str(N),
            "!matrix size [3]": str(N),
            "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [3]": str(VOXEL_MM),
        },
        backend="sirf",
    )
    builder.set_pixel_array(array.astype(np.float32))
    return builder.build()


def _distances():
    z, y, x = np.indices((N, N, N))
    c = N / 2
    to_axis = (x - c) ** 2 + (y - c) ** 2
    to_hot = (x - c - HOT_CENTRE_OFFSET) ** 2 + (y - c) ** 2 + (z - c) ** 2
    return to_axis, to_hot, np.abs(z - c)


def phantom() -> tuple[np.ndarray, np.ndarray]:
    """Water cylinder with a hot sphere (5:1: four above a background of one);
    (activity, mu)."""
    to_axis, to_hot, from_centre_z = _distances()
    body = (to_axis <= (0.35 * N) ** 2) & (from_centre_z <= 0.3 * N)
    activity = np.where(body, 1.0, 0.0) + np.where(to_hot <= 3**2, 4.0, 0.0)
    mu = np.where(body, 0.15, 0.0)
    return activity, mu


def _rois() -> tuple[np.ndarray, np.ndarray]:
    """(hot core, background) masks."""
    to_axis, to_hot, from_centre_z = _distances()
    core = to_hot <= 2**2
    background = (to_hot >= 6**2) & (to_axis <= 8**2) & (from_centre_z <= 5)
    return core, background


def contrast_error(reconstruction: sirf.ImageData, activity: np.ndarray) -> float:
    """Hot-sphere contrast error: relative error of the hot/background ratio."""
    core, background = _rois()
    recon = reconstruction.as_array()
    recon_ratio = recon[core].mean() / recon[background].mean()
    true_ratio = activity[core].mean() / activity[background].mean()
    return abs(recon_ratio - true_ratio) / true_ratio


def contrast_recovery(reconstruction: sirf.ImageData, activity: np.ndarray) -> float:
    """NEMA contrast recovery of the hot core: 1 is perfect."""
    core, background = _rois()
    recon = reconstruction.as_array()
    measured = recon[core].mean() / recon[background].mean() - 1.0
    true = activity[core].mean() / activity[background].mean() - 1.0
    return measured / true


def background_cov(reconstruction: sirf.ImageData) -> float:
    """Coefficient of variation in the background region."""
    _, background = _rois()
    values = reconstruction.as_array()[background]
    return float(values.std() / values.mean())


def template(output_dir: Path) -> sirf.AcquisitionData:
    sirf.AcquisitionData.set_storage_scheme("memory")
    return STIRSPECTAcquisitionDataBuilder(
        header_overrides={
            "!matrix size [1]": str(N),
            "!matrix size [2]": str(N),
            "!number of projections": str(NUM_VIEWS),
            "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
            "!direction of rotation": "CW",
            "start angle": "180",
            "Radius": "200",
        },
        backend="sirf",
    ).build(output_path=output_dir / "template")


def simind_adaptor(
    output_dir: Path,
    prefix: str,
    photon_energy_kev: float = 140.0,
    window_kev: tuple[float, float] = (126.0, 154.0),
    collimator: str = "ma-lehr",
) -> SirfSimindAdaptor:
    """PENETRATE with one photon energy, an energy window and a collimator."""
    adaptor = SirfSimindAdaptor(
        configs.get("Example.yaml"),
        str(output_dir),
        prefix,
        scoring_routine=ScoringRoutine.PENETRATE,
    )
    adaptor.add_runtime_switch("CC", collimator)
    adaptor.add_config_value(1, photon_energy_kev)
    adaptor.add_config_value(20, window_kev[1])  # upper energy threshold (keV)
    adaptor.add_config_value(21, window_kev[0])  # lower energy threshold (keV)
    return adaptor


def simulate_measured(
    output_dir, template_data, activity_image, mu_image, seed=1, **adaptor_settings
):
    """SIMIND data with a known activity (A2 route 1) plus Poisson noise."""
    adaptor = simind_adaptor(output_dir / "measured", "measured", **adaptor_settings)
    adaptor.set_template(template_data)
    adaptor.set_source(activity_image)
    adaptor.set_mu_map(mu_image)
    adaptor.set_activity(ACTIVITY_MBQ, TIME_PER_VIEW_S)
    adaptor.add_runtime_switch("RR", seed)
    expected = adaptor.run()["all_interactions"]
    rng = np.random.default_rng(seed)
    measured = expected.clone()
    measured.fill(rng.poisson(np.maximum(expected.as_array(), 0)).astype(np.float32))
    return measured


def fast_model_factory(mu_image: sirf.ImageData):
    """SPECTUB with attenuation and no PSF;
    it omits the collimator and detector blur that SIMIND's b02 includes."""

    def make_model():
        matrix = sirf.SPECTUBMatrix()
        matrix.set_attenuation_image(mu_image)
        matrix.set_keep_all_views_in_cache(True)
        matrix.set_resolution_model(0.0, 0.0, False)
        return sirf.AcquisitionModelUsingMatrix(matrix)

    return make_model


def scatter_updater(
    output_dir,
    template_data,
    mu_image,
    measured,
    full_model,
    every,
    stop_at=None,
    **adaptor_settings,
):
    """SIMIND scatter scaled to the fast model (A2 route 2), from a zero start."""
    projector = SimindProjector(
        simind_adaptor(output_dir / "estimate", "estimate", **adaptor_settings),
        template_data,
        mu_image,
        normalise=reference_normaliser(full_model),
        seed=100,
    )
    return AdditiveUpdater(
        ScatterCorrection(projector),
        UpdateSchedule(every=every, stop_at=stop_at),
        measured.get_uniform_copy(0),
    )


def print_updates(updater: AdditiveUpdater) -> None:
    for record in updater.history:
        print(
            f"update at iteration {record.iteration}: scale {record.scale:.4g}, "
            f"scatter sum {record.additive_sum:.4g}, {record.seconds:.0f} s"
        )
