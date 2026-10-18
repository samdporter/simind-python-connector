import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simind_python_connector import configs
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
)


pytestmark = [pytest.mark.validation, pytest.mark.requires_simind]

N = 32
VOXEL_MM = 4.42
VIEWS = 30
RADIUS_MM = 200
QUANTIZATION_SCALE = 0.1 * DEFAULT_QUANTIZATION_SCALE
SUBSETS = 5
ITERATIONS = 4
HOT_CENTRE = (16, 16, 22)
TRUE_RATIO = 2.5


def _phantom():
    """Activity and attenuation in (z, y, x), the 07 phantom with a 2.5:1 sphere."""
    z, y, x = np.indices((N, N, N))
    body = ((x - 16) ** 2 + (y - 16) ** 2 <= 10**2) & (np.abs(z - 16) <= 8)
    sphere_distance2 = (
        (z - HOT_CENTRE[0]) ** 2 + (y - HOT_CENTRE[1]) ** 2 + (x - HOT_CENTRE[2]) ** 2
    )
    hot = sphere_distance2 <= 4**2
    background = body & (sphere_distance2 > 8**2)
    activity = np.where(body, 1.0, 0.0) + np.where(hot, 1.5, 0.0)
    return activity.astype(np.float32), np.where(body, 0.15, 0.0), body, hot, background


def _configure(adaptor):
    adaptor.set_energy_windows([126], [154], [0])
    adaptor.add_runtime_switch("FI", "tc99m")
    adaptor.add_runtime_switch("CC", "ma-lehr")
    adaptor.add_runtime_switch("RR", 12345)
    adaptor.add_config_value(1, 140.0)
    adaptor.add_config_value(19, 2)
    adaptor.add_config_value(53, 0)


def _agreement_dir(engine):
    root = Path(os.environ.get("SIMIND_VALIDATION_OUT", "output/validation"))
    path = root / "agreement" / engine.label
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.requires_sirf
def test_sirf_reconstruction_is_saved(tmp_path, engine):
    import sirf.STIR as sirf

    from simind_python_connector import SirfSimindAdaptor
    from simind_python_connector.builders import (
        STIRSPECTAcquisitionDataBuilder,
        STIRSPECTImageDataBuilder,
    )

    def image(array):
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

    activity, mu, *_ = _phantom()
    source_image = image(activity)
    mu_image = image(mu)

    sirf.AcquisitionData.set_storage_scheme("memory")
    template = STIRSPECTAcquisitionDataBuilder(
        header_overrides={
            "!matrix size [1]": str(N),
            "!matrix size [2]": str(N),
            "!number of projections": str(VIEWS),
            "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
            "!direction of rotation": "CW",
            "start angle": "180",
            "Radius": str(RADIUS_MM),
        },
        backend="sirf",
    ).build(output_path=tmp_path / "template")

    adaptor = SirfSimindAdaptor(
        configs.get("Example.yaml"),
        str(tmp_path / "sirf"),
        "sirf",
        quantization_scale=QUANTIZATION_SCALE,
        scoring_routine=ScoringRoutine.PENETRATE,
    )
    _configure(adaptor)
    adaptor.add_runtime_switch("NN", 1)
    adaptor.set_template(template)
    adaptor.set_source(source_image)
    adaptor.set_mu_map(mu_image)
    data = adaptor.run()["all_interactions"]

    matrix = sirf.SPECTUBMatrix()
    # STIR SPECTUB mirrors x relative to the mu map (UCL/STIR#452)
    matrix.set_attenuation_image(image(mu[..., ::-1]))
    matrix.set_keep_all_views_in_cache(True)
    matrix.set_resolution_model(0.0, 0.0, False)
    acq_model = sirf.AcquisitionModelUsingMatrix(matrix)
    objective = sirf.make_Poisson_loglikelihood(data)
    objective.set_acquisition_model(acq_model)
    objective.set_num_subsets(SUBSETS)
    recon = sirf.OSMAPOSLReconstructor()
    recon.set_objective_function(objective)
    recon.set_num_subsets(SUBSETS)
    recon.set_num_subiterations(SUBSETS * ITERATIONS)
    recon.set_input(data)
    reconstruction = source_image.get_uniform_copy(1.0)
    recon.set_up(reconstruction)
    recon.reconstruct(reconstruction)

    out = _agreement_dir(engine)
    np.save(out / "sirf_recon.npy", reconstruction.as_array())
    # (tof, axial, view, bin) -> (view, axial, bin), the connector's projection layout
    np.save(out / "sirf_data.npy", data.as_array()[0].transpose(1, 0, 2))
    np.save(out / "truth.npy", activity)


@pytest.fixture(scope="module")
def pytomography_run(tmp_path_factory, engine):
    import torch
    from pytomography import algorithms, likelihoods
    from pytomography.io.SPECT import simind as pytomo_simind
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    from pytomography.transforms.SPECT import SPECTAttenuationTransform

    from simind_python_connector import PyTomographySimindAdaptor

    out = _agreement_dir(engine)
    truth = np.load(out / "truth.npy")
    _, mu, *_ = _phantom()

    to_object = PyTomographySimindAdaptor.from_simind_image_axes
    mu_tensor = to_object(torch.from_numpy(mu.astype(np.float32)))

    adaptor = PyTomographySimindAdaptor(
        configs.get("Example.yaml"),
        str(tmp_path_factory.mktemp("pytomography")),
        "pytomography",
        # SIRF stores the voxel size as float32; SIMIND's run depends on the rounding
        voxel_size_mm=float(np.float32(VOXEL_MM)),
        quantization_scale=QUANTIZATION_SCALE,
        scoring_routine=ScoringRoutine.PENETRATE,
    )
    _configure(adaptor)
    adaptor.add_runtime_switch("NN", 1)
    adaptor.add_config_value(12, RADIUS_MM / 10)
    adaptor.add_config_value(28, VOXEL_MM / 10)
    adaptor.add_config_value(29, VIEWS)
    adaptor.add_config_value(30, 360)
    adaptor.add_config_value(41, 0)
    adaptor.add_config_value(76, N)
    adaptor.add_config_value(77, N)
    adaptor.set_source(to_object(torch.from_numpy(truth)))
    adaptor.set_mu_map(mu_tensor)
    adaptor.run()
    projections = adaptor.get_outputs()["all_interactions"]

    object_meta, proj_meta = pytomo_simind.get_metadata(
        str(adaptor.get_output_header_path("all_interactions"))
    )
    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms=[SPECTAttenuationTransform(attenuation_map=mu_tensor)],
        proj2proj_transforms=[],
        object_meta=object_meta,
        proj_meta=proj_meta,
    )
    likelihood = likelihoods.PoissonLogLikelihood(
        system_matrix=system_matrix, projections=projections
    )
    reconstruction = algorithms.OSEM(likelihood=likelihood)(
        ITERATIONS, n_subsets=SUBSETS
    )
    return SimpleNamespace(
        out=out,
        data=projections.numpy(),
        recon=PyTomographySimindAdaptor.to_simind_image_axes(
            reconstruction.detach().cpu()
        ).numpy(),
    )


@pytest.fixture(scope="module")
def images(pytomography_run):
    _, _, body, hot, background = _phantom()
    sirf_recon = np.load(pytomography_run.out / "sirf_recon.npy")
    np.save(pytomography_run.out / "pytomography_recon.npy", pytomography_run.recon)

    def ratio(image):
        return float(image[hot].mean() / image[background].mean())

    return SimpleNamespace(
        body=body,
        sirf=sirf_recon,
        pytomography=pytomography_run.recon,
        sirf_ratio=ratio(sirf_recon),
        pytomography_ratio=ratio(pytomography_run.recon),
    )


@pytest.mark.requires_pytomography
def test_identical_simind_data(pytomography_run, validate):
    sirf_data = np.load(pytomography_run.out / "sirf_data.npy")
    # (view, bin, axial) with the axial direction reversed -> (view, axial, bin)
    pytomography_data = pytomography_run.data.transpose(0, 2, 1)[:, ::-1, :]
    validate(
        "identical_simind_data",
        float(np.max(np.abs(pytomography_data - sirf_data)) / np.max(sirf_data)),
        expected=0,
        abs_tol=1e-6,
    )


@pytest.mark.requires_pytomography
def test_hot_to_background_ratio_agrees(images, validate):
    validate(
        "hot_to_background_ratio_agrees",
        images.pytomography_ratio / images.sirf_ratio,
        expected=1.0,
        rel=0.05,
        note=(
            f"SIRF {images.sirf_ratio:.3f}, "
            f"PyTomography {images.pytomography_ratio:.3f}"
        ),
    )


@pytest.mark.requires_pytomography
def test_image_correlation(images, validate):
    validate(
        "image_correlation",
        float(
            np.corrcoef(images.sirf[images.body], images.pytomography[images.body])[
                0, 1
            ]
        ),
        lower=0.95,
    )


@pytest.mark.requires_pytomography
def test_total_activity_agrees(images, validate):
    validate(
        "total_activity_agrees",
        float(images.pytomography[images.body].sum() / images.sirf[images.body].sum()),
        expected=1.0,
        rel=0.05,
    )


@pytest.mark.requires_pytomography
def test_recovery_vs_truth_sirf(images, validate):
    validate(
        "recovery_vs_truth_sirf",
        images.sirf_ratio / TRUE_RATIO,
        lower=0,
        note=f"hot-to-background {images.sirf_ratio:.3f} against {TRUE_RATIO}",
    )


@pytest.mark.requires_pytomography
def test_recovery_vs_truth_pytomography(images, validate):
    validate(
        "recovery_vs_truth_pytomography",
        images.pytomography_ratio / TRUE_RATIO,
        lower=0,
        note=f"hot-to-background {images.pytomography_ratio:.3f} against {TRUE_RATIO}",
    )
