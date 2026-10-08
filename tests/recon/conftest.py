import numpy as np
import pytest


@pytest.fixture(scope="session")
def sirf_scene(tmp_path_factory):
    """Small SPECT problem: (measured data, image, SPECTUB model factory)."""
    import sirf.STIR as sirf

    from simind_python_connector.builders import (
        STIRSPECTAcquisitionDataBuilder,
        STIRSPECTImageDataBuilder,
    )

    sirf.AcquisitionData.set_storage_scheme("memory")
    n, voxel = 16, "4.42"
    tmp_dir = tmp_path_factory.mktemp("sirf_scene")
    template = STIRSPECTAcquisitionDataBuilder(
        header_overrides={
            "!matrix size [1]": str(n),
            "!matrix size [2]": str(n),
            "!number of projections": "12",
            "scaling factor (mm/pixel) [1]": voxel,
            "scaling factor (mm/pixel) [2]": voxel,
            "Radius": "200",
        },
        backend="sirf",
    ).build(output_path=tmp_dir / "template")

    z, y, x = np.indices((n, n, n))
    phantom = ((x - n / 2) ** 2 + (y - n / 2) ** 2 <= (0.3 * n) ** 2).astype(np.float32)
    image_builder = STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": str(n),
            "!matrix size [2]": str(n),
            "!matrix size [3]": str(n),
            "scaling factor (mm/pixel) [1]": voxel,
            "scaling factor (mm/pixel) [2]": voxel,
            "scaling factor (mm/pixel) [3]": voxel,
        },
        backend="sirf",
    )
    image_builder.set_pixel_array(10.0 * phantom)
    image = image_builder.build(output_path=tmp_dir / "image")

    def make_model():
        matrix = sirf.SPECTUBMatrix()
        matrix.set_keep_all_views_in_cache(True)
        matrix.set_resolution_model(0.0, 0.0, False)
        return sirf.AcquisitionModelUsingMatrix(matrix)

    model = make_model()
    model.set_up(template, image)
    expected = model.forward(image)
    measured = expected.clone()
    rng = np.random.default_rng(0)
    measured.fill(rng.poisson(expected.as_array() + 1.0).astype(np.float32))
    return measured, image, make_model


@pytest.fixture(scope="session")
def psf_scene(tmp_path_factory):
    """Fu & Qi's setting: the accurate model is the fast one after a 6 mm blur."""
    from types import SimpleNamespace

    import sirf.STIR as sirf

    from simind_python_connector.builders import (
        STIRSPECTAcquisitionDataBuilder,
        STIRSPECTImageDataBuilder,
    )

    n, voxel = 32, "4.42"
    tmp_dir = tmp_path_factory.mktemp("psf_scene")
    template = STIRSPECTAcquisitionDataBuilder(
        header_overrides={
            "!matrix size [1]": str(n),
            "!matrix size [2]": str(n),
            "!number of projections": "32",
            "scaling factor (mm/pixel) [1]": voxel,
            "scaling factor (mm/pixel) [2]": voxel,
            "Radius": "200",
        },
        backend="sirf",
    ).build(output_path=tmp_dir / "template")

    z, y, x = np.indices((n, n, n))
    c = n / 2
    body = ((x - c) ** 2 + (y - c) ** 2 <= (0.35 * n) ** 2) & (np.abs(z - c) <= 0.3 * n)
    hot = ((x - c - 6) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= 9) | (
        (x - c + 5) ** 2 + (y - c - 4) ** 2 + (z - c) ** 2 <= 4
    )
    truth = np.where(body, 1.0, 0.0) + np.where(hot, 4.0, 0.0)
    image_builder = STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": str(n),
            "!matrix size [2]": str(n),
            "!matrix size [3]": str(n),
            "scaling factor (mm/pixel) [1]": voxel,
            "scaling factor (mm/pixel) [2]": voxel,
            "scaling factor (mm/pixel) [3]": voxel,
        },
        backend="sirf",
    )
    image_builder.set_pixel_array(truth.astype(np.float32))
    x_true = image_builder.build(output_path=tmp_dir / "truth")

    def make_fast():
        matrix = sirf.SPECTUBMatrix()
        matrix.set_keep_all_views_in_cache(True)
        matrix.set_resolution_model(0.0, 0.0, False)
        return sirf.AcquisitionModelUsingMatrix(matrix)

    def make_accurate():
        model = make_fast()
        blur = sirf.SeparableGaussianImageFilter()
        blur.set_fwhms((6.0, 6.0, 6.0))
        blur.set_up(x_true)
        model.set_image_data_processor(blur)
        return model

    fast, accurate = make_fast(), make_accurate()
    fast.set_up(template, x_true)
    accurate.set_up(template, x_true)
    background = template.get_uniform_copy(0.1)
    expected = accurate.forward(x_true) + background
    measured = expected.clone()
    rng = np.random.default_rng(1)
    measured.fill(rng.poisson(expected.as_array()).astype(np.float32))
    return SimpleNamespace(
        measured=measured,
        background=background,
        x_true=x_true,
        make_fast=make_fast,
        make_accurate=make_accurate,
        fast=fast,
        accurate=accurate,
    )
