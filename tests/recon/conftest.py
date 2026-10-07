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
