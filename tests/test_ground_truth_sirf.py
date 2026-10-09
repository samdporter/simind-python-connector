import numpy as np
import pytest


pytestmark = pytest.mark.requires_sirf


def test_ground_truth_images_are_sirf_images_on_the_template_grid(tmp_path):
    import sirf.STIR as sirf

    from simind_python_connector import SirfSimindAdaptor
    from simind_python_connector.builders import STIRSPECTImageDataBuilder
    from simind_python_connector.configs import get
    from simind_python_connector.phantoms import AnalyticPhantom, Box, voxelise

    builder = STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": "16",
            "!matrix size [2]": "16",
            "!matrix size [3]": "12",
            "scaling factor (mm/pixel) [1]": "4",
            "scaling factor (mm/pixel) [2]": "4",
            "scaling factor (mm/pixel) [3]": "4",
        },
        backend="sirf",
    )
    template = builder.build(output_path=tmp_path / "template")
    phantom = AnalyticPhantom(Box((1.0, 2.0, 1.5)), Box((3.0, 3.0, 3.0)))
    adaptor = SirfSimindAdaptor(get("Example.yaml"), str(tmp_path / "sim"), "case01")
    adaptor.set_phantom(phantom)

    activity, mu = adaptor.get_ground_truth(template)

    assert isinstance(activity, sirf.ImageData) and isinstance(mu, sirf.ImageData)
    expected, _ = voxelise(phantom, (12, 16, 16), (4.0, 4.0, 4.0))
    np.testing.assert_allclose(activity.as_array(), expected)
    assert activity.voxel_sizes() == template.voxel_sizes()
    assert mu.as_array().max() > 0
