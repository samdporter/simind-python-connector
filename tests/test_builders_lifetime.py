import numpy as np
import pytest

from simind_python_connector.builders import (
    STIRSPECTAcquisitionDataBuilder,
    STIRSPECTImageDataBuilder,
)


_IMAGE_HEADER = {
    "!matrix size [1]": "4",
    "!matrix size [2]": "3",
    "!matrix size [3]": "2",
}
_ACQ_HEADER = {
    "!matrix size [1]": "4",
    "!matrix size [2]": "2",
    "!number of projections": "3",
}


def _to_numpy(obj):
    if hasattr(obj, "as_array"):
        return obj.as_array()
    import stirextra

    return stirextra.to_numpy(obj)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("sirf", marks=pytest.mark.requires_sirf),
        pytest.param("stir", marks=pytest.mark.requires_stir),
    ],
)
def test_image_built_into_a_temporary_directory_stays_readable(backend):
    builder = STIRSPECTImageDataBuilder(header_overrides=_IMAGE_HEADER, backend=backend)
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    builder.set_pixel_array(data)
    image = builder.build()  # temporary directory is deleted on return
    assert np.array_equal(np.asarray(_to_numpy(image)).reshape(data.shape), data)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("sirf", marks=pytest.mark.requires_sirf),
        pytest.param("stir", marks=pytest.mark.requires_stir),
    ],
)
def test_acquisition_built_into_a_temporary_directory_stays_readable(backend):
    builder = STIRSPECTAcquisitionDataBuilder(
        header_overrides=_ACQ_HEADER, backend=backend
    )
    data = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    builder.pixel_array = data
    acquisition = builder.build()
    assert np.array_equal(np.asarray(_to_numpy(acquisition)).reshape(data.shape), data)
