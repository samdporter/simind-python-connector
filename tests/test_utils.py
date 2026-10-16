import numpy as np
import pytest

from simind_python_connector.backends import (
    create_image_data,
    reset_backend,
    set_backend,
)
from simind_python_connector.utils import get_array
from simind_python_connector.utils.stir_utils import (
    create_attenuation_map,
    create_simple_phantom,
    create_stir_acqdata,
    create_stir_image,
    parse_interfile,
)


@pytest.fixture(
    params=[
        pytest.param("sirf", marks=pytest.mark.requires_sirf),
        pytest.param("stir", marks=pytest.mark.requires_stir),
    ]
)
def backend(request):
    reset_backend()
    set_backend(request.param)
    yield request.param
    reset_backend()


def test_create_simple_phantom(backend):
    phantom = create_simple_phantom()
    phantom_array = get_array(phantom)
    assert phantom_array.shape == (64, 64, 64)
    assert np.unique(phantom_array).tolist() == [0, 10, 40]
    assert phantom_array[25, 32, 32] == 10
    assert phantom_array[32, 32, 36] == 40
    assert phantom_array[0, 0, 0] == 0


def test_create_attenuation_map(backend):
    phantom = create_simple_phantom()
    original = get_array(phantom).copy()
    mu_map = create_attenuation_map(phantom)
    mu_array = get_array(mu_map)
    assert mu_array.shape == original.shape
    np.testing.assert_allclose(np.unique(mu_array), [0, 0.15])
    assert mu_array[25, 32, 32] == pytest.approx(0.15)
    assert mu_array[32, 32, 36] == pytest.approx(0.15)
    assert mu_array[0, 0, 0] == 0
    np.testing.assert_array_equal(get_array(phantom), original)


def test_create_stir_image(backend):
    matrix_dim = [32, 32, 32]
    voxel_size = [2.0, 2.0, 2.0]

    image = create_image_data(
        create_stir_image(matrix_dim, voxel_size, backend=backend)
    )

    dims = image.dimensions()
    assert dims == tuple(matrix_dim)

    voxels = image.voxel_sizes()
    assert np.allclose(voxels, voxel_size, rtol=1e-6)


def test_create_stir_acqdata(backend):
    proj_matrix = [64, 32]  # (bins, axial)
    num_projections = 60
    pixel_size = [4.42, 4.42]

    acqdata = create_stir_acqdata(
        proj_matrix, num_projections, pixel_size, backend=backend
    )

    # STIR/SIRF arrays are (tof, axial, view, bin).
    assert get_array(acqdata).shape == (1, 32, num_projections, 64)
    assert np.count_nonzero(get_array(acqdata)) == 0


@pytest.mark.unit
def test_parse_interfile(tmp_path):
    header = tmp_path / "image.hv"
    header.write_text(
        "!INTERFILE :=\n"
        "name of data file := test.v\n"
        "!matrix size [1] := 64\n"
        "scaling factor (mm/pixel) [1] := 4.42\n"
        "!END OF INTERFILE :=\n"
    )
    values = parse_interfile(header)
    assert values == {
        "name of data file": "test.v",
        "!matrix size [1]": "64",
        "scaling factor (mm/pixel) [1]": "4.42",
    }
