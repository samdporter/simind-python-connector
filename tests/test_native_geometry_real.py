"""Geometry rejection using installed SIRF/STIR bindings, without SIMIND."""

import pytest

from simind_python_connector.configs import get


pytestmark = pytest.mark.integration


@pytest.mark.requires_sirf
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("spacing", [(4, 4, 4), (8, 4, 4)])
def test_real_sirf_adaptor_rejects_shifted_attenuation(tmp_path, wrapped, spacing):
    sirf = pytest.importorskip("sirf.STIR")
    from simind_python_connector.backends.sirf_backend import SirfImageData
    from simind_python_connector.connectors.sirf_adaptor import SirfSimindAdaptor

    source = sirf.ImageData()
    source.initialise((8, 8, 8), spacing, (0, 0, 0))
    aligned = source.clone()
    shifted = sirf.ImageData()
    shifted.initialise((8, 8, 8), spacing, (0, 0, 12))
    wrap = SirfImageData if wrapped else lambda image: image
    adaptor = SirfSimindAdaptor(get("Example.yaml"), str(tmp_path))
    adaptor.set_source(wrap(source))
    adaptor.set_mu_map(wrap(aligned))
    adaptor._validate_inputs()
    adaptor.set_mu_map(wrap(shifted))
    with pytest.raises(ValueError, match="same spatial geometry"):
        adaptor._validate_inputs()


@pytest.mark.requires_stir
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("spacing", [(4, 4, 4), (8, 4, 4)])
def test_real_stir_adaptor_rejects_shifted_attenuation(tmp_path, wrapped, spacing):
    stir = pytest.importorskip("stir")
    from simind_python_connector.backends.stir_backend import StirImageData
    from simind_python_connector.connectors.stir_adaptor import StirSimindAdaptor
    from simind_python_connector.utils.stir_utils import create_stir_image

    source = create_stir_image([8, 8, 8], list(spacing), backend="stir")
    aligned = source.get_empty_copy()
    shifted = source.get_empty_copy()
    shifted.set_origin(stir.FloatCartesianCoordinate3D(0, 0, 12))
    wrap = StirImageData if wrapped else lambda image: image
    adaptor = StirSimindAdaptor(get("Example.yaml"), str(tmp_path))
    adaptor.set_source(wrap(source))
    adaptor.set_mu_map(wrap(aligned))
    adaptor._validate_inputs()
    adaptor.set_mu_map(wrap(shifted))
    with pytest.raises(ValueError, match="same spatial geometry"):
        adaptor._validate_inputs()
