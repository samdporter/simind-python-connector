"""All backend examples must simulate the same physical asymmetric phantom."""

import importlib

import numpy as np
import pytest

from simind_python_connector.utils import get_array
from simind_python_connector.utils.geometry_phantom import build_asymmetric_phantom


pytestmark = pytest.mark.integration


@pytest.mark.requires_stir
def test_stir_example_uses_canonical_asymmetric_phantom(tmp_path):
    pytest.importorskip("stir")
    module = importlib.import_module("examples.07A_stir_adaptor_osem")
    source, mu = module._build_small_phantom(tmp_path)
    expected_source, expected_mu, _ = build_asymmetric_phantom()
    np.testing.assert_array_equal(get_array(source), expected_source)
    np.testing.assert_array_equal(get_array(mu), expected_mu)


@pytest.mark.requires_sirf
def test_sirf_example_uses_canonical_asymmetric_phantom(tmp_path):
    pytest.importorskip("sirf.STIR")
    module = importlib.import_module("examples.07B_sirf_adaptor_osem")
    source, mu = module._build_small_phantom(tmp_path)
    expected_source, expected_mu, _ = build_asymmetric_phantom()
    np.testing.assert_array_equal(get_array(source), expected_source)
    np.testing.assert_array_equal(get_array(mu), expected_mu)


@pytest.mark.requires_pytomography
def test_pytomography_example_uses_same_physical_asymmetric_phantom():
    pytest.importorskip("pytomography")
    module = importlib.import_module("examples.07C_pytomography_adaptor_osem")
    source, mu = module._build_small_tensors()
    to_simind = module.PyTomographySimindAdaptor.to_simind_image_axes
    expected_source, expected_mu, _ = build_asymmetric_phantom()
    np.testing.assert_array_equal(to_simind(source).numpy(), expected_source)
    np.testing.assert_array_equal(to_simind(mu).numpy(), expected_mu)
