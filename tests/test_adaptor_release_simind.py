"""SCATTWIN output access and arithmetic through real reconstruction backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simind_python_connector import RuntimeOperator
from simind_python_connector.configs import get


pytestmark = [pytest.mark.integration, pytest.mark.requires_simind]

RUN_TIMEOUT = 60.0
PHOTON_MULTIPLIER = 5
MU_WATER = 0.15  # cm^-1 at 140 keV, so the run has phantom scatter
VIEWS = 4
SHAPE = (8, 8, 8)
VOXEL_MM = (4.0, 4.0, 4.0)


def _point_array(shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    source = np.zeros(shape, np.float32)
    source[tuple(dim // 2 for dim in shape)] = 1.0
    return source


def _configure_scattwin(adaptor):
    adaptor.get_config().set_flag(8, False)
    adaptor.set_energy_windows([126], [154], [0])
    adaptor.add_config_value(29, VIEWS)
    adaptor.add_runtime_switch("FI", "tc99m")
    adaptor.add_runtime_switch("CC", "ma-lehr")
    adaptor.add_runtime_switch("CA", 1)


def _stir_or_sirf_images(create_stir_image, backend: str):
    # Native STIR images fill from an iterator; SIRF images take the array.
    def values(array):
        return array.flat if backend == "stir" else array

    source = create_stir_image(list(SHAPE), list(VOXEL_MM), backend=backend)
    source.fill(values(_point_array()))
    mu_map = create_stir_image(list(SHAPE), list(VOXEL_MM), backend=backend)
    mu_map.fill(values(np.full(SHAPE, MU_WATER, np.float32)))
    return source, mu_map


@pytest.mark.requires_stir
def test_stir_adaptor_real_scattwin_run(tmp_path: Path):
    from simind_python_connector.connectors.stir_adaptor import StirSimindAdaptor
    from simind_python_connector.utils.stir_utils import create_stir_image

    source, mu_map = _stir_or_sirf_images(create_stir_image, "stir")
    adaptor = StirSimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir=str(tmp_path),
        output_prefix="stircase",
        photon_multiplier=PHOTON_MULTIPLIER,
    )
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)
    _configure_scattwin(adaptor)

    with pytest.raises(RuntimeError, match="Run the adaptor first"):
        adaptor.get_primary_output(window=1)

    outputs = adaptor.run(RuntimeOperator(timeout=RUN_TIMEOUT))

    assert {"tot_w1", "sca_w1"} <= set(outputs)
    assert float(adaptor.get_total_output(window=1).sum()) > 0
    assert float(outputs["sca_w1"].sum()) > 0
    assert adaptor.get_scatter_output(window=1) is outputs["sca_w1"]

    from simind_python_connector.backends.stir_backend import StirAcquisitionData

    primary = StirAcquisitionData(adaptor.get_primary_output(window=1)).as_array()
    total = StirAcquisitionData(outputs["tot_w1"]).as_array()
    scatter = StirAcquisitionData(outputs["sca_w1"]).as_array()
    assert np.allclose(primary, total - scatter)

    assert adaptor.get_outputs() is outputs
    assert "pri_w1" not in adaptor.get_outputs()


@pytest.mark.requires_sirf
def test_sirf_adaptor_real_scattwin_run(tmp_path: Path):
    from simind_python_connector.connectors.sirf_adaptor import SirfSimindAdaptor
    from simind_python_connector.utils.stir_utils import create_stir_image

    source, mu_map = _stir_or_sirf_images(create_stir_image, "sirf")
    adaptor = SirfSimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir=str(tmp_path),
        output_prefix="sirfcase",
        photon_multiplier=PHOTON_MULTIPLIER,
    )
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)
    _configure_scattwin(adaptor)

    with pytest.raises(RuntimeError, match="Run the adaptor first"):
        adaptor.get_primary_output(window=1)

    outputs = adaptor.run(RuntimeOperator(timeout=RUN_TIMEOUT))

    assert {"tot_w1", "sca_w1"} <= set(outputs)
    assert float(adaptor.get_total_output(window=1).as_array().sum()) > 0
    assert float(outputs["sca_w1"].as_array().sum()) > 0
    assert adaptor.get_scatter_output(window=1) is outputs["sca_w1"]

    assert np.allclose(
        adaptor.get_primary_output(window=1).as_array(),
        outputs["tot_w1"].as_array() - outputs["sca_w1"].as_array(),
    )

    assert adaptor.get_outputs() is outputs
    assert "pri_w1" not in adaptor.get_outputs()


@pytest.mark.requires_pytomography
def test_pytomography_adaptor_real_scattwin_run(tmp_path: Path):
    import torch

    from simind_python_connector.connectors.pytomography_adaptor import (
        PyTomographySimindAdaptor,
    )

    source = torch.zeros(SHAPE, dtype=torch.float32)
    source[tuple(dim // 2 for dim in SHAPE)] = 1.0
    adaptor = PyTomographySimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="pytcase",
        photon_multiplier=PHOTON_MULTIPLIER,
        voxel_size_mm=VOXEL_MM[0],
    )
    adaptor.set_source(source)
    adaptor.set_mu_map(torch.full_like(source, MU_WATER))
    _configure_scattwin(adaptor)

    with pytest.raises(RuntimeError, match="Run the connector first"):
        adaptor.get_primary_output(window=1)

    outputs = adaptor.run(RuntimeOperator(timeout=RUN_TIMEOUT))

    assert {"tot_w1", "sca_w1"} <= set(outputs)
    assert float(adaptor.get_total_output(window=1).sum()) > 0
    assert float(outputs["sca_w1"].sum()) > 0
    assert adaptor.get_scatter_output(window=1) is outputs["sca_w1"]

    assert torch.allclose(
        adaptor.get_primary_output(window=1), outputs["tot_w1"] - outputs["sca_w1"]
    )

    assert adaptor.get_outputs() is outputs
    assert "pri_w1" not in adaptor.get_outputs()
