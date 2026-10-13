"""Real-SIMIND SCATTWIN runs through the STIR, SIRF and PyTomography adaptors.

Each test runs one small SIMIND simulation through the adaptor, checks that the
total and scatter accessors return usable backend objects, and pins the primary
projection contract: ``RuntimeError`` before a run and ``NotImplementedError``
after one, without the output cache being mutated. The random seed is fixed so
the runs are reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simind_python_connector import RuntimeOperator
from simind_python_connector.configs import get


pytestmark = [pytest.mark.integration, pytest.mark.requires_simind]

RUN_TIMEOUT = 600.0
SEED = 12345
VIEWS = 4
SHAPE = (8, 8, 8)
VOXEL_MM = (4.0, 4.0, 4.0)


def _point_array(shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    source = np.zeros(shape, np.float32)
    source[tuple(dim // 2 for dim in shape)] = 1.0
    return source


def _stir_or_sirf_images(create_stir_image, backend: str):
    source = create_stir_image(list(SHAPE), list(VOXEL_MM), backend=backend)
    source.fill(_point_array())
    mu_map = create_stir_image(list(SHAPE), list(VOXEL_MM), backend=backend)
    mu_map.fill(np.zeros(SHAPE, np.float32))
    return source, mu_map


@pytest.mark.requires_stir
def test_stir_adaptor_real_scattwin_run(tmp_path: Path):
    pytest.importorskip("stir")
    from simind_python_connector.connectors.stir_adaptor import StirSimindAdaptor
    from simind_python_connector.utils.stir_utils import create_stir_image

    source, mu_map = _stir_or_sirf_images(create_stir_image, "stir")
    adaptor = StirSimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir=str(tmp_path),
        output_prefix="stircase",
        photon_multiplier=1,
    )
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)
    adaptor.set_energy_windows([126], [154], [0])
    adaptor.add_config_value(29, VIEWS)
    adaptor.add_runtime_switch("RR", SEED)

    with pytest.raises(RuntimeError, match="Run the adaptor first"):
        adaptor.get_primary_output(window=1)

    outputs = adaptor.run(RuntimeOperator(timeout=RUN_TIMEOUT))

    assert {"tot_w1", "sca_w1"} <= set(outputs)
    assert float(adaptor.get_total_output(window=1).sum()) > 0
    assert adaptor.get_scatter_output(window=1) is outputs["sca_w1"]

    with pytest.raises(NotImplementedError, match="standalone primary projection"):
        adaptor.get_primary_output(window=1)

    assert adaptor.get_outputs() is outputs
    assert "pri_w1" not in adaptor.get_outputs()


@pytest.mark.requires_sirf
def test_sirf_adaptor_real_scattwin_run(tmp_path: Path):
    pytest.importorskip("sirf.STIR")
    from simind_python_connector.connectors.sirf_adaptor import SirfSimindAdaptor
    from simind_python_connector.utils.stir_utils import create_stir_image

    source, mu_map = _stir_or_sirf_images(create_stir_image, "sirf")
    adaptor = SirfSimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir=str(tmp_path),
        output_prefix="sirfcase",
        photon_multiplier=1,
    )
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)
    adaptor.set_energy_windows([126], [154], [0])
    adaptor.add_config_value(29, VIEWS)
    adaptor.add_runtime_switch("RR", SEED)

    with pytest.raises(RuntimeError, match="Run the adaptor first"):
        adaptor.get_primary_output(window=1)

    outputs = adaptor.run(RuntimeOperator(timeout=RUN_TIMEOUT))

    assert {"tot_w1", "sca_w1"} <= set(outputs)
    assert float(adaptor.get_total_output(window=1).as_array().sum()) > 0
    assert adaptor.get_scatter_output(window=1) is outputs["sca_w1"]

    with pytest.raises(NotImplementedError, match="standalone primary projection"):
        adaptor.get_primary_output(window=1)

    assert adaptor.get_outputs() is outputs
    assert "pri_w1" not in adaptor.get_outputs()


@pytest.mark.requires_pytomography
def test_pytomography_adaptor_real_scattwin_run(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytomography")
    from simind_python_connector.connectors.pytomography_adaptor import (
        PyTomographySimindAdaptor,
    )

    source = torch.zeros(SHAPE, dtype=torch.float32)
    source[tuple(dim // 2 for dim in SHAPE)] = 1.0
    adaptor = PyTomographySimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="pytcase",
        photon_multiplier=1,
        voxel_size_mm=VOXEL_MM[0],
    )
    adaptor.set_source(source)
    adaptor.set_mu_map(torch.zeros_like(source))
    adaptor.set_energy_windows([126], [154], [0])
    adaptor.add_config_value(29, VIEWS)
    adaptor.add_runtime_switch("RR", SEED)

    with pytest.raises(RuntimeError, match="Run the connector first"):
        adaptor.get_primary_output(window=1)

    outputs = adaptor.run(RuntimeOperator(timeout=RUN_TIMEOUT))

    assert {"tot_w1", "sca_w1"} <= set(outputs)
    assert float(adaptor.get_total_output(window=1).sum()) > 0
    assert adaptor.get_scatter_output(window=1) is outputs["sca_w1"]

    with pytest.raises(NotImplementedError, match="standalone primary projection"):
        adaptor.get_primary_output(window=1)

    assert adaptor.get_outputs() is outputs
    assert "pri_w1" not in adaptor.get_outputs()
