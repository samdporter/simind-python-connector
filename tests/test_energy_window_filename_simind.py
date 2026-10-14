"""Real-engine evidence for the SCATTWIN 19-character filename buffer."""

import numpy as np
import pytest

from simind_python_connector import RuntimeOperator, SimindPythonConnector
from simind_python_connector.configs import get
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE


pytestmark = [pytest.mark.integration, pytest.mark.requires_simind]


@pytest.mark.parametrize("prefix", ["windowcase", "abcdefghijklmno"])
def test_simind_honors_energy_window_through_maximum_safe_prefix(tmp_path, prefix):
    connector = SimindPythonConnector(
        get("Example.yaml"),
        tmp_path,
        prefix,
        quantization_scale=0.02 * DEFAULT_QUANTIZATION_SCALE,
    )
    source = np.zeros((8, 8, 8), np.float32)
    source[3:5, 3:5, 3:5] = 1
    connector.configure_voxel_phantom(source, np.zeros_like(source), 4)
    connector.set_energy_windows([126], [154], [0])
    for index, value in ((19, 2), (29, 4), (53, 0), (28, 0.4), (76, 8), (77, 8)):
        connector.add_config_value(index, value)
    connector.add_runtime_switch("FI", "tc99m")
    connector.add_runtime_switch("RR", 12345)
    result = connector.run(RuntimeOperator(timeout=60))
    assert "sca_w1" in result  # Proves the custom two-line window was read.
    header = (tmp_path / f"{prefix}_tot_w1.h00").read_text()
    assert ";energy window lower level := 126.00" in header
    assert ";energy window upper level := 154.00" in header
