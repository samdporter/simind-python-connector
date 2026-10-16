"""Real-SIMIND checks of the projections written for one SCATTWIN energy window.

The window file holds exactly the requested window. SIMIND's CA runtime switch
selects which projections are written for it: without CA only ``tot``, with
``CA:1`` also ``sca`` and ``air``, with ``CA:2`` also ``pri`` and ``air``.
"""

import os
import shutil

import numpy as np
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.requires_simind]

_RUN_TIMEOUT = 10.0


def _simind_available() -> bool:
    return bool(os.environ.get("SIMIND_BIN")) or shutil.which("simind") is not None


@pytest.mark.parametrize(
    "switches, written, not_written",
    [
        ({}, {"tot_w1"}, {"sca_w1", "pri_w1", "air_w1"}),
        ({"CA": 1}, {"tot_w1", "sca_w1", "air_w1"}, {"pri_w1"}),
        ({"CA": 2}, {"tot_w1", "pri_w1", "air_w1"}, set()),
    ],
)
def test_single_window_outputs_follow_ca_switch(
    switches, written, not_written, tmp_path
):
    if not _simind_available():
        pytest.skip("SIMIND not available")

    from simind_python_connector import RuntimeOperator, SimindPythonConnector
    from simind_python_connector.configs import get
    from simind_python_connector.core.types import MAX_SOURCE

    conn = SimindPythonConnector(
        get("Example.yaml"),
        tmp_path,
        output_prefix="sw",
        quantization_scale=1 / MAX_SOURCE,
    )
    conn.add_config_value(29, 4)
    source = np.zeros((8, 8, 8), dtype=np.float32)
    source[4, 4, 4] = 1.0
    conn.configure_voxel_phantom(
        source, np.zeros_like(source), voxel_size_mm=4.0, scoring_routine=1
    )
    conn.set_energy_windows([126.0], [154.0], [0])
    conn.set_runtime_switches(switches)

    outputs = set(conn.run(RuntimeOperator(timeout=_RUN_TIMEOUT)))

    assert (tmp_path / "sw.win").read_text().splitlines() == ["126.0,154.0,0"]
    assert written <= outputs
    assert not (not_written & outputs)
    assert not [key for key in outputs if key.endswith("_w2")]
