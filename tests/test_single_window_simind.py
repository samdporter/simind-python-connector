"""Real-SIMIND evidence for the scatter-order workaround in ``create_window_file``.

``create_window_file`` appends an extra ``lower,upper,1`` line when every requested
scatter order is ``0``. SIMIND's SCATTWIN routine only writes scatter (``sca_*``)
and air (``air_*``) projections when the window file contains a positive scatter
order, so without the appended line a single-window run yields only the total
projection. Measured on SIMIND V8.0 (Linux x86-64), ``Example.yaml``, SCATTWIN,
32^3 point source in mu=0.15 water, 8 views, ``/NN:20/RR:12345``:

  A) ``126.0,154.0,0\\n126.0,154.0,1`` -> ``air_w1``, ``sca_w1``, ``sca_w2``, ``tot_w1``
     (``sca_w1`` and ``sca_w2`` are byte-identical duplicates)
  B) ``126.0,154.0,0\\n``               -> ``tot_w1`` only (no ``sca_w1``/``air_w1``)

The A/B pair below re-checks that claim so the retained workaround has
executable evidence rather than a comment alone.
"""

import os
import shutil

import numpy as np
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.requires_simind]

_SEED = 12345
_NN = 20


def _simind_available() -> bool:
    return bool(os.environ.get("SIMIND_BIN")) or shutil.which("simind") is not None


def _configured_connector(out_dir):
    from simind_python_connector import SimindPythonConnector
    from simind_python_connector.configs import get

    conn = SimindPythonConnector(get("Example.yaml"), out_dir, output_prefix="sw")
    conn.add_config_value(26, 0.02)
    conn.add_config_value(29, 8)

    source = np.zeros((32, 32, 32), dtype=np.float32)
    source[16, 16, 16] = 1.0
    density = np.full((32, 32, 32), 0.15, dtype=np.float32)
    conn.configure_voxel_phantom(source, density, voxel_size_mm=4.0, scoring_routine=1)
    return conn


@pytest.mark.integration
@pytest.mark.requires_simind
def test_single_window_scatter_output_requires_appended_line(tmp_path):
    if not _simind_available():
        pytest.skip("SIMIND not available")

    from simind_python_connector import RuntimeOperator

    switches = RuntimeOperator(switches={"NN": _NN, "RR": _SEED})

    with_line = _configured_connector(tmp_path / "with_line")
    with_line.set_energy_windows([126.0], [154.0], [0])
    outputs_a = with_line.run(switches)

    without_line = _configured_connector(tmp_path / "without_line")
    without_line.set_energy_windows([126.0], [154.0], [0])
    # Same connector output as A, minus the appended scatter-order line.
    (tmp_path / "without_line" / "sw.win").write_text("126.0,154.0,0\n")
    outputs_b = without_line.run(switches)

    # A: the retained workaround scores scatter and air.
    assert "tot_w1" in outputs_a
    assert "sca_w1" in outputs_a
    assert "air_w1" in outputs_a
    assert "sca_w2" in outputs_a
    assert outputs_a["sca_w1"].projection.sum() > 0
    assert np.array_equal(
        outputs_a["sca_w2"].projection, outputs_a["sca_w1"].projection
    )

    # B: dropping the appended line makes SIMIND omit scatter/air entirely, so
    # the workaround is load-bearing rather than a harmless duplication.
    assert "tot_w1" in outputs_b
    assert not [key for key in outputs_b if key.startswith("sca_")]
    assert not [key for key in outputs_b if key.startswith("air_")]
