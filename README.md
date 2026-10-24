# simind-python-connector

Python tools for running SIMIND and using its projections with STIR, SIRF, and
PyTomography. Python 3.10 or newer is required. SIMIND is licensed separately
and is not included with this package; this project is independent of the
SIMIND project and Lund University.

## Start here

See the [workflow tutorial](docs/tutorial.rst) for an ordered path through a
core simulation, known-activity NEMA phantom, reconstruction, scatter
correction, and measured-data calibration. The [gallery](docs/auto_examples/index.rst)
has every tutorial with its figures and printed output. The
[documentation](https://simind-python-connector.readthedocs.io/) explains
installation, geometry, normalisation, backend options, and the
[observed SIMIND output-component mapping](docs/simind_validation.md).

Install the package and example plotting dependencies:

```bash
pip install "simind-python-connector[examples]"
```

Select a SIMIND executable and its matching data directory. The binary and
`smc_dir` must come from the same release. Set `SIMIND_BIN` explicitly or put
`simind` on `PATH`; set `SMC_DIR` when required by your installation.

```bash
export SIMIND_BIN=/path/to/simind-v9/simind
export PATH=/path/to/simind-v9:$PATH
export SMC_DIR=/path/to/simind-v9/smc_dir/
export SIMIND_VERSION=9
```

Select version `8` or `9` explicitly. The connector uses this choice to place
voxel source and density maps in the same physical coordinates; the default
is version 8. A constructor's `simind_version` overrides `SIMIND_VERSION`.

Use the installed connector directly:

```python
import numpy as np
from simind_python_connector import SimindPythonConnector, configs

connector = SimindPythonConnector(
    configs.get("Example.yaml"), "output/basic", "case01", simind_version=9
)
source = np.zeros((32, 32, 32), dtype=np.float32)
source[12:20, 12:20, 12:20] = 1
connector.configure_voxel_phantom(source, np.zeros_like(source), voxel_size_mm=4)
connector.set_energy_windows([126], [154], [0])
connector.add_runtime_switch("NN", 1)
outputs = connector.run()
print(outputs["tot_w1"].projection.shape)
```

Run a complete core simulation example from a source checkout:

```bash
python examples/getting_started/plot_01_basic_simulation.py
```

The core connector does not require STIR, SIRF, or PyTomography. Install only
the optional backend used by your workflow; see
[installation](docs/installation.rst). The NEMA examples need the optional
phantomgen package; the install command is in the tutorial.

## Example map

- `examples/getting_started/plot_01`–`plot_06`: core NumPy simulations, windows, configurations, scoring,
  and density conversion.
- `examples/backends/plot_07a` / `07b` / `07c`: STIR / SIRF / PyTomography adaptors.
- `examples/backends/plot_08a` / `08b` / `08c`: DICOM setup for STIR / SIRF / PyTomography.
- `examples/phantoms/plot_09` / `plot_10`: the SIMIND analytic phantom library,
  and a known-activity NEMA/IEC simulation reconstructed with SIRF OSEM.
- `examples/mc_scatter/plot_11`–`plot_13`: Monte Carlo scatter in SIRF OSEM,
  scatter normalisation, and residual correction.
- `examples/cil/plot_14` / `plot_15`: CIL scatter correction, and an
  RDP-regularised SVRG reconstruction.
- `examples/real_data/plot_16a` / `16b` / `16c`: CT-to-SPECT conversion with
  STIR / SIRF / PyTomography for the measured Y-90 phantom.
- `examples/real_data/plot_17`: the Y-90 effective attenuation energy with SIRF.

The real-data pages are rendered case studies on restricted Y-90 input, not
reproducible tutorials. See [datasets](docs/datasets.rst) for the availability
statement.

Lower `quantization_scale`, fewer projections, or a mostly-zero source map can
make smoke runs faster. These settings reduce history statistics; they do not
set projection counts. For quantitative output, use known activity and
acquisition time or normalise against a measured or analytic reference. Keep
`NN >= 1`.

## License

Apache 2.0. See [LICENSE](LICENSE).
