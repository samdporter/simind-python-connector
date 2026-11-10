# simind-python-connector

Python tools for running SIMIND and using its projections with STIR, SIRF, and
PyTomography. Python 3.10 or newer is required. SIMIND is licensed separately
and is not included with this package; this project is independent of the
SIMIND project and Lund University.

## Start here

The [workflow tutorial](docs/tutorial.rst) walks through a core simulation,
known-activity NEMA phantom, reconstruction, scatter correction, and
measured-data calibration; the [example pages](docs/example_pages/index.rst)
collect the tutorials and figures. The
[documentation](https://simind-python-connector.readthedocs.io/) covers
installation, geometry, normalisation and backends.

Install the package and example plotting dependencies:

```bash
pip install "simind-python-connector[examples]"
```

Select a SIMIND executable and its matching data directory. The binary and
`smc_dir` must come from the same release. Set `SIMIND_BIN` explicitly or put
`simind` on `PATH`; set `SMC_DIR` when required by your installation.

```bash
export SIMIND_BIN=/path/to/simind-v8/simind
export PATH=/path/to/simind-v8:$PATH
export SMC_DIR=/path/to/simind-v8/smc_dir/
export SIMIND_VERSION=8
```

SIMIND v8 is the supported engine. A constructor's `simind_version` overrides
`SIMIND_VERSION`; v9 selection is experimental. See
[installation](docs/installation.rst).

Use the installed connector directly:

```python
import numpy as np
from simind_python_connector import SimindPythonConnector, configs

connector = SimindPythonConnector(
    configs.get("Example.yaml"), "output/basic", "case01", simind_version=8
)
source = np.zeros((32, 32, 32), dtype=np.float32)
source[12:20, 12:20, 12:20] = 1
connector.configure_voxel_phantom(source, np.zeros_like(source), voxel_size_mm=4)
connector.set_energy_windows([126], [154], [0])
connector.add_runtime_switch("NN", 1)
outputs = connector.run()
print(outputs["tot_w1"].projection.shape)
```

The core connector does not require STIR, SIRF, or PyTomography. Install only
the optional backend your workflow uses; the NEMA examples also need
phantomgen. See [installation](docs/installation.rst) and the tutorial.

## Example map

- `examples/getting_started/` and `examples/phantoms/`: core simulations,
  windows, configurations, scoring, density conversion and phantoms.
- `examples/backends/`: STIR, SIRF and PyTomography adaptors, plus the DICOM
  projection recipe in `docs/example_pages/dicom_recipe.rst`.
- `examples/mc_scatter/` and `examples/cil/`: Monte Carlo scatter and
  residual correction, including CIL.
- `examples/real_data/` and `docs/example_pages/y90_*.rst`: restricted Y-90
  CT conversion and effective attenuation energy, from data that are not
  redistributed; see [datasets](docs/datasets.rst).

Lower `quantization_scale`, fewer projections, or a mostly-zero source map make
smoke runs faster without changing projection counts. For quantitative output,
use known activity and acquisition time or normalise against a measured or
analytic reference. Keep `NN >= 1`.

## License

Apache 2.0. See [LICENSE](LICENSE).
