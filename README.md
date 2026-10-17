# simind-python-connector

Python tools for running SIMIND and using its projections with STIR, SIRF, and
PyTomography. Python 3.10 or newer is required. SIMIND is licensed separately
and is not included with this package; this project is independent of the
SIMIND project and Lund University.

## Start here

See the [workflow tutorial](docs/tutorial.rst) for an ordered path through a
core simulation, known-activity NEMA phantom, reconstruction, scatter
correction, and measured-data calibration. The [documentation](https://simind-python-connector.readthedocs.io/)
explains installation, geometry, normalisation, backend options, and the
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
python examples/01_basic_simulation.py
```

The core connector does not require STIR, SIRF, or PyTomography. Install only
the optional backend used by your workflow; see
[installation](docs/installation.rst). The NEMA examples need the optional
phantomgen package; the install command is in the tutorial.

## Example map

- `examples/01`–`06`: core NumPy simulations, windows, configurations, scoring,
  and density conversion.
- `examples/07A` / `07B` / `07C`: STIR / SIRF / PyTomography adaptors.
- `examples/08A` / `08B` / `08C`: DICOM setup for STIR / SIRF / PyTomography.
- `examples/09`–`11`: SIRF scatter and residual correction, plus CIL in `10`.
- `examples/12`: phantomgen NEMA/IEC with known activity, with SIMIND's
  multiple-inserts phantom as a geometry-only fallback.
- `examples/13`: one known-activity NEMA simulation reconstructed with STIR or SIRF.

Lower `quantization_scale`, fewer projections, or a mostly-zero source map can
make smoke runs faster. These settings reduce history statistics; they do not
set projection counts. For quantitative output, use known activity and
acquisition time or normalise against a measured or analytic reference. Keep
`NN >= 1`.

## License

Apache 2.0. See [LICENSE](LICENSE).
