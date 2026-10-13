# simind-python-connector

Python SIMIND Monte Carlo Connector.

[![Tests](https://github.com/samdporter/simind-python-connector/workflows/Tests/badge.svg)](https://github.com/samdporter/simind-python-connector/actions)
[![Documentation Status](https://readthedocs.org/projects/simind-python-connector/badge/?version=latest)](https://simind-python-connector.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A Python toolkit that lets you run SIMIND from Python and use the outputs in
common reconstruction ecosystems (STIR, SIRF, PyTomography).

## Disclaimer

This project is an independent Python connector toolkit and is **not
affiliated with, endorsed by, or maintained by** the SIMIND project or Lund
University.

SIMIND is **not distributed** with this package. You must separately obtain and
install a licensed SIMIND installation.

## Quick Links
- [Full Documentation](https://simind-python-connector.readthedocs.io/)
- [Installation](https://simind-python-connector.readthedocs.io/en/latest/installation.html)
- [Backend Support](https://simind-python-connector.readthedocs.io/en/latest/backends.html) - adaptor dependency matrix

## What This Package Does
1. Runs SIMIND from Python with a minimal, explicit API.
2. Adapts SIMIND data for widely used Python reconstruction packages.

## Key Features
- **Connector-first API** - `SimindPythonConnector` for direct SIMIND execution from Python
- **Package Adaptors** - STIR/SIRF/PyTomography adaptors for reconstruction workflows
- **Native reconstruction workflows** - Use STIR/SIRF/PyTomography reconstruction tools with generated SIMIND data
- **Dual scoring support** - SCATTWIN and PENETRATE
- **DICOM builders** - DICOM-driven setup utilities for scanner/input preparation
- **Advanced Schneider2000 density conversion** - 44-segment HU-to-density mapping

## Installation

### Basic Installation

```bash
pip install simind-python-connector
```

Import with:

```python
import simind_python_connector
```

For example plotting dependencies:

```bash
pip install "simind-python-connector[examples]"
```

**NumPy**: the package requires `numpy>=1.20.0,<2.0`. The `<2.0` cap is a
binary-ABI / backend-compatibility guard for the legacy 1.1.x line: the STIR
and SIRF bindings are compiled against NumPy 1.x and crash with NumPy 2, so the
cap applies package-wide. It is not a Python-version constraint; interpreter
support is stated separately as Python 3.10+.

### Adaptor Dependencies

`SimindPythonConnector` works without SIRF/STIR/PyTomography.

Install optional packages only for the adaptor paths you need:

- **STIR Python** for `StirSimindAdaptor` workflows (example 07A)
- **SIRF** for `SirfSimindAdaptor` workflows (example 07B)
- **PyTomography** for `PyTomographySimindAdaptor` workflows (example 07C)

#### Option 1: STIR Python (Recommended for basic usage)

Install from conda:
```bash
conda install -c conda-forge stir "numpy<2.0"
```

**Important**: As of November 2025, the STIR conda package (v6.3.0) requires NumPy < 2.0 due to binary compatibility issues with NumPy 2.x. The package was compiled against NumPy 1.x and will crash with memory errors if NumPy 2.x is installed. This should be resolved in future STIR releases. Like the package-wide pin, this is a backend binary-ABI compatibility guard, not a Python-version constraint.

Or build from source:
```bash
git clone https://github.com/UCL/STIR.git
cd STIR
# Follow build instructions in the repository
```

#### Option 2: SIRF

Install from source:
```bash
git clone https://github.com/SyneRBI/SIRF.git
cd SIRF
# Follow build instructions in the repository
```

**Note**: SIRF includes STIR, so a separate STIR install is usually unnecessary.

#### Option 3: PyTomography

Install PyTomography for the PyTomography adaptor workflow:

```bash
pip install pytomography
```

### SIMIND Requirement

SIMIND is **not included** with this package and must be installed separately.

Use the official SIMIND resources:

- SIMIND site (Medical Radiation Physics, Lund University): https://www.msf.lu.se/en/research/simind-monte-carlo-program
- SIMIND manual/docs: https://www.msf.lu.se/en/research/simind-monte-carlo-program/manual

For local use with this package's scripts, place your SIMIND installation under:

```text
./simind
```

and make sure the executable is available at:

```text
./simind/simind
```

and ensure SIMIND data files are available under:

```text
./simind/smc_dir/
```

The Docker helper scripts use this layout and automatically wire SIMIND paths
when the binary exists at `./simind/simind`.

For direct local runs outside the repository helpers, make sure the SIMIND
executable is available as `simind` on `PATH`.

## Quick Start
```python
import numpy as np
from simind_python_connector import SimindPythonConnector
from simind_python_connector.configs import get
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE

source = np.zeros((32, 32, 32), dtype=np.float32)  # z, y, x
source[12:20, 12:20, 12:20] = 1.0
mu_map = np.zeros_like(source)
mu_map[source > 0] = 0.15

connector = SimindPythonConnector(
    config_source=get("Example.yaml"),
    output_dir="output/basic",
    output_prefix="case01",
    quantization_scale=0.05 * DEFAULT_QUANTIZATION_SCALE,
)

connector.configure_voxel_phantom(
    source=source,
    mu_map=mu_map,
    voxel_size_mm=4.0,
)
connector.set_energy_windows([126], [154], [0])  # Tc-99m ± 10%
connector.add_runtime_switch("FI", "tc99m")
connector.add_runtime_switch("CC", "ma-lehr")
connector.add_runtime_switch("NN", 1)
connector.add_runtime_switch("RR", 12345)

outputs = connector.run()
total = outputs["tot_w1"].projection
print(total.shape)
```

### Advanced Density Conversion

```python
from simind_python_connector.converters.attenuation import hu_to_density_schneider
import numpy as np

# Convert HU image to densities using Schneider2000 model
hu_image = np.array([[-1000, 0, 500], [800, 1200, 2000]])
density_map = hu_to_density_schneider(hu_image)  # 44-segment clinical model
```

### Pure Python Connector (NumPy Outputs)

```python
from simind_python_connector import RuntimeOperator, SimindPythonConnector
from simind_python_connector.configs import get

connector = SimindPythonConnector(
    config_source=get("AnyScan.yaml"),
    output_dir="output/python_connector",
    output_prefix="case01",
    quantization_scale=1.0,
)

outputs = connector.run(RuntimeOperator(switches={"NN": 1, "RR": 12345}))
total = outputs["tot_w1"]
print(total.projection.shape)
print(total.header_path)
```

For minimal toy runs with smaller projection/image settings, use
`get("Example.yaml")`.

`quantization_scale` controls source integer quantization before writing SIMIND
`.smi` files:

- the default (`500 / 32767`) preserves the legacy 500-weight source budget;
- `1.0` uses the full signed 16-bit source range (`32767`), the best numeric
  precision;
- `< 1.0` reduces the integer scale for faster toy runs but increases rounding
  error.

SIMIND reads the source and density maps as signed 16-bit integers, so values
of `32768` and above wrap negative and silently corrupt the map; the connector
caps both maps at `32767`. Explicit `quantization_scale` values are used as
given, with no automatic migration of scales calibrated for the former
500-weight maximum: rescale an old scale with
`old_scale * DEFAULT_QUANTIZATION_SCALE`. SIMIND treats source maps as integer
weights; absolute activity/time scaling is controlled separately by simulation
settings.

## Docker Environments

Dedicated container definitions are provided in [`docker/`](docker/):

- `docker/stir/Dockerfile`
- `docker/sirf/Dockerfile`
- `docker/pytomography/Dockerfile`

Use Docker Compose:

```bash
docker compose -f docker/compose.yaml build stir
docker compose -f docker/compose.yaml run --rm stir
```

Run per-backend validation and separated examples:

```bash
bash scripts/run_container_validation.sh
bash scripts/run_container_validation.sh --with-simind
bash scripts/run_container_examples.sh
```

## Contributing
Please see the [Contributing Guide](docs/contributing.rst).

## License
Apache License 2.0
