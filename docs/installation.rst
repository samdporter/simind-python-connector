.. _installation:

Installation
============

Prerequisites
-------------

- Python 3.10+
- A separate SIMIND installation for running simulations. SIMIND is not
  distributed with this package or installed by pip.
- Optional backend libraries depending on workflow:
  - STIR Python for STIR-based paths
  - SIRF for SIRF-based paths
  - PyTomography for PyTomography adaptor paths

Install the Python Package
--------------------------

Install the released package from PyPI:

.. code-block:: bash

    pip install simind-python-connector

Import it with:

.. code-block:: python

    import simind_python_connector

Install the examples extra if you want to run the plotting examples:

.. code-block:: bash

    pip install "simind-python-connector[examples]"

Development Install
-------------------

For local development from a checkout:

.. code-block:: bash

    git clone https://github.com/samdporter/simind-python-connector.git
    cd simind-python-connector
    pip install -e ".[dev,examples]"

SIMIND Requirement (External Dependency)
----------------------------------------

Disclaimer
~~~~~~~~~~

This project is independent and is **not affiliated with, endorsed by, or
maintained by** the SIMIND project or Lund University.

SIMIND is **not** distributed with this package (or on PyPI) and must be
installed separately.

Use the official SIMIND resources for installation and manual/reference
documentation:

- SIMIND site (Medical Radiation Physics, Lund University):
  https://www.msf.lu.se/en/research/simind-monte-carlo-program
- SIMIND manual page:
  https://www.msf.lu.se/en/research/simind-monte-carlo-program/manual

Repo/Docker Layout
~~~~~~~~~~~~~~~~~~

For the repository helper scripts and Docker setup, place SIMIND under:

.. code-block:: text

    ./simind

and ensure these paths exist:

.. code-block:: text

    ./simind/simind
    ./simind/smc_dir/

If needed, make the binary executable:

.. code-block:: bash

    chmod +x ./simind/simind

Local Runtime Environment
~~~~~~~~~~~~~~~~~~~~~~~~~

For direct local runs, pair the SIMIND executable with the ``smc_dir`` data
from the same release. Set ``SIMIND_BIN`` to select the executable explicitly;
otherwise the connector searches for ``simind`` on ``PATH``. Set
``SMC_DIR`` to that release's data directory, with a trailing slash:

.. code-block:: bash

    export SIMIND_BIN="/path/to/simind/simind"
    export SMC_DIR="/path/to/simind/smc_dir/"
    export PATH="/path/to/simind:$PATH"
    export SIMIND_VERSION=9

Set ``SIMIND_VERSION`` to ``8`` or ``9`` to match the executable. The default
is ``8``; ``simind_version=9`` on a connector or adaptor overrides the
environment. This explicit selection accounts for the different voxel-map
origins. It does not replace selecting the matching executable and data.

.. warning::

   The SIMIND v9.0 Linux Intel build has a SCATTWIN defect: it can abort with
   heap corruption or exit normally with silently corrupted projections. For
   SCATTWIN simulations use SIMIND v8 or the macOS v9 build; the PENETRATE
   routine is unaffected. The connector emits a ``RuntimeWarning`` when asked to
   run SCATTWIN with v9 on Linux. See :doc:`simind_validation`.

When working from this repository with the layout above, those commands become:

.. code-block:: bash

    export SIMIND_BIN="$PWD/simind/simind"
    export PATH="$PWD/simind:$PATH"
    export SMC_DIR="$PWD/simind/smc_dir/"
    export SIMIND_VERSION=8

Docker Behavior
~~~~~~~~~~~~~~~

The Docker Compose services and container helper scripts use the local SIMIND
layout above. They automatically wire SIMIND paths inside the containers when
``./simind/simind`` is present.

Quick Verification
------------------

Verify basic Python package import:

.. code-block:: bash

    python -c "import simind_python_connector as s; print(s.__version__)"

Verify SIMIND detection from this repo root:

.. code-block:: bash

    test -x ./simind/simind && echo "SIMIND found" || echo "SIMIND missing"
