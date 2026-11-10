.. _installation:

Installation
============

- Python 3.10+
- A separate SIMIND installation to run simulations. SIMIND is not distributed
  with this package or installed by pip.
- Optional for your workflow: STIR Python, SIRF or PyTomography.

Install the Python Package
--------------------------

Install the released package from PyPI:

.. code-block:: bash

    pip install simind-python-connector

Install the examples extra to run the plotting examples:

.. code-block:: bash

    pip install "simind-python-connector[examples]"

Development Install
-------------------

For local development from a checkout:

.. code-block:: bash

    git clone https://github.com/samdporter/simind-python-connector.git
    cd simind-python-connector
    pip install -e ".[dev,examples]"

SIMIND
------

This project is independent and is **not affiliated with, endorsed by, or
maintained by** the SIMIND project or Lund University. SIMIND is **not**
distributed with this package (or on PyPI) and must be installed separately.
Use the official SIMIND site and manual:

- https://www.msf.lu.se/en/research/simind-monte-carlo-program
- https://www.msf.lu.se/en/research/simind-monte-carlo-program/manual

Runtime Environment
-------------------

Pair the SIMIND executable with the ``smc_dir`` data from the same release.
Set ``SIMIND_BIN`` to select the executable explicitly; otherwise the connector
searches for ``simind`` on ``PATH``. Set ``SMC_DIR`` to that release's data
directory, with a trailing slash:

.. code-block:: bash

    export SIMIND_BIN="/path/to/simind-v8/simind"
    export SMC_DIR="/path/to/simind-v8/smc_dir/"
    export PATH="/path/to/simind-v8:$PATH"
    export SIMIND_VERSION=8

SIMIND v8 is the supported engine. Set ``SIMIND_VERSION`` to ``8``; an explicit
``simind_version=8`` on a connector or adaptor overrides the environment.

.. warning::

   SIMIND v9 is experimental. The v9.0 Linux Intel build can abort or exit
   normally with silently corrupted SCATTWIN projections. Use SIMIND v8 for
   supported workflows; see :doc:`simind_validation`.

Repo/Docker Layout
------------------

For the repository helper scripts and Docker setup, place SIMIND under
``./simind``, with ``./simind/simind`` and ``./simind/smc_dir/`` present. Make
the binary executable if needed:

.. code-block:: bash

    chmod +x ./simind/simind

The container scripts wire these paths inside the containers when
``./simind/simind`` is present.

Quick Verification
------------------

.. code-block:: bash

    python -c "import simind_python_connector as s; print(s.__version__)"
    test -x ./simind/simind && echo "SIMIND found" || echo "SIMIND missing"
