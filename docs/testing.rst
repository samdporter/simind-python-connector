.. _testing:

Testing
=======

This document explains the testing strategy for simind-python-connector, which handles the
challenge of testing code that depends on optional external dependencies (SIRF,
STIR, SIMIND, and PyTomography) that may not be available in every
environment.

Test Organization
-----------------

Test Markers
~~~~~~~~~~~~

Tests are organized using pytest markers to indicate their dependencies:

* ``@pytest.mark.unit`` - Pure unit tests with no external dependencies (CI-friendly)
* ``@pytest.mark.requires_sirf`` - Tests that require SIRF to be installed
* ``@pytest.mark.requires_stir`` - Tests that require STIR Python to be installed
* ``@pytest.mark.requires_simind`` - Tests that require SIMIND command-line tool
* ``@pytest.mark.requires_pytomography`` - Tests that require PyTomography to be installed
* ``@pytest.mark.integration`` - Integration tests (may be slow)
* ``@pytest.mark.ci_skip`` - Tests to skip in CI environments

Automatic Dependency Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The test suite automatically detects available dependencies:

* **SIRF detection**: Attempts to ``import sirf.STIR``
* **STIR detection**: Attempts to ``import stir``, then imports the STIR
  conversion helpers the way the package itself does (``stir.extra`` preferred,
  the legacy ``stirextra`` module only as a fallback), so a broken
  ``stir.extra`` dependency is reported as unavailable instead of being masked
  by the legacy module
* **PyTomography detection**: Attempts to ``import pytomography``
* **SIMIND detection**: Uses ``$SIMIND_BIN`` when set, otherwise looks for
  ``simind`` on ``PATH``
* **CI detection**: Checks ``CI`` or ``GITHUB_ACTIONS`` environment variables

Tests requiring unavailable dependencies are automatically skipped with informative messages.

Running Tests
-------------

Local Development (All Tests)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run all available tests
    pytest

    # Run only unit tests (fast)
    pytest -m unit

    # Run only SIRF-dependent tests
    pytest -m requires_sirf

    # Run with coverage
    pytest --cov=simind_python_connector --cov-report=html

CI Environment (GitHub Actions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GitHub Actions workflow runs only CI-friendly tests:

.. code-block:: bash

    # CI command (no SIRF/SIMIND dependencies)
    pytest -m "not integration and not requires_sirf and not requires_stir and not requires_simind and not requires_pytomography and not requires_cil and not requires_setr and not ci_skip"

Docker Backend Isolation
~~~~~~~~~~~~~~~~~~~~~~~~

Use the dedicated backend containers to validate connector separation and run one
example per backend in isolated environments:

.. code-block:: bash

    bash scripts/run_container_validation.sh
    bash scripts/run_container_examples.sh

Run only selected groups:

.. code-block:: bash

    bash scripts/run_container_validation.sh --only-core
    bash scripts/run_container_validation.sh --only-pytomography
    bash scripts/run_container_examples.sh --only-osem

Override target architecture (or let scripts auto-detect from SIMIND binary):

.. code-block:: bash

    bash scripts/run_container_examples.sh --only-core --docker-platform linux/amd64
    bash scripts/run_container_validation.sh --with-simind --docker-platform linux/amd64

To include SIMIND-dependent integration/example checks in the validation run:

.. code-block:: bash

    bash scripts/run_container_validation.sh --with-simind

The container scripts check for a repo-local SIMIND executable at
``./simind/simind`` before running SIMIND-dependent checks. If missing, they
skip those checks by default. Use ``--require-simind`` to fail fast instead.
``input.smc`` remains packaged in ``simind_python_connector/configs`` and is not
part of the SIMIND runtime availability check.

SIMIND itself is not bundled with this package; install it separately from
the official SIMIND site and manual:

* https://www.msf.lu.se/en/research/simind-monte-carlo-program
* https://www.msf.lu.se/en/research/simind-monte-carlo-program/manual

The container suite includes ``tests/test_container_library_isolation.py``,
which asserts that each container can import only its target backend library.

Test Categories
---------------

CI-Friendly Test Suites
~~~~~~~~~~~~~~~~~~~~~~~

The default CI job runs the tests that avoid heavyweight dependencies. These focus on:

* **Conversion maths** – ``tests/test_schneider_density.py`` exercises the HU-to-density pipelines.
* **Configuration & builders** – ``tests/test_simulation_config.py`` and ``tests/test_acquisition_builder_unit.py`` validate YAML/SMC handling plus Interfile header generation without a SIRF runtime.
* **Connector behavior** – ``tests/test_python_connector.py`` and ``tests/test_connector_backend_separation.py`` cover the connector-first API without requiring reconstruction packages.
* **Utility helpers** – ``tests/test_utils_small.py``, ``tests/test_interfile_parser.py``, and ``tests/test_interfile_numpy.py`` cover file parsing and small helper functions.
* **Backend dispatch** – ``tests/test_backends.py`` covers backend detection and dispatch with stand-in backends.

Backend-Dependent Suites
~~~~~~~~~~~~~~~~~~~~~~~~

Tests that construct real STIR/SIRF/PyTomography objects carry dependency
markers and are skipped when those packages are unavailable. These include:

* ``tests/test_native_geometry_real.py`` for native STIR/SIRF image geometry.
* ``tests/test_utils.py`` for phantom, image and acquisition helpers on SIRF and STIR.
* ``tests/test_pytomography_adaptor.py`` for PyTomography tensor and output adaptation.
* ``tests/test_arithmetic_operations.py`` and ``tests/test_stir_backend_arithmetic.py`` for arithmetic on real backend objects.

SIMIND and Container Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SIMIND-dependent tests use ``requires_simind`` and are skipped unless SIMIND
is found. Container and geometry diagnostics include:

* ``tests/test_integration.py`` for full workflow checks.
* ``tests/test_container_library_isolation.py`` for backend container import isolation.

The MPI cases in ``tests/test_patch_release_simind.py`` additionally require
``mpirun``, the ``simind_mpi`` binary and its ``libmpi`` runtime libraries; the
latter two are provided by ``docker/python/Dockerfile``, which installs MPICH
and exposes the ``libmpi.so.12``/``libmpifort.so.12`` names SIMIND links
against. They never run in GitHub CI because the SIMIND binary is not tracked.

Configuration Files
-------------------

pyproject.toml
~~~~~~~~~~~~~~

* ``[tool.pytest.ini_options]`` registers every marker and sets the default
  options for local development
* The CI dependency selection is passed with ``-m`` in the GitHub Actions
  workflow, not from the default options

tests/conftest.py
~~~~~~~~~~~~~~~~~

* Implements automatic dependency detection and test skipping
* Handles CI environment detection

Adding New Tests
----------------

Use ``unit`` for isolated behavior and ``integration`` for workflows involving
external systems. Add the relevant dependency markers whenever a test needs a
reconstruction library or SIMIND. The distribution build check runs once in
the CI build job, separately from the unit test matrix.

Assert observable results: array values and orientation, saved file contents,
metadata or a useful failure. Stub external I/O at the boundary when needed.
Tests that only check whether a function exists, repeat its implementation or
assert private storage details do not establish that a workflow works.

SIMIND geometry comparisons use Flag 8 set to false for a repeatable initial
random state. Choose photon statistics to support the numerical tolerance;
setting ``RR`` alone does not make V8 simulations reproducible.

Continuous Integration
----------------------

GitHub Actions is used to run tests automatically. The CI workflow:

1. **Installs only basic Python dependencies** (no SIRF/SIMIND)
2. **Runs code quality checks** (``ruff check`` and ``ruff format --check``)
3. **Executes CI-friendly tests** using dependency markers
4. **Generates coverage reports** for the tested code
5. **Builds and validates the package**
