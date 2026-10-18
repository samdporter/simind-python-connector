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

* **SIRF detection**: Attempts to ``import sirf``
* **STIR detection**: Attempts to ``import stir``, then imports the STIR
  conversion helpers the way the package itself does (``stir.extra`` preferred,
  the legacy ``stirextra`` module only as a fallback), so a broken
  ``stir.extra`` dependency is reported as unavailable instead of being masked
  by the legacy module
* **PyTomography detection**: Attempts to ``import pytomography``
* **SIMIND detection**: Checks if ``simind`` command is available
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
    pytest -m "not requires_sirf and not requires_stir and not requires_simind and not requires_pytomography and not ci_skip"

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
With SIMIND enabled, ``tests/test_measured_geometry_simind.py`` checks known
positions, rotation directions, non-circular orbits and anisotropic spacing
against SPECTUB. These cases generate their own inputs.

Alternative CI Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m pytest -m "not integration and not requires_sirf and not requires_stir and not requires_simind and not requires_pytomography and not requires_cil and not ci_skip"

Test Categories
---------------

CI-Friendly Test Suites
~~~~~~~~~~~~~~~~~~~~~~~

The default CI job runs the tests that avoid heavyweight dependencies. These focus on:

* **Conversion maths** – ``tests/test_schneider_density.py`` exercises the HU-to-density pipelines.
* **Configuration & builders** – ``tests/test_simulation_config.py`` and ``tests/test_acquisition_builder_unit.py`` validate YAML/SMC handling plus Interfile header generation without a SIRF runtime.
* **Connector behavior** – ``tests/test_python_connector.py``, ``tests/test_connectors_base.py``, and ``tests/test_connector_backend_separation.py`` cover the connector-first API without requiring reconstruction packages.
* **Utility helpers** – ``tests/test_interfile.py`` and ``tests/test_utils_small.py`` cover Interfile parsing and small helper functions.
* **Backend guards** – ``tests/test_builders_native.py``, ``tests/test_marker_policy.py``, and ``tests/test_examples_backend_separation.py`` ensure optional backend imports remain isolated.

Backend-Dependent Suites
~~~~~~~~~~~~~~~~~~~~~~~~

Tests that construct real STIR/SIRF/PyTomography objects carry dependency
markers and are skipped when those packages are unavailable. These include:

* ``tests/test_native_adaptors.py`` for STIR/SIRF adaptor behavior.
* ``tests/test_pytomography_adaptor.py`` for PyTomography tensor and output adaptation.

SIMIND and Container Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SIMIND-dependent tests use ``requires_simind`` and are skipped unless the
``simind`` command is available. Container and geometry diagnostics include:

* ``tests/test_integration.py`` for full workflow checks.
* ``tests/test_container_library_isolation.py`` for backend container import isolation.
* ``tests/test_measured_geometry_simind.py`` for controlled geometry comparisons.
* ``tests/recon/test_residual_sirf.py`` for interleaved-model cache isolation.

A3 Measured-Geometry Validation Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/test_measured_geometry_simind.py`` forwards a phantom through SIMIND and
through SIRF's SPECTUB model in the same template geometry, covering CW, CCW,
non-circular orbit and anisotropic-voxel cases. It selects its compute budget
from an environment switch so the same checks can run cheaply or at full
statistics:

* **Smoke** (default) - 16x16x16 grid, 8 views,
  ``quantization_scale=0.02 x (500 / 32767)``, correlation > 0.5, centre gap
  <= 1 bin. Fast enough for a local smoke check.
* **Full** (``SIMIND_A3_FULL_VALIDATION=1``) - 64x64x64 grid, 60 views, the
  default ``quantization_scale`` (the legacy 500-history budget), correlation
  > 0.9, centre gap <= 1 bin. This is the standing A3 release gate.

The grid, view count, phantom offsets, template and thresholds are all derived
from the selected profile, and each run prints the profile, its settings and the
measured correlation and centre gap. Put the SIMIND executable on ``PATH`` so
the ``requires_simind`` check passes:

.. code-block:: bash

    export PATH="$PWD/simind:$PATH"
    export SMC_DIR="$PWD/simind/smc_dir/"

    # Smoke profile (default)
    pytest tests/test_measured_geometry_simind.py -s

    # Full validation profile (standing A3 release gate)
    SIMIND_A3_FULL_VALIDATION=1 pytest tests/test_measured_geometry_simind.py -s

Feature validation
~~~~~~~~~~~~~~~~~~

``tests/validation`` holds one module per feature. Each case runs a small SIMIND
simulation, compares the result with a stated tolerance, and appends the
measured value to ``output/validation/<engine>.jsonl``. The runner executes the
cases on three engines:

* ``v8``: SIMIND v8 on Linux (Docker, amd64).
* ``v9``: SIMIND v9 on Linux (Docker, amd64). SCATTWIN cases that check
  projection data are recorded as not run because of a SIMIND v9 Linux engine
  defect.
* ``v9-macos``: the native macOS SIMIND v9 build, for the cases marked
  ``macos_v9``.

.. code-block:: bash

    bash scripts/run_validation.sh --engine v8
    bash scripts/run_validation.sh --engine v9
    bash scripts/run_validation.sh --engine v9-macos

    # Re-run one feature only
    bash scripts/run_validation.sh --engine v8 --feature recon

The runs take hours. Rebuild the published tables from the recorded results
with:

.. code-block:: bash

    python scripts/validation_report.py --write docs/simind_validation.md

The results are published in ``docs/simind_validation.md``.

Integration Test
~~~~~~~~~~~~~~~~

``tests/test_integration.py`` orchestrates a full example run and needs
external runtime dependencies. It is marked with dependency markers so it is
skipped in lightweight CI environments.

Configuration Files
-------------------

pyproject.toml
~~~~~~~~~~~~~~

* ``[tool.pytest.ini_options]`` registers every marker and sets default options
* The CI selection is passed with ``-m`` (see above)

tests/conftest.py
~~~~~~~~~~~~~~~~~

* Implements automatic dependency detection and test skipping
* Handles CI environment detection

Adding New Tests
----------------

When adding new tests, use appropriate markers:

.. code-block:: python

    import pytest

    @pytest.mark.unit
    def test_pure_python_logic():
        """Test that doesn't need external dependencies."""
        assert True

    @pytest.mark.requires_sirf  
    def test_sirf_functionality():
        """Test that uses SIRF objects."""
        from sirf.STIR import ImageData
        # ... test code

    @pytest.mark.requires_simind
    def test_simind_execution():
        """Test that calls SIMIND command."""
        # ... test code that runs simind

    @pytest.mark.requires_pytomography
    def test_pytomography_path():
        """Test that uses PyTomography APIs."""
        # ... test code that imports pytomography

This ensures your tests will be properly categorized and run in the appropriate environments.

Continuous Integration
----------------------

GitHub Actions is used to run tests automatically. The CI workflow:

1. **Installs only basic Python dependencies** (no SIRF/SIMIND)
2. **Runs code quality checks** (``ruff check`` and ``ruff format --check``)
3. **Executes CI-friendly tests** using dependency markers
4. **Generates coverage reports** for the tested code
5. **Builds and validates the package**

This approach ensures reliable CI while maintaining comprehensive test coverage for local development.
