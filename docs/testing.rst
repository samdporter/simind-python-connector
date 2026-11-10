.. _testing:

Testing
=======

Tests carry markers for their dependencies; ``conftest.py`` skips a test when
its dependency is missing.

Test Markers
------------

- ``unit``: pure Python, no external dependency.
- ``integration``: slower integration tests.
- ``slow``: more than 10 seconds.
- ``ci_skip``: skipped in CI.
- ``requires_sirf``, ``requires_stir``, ``requires_simind``,
  ``requires_pytomography``, ``requires_cil``, ``requires_phantomgen``:
  skipped when that dependency is missing.
- ``validation``: SIMIND feature validation with recorded numbers.
- ``scattwin_data``: validation that checks SCATTWIN projection data.

Dependency detection imports ``sirf``, ``stir`` (preferring ``stir.extra`` and
falling back to ``stirextra`` only when it is absent), ``pytomography`` and
``phantomgen``, and checks for the ``simind`` command. ``requires_simind``
checks ``PATH`` only.

Running Tests
-------------

.. code-block:: bash

    pytest                                  # all available tests
    pytest -m unit                          # fast unit tests
    pytest -m requires_sirf                 # SIRF-dependent tests
    pytest --cov=simind_python_connector --cov-report=html

The CI selection is:

.. code-block:: bash

    python -m pytest -m "not integration and not requires_sirf and not requires_stir and not requires_simind and not requires_pytomography and not requires_cil and not ci_skip"

Docker Backend Isolation
------------------------

.. code-block:: bash

    bash scripts/run_container_validation.sh
    bash scripts/run_container_examples.sh
    bash scripts/run_container_validation.sh --only-core
    bash scripts/run_container_validation.sh --only-pytomography
    bash scripts/run_container_examples.sh --only-osem
    bash scripts/run_container_examples.sh --only-core --docker-platform linux/amd64
    bash scripts/run_container_validation.sh --with-simind --docker-platform linux/amd64

The scripts look for ``./simind/simind`` and skip SIMIND checks when it is
absent; ``--require-simind`` fails fast instead.
``tests/test_container_library_isolation.py`` asserts that each container
imports only its target backend.

Measured-Geometry Validation Profiles
-------------------------------------

``tests/test_measured_geometry_simind.py`` forwards a phantom through SIMIND
and SIRF SPECTUB in the same template geometry for clockwise,
counter-clockwise, non-circular and anisotropic-voxel cases. It selects a
compute budget from an environment switch:

- **Smoke** (default): 16³ voxels, 8 views,
  ``quantization_scale=0.02 x (500 / 32767)``, correlation > 0.5, centre gap
  <= 1 bin.
- **Full** (``SIMIND_A3_FULL_VALIDATION=1``): 64³ voxels, 60 views, default
  ``quantization_scale``, correlation > 0.9, centre gap <= 1 bin.

Put SIMIND on ``PATH``, then run either profile:

.. code-block:: bash

    export PATH="$PWD/simind:$PATH"
    export SMC_DIR="$PWD/simind/smc_dir/"
    pytest tests/test_measured_geometry_simind.py -s
    SIMIND_A3_FULL_VALIDATION=1 pytest tests/test_measured_geometry_simind.py -s

Feature validation
------------------

``tests/validation`` holds one module per feature. Each case runs a small
SIMIND simulation, compares the result with a stated tolerance, and appends the
measured value to ``output/validation/<engine>.jsonl``. Release validation runs
against SIMIND v8 on Linux (Docker, amd64); the runner rejects v9.

.. code-block:: bash

    bash scripts/run_validation.sh --engine v8
    bash scripts/run_validation.sh --engine v8 --feature recon

The runs take hours. Rebuild the published tables with:

.. code-block:: bash

    python scripts/validation_report.py --write docs/simind_validation.md

Configuration
-------------

``pyproject.toml`` registers every marker and sets default options.
``tests/conftest.py`` implements dependency detection and skipping.

Adding New Tests
----------------

Mark every new test:

.. code-block:: python

    import pytest

    @pytest.mark.unit
    def test_pure_python_logic():
        assert True

    @pytest.mark.requires_sirf
    def test_sirf_functionality():
        from sirf.STIR import ImageData
