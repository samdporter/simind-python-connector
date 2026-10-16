.. _contributing:

Contributing
============

Contributions are welcome. For code changes, please follow the development and
testing checklist below before opening a pull request.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a development branch
4. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

5. Make your changes
6. Run tests to ensure everything works
7. Submit a pull request

Testing Your Changes
--------------------

Before submitting:

.. code-block:: bash

    # Run all tests
    python -m pytest tests/ -v

    # Check linting and formatting
    ruff check .
    ruff format --check .

Documentation Updates
---------------------

If you make changes that affect the documentation:

.. code-block:: bash

    sphinx-build -b html docs docs/_build/html

Preparing a Release
-------------------

The version is defined once in ``simind_python_connector/_version.py``.
Package builds, ``simind_python_connector.__version__``, documentation and the
publishing workflow all read it directly.

Write the release notes for the next version in ``CHANGELOG.md`` and
``docs/changelog.rst``, then run:

.. code-block:: bash

    bash scripts/release.sh patch

Use ``minor``, ``major`` or an exact version such as ``1.2.0`` instead of
``patch`` when appropriate. The script updates the shared version and the
Zenodo and citation metadata, including the citation release date, then creates
a local release commit and tag. Review your working tree first; the script also
commits the release notes. It prints the commands for pushing and publishing.
No other version values need editing by hand.
