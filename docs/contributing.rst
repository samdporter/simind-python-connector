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

Releasing
---------

Release notes are written by hand; the release tooling only validates them and
updates the machine version surfaces.

1. Write the release notes. In ``CHANGELOG.md``, move the entries out of
   ``## [Unreleased]`` into a new ``## [X.Y.Z] - YYYY-MM-DD`` section and leave
   ``## [Unreleased]`` empty. Add the matching ``X.Y.Z - YYYY-MM-DD`` section to
   ``docs/changelog.rst``.
2. Run the preparation script from the repository root:

   .. code-block:: bash

      bash scripts/release.sh X.Y.Z

   This validates both changelogs and rewrites the version in ``pyproject.toml``,
   ``uv.lock`` (the root package only), ``CITATION.cff``, ``.zenodo.json``,
   ``simind_python_connector/__init__.py`` and ``docs/conf.py``. It fails without
   changing anything when a note or field is missing or malformed.
3. Review the changes and commit them, then tag the release commit and push the
   branch and the tag:

   .. code-block:: bash

      git add -A
      git commit -m "chore: release X.Y.Z"
      git tag -a vX.Y.Z -m "Release X.Y.Z"
      git push origin HEAD vX.Y.Z

4. Publish the GitHub release from that tag:

   .. code-block:: bash

      gh release create vX.Y.Z --verify-tag --title "X.Y.Z" --notes-file notes.md

   Publishing the GitHub release starts ``publish.yml``, which reruns the test
   suite, checks that the tag matches the package version and uploads to PyPI.
   Pushing the tag alone does **not** publish to PyPI.

You can verify the eight version surfaces without changing anything:

.. code-block:: bash

    python scripts/prepare_release.py X.Y.Z --date YYYY-MM-DD --check
