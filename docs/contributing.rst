.. _contributing:

Contributing
============

Contributions are welcome. Open a pull request once the tests and the `-W`
documentation build pass.

Development Setup
-----------------

.. code-block:: bash

    pip install -e ".[dev]"

Testing and Linting
-------------------

.. code-block:: bash

    python -m pytest tests/ -v
    ruff check .
    ruff format --check .

Documentation
-------------

.. code-block:: bash

    sphinx-build -b html -W --keep-going docs docs/_build/html

Releasing
---------

Release notes are written by hand; the release tooling validates them and
updates the machine version surfaces. Only after the validation run passes and
the release is approved, tag and publish.

1. Write the release notes. Move entries out of ``## [Unreleased]`` in
   ``CHANGELOG.md`` into a new ``## [X.Y.Z] - YYYY-MM-DD`` section, and add the
   matching section to ``docs/changelog.rst``.
2. Run the preparation script:

   .. code-block:: bash

      bash scripts/release.sh X.Y.Z

   It rewrites the version in ``pyproject.toml``, ``uv.lock``,
   ``CITATION.cff``, ``.zenodo.json``,
   ``simind_python_connector/__init__.py`` and ``docs/conf.py``, and fails
   without changing anything on a missing or malformed note or field.
3. Inspect the working tree and the index. ``git status`` lists the files the
   preparation script rewrote, and ``git diff --cached`` shows what is already
   staged. Implementation or test corrections are committed separately, in
   their own commit, before the metadata-preparation commit.

   .. code-block:: bash

      git status
      git diff --cached

4. Stage exactly the eight version surfaces, then read back what is staged:

   .. code-block:: bash

      git add pyproject.toml uv.lock CHANGELOG.md docs/changelog.rst \
          CITATION.cff .zenodo.json simind_python_connector/__init__.py docs/conf.py
      git diff --cached --stat
      git diff --cached

5. Commit the staged metadata:

   .. code-block:: bash

      git commit -m "chore: release X.Y.Z"

6. Tag and push:

   .. code-block:: bash

      git tag -a vX.Y.Z -m "Release X.Y.Z"
      git push origin HEAD vX.Y.Z

7. Publish the GitHub release from the pushed tag:

   .. code-block:: bash

      gh release create vX.Y.Z --verify-tag --title "X.Y.Z" --notes-file notes.md

   Publishing starts ``publish.yml``, which reruns the tests, builds the
   documentation at the release tag, checks the tag against the package version
   and uploads to PyPI. Pushing the tag alone does not publish to PyPI.

You can verify the version surfaces without changing anything:

.. code-block:: bash

    python scripts/prepare_release.py X.Y.Z --date YYYY-MM-DD --check
