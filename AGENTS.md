# Repository Instructions

## Architecture

- The distribution is `simind-python-connector`; the import package is `simind_python_connector`.
- `SimindPythonConnector` in `simind_python_connector/connectors/python_connector.py` is the current execution path. `StirSimindAdaptor`, `SirfSimindAdaptor`, and `PyTomographySimindAdaptor` build on it; `core/executor.py` invokes the external `simind` command.
- Keep SIRF, STIR, and PyTomography imports optional and lazy. The pure-Python connector and its tests must work without those libraries, and backend-specific examples/containers must import only their target backend.
- Use `from simind_python_connector.configs import get` for packaged YAML/SMC presets; do not assume a checkout-relative path for package data.
- Do not use `SimindSimulator` as a current API: it was removed in 1.0, and `scripts/simulation.py` still references that legacy surface.

## Runtime

- `simind/` is an ignored, repo-local SIMIND installation, not tracked source. SIMIND runs require an executable at `./simind/simind`, its data under `./simind/smc_dir/`, or a `simind` executable on `PATH` for direct local runs.
- Container scripts skip SIMIND-dependent checks/examples when that executable is absent; add `--require-simind` to fail instead. Use `--docker-platform linux/amd64` when the SIMIND binary or target image requires x86-64.

## Development Commands

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m pytest tests/ -v
python -m pytest tests/test_python_connector.py::test_name -q
python -m pytest -c pytest-ci.ini
python -m build
twine check dist/*
sphinx-build -b html docs docs/_build/html
```

- `bash scripts/fix.sh` applies Ruff lint fixes, formats the tree, then runs the lint check; it modifies files.
- CI runs Ruff checks before the CI-friendly pytest selection, then separately builds and runs `twine check`.
- The exact CI test command is `python -m pytest tests/ -v -m "not integration and not requires_sirf and not requires_stir and not requires_simind and not requires_pytomography and not requires_cil and not requires_setr and not ci_skip" --cov=simind_python_connector --cov-report=xml`.
- Backend/container checks use `bash scripts/run_container_validation.sh` and `bash scripts/run_container_examples.sh`; both support `--only-core`, `--only-stir`, `--only-sirf`, `--only-pytomography`, `--only-osem`, and `--no-build`.

## Tests

- `tests/conftest.py` auto-skips unavailable SIRF, STIR, CIL, SETR, PyTomography, and SIMIND dependencies. Use the existing markers: `unit`, `integration`, `requires_sirf`, `requires_stir`, `requires_simind`, `requires_pytomography`, `requires_cil`, `requires_setr`, and `ci_skip`.
- Every test module/function needs one of the supported category/dependency markers; `tests/test_marker_policy.py` enforces this.
- `tests/test_examples_backend_separation.py` enforces that examples 01-06 are backend-free and 07A/07B/07C are isolated to STIR/SIRF/PyTomography respectively.

## Dependency Changes

- `pyproject.toml` is the package/dependency source of truth and `uv.lock` is checked in; keep the lockfile synchronized when changing dependencies.
