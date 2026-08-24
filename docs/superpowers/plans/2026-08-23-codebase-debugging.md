# Codebase Debugging and Test Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce, fix, and permanently test the confirmed correctness, packaging, backend-isolation, and container-validation defects found in `simind-python-connector`.

**Architecture:** Keep `SimindPythonConnector` as the backend-independent execution path. Make configuration parsing, command execution, Interfile loading, and output discovery deterministic and independently testable. Keep STIR, SIRF, and PyTomography optional and lazy, and move container checks behind deterministic shell and package smoke tests.

**Tech Stack:** Python 3.9-3.12, NumPy, pytest, pydicom, PyYAML, SIRF/STIR/PyTomography optional backends, Docker Compose, Ruff, setuptools, `uv.lock`.

---

## Execution Rules

- Run every command from `/Users/samd-work/Projects/SIRF-SIMIND-Connection`.
- Use test-first development for every production-code defect: write one regression test, run it and confirm the expected failure, make the smallest fix, then rerun the test.
- Do not weaken an assertion merely to make an existing implementation pass.
- Do not change or delete the pre-existing untracked `AGENTS.md` file.
- Do not commit, amend, push, or create a pull request. The user requested a plan and has not requested commits.
- Preserve the public `SimindPythonConnector`, `StirSimindAdaptor`, `SirfSimindAdaptor`, and `PyTomographySimindAdaptor` APIs unless a task explicitly says otherwise.
- Keep SIMIND-dependent tests marked `requires_simind`; keep backend-specific tests marked with their existing backend markers.
- Treat the existing `coverage.xml` as historical evidence only. A fresh test run is required before claiming coverage or test success.

## Confirmed Findings

The following findings were established by source inspection, clean-git archive inspection, existing coverage data, shell validation, and container-runner diagnostics. The line references are the current baseline and may move during implementation.

| ID | Priority | Finding | Primary evidence |
|---|---|---|---|
| D-01 | P0 | `input.smc` is ignored by Git and absent from a clean `HEAD` archive, although package data, documentation, and example 04 require it. | `.gitignore:153-154`, `pyproject.toml:66-70`, `configs/__init__.py:21-30`, `examples/04_custom_config.py` |
| D-02 | P0 | `--no-build` skips only the explicit Compose build command; `docker compose run` can build a missing image anyway. | `scripts/run_container_validation.sh:169-182`, `scripts/run_container_examples.sh:165-178` |
| D-03 | P0 | The import guard raises plain `ImportError`, which breaks module collection at `pytest.importorskip("sirf.STIR")` in an isolated container. | `docker/import_guard/sitecustomize.py:25-48`, `tests/test_geometry_isolation_forward_projection.py:18` |
| D-04 | P1 | Docker uses the repository root as context without a `.dockerignore`, sending local SIMIND files, archives, caches, outputs, and virtual environments to the daemon. | `docker/compose.yaml:3-5`, all backend Dockerfiles use `COPY .` |
| D-05 | P1 | GitHub Actions does not install and import the built wheel, run backend containers, or run the container example suite. | `.github/workflows/tests.yml:21-57` |
| D-06 | P1 | Docker bases and dependency installs are mutable and do not use the checked-in lockfile. | Dockerfiles use `python:3.12-slim`, `synerbi/sirf:latest`, latest pip/packages; `uv.lock` is unused by CI/Docker |
| D-07 | P1 | `pytest-ci.ini` uses the wrong section name and omits the `slow` marker; the marker-policy AST walk ignores test methods inside classes. | `pytest-ci.ini:1`, `tests/test_marker_policy.py:52-67` |
| D-08 | P1 | Voxel map dimensions are assigned to the wrong SIMIND indices and indices `81` and `82` remain stale. | `connectors/python_connector.py:149-160`, `core/config.py:108-115` |
| D-09 | P1 | SMC parsing uses fixed line offsets, weak numeric matching, and no exact count validation; SMC writing emits inconsistent flag/data-file counts and omits parameter 11 from YAML export. | `core/config.py:223-263,281-309,778-825` |
| D-10 | P1 | Data-file access by descriptive name compares the description to stored filenames and silently ignores absent integer entries. | `core/config.py:722-757` |
| D-11 | P1 | Uppercase `.YAML` is accepted by the connector but parsed as SMC by `SimulationConfig`. | `connectors/python_connector.py:84-95`, `core/config.py:201-208` |
| D-12 | P1 | Output prefixes allow traversal/absolute paths; broad globs can ingest stale or similarly named outputs; `run()` mutates the process-global working directory. | `connectors/python_connector.py:65-66,219-229,259-281` |
| D-13 | P1 | Runtime switches supplied through `RuntimeOperator` are merged into persistent connector state and affect later runs. | `connectors/python_connector.py:214-217` |
| D-14 | P1 | Numeric validation accepts NaN/Inf/negative map values and produces an unhelpful `AttributeError` for invalid scoring values. | `connectors/python_connector.py:68-70,118-143` |
| D-15 | P1 | Energy-window validation uses `assert`, allows reversed bounds and invalid scatter orders, and raises `IndexError`/`ValueError` for empty input depending on the path. | `utils/simind_utils.py:44-70` |
| D-16 | P1 | Native adaptor output caches are not cleared before validation or execution, so a failed rerun can expose results from a previous successful run. | `connectors/stir_adaptor.py:81-101`, `sirf_adaptor.py:81-101`, `pytomography_adaptor.py:134-177` |
| D-17 | P1 | NumPy Interfile loading ignores data offsets and silently truncates extra payload bytes. | `utils/interfile_numpy.py:145-185`, existing truncation assertion in `tests/test_interfile_numpy.py:91-116` |
| D-18 | P1 | Acquisition builders accept payload/header shape mismatches and split non-divisible projection counts into headers with incorrect counts. | `builders/acquisition_builder.py:83-118,197-216`, `builders/image_builder.py:89-115` |
| D-19 | P1 | DICOM conversion uses `time_per_projection` when its tag is absent and assumes pixel data is always 3D. | `builders/acquisition_builder.py:320-337,519-526` |
| D-20 | P1 | The backend spacing contract is contradictory: the interface and STIR wrapper document `(z,y,x)`, while `_spacing.py`, its tests, and `docs/geometry.rst` use the last spatial element as z. | `backends/base.py:73-87`, `backends/stir_backend.py:92-104`, `connectors/_spacing.py:46-54`, `docs/geometry.rst:38-41` |
| D-21 | P1 | Converter ignore rules run before data-file rules and match arbitrary substrings, so a data file named `patient.a00` can be commented out. | `converters/simind_to_stir.py:38-50,256-297` |
| D-22 | P1 | `get_sirf_attenuation_from_simind()` selects `rho*1000` but fills only `rho*100`; custom attenuation directories are joined to an already absolute packaged path. | `utils/stir_utils.py:104-136`, `converters/attenuation.py:52-65` |
| D-23 | P2 | The connectors package eagerly imports every adaptor, and `get_stir_types()` checks `stir` without checking `stirextra`. | `connectors/__init__.py:3-12`, `utils/import_helpers.py:27-39` |
| D-24 | P2 | Backend factories accept `str` paths but reject `Path` objects. | `backends/__init__.py:250-256,278-279,331-334,362-363` |
| D-25 | P2 | The all-examples Python runner prints failures but exits successfully; the shell runner passes `--backend` to examples that do not accept it. | `scripts/run_all_examples.py:196-205`, `examples/run_all_examples.sh:143-149` |
| D-26 | P2 | `scripts/simulation.py` still imports and constructs removed `SimindSimulator`. | `scripts/simulation.py:23,207`, removal documented in `docs/changelog.rst:32-40` |
| D-27 | P2 | PyTomography output handling has separate `.h00` and `.hs` branches that have not been tested with asymmetric data or PENETRATE components. | `connectors/pytomography_adaptor.py:157-172`, `docs/geometry.rst:10-18` |
| D-28 | P1 | The acquisition builder writes the supplied array and then replaces the backend object with an unconditional final-axis flip, so the on-disk payload and returned object can disagree about orientation. | `builders/acquisition_builder.py:107-118` |

The following items are not confirmed defects yet. They must be tested before being promoted to the issue list: D-27, `configs.get()` behavior from a zipped wheel, lifetime of backend objects returned after builder temporary-directory cleanup, SIMIND MPI switch grammar, and native-map origin/spacing mismatch when source and attenuation grids have equal shapes.

## Files to Change

### Production and packaging files

- Modify: `.gitignore`
- Modify: `simind_python_connector/core/config.py`
- Modify: `simind_python_connector/core/executor.py`
- Modify: `simind_python_connector/connectors/python_connector.py`
- Modify: `simind_python_connector/connectors/__init__.py`
- Modify: `simind_python_connector/connectors/_spacing.py`
- Modify: `simind_python_connector/connectors/stir_adaptor.py`
- Modify: `simind_python_connector/connectors/sirf_adaptor.py`
- Modify: `simind_python_connector/connectors/pytomography_adaptor.py`
- Modify: `simind_python_connector/utils/simind_utils.py`
- Modify: `simind_python_connector/utils/interfile_numpy.py`
- Modify: `simind_python_connector/utils/import_helpers.py`
- Modify: `simind_python_connector/utils/stir_utils.py`
- Modify: `simind_python_connector/converters/attenuation.py`
- Modify: `simind_python_connector/converters/simind_to_stir.py`
- Modify: `simind_python_connector/builders/acquisition_builder.py`
- Modify: `simind_python_connector/builders/image_builder.py`
- Delete: `scripts/simulation.py` after the reference scan in Task 10 confirms it is not supported documentation or public API.

### Tests to add or modify

- Modify: `tests/test_simulation_config.py`
- Create: `tests/test_executor.py`
- Modify: `tests/test_python_connector.py`
- Modify: `tests/test_utils_small.py`
- Modify: `tests/test_interfile_numpy.py`
- Modify: `tests/test_simind_to_stir_converter.py`
- Create: `tests/test_attenuation_utils.py`
- Modify: `tests/test_acquisition_builder_unit.py`
- Modify: `tests/test_image_builder_unit.py`
- Create: `tests/test_dicom_builder.py`
- Modify: `tests/test_native_adaptors.py`
- Modify: `tests/test_pytomography_adaptor.py`
- Modify: `tests/test_marker_policy.py`
- Modify: `tests/test_schneider_density.py`
- Create: `tests/test_packaging_resources.py`
- Create: `tests/test_container_scripts.py`
- Create: `tests/test_legacy_surface_policy.py`

### Container and CI files

- Create: `.dockerignore`
- Modify: `docker/import_guard/sitecustomize.py`
- Modify: `docker/compose.yaml`
- Modify: `docker/python/Dockerfile`
- Modify: `docker/stir/Dockerfile`
- Modify: `docker/sirf/Dockerfile`
- Modify: `docker/pytomography/Dockerfile`
- Modify: `scripts/run_container_validation.sh`
- Modify: `scripts/run_container_examples.sh`
- Modify: `scripts/run_all_examples.py`
- Modify: `examples/run_all_examples.sh`
- Modify: `pytest-ci.ini`
- Modify: `.github/workflows/tests.yml`
- Modify: `docs/testing.rst`
- Modify: `docs/geometry.rst`

## Task 1: Add Configuration Regression Tests

**Files:**
- Modify: `tests/test_simulation_config.py`
- Create: `tests/fixtures/minimal_variable_sections.smc`

- [x] **Step 1: Add a valid SMC fixture with non-default section lengths.**

The fixture must contain these exact sections in order:

```text
SMCV2
variable-section-test
   120  # Basic Change data
```

Follow the count line with exactly 24 lines containing five scientific-notation numbers each, then:

```text
1.0E+00 2.0E+00 3.0E+00 4.0E+00 5.0E+00
6.0E+00 7.0E+00 8.0E+00 9.0E+00 10.0E+00
11.0e+00 12.0E+00 13.0E+00 14.0E+00 15.0E+00
16.0E+00 17.0E+00 18.0E+00 19.0E+00 20.0E+00
21.0E+00 22.0E+00 23.0E+00 24.0E+00 25.0E+00
26.0E+00 27.0E+00 28.0E+00 29.0E+00 30.0E+00
31.0E+00 32.0E+00 33.0E+00 34.0E+00 35.0E+00
36.0E+00 37.0E+00 38.0E+00 39.0E+00 40.0E+00
41.0E+00 42 43.0E+00 44.0E+00 45.0E+00
46.0E+00 47.0E+00 48.0E+00 49.0E+00 50.0E+00
51.0E+00 52.0E+00 53.0E+00 54.0E+00 55.0E+00
56.0E+00 57.0E+00 58.0E+00 59.0E+00 60.0E+00
61.0E+00 62.0E+00 63.0E+00 64.0E+00 65.0E+00
66.0E+00 67.0E+00 68.0E+00 69.0E+00 70.0E+00
71.0E+00 72.0E+00 73.0E+00 74.0E+00 75.0E+00
76.0E+00 77.0E+00 78.0E+00 79.0E+00 80.0E+00
81.0E+00 82.0E+00 83.0E+00 84.0E+00 85.0E+00
86.0E+00 87.0E+00 88.0E+00 89.0E+00 90.0E+00
91.0E+00 92.0E+00 93.0E+00 94.0E+00 95.0E+00
96.0E+00 97.0E+00 98.0E+00 99.0E+00 100.0E+00
101.0E+00 102.0E+00 103.0E+00 104.0E+00 105.0E+00
106.0E+00 107.0E+00 108.0E+00 109.0E+00 110.0E+00
111.0E+00 112.0E+00 113.0E+00 114.0E+00 115.0E+00
116.0E+00 117.0E+00 118.0E+00 119.0E+00 120.0E+00
```

```text
     30  # Simulation flags
TFTFFFFFFFFFFFFFFFFFFFFFFFFFFF
      2  # Text Variables
first text variable
second text variable
     3  # Data files
phantom.dat
source.dat
attenuation.dat
```

The 24 numeric lines must produce exactly 120 values, with index 11 equal to `11.0` and indices 78, 79, 81, and 82 equal to `78.0`, `79.0`, `81.0`, and `82.0` respectively. Use lowercase `e` on one value and a plain integer such as `42` on another value so the parser test covers both forms.

- [x] **Step 2: Write the failing parser test.**

Add this test to `tests/test_simulation_config.py`:

```python
FIXTURES = Path(__file__).parent / "fixtures"


def test_smc_parser_uses_section_cursor_and_preserves_counts():
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")

    assert len(config.data) == 120
    assert config.get_value(11) == pytest.approx(11.0)
    assert config.flags == "TFTFFFFFFFFFFFFFFFFFFFFFFFFFFF"
    assert config.text_variables == {
        1: "first text variable",
        2: "second text variable",
    }
    assert config.data_files == {
        1: "phantom.dat",
        2: "source.dat",
        3: "attenuation.dat",
    }
```

Run: `python -m pytest tests/test_simulation_config.py::test_smc_parser_uses_section_cursor_and_preserves_counts -q`

Expected: FAIL because the current parser reads data files from hard-coded line 38 and does not reliably parse all numeric formats.

- [x] **Step 3: Add failing tests for the public configuration behaviors.**

Add these test names and assertions:

```python
def test_yaml_smc_yaml_round_trip_preserves_parameter_flags_and_files(tmp_path):
    original = SimulationConfig(get("AnyScan.yaml"))
    original.set_value(11, 5.0)
    original.set_data_file("phantom_soft_tissue", "custom_phantom")
    original.set_data_file(12, "custom_last_file")
    smc_path = original.save_file(tmp_path / "round_trip.smc")
    restored = SimulationConfig(smc_path)

    assert restored.get_value(11) == pytest.approx(5.0)
    assert restored.get_data_file("phantom_soft_tissue") == "custom_phantom"
    assert restored.get_data_file(12) == "custom_last_file"
    assert len(restored.flags) == len(original.flags)


def test_data_file_description_accessors_create_and_return_entries(tmp_path):
    config = SimulationConfig(get("AnyScan.yaml"))
    config.set_data_file(12, "new_file")

    assert config.get_data_file(12) == "new_file"
    config.set_data_file("unknown_file_4", "replacement")
    assert config.get_data_file("unknown_file_4") == "replacement"


def test_uppercase_yaml_suffix_is_loaded_as_yaml(tmp_path):
    source = tmp_path / "config.YAML"
    source.write_text((get("AnyScan.yaml")).read_text())

    config = SimulationConfig(source)

    assert config.get_value("photon_energy") == pytest.approx(150.0)
```

Run: `python -m pytest tests/test_simulation_config.py -q`

Expected: the new tests fail before implementation; the current data-file description test either returns `None` or raises, and the uppercase extension test treats YAML as SMC.

- [x] **Step 4: Add malformed-file tests.**

Cover a missing basic value, a flag count that does not match the flag string, a text count larger than the available lines, a data-file count larger than the available lines, and a non-numeric basic value. Each must raise `ValueError` with the section name in the message. Do not accept a partially populated `SimulationConfig`.

- [x] **Step 5: Run the new tests and record the red failures.**

Run: `python -m pytest tests/test_simulation_config.py -q`

Expected: all newly added tests fail for the currently identified reasons while the existing tests continue to collect.

## Task 2: Fix Configuration Parsing, YAML Export, and Data-File Access

**Files:**
- Modify: `simind_python_connector/core/config.py`
- Modify: `.gitignore`
- Add to source control: `simind_python_connector/configs/input.smc`

- [x] **Step 1: Make file-type detection case-insensitive.**

Convert the suffix once with `Path(filepath).suffix.lower()` and dispatch `.yaml` and `.yml` to YAML parsing. Dispatch `.smc` to SMC parsing. Raise `ValueError` for every other suffix instead of silently treating it as SMC.

- [x] **Step 2: Replace fixed SMC line offsets with a cursor parser.**

Implement the parser in this order:

1. Require the `SMCV2` first line and a comment second line.
2. Read the basic-data count from the next section header.
3. Consume lines until exactly that many numeric values have been parsed. Use this regex:

```python
number_pattern = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")
```

4. Raise `ValueError("basic data ...")` if the section ends before the declared count or contains an invalid token.
5. Read the flag count and one flag string; require the string length to equal the declared count.
6. Read the text-variable count and exactly that many full lines, preserving spaces inside each value and trimming only the line terminator.
7. Read the data-file count and exactly that many lines, trimming the fixed-width padding.
8. Permit only blank trailing lines. Raise `ValueError` for unexpected nonblank content.

Store the parsed basic-data count, flag count, text-variable count, and data-file count on the object so `save_file()` can preserve them.

- [x] **Step 3: Preserve all YAML/SMC values during export.**

Make these changes:

- Initialize YAML flags to 30 `F` characters, because the packaged SMC reference declares 30 flags.
- Initialize YAML data files for indices 1 through 12 to `"none"`, then overwrite entries supplied by YAML. This ensures connector source/density assignments do not silently disappear from a minimal YAML file.
- Add parameter index 11 to the `detector_crystal` parameter group.
- Preserve unmapped flag positions through a `raw_flags` YAML field. Named flags 1 through 15 still use the existing structured entries; importing YAML must load `raw_flags` first and then apply named overrides.
- Export every known data-file slot that exists in `self.data_files`, including slots whose value is `"none"`.
- Write the actual flag count and actual flag string, not a hard-coded `30` followed by a shorter string.
- Write exactly the declared number of text-variable and data-file lines.
- Always write 120 basic values, padding missing internal values with `0.0`, and always parse exactly 120 values from a standard SMC file.

The resulting writer must satisfy this invariant:

```python
loaded = SimulationConfig(config.save_file(path))
assert loaded.flags == config.flags
assert loaded.text_variables == config.text_variables
assert loaded.data_files == config.data_files
assert loaded.data[:120] == pytest.approx(config.data[:120])
```

- [x] **Step 4: Correct descriptive data-file access.**

Resolve a string description by searching `self.data_file_dict`, not `self.data_files` values. For an integer index in `data_file_dict`, `set_data_file()` must assign the entry even if it did not previously exist. `get_data_file()` must return the path string for both integer and descriptive access. Raise `ValueError` for an invalid index or description and `KeyError` only when a valid slot is deliberately absent.

- [x] **Step 5: Track the packaged SMC preset.**

Add this exception after the existing `*.smc` ignore rule in `.gitignore`:

```gitignore
!simind_python_connector/configs/input.smc
```

Do not change the contents of `input.smc` unless the round-trip tests prove that the fixture itself is malformed.

- [x] **Step 6: Run the configuration tests green.**

Run: `python -m pytest tests/test_simulation_config.py -q`

Expected: all configuration tests pass, including malformed-input tests and the existing YAML tests.

## Task 3: Add Executor and Connector Regression Tests

**Files:**
- Create: `tests/test_executor.py`
- Modify: `tests/test_python_connector.py`

- [x] **Step 1: Add executor tests before changing production code.**

Create `tests/test_executor.py` with these behaviors:

```python
def test_executor_uses_explicit_executable_cwd_and_non_shell(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "simind_python_connector.core.executor.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    SimindExecutor(executable=tmp_path / "custom-simind").run_simulation(
        "case01", cwd=tmp_path
    )

    assert calls[0][0][0] == str(tmp_path / "custom-simind")
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["check"] is True
```

Add separate tests for MPI command construction, whitespace/NUL token rejection, `OSError` translation to `SimulationError`, and `CalledProcessError` translation.

Run: `python -m pytest tests/test_executor.py -q`

Expected: the test fails at the constructor call because `SimindExecutor` currently has no `executable` support. This is the required red state.

- [x] **Step 2: Add connector geometry and validation regression tests.**

Extend `tests/test_python_connector.py` with these exact behaviors:

- A source/mu array of shape `(2, 3, 4)` sets `76=4`, `77=3`, `78=4`, `79=4`, `81=3`, and `82=3`.
- `output_prefix="../escape"`, `output_prefix="/absolute"`, `output_prefix=""`, and a prefix containing a path separator raise `ValueError` before any output file is written.
- `quantization_scale` and `voxel_size_mm` reject `math.nan`, positive infinity, negative infinity, zero, and negative values.
- Source and attenuation arrays reject empty arrays, NaN, infinity, negative source values, and negative attenuation values.
- A float scoring value such as `1.0` and an invalid integer such as `999` raise `ValueError` with `scoring_routine` in the message.
- A failed second run clears the cached outputs and makes `get_outputs()` raise `RuntimeError`.
- Runtime switches supplied in one `RuntimeOperator` do not appear in a later run that has no operator.
- Two fake connectors can run with different output directories without changing `Path.cwd()`.

Update the existing fake `run_simulation()` callbacks in this file to accept a `cwd=None` keyword before running the new connector tests.

- [x] **Step 3: Run the connector tests and confirm red failures.**

Run: `python -m pytest tests/test_python_connector.py tests/test_executor.py -q`

Expected: new tests fail against the current index assignments, validation, CWD mutation, and executor signature.

## Task 4: Fix Command Execution, Connector State, Output Isolation, and Validation

**Files:**
- Modify: `simind_python_connector/core/executor.py`
- Modify: `simind_python_connector/connectors/python_connector.py`
- Modify: `simind_python_connector/converters/simind_to_stir.py`

- [x] **Step 1: Make the executor configurable and CWD-safe.**

Change the executor constructor to accept `executable: str | Path | None = None`. Resolve the executable in this order: explicit constructor argument, `SIMIND_BIN` environment variable, then the literal `"simind"` for existing PATH-based behavior. Add `cwd: str | Path | None = None` to `run_simulation()` and pass it directly to `subprocess.run()`.

Keep `shell=False` and `check=True`. Keep command-token validation. Do not use `os.chdir()` anywhere in the connector execution path.

- [x] **Step 2: Validate connector constructor and voxel inputs.**

Add a private prefix validator that accepts a nonempty basename containing no `/`, `\\`, `.` path component, or `..` component. Reject absolute paths and traversal before creating files. Validate all scalar numeric inputs with `math.isfinite()`.

In `configure_voxel_phantom()`:

```python
if source_array.size == 0 or mu_map_array.size == 0:
    raise ValueError("source and mu_map must not be empty")
if not np.isfinite(source_array).all() or not np.isfinite(mu_map_array).all():
    raise ValueError("source and mu_map must contain only finite values")
if (source_array < 0).any() or (mu_map_array < 0).any():
    raise ValueError("source and mu_map must be non-negative")
```

Convert invalid scoring values with `ScoringRoutine(value)` inside a `try` block and raise `ValueError("scoring_routine ...")` from the enum error.

- [x] **Step 3: Set all six voxel-map dimensions consistently.**

For an array shape `(dim_z, dim_y, dim_x)`, use this mapping:

```python
cfg.set_value(76, dim_x)  # image i
cfg.set_value(77, dim_y)  # image j
cfg.set_value(78, dim_x)  # density map i
cfg.set_value(79, dim_x)  # source map i
cfg.set_value(81, dim_y)  # density map j
cfg.set_value(82, dim_y)  # source map j
```

Keep `number_density_images` equal to `dim_z`. The rectangular `(2, 3, 4)` regression test must pass exactly.

- [x] **Step 4: Keep runtime-operator switches one-shot.**

Do not call `set_runtime_switches()` on the persistent `RuntimeSwitches` object from `run()`. Build a local dictionary:

```python
run_switches = dict(self.runtime_switches.switches)
if runtime_operator is not None:
    run_switches.update(runtime_operator.switches)
```

Pass `run_switches` to the executor without mutating `self.runtime_switches.switches`.

- [x] **Step 5: Remove global CWD mutation.**

Replace the `os.chdir()` block in `SimindPythonConnector.run()` with:

```python
self.executor.run_simulation(
    self.output_prefix,
    orbit_file,
    run_switches,
    cwd=self.output_dir,
)
```

Remove the now-unused CWD restoration code and import.

- [x] **Step 6: Remove stale same-prefix outputs before each run and use exact globs.**

Add a helper that removes only files whose stem is exactly the prefix or starts with `f"{prefix}_"`. Cover `.h00`, `.hs`, `.a00`, `.bNN`, `.s`, `.smi`, `.dmi`, and `.win`. Call it after input preparation and immediately before executor invocation.

Change normal output discovery from `*{prefix}*.h00` and `*{prefix}*.hs` to `f"{prefix}_*.h00"` and `f"{prefix}_*.hs"`, also allowing the exact `f"{prefix}.h00"` PENETRATE template. Reject ambiguous PENETRATE templates instead of returning the first result.

- [x] **Step 7: Run the focused suite green.**

Run: `python -m pytest tests/test_executor.py tests/test_python_connector.py -q`

Expected: all focused executor and connector tests pass, with no change to the process working directory after a run.

## Task 5: Fix Energy-Window Validation

**Files:**
- Modify: `simind_python_connector/utils/simind_utils.py`
- Modify: `tests/test_utils_small.py`
- Modify: `tests/test_python_connector.py`

- [x] **Step 1: Add failing validation tests.**

Test `create_window_file()` and `SimindPythonConnector.set_energy_windows()` for:

- empty lists;
- mismatched list lengths;
- a lower bound greater than or equal to its upper bound;
- NaN and infinity bounds;
- negative scatter orders;
- non-integral scatter orders such as `1.5`.

Each invalid call must raise `ValueError`. A valid scalar call and a valid multiple-window call must preserve the current output format.

- [x] **Step 2: Implement one validation path.**

Convert scalar `Number` arguments to one-element lists, convert sequences to lists, reject empty lists, require equal lengths, require finite bounds, require `lower < upper`, and require each scatter order to be an integer greater than or equal to zero. Replace the current `assert` and `max(scatter_orders)` logic with explicit validation. Use `all(order < 1 for order in scatter_orders)` when deciding whether to append the extra scatter-order line.

- [x] **Step 3: Run the utility and connector tests.**

Run: `python -m pytest tests/test_utils_small.py tests/test_python_connector.py -q`

Expected: all valid-window tests pass and every invalid input produces the documented `ValueError`.

## Task 6: Harden Interfile Loading and SIMIND-to-STIR Conversion

**Files:**
- Modify: `simind_python_connector/utils/interfile_numpy.py`
- Modify: `simind_python_connector/converters/simind_to_stir.py`
- Modify: `tests/test_interfile_numpy.py`
- Modify: `tests/test_simind_to_stir_converter.py`

- [x] **Step 1: Add Interfile offset and strict-size tests.**

Add tests that write a four-byte prefix before a known float payload and set `!data offset in bytes := 4`; the loader must return the payload without the prefix. Add tests for `data offset in bytes` and `data_offset_in_bytes` spelling. Add tests that reject both truncated and trailing payloads with a message containing `Data size mismatch`.

Replace the existing test that expects silent truncation at `tests/test_interfile_numpy.py:91-116` with the rejection assertion.

- [x] **Step 2: Implement offset and exact-payload handling.**

Look up these case-insensitive keys:

```python
_lookup_header_value(
    "!data offset in bytes",
    "data offset in bytes",
    "data_offset_in_bytes",
)
```

Parse a non-negative integer offset, require the offset to be aligned to the dtype item size, read with `np.fromfile(data_path, dtype=dtype, offset=offset)`, and require `flat.size == expected_elements`. Strip matching single or double quotes from the data filename before resolving it.

- [x] **Step 3: Add the converter regression test.**

Create an `.h00` containing `!name of data file := patient.a00`, convert it, and assert that the output contains an active `!name of data file := patient.a00` line rather than a semicolon-prefixed comment.

- [x] **Step 4: Fix converter rule ordering.**

Ensure `DataFileNameRule` runs before `IgnorePatternRule`, or make `IgnorePatternRule.matches()` ignore structured data-file keys. Preserve comments for actual patient/institution metadata values. Add a test with a path containing `patient` and another containing uppercase `ID`.

- [x] **Step 5: Make radius scaling obey its configured factor.**

`ConversionConfig.radius_scale_factor` is documented as cm-to-mm and defaults to `10.0`. Apply `self.scale_factor` in `RadiusConversionRule.convert()`. Update the existing rule test to assert the scaled result and add a factor `1.0` test. Keep orbit-file conversion consistent with the same unit convention.

- [x] **Step 6: Run the focused I/O/converter suite.**

Run: `python -m pytest tests/test_interfile_numpy.py tests/test_simind_to_stir_converter.py -q`

Expected: all tests pass, including offset, strict-size, filename, and scaling regressions.

## Task 7: Fix Attenuation Utilities

**Files:**
- Modify: `simind_python_connector/utils/stir_utils.py`
- Modify: `simind_python_connector/converters/attenuation.py`
- Create: `tests/test_attenuation_utils.py`

- [x] **Step 1: Add failing attenuation tests.**

In `tests/test_attenuation_utils.py`, monkeypatch `ImageData` with a fake object exposing `initialise()` and `fill()`. Create temporary `.ict` and `.hct` files. Assert that `attn_type="rho*1000"` fills the image and that an unsupported selector raises `ValueError`.

Add tests for `get_attenuation_coefficient("water", energy, file_path=tmp_path)` and `get_attenuation_coefficient("bone", energy, file_path=tmp_path)` using temporary table files. The function must read the override directory, not the packaged table.

- [x] **Step 2: Normalize attenuation selectors and fill behavior.**

Accept exactly `"mu"` and `"rho*1000"`. Select `np.float32` for `mu` and `np.uint16` for `rho*1000`. Fill the image in both branches. Raise `ValueError` before reading files for every other selector.

- [x] **Step 3: Resolve custom attenuation paths correctly.**

When `file_path` is supplied as a directory, join it with `Path(filename).name`. When it is supplied as a file path, use that file directly for the selected material. Keep the packaged-resource path for the default case.

- [x] **Step 4: Run attenuation tests.**

Run: `python -m pytest tests/test_attenuation_utils.py tests/test_schneider_density.py -q`

Expected: all attenuation and Schneider tests pass without requiring SIRF.

## Task 8: Validate Builders and DICOM Conversion

**Files:**
- Modify: `simind_python_connector/builders/image_builder.py`
- Modify: `simind_python_connector/builders/acquisition_builder.py`
- Modify: `tests/test_image_builder_unit.py`
- Modify: `tests/test_acquisition_builder_unit.py`
- Create: `tests/test_dicom_builder.py`

- [x] **Step 1: Add failing shape and multi-energy tests.**

Add these cases to the builder tests:

- Image data shape `(2, 3, 5)` with header dimensions `(2, 3, 4)` raises `ValueError`.
- Acquisition data shape different from `(1, matrix_size_1, projections, matrix_size_2)` raises `ValueError`.
- Five projections split over two energy windows raises `ValueError` before writing files.
- `build_multi_energy()` with `pixel_array is None` raises `ValueError`.
- A valid four-projection/two-window input writes two outputs with two projections each.
- An asymmetric one-hot array is written and read back with the documented axis order; the test must assert the exact one-hot coordinate, not only the shape.

- [x] **Step 2: Validate builder shapes before writing.**

In `STIRSPECTImageDataBuilder.build()`, compare the resolved data shape to `(dim_z, dim_y, dim_x)` from the header. In `STIRSPECTAcquisitionDataBuilder.build()`, compare the resolved data shape to `(1, matrix_size_1, num_projections, matrix_size_2)`. Raise `ValueError` containing both expected and received shapes.

Remove the unconditional `np.flip(..., axis=-1)` from acquisition building. The canonical input and output array order is `(tof, bin, view, axial)` as documented in `docs/geometry.rst`; write and fill the same array without an undocumented transformation. The asymmetric one-hot test must pass for both the raw file and the returned backend object.

- [x] **Step 3: Make multi-energy splitting deterministic.**

Require a non-`None` four-dimensional pixel array, require `num_projections > 0`, require `num_projections % len(self.energy_windows) == 0`, and split only after those checks. Restore the original pixel array after the loop so a builder can be reused. Do not emit any file before validation succeeds.

- [x] **Step 4: Add DICOM fixtures in memory.**

Create minimal `pydicom.dataset.FileDataset` objects in `tests/test_dicom_builder.py` with `Rows`, `Columns`, `NumberOfFrames`, `PixelSpacing`, energy-window sequence, and rotation sequence. Cover:

- a single-frame 2D pixel array;
- a multi-frame 3D pixel array;
- a rotation sequence without `(0018,1242)` time-per-projection;
- missing energy-window metadata;
- malformed pixel dimensionality.

Use `pytest.warns` only for metadata that is intentionally optional. Missing timing must not prevent start angle, rotation direction, extent, or radius from being populated.

- [x] **Step 5: Make DICOM handling explicit.**

Initialize `num_frames` and `time_per_projection` to `None` before reading the rotation sequence. Process timing only when present. Process angle, direction, extent, and radius independently. Normalize pixel data to an explicit internal shape for 2D and 3D inputs; raise `ValueError("pixel data must be 2D or 3D")` for other ranks instead of allowing a raw transpose error.

- [x] **Step 6: Run builder tests.**

Run: `python -m pytest tests/test_image_builder_unit.py tests/test_acquisition_builder_unit.py tests/test_dicom_builder.py -q`

Expected: all shape, split, persistence, and DICOM tests pass.

## Task 9: Resolve Backend Geometry, Lazy Imports, and Adaptor State

**Files:**
- Modify: `simind_python_connector/connectors/_spacing.py`
- Modify: `simind_python_connector/backends/base.py`
- Modify: `simind_python_connector/backends/stir_backend.py`
- Modify: `simind_python_connector/backends/__init__.py`
- Modify: `simind_python_connector/utils/import_helpers.py`
- Modify: `simind_python_connector/connectors/__init__.py`
- Modify: `simind_python_connector/connectors/stir_adaptor.py`
- Modify: `simind_python_connector/connectors/sirf_adaptor.py`
- Modify: `simind_python_connector/connectors/pytomography_adaptor.py`
- Modify: `tests/test_native_adaptors.py`
- Modify: `tests/test_pytomography_adaptor.py`
- Modify: `tests/test_connector_backend_separation.py`
- Modify: `docs/geometry.rst`

- [x] **Step 1: Add an anisotropic spacing regression test.**

Use a fake image with documented spacing `(2.0, 3.0, 5.0)` and assert the value passed to `configure_voxel_phantom()` is the z spacing defined by the package contract. Add the same assertion for a four-element grid spacing object. The test must name the chosen convention in its docstring.

- [x] **Step 2: Establish one spacing contract.**

Use `(z, y, x)` consistently across `ImageDataInterface`, `StirImageData.voxel_sizes()`, SIRF adaptor extraction, documentation, and tests. If the real SIRF API returns a different order, convert it exactly once in the backend wrapper and keep the public interface `(z, y, x)`. For a three-item public tuple, `_spacing.py` must select index `0`; for a four-item raw sequence in `(unused, z, y, x)` order, it must select index `1`; for a STIR coordinate object, use the documented z coordinate accessor and cover it with a test. Update `docs/geometry.rst` to remove the contradictory `voxel_sizes()[2]` statement.

- [x] **Step 3: Add failed-rerun cache tests.**

For STIR, SIRF, and PyTomography adaptors, set `_outputs` to a known successful result, make the underlying connector raise on the next run, and assert `get_outputs()` raises the adaptor's existing `RuntimeError` message after the failure. For PyTomography also assert `_output_metadata` and `_output_header_paths` are cleared.

- [x] **Step 4: Clear adaptor caches at run start.**

At the first executable line of each adaptor `run()` method, set all adaptor output caches to `None`. Keep input validation after cache clearing so validation failures cannot expose stale results.

- [x] **Step 5: Make optional adaptor imports lazy.**

Replace eager adaptor imports in `connectors/__init__.py` with a module-level `__getattr__` mapping. The package may eagerly import `BaseConnector` and the pure-Python connector, but importing `simind_python_connector.connectors` must not import `stir`, `sirf`, or `pytomography`.

Add an import-isolation test that installs a temporary import blocker and imports the pure Python connector successfully while each optional backend is unavailable.

- [x] **Step 6: Require both STIR modules in the helper.**

Update `get_stir_types()` to import both `stir` and `stirextra` before returning `STIR_AVAILABLE=True`. Add a test that blocks `stirextra` and asserts the helper returns `(type(None), type(None), False)`.

- [x] **Step 7: Accept path-like backend inputs.**

Normalize `str` and `os.PathLike` inputs with `os.fspath()` in `create_image_data()` and `create_acquisition_data()`. Add `Path` tests for both factories using monkeypatched backend wrappers.

- [x] **Step 8: Test both PyTomography output branches.**

Use asymmetric projection data with different values on every axis. Test the `.h00`/`pytomo_simind.get_projections()` branch and the `.hs` fallback branch. Both must expose the documented `(theta, r, z)` order, preserve values, and report the header path actually used. Test at least one PENETRATE component where only the generated `.hs` header exists.

- [x] **Step 9: Run backend-independent adaptor tests.**

Run: `python -m pytest tests/test_native_adaptors.py tests/test_pytomography_adaptor.py tests/test_connector_backend_separation.py -q`

Expected: tests requiring unavailable packages skip cleanly; pure import and mocked adaptor tests pass.

## Task 10: Fix Test Configuration and Legacy/Example Runners

**Files:**
- Modify: `pytest-ci.ini`
- Modify: `tests/test_marker_policy.py`
- Modify: `tests/test_schneider_density.py`
- Create: `tests/test_container_scripts.py`
- Create: `tests/test_legacy_surface_policy.py`
- Modify: `scripts/run_all_examples.py`
- Modify: `examples/run_all_examples.sh`
- Delete: `scripts/simulation.py`

- [x] **Step 1: Add tests for CI configuration and marker coverage.**

Add a marker-policy fixture containing a class with a test method and no marker. Assert the policy reports the class method. Add a test that parses `pytest-ci.ini` and asserts it has a `[pytest]` section, registers `slow`, and contains the CI marker expression.

- [x] **Step 2: Fix pytest CI configuration.**

Change the first line of `pytest-ci.ini` from `[tool:pytest]` to `[pytest]`. Register `slow`, `requires_stir`, `requires_pytomography`, `requires_cil`, and `requires_setr` in the file so the config works without relying on `conftest.py` side effects. Keep the existing CI exclusion expression.

- [x] **Step 3: Make marker policy traverse class methods.**

Walk both module-level test functions and `ast.ClassDef` bodies. A class decorator or module `pytestmark` satisfies all methods in that class. Report missing markers using `path.py:ClassName.test_name`.

- [x] **Step 4: Mark Schneider tests explicitly.**

Add `pytestmark = pytest.mark.unit` near the imports in `tests/test_schneider_density.py`. Keep the integration test's explicit `pytest.mark.integration` marker so it remains selectable by integration runs.

- [x] **Step 5: Add failure-propagation tests for the Python runner.**

Refactor `scripts/run_all_examples.py` so `main()` returns `0` for all-success and `1` when any example result has `success=False`; call `raise SystemExit(main())` in the module entry point. Add a test that monkeypatches `EXAMPLES` and `run_example_subprocess()` to return one failure, then asserts the process exits with status 1.

- [x] **Step 6: Fix the shell runner's backend argument handling.**

Only append `--backend` to examples `04_custom_config.py` and `06_schneider_density_conversion.py`, which define that option. Do not pass it to examples 01, 02, 03, or 05. Add a shell test with a fake `python3` executable that records arguments and assert no unsupported argument is sent.

- [x] **Step 7: Retire the removed API script.**

Run `git grep -n "SimindSimulator" -- ':!docs/changelog.rst' ':!AGENTS.md'`. After confirming the only executable reference is `scripts/simulation.py`, delete that unsupported script. Add `tests/test_legacy_surface_policy.py` that scans `scripts`, `examples`, and `simind_python_connector` for `SimindSimulator` and fails if the removed symbol is referenced.

- [x] **Step 8: Run script and marker tests.**

Run: `python -m pytest tests/test_marker_policy.py tests/test_schneider_density.py tests/test_container_scripts.py tests/test_legacy_surface_policy.py -q`

Expected: class tests are selected by `-m unit`, runner failures return nonzero, and no current executable code references `SimindSimulator`.

## Task 11: Harden Docker Containers, Compose, and CI

**Files:**
- Create: `.dockerignore`
- Modify: `docker/import_guard/sitecustomize.py`
- Modify: `docker/compose.yaml`
- Modify: `docker/python/Dockerfile`
- Modify: `docker/stir/Dockerfile`
- Modify: `docker/sirf/Dockerfile`
- Modify: `docker/pytomography/Dockerfile`
- Modify: `scripts/run_container_validation.sh`
- Modify: `scripts/run_container_examples.sh`
- Modify: `.github/workflows/tests.yml`
- Modify: `tests/test_container_scripts.py`

- [x] **Step 1: Add Docker context policy tests.**

Assert `.dockerignore` excludes `.git`, `.venv`, `simind`, `dist`, `build`, `*.tar.gz`, `.pytest_cache`, `.ruff_cache`, `.uv-cache`, `coverage.xml`, `docs/_build`, `output`, and `results`. Do not exclude `simind_python_connector/configs/input.smc`.

- [x] **Step 2: Create the Docker ignore file.**

Use this minimum content:

```text
.git
.venv
simind
dist
build
*.tar.gz
*.egg-info
.pytest_cache
.ruff_cache
.uv-cache
.uv-tools
.coverage
.coverage.*
coverage.xml
docs/_build
output
results
```

- [x] **Step 3: Make blocked imports look like missing modules.**

In `docker/import_guard/sitecustomize.py`, raise `ModuleNotFoundError(message, name=root)` from both the meta-path finder and guarded `__import__` function. Keep the existing text in the message. Add an isolated-container test that calls `pytest.importorskip("sirf.STIR")` under the guard and verifies it skips rather than aborts collection.

- [x] **Step 4: Remove the hardcoded PyTomography platform.**

Remove `platform: linux/amd64` from `docker/compose.yaml`. Let the runners set `DOCKER_DEFAULT_PLATFORM` from `--docker-platform` or SIMIND binary detection. Add a Compose config test that confirms `linux/arm64` is not overridden by a service-level amd64 value.

- [x] **Step 5: Make container `--no-build` deterministic.**

Define an image-name map in each runner. When `DO_BUILD=0`, run `docker image inspect image-name` for every selected service and exit 1 with the missing image name if any image is absent. Add `--no-build` to every `docker compose run` invocation when `DO_BUILD=0`. When `DO_BUILD=1`, retain the explicit build step.

- [x] **Step 6: Pass the selected SIMIND executable and data directory into containers.**

Set `SIMIND_BIN` and `SIMIND_SMC_DIR` explicitly for each `docker compose run` call using the repo-relative container path. Keep the existing rejection for an executable outside the repository mount. Make `docker/compose.yaml` use these environment values as defaults rather than fixed paths.

- [x] **Step 7: Add fake-Docker runner tests.**

Create fake `docker` and `docker compose` executables in a temporary directory. Test:

- `--no-build` never records a `compose build` call;
- a missing selected image exits 1;
- `--require-simind` exits 1 when the executable is absent;
- `--docker-platform linux/arm64` reaches Compose through `DOCKER_DEFAULT_PLATFORM`;
- `--only-osem` with no SIMIND prints an explicit skip message and does not silently claim that examples ran.

- [x] **Step 8: Pin container dependencies after recording tested versions.**

Build each image once in the execution environment, record the resolved base image digests and package versions, then replace mutable image tags and unbounded package installs with those digests/versions or a checked-in constraints file. The Python container must use the lock-compatible pytest and NumPy versions. The STIR/SIRF containers must use a NumPy version compatible with their installed bindings. The PyTomography container must pin Torch and PyTomography together.

- [x] **Step 9: Add package and core-container jobs to GitHub Actions.**

Update `.github/workflows/tests.yml` as follows:

- Run the CI test command with `-c pytest-ci.ini`.
- In the build job, install the generated wheel into a clean virtual environment and run:

```bash
python -c 'from simind_python_connector.configs import get; assert get("input.smc").is_file()'
python -c 'from simind_python_connector.data import data_path; assert data_path("h2o.atn").is_file()'
```

- Add a core Docker job that runs `bash scripts/run_container_validation.sh --only-core`.
- Keep backend Docker jobs separate from the basic matrix if their base images are unavailable on every runner; run them on a scheduled or manually dispatched workflow with the same validation script.

- [x] **Step 10: Run container configuration checks.**

Run: `bash -n scripts/run_container_validation.sh`

Run: `bash -n scripts/run_container_examples.sh`

Run: `docker compose -f docker/compose.yaml config --quiet`

Expected: all commands exit 0. Do not run `--no-build` until the image-inspection behavior is implemented and the image availability is known.

## Task 12: Add Packaging and Optional-Dependency Tests

**Files:**
- Create: `tests/test_packaging_resources.py`
- Modify: `simind_python_connector/configs/__init__.py` only if the resource test exposes a Traversable/path bug
- Modify: `tests/test_connector_backend_separation.py`

- [x] **Step 1: Add resource inventory tests.**

Assert `configs.list()` includes `AnyScan.yaml`, `Example.yaml`, and `input.smc`. Assert `configs.get()` returns an existing resource for each YAML, SMC, ATN, and JSON file used by public examples.

- [x] **Step 2: Add a clean distribution smoke test.**

The test may be marked `ci_skip` if it invokes a nested build, but the CI build job must run the equivalent commands. Build into a temporary directory, install the wheel into a fresh virtual environment, and run the two resource assertions from Task 11. The test must build from tracked files, not the current ignored working tree.

- [x] **Step 3: Add blocked-backend import tests.**

Run a subprocess with `PYTHONPATH=docker/import_guard` and `SIMIND_PYTHON_CONNECTOR_BLOCK_IMPORTS=sirf,stir,stirextra,pytomography`. Import `simind_python_connector.connectors.python_connector` and `SimulationConfig`; both must succeed. Importing a backend adaptor must either be lazy until attribute access or raise the adaptor's documented `ImportError` at construction, not at pure-core import time.

- [x] **Step 4: Run packaging/import tests.**

Run: `python -m pytest tests/test_packaging_resources.py tests/test_connector_backend_separation.py -q`

Expected: package resources and pure-core imports work without optional libraries.

## Task 13: Investigate Deferred Backend and SIMIND Behaviors

**Files:**
- Modify: `tests/test_packaging_resources.py` for zipped-resource behavior.
- Modify: `tests/test_native_adaptors.py` for backend object lifetime and source/mu geometry.
- Modify: `tests/test_pytomography_adaptor.py` for backend object lifetime if PyTomography is affected.
- Modify: `tests/test_executor.py` for MPI command behavior.
- Modify production files only when one of these tests reproduces the behavior.

- [x] **Step 1: Verify zipped-resource loading.**

Build a wheel, place it in a temporary isolated environment, and import the package from the installed wheel. If `SimulationConfig(get("AnyScan.yaml"))` fails because `get()` returns a non-filesystem `Traversable`, use `importlib.resources.as_file()` around resource reads and update the return annotation/documentation. Add a regression test that exercises the installed wheel.

- [ ] **Step 2: Verify builder object lifetime.**

In a STIR and SIRF container, call `STIRSPECTImageDataBuilder.build()` and `STIRSPECTAcquisitionDataBuilder.build()` without `output_path`, then read dimensions and arrays after the temporary-directory context has ended. If either backend lazily needs deleted files, retain files in a caller-owned output directory or load into a backend-owned in-memory object before cleanup, then add a backend-marked regression test.

- [ ] **Step 3: Verify source/mu geometry metadata.**

Use equal-shaped source and attenuation images with different spacing and origins in a backend container. Confirm whether the API permits resampling or requires exact geometry equality. If exact equality is required, reject mismatched spacing/origin with `ValueError` and add tests. Do not silently use source geometry for both maps.

- [x] **Step 4: Verify SIMIND MPI grammar.**

With a disposable fake `simind`/`mpirun` executable, assert the generated command for `MP=2`, an orbit file, and ordinary switches. Then run one real SIMIND smoke case if the repo-local executable is available. Update `tests/test_executor.py` and the command builder only if the fake or real invocation demonstrates a mismatch.

## Task 14: Full Verification and Report Update

**Files:**
- Modify: `docs/superpowers/plans/2026-08-23-codebase-debugging.md` only to update implementation status and evidence after execution.

- [x] **Step 1: Run formatting and lint checks.**

Run: `ruff check .`

Run: `ruff format --check .`

Expected: both exit 0.

- [x] **Step 2: Run the complete dependency-light test suite.**

Run: `python -m pytest tests/ -v -m "not integration and not requires_sirf and not requires_stir and not requires_simind and not requires_pytomography and not requires_cil and not requires_setr and not ci_skip" --cov=simind_python_connector --cov-report=xml`

Expected: zero failures and zero collection errors. Record the final number of passed, skipped, and covered lines in the report.

- [x] **Step 3: Run the dedicated CI configuration.**

Run: `python -m pytest -c pytest-ci.ini -q`

Expected: the CI exclusions are applied, `slow` is known, class-based unit tests are selected when requested, and no unknown-marker warning is emitted.

- [x] **Step 4: Build and validate distributions.**

Run: `python -m build`

Run: `twine check dist/*`

Install the newly built wheel in a clean temporary environment and run the package-resource smoke commands from Task 11.

Expected: both distributions contain `simind_python_connector/configs/input.smc` and all packaged data resources.

- [x] **Step 5: Run container validation.**

Run: `bash scripts/run_container_validation.sh --only-core`

Run: `bash scripts/run_container_validation.sh --only-stir` when STIR image dependencies are available.

Run: `bash scripts/run_container_validation.sh --only-sirf` when SIRF image dependencies are available.

Run: `bash scripts/run_container_validation.sh --only-pytomography --docker-platform linux/amd64` when PyTomography image dependencies are available.

Run: `bash scripts/run_container_examples.sh --only-core` when SIMIND is available; otherwise verify the explicit skip behavior and run with `--require-simind` to verify the expected failure.

- [x] **Step 6: Review the final worktree.**

Run: `git status --short`

Run: `git diff --check`

Run: `git diff --stat`

Expected: only files named in this plan are modified, apart from the pre-existing untracked `AGENTS.md`; no generated output, local SIMIND installation, coverage artifact, or secret is staged or included in the implementation diff.

## Definition of Done

- Every confirmed finding D-01 through D-26 and D-28 has either a passing regression test and implementation fix or an explicit documented reason for deferral.
- D-27 and all deferred hypotheses have been tested in the specified backend or container environment before being labeled fixed or rejected.
- `input.smc` is present in a clean source archive and built wheel.
- SMC/YAML round trips preserve values, flags, text variables, data files, and parameter 11.
- Connector runs do not mutate process-global CWD, retain one-shot runtime switches, ingest stale outputs, or accept unsafe prefixes.
- Interfile offsets and exact payload sizes are validated.
- Backend imports remain optional and lazy, and spacing conventions are documented consistently.
- Builder and DICOM tests cover asymmetric data, malformed metadata, single-frame data, and non-divisible energy windows.
- Container runners honor `--no-build`, report skipped/failed work accurately, and do not leak local installations or licensed files into build context.
- Fresh Ruff, pytest, package, and applicable Docker validation commands have been run and their outputs recorded in the report.

---

## Execution Evidence (2026-08-24)

All commands run from the repository root with `.venv/bin/python` (Python 3.9.6).

| Check | Command | Result |
|---|---|---|
| Lint | `ruff check .` | All checks passed |
| Format | `ruff format --check .` | Clean |
| Dependency-light suite | full `-m` filter + coverage | 193 passed, 6 skipped, 0 failed; line-rate **0.6522** (2134/3272) vs baseline 0.4766 (1692/3550, inflated by the removed legacy script) |
| CI config suite | `python -m pytest -c pytest-ci.ini -q` | 192 passed; `[pytest]` section honoured; no unknown-marker warnings |
| Distribution build | `python -m build && twine check dist/*` | Both artifacts PASSED; wheel contains `configs/input.smc`, all data files |
| Core container validation | `bash scripts/run_container_validation.sh --only-core` | 194 passed in-container + import-isolation test passed (image rebuilt with `docker/constraints.txt`) |
| SIMIND end-to-end | `bash scripts/run_container_validation.sh --only-core --with-simind --require-simind` | Examples 01/02/03/05 executed against real SIMIND inside the container; geometry diagnostics 4 passed |

### Integration bug caught by the SIMIND run

Example 03 initially lost its second energy window: `_clear_previous_outputs()`
was deleting the freshly written `{prefix}.win` input before execution.
Fixed by protecting `{prefix}.win`; regression added
(`test_python_connector_cleanup_preserves_window_file`). Re-run of the
SIMIND-dependent validation passes fully.

### Deviations from plan

- `raw_flags` YAML field was not added: initialising YAML configs with a
  full 30-character flag string plus named-flag overlay already preserves
  unmapped positions on round trips (covered by round-trip test).
- `ConversionConfig.radius_scale_factor` default changed **10.0 → 1.0**
  while still honouring any explicitly configured factor. The previous
  default was dead code that would have rescaled millimetre radii if ever
  wired up; inline evidence in the converter states SIMIND writes mm.
- Base-image digest pinning is deferred (requires network image pulls);
  pip installs are constrained via the new `docker/constraints.txt`.
- Backend containers for STIR/SIRF/PyTomography and the remaining Task 13
  investigations were not run in this session; commands are listed in
  Task 14 Step 5.

### Worktree

Only files enumerated by this plan are modified or created, plus the
pre-existing untracked `AGENTS.md`. `git diff --check` is clean. Nothing
has been committed or staged, per the execution rules.
