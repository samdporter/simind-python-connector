# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-10-16

### Changed
- **Breaking:** without the `CA` runtime switch, SCATTWIN runs return only the
  total projection. The energy-window file now holds exactly the requested
  windows; 1.1.x appended a scatter-order-1 line when every order was 0, which
  made SIMIND also write `sca`, `air` and a duplicate `sca_w2`. Code that reads
  `outputs["sca_w1"]` or `get_scatter_output()` needs
  `add_runtime_switch("CA", 1)`.
- Internal tidy-up with no behaviour change: type hints use `X | None` and
  built-in generics, `__all__` lists are sorted, and redundant `pass`
  statements are removed.

### Added
- The `CA` runtime switch is accepted (`add_runtime_switch("CA", n)`). It
  selects the SCATTWIN projections SIMIND V8 writes: without it only `tot`;
  `1` adds `sca` and `air`; `2` adds `pri`, `sca` and `air`.
- A SCATTWIN run without the `CA` switch, and with every scatter order 0,
  issues a `UserWarning` saying that only total projections will be written.
- `get_primary_output()` returns the primary projection written with `CA` = 2.
- `get_scatter_output()` and `get_primary_output()` return the total minus the
  other component when only that one was written, and warn and return `None`
  when neither was.

### Fixed
- `number - x` on SIRF image and acquisition wrappers returned wrong values.
- `StirImageData` arithmetic raised `TypeError` on STIR 6.2, and
  `create_simple_phantom()` and `create_attenuation_map()` failed there.
- `StirAcquisitionData.get_info()` returns STIR's parameter text instead of the
  SWIG object's address.
- `from simind_python_connector.utils import *` raised `AttributeError`.
- Interfile data with a partial trailing pixel is rejected instead of the extra
  bytes being ignored.
- The DICOM acquisition builder no longer prints debug lines, and its warnings
  name the missing `AcquisitionDate` and `AcquisitionTime` fields.

### Deprecated
- STIR 6.2 and earlier, which have no image arithmetic. `StirImageData`
  computes it in NumPy there and issues a `DeprecationWarning`; support will be
  removed in a future release.

### Notes
- This release supports SIMIND V8.0. SIMIND V9.0 is not supported yet: it
  changes several output formats and behaviours the connector relies on, so
  V9 support will follow in a later release.

## [1.1.2] - 2026-10-15

### Added
- Voxel phantoms accept separate axial and in-plane spacing via
  `voxel_size_mm=(z, y, x)` in mm. SIRF/STIR adaptors read all three spacings
  from their images; PyTomography accepts the same tuple order. In-plane
  spacing must remain square (`y == x`), and scalar spacing remains supported.
- `RuntimeOperator(timeout=...)` takes a limit in seconds; a run that exceeds it
  stops and raises `SimulationError`. The default remains unlimited.

### Changed
- Python ≥3.10 is now required (`requires-python` and the metadata/classifiers
  are updated accordingly); previously the metadata claimed 3.9+.
- NumPy is capped at `>=1.20.0,<2.0` package-wide: this cap is a binary-ABI /
  backend-compatibility guard for the legacy 1.1.x line, because the STIR and
  SIRF bindings are compiled against NumPy 1.x and crash with NumPy 2. It is not
  a Python-version constraint.
- PENETRATE per-component fallback diagnostics distinguish a missing backend from
  a real failure: falling back because the backend is not installed is logged at
  DEBUG (quiet by default), while a load failure in an installed backend is
  logged at WARNING.
- The duplicate single-window scatter file was investigated on a real SIMIND V8.0
  run: the extra `.win` line is REQUIRED (without it SIMIND emits no scatter/air
  outputs), so the behaviour is intentionally retained and is now documented and
  covered by a real-binary regression test.
- Input narrowing is deliberate, even in a patch release: the Interfile connector
  now rejects up front what it could never consume, instead of failing obscurely
  after SIMIND ran. Only the scoring routines DUMMY (0), SCATTWIN (1) and
  PENETRATE (4) are supported; LIST_MODE (2) and FORCED_COLLIMATION (3) are
  rejected at preflight. A literal 1x1x1 phantom is rejected (pad to at least two
  voxels in one dimension), as is a non-zero source that `quantization_scale`
  rounds to all zeros. `output_prefix` is capped at 47 characters and at most 99
  energy windows are accepted: SIMIND V8 copies output names into a fixed
  60-character buffer and silently truncates longer ones, dropping the final
  `.h00`/`.a00` character so no readable SPECT header remains. Measured on a
  real SIMIND V8.0 run, 49 characters is the exact maximum prefix for
  `{prefix}_tot_w1.a00` (50 truncates it); the 47-character cap is a
  conservative policy with a small margin below that boundary, not an
  engine-measured limit. The longest per-window name adds `_tot_w{N}.a00`, and
  when every scatter order is below 1 the connector writes the required extra
  window line, which adds one more output set, so with the 99-window maximum the
  suffix is 13 characters and 47 + 13 = 60 fills the buffer exactly. `MP` takes
  positive integers (including NumPy integers), integral floats and ASCII digit
  strings; `NN` takes positive integers and integral floats, and an
  integral-float `NN` is written to the command line as an integer (for example
  `/NN:2`) while the stored switch value is unchanged. `orbit_file` must be a
  regular file. Index 26 (voxel history) and the `/NN` *simulation semantics*
  are unchanged — only validation was added.
- The direct executor validates every runtime switch value, not only `MP` and
  `NN`, before it builds the SIMIND command, so a caller driving
  `SimindExecutor` directly gets the same checks the connector applies;
  invalid values raise `SimulationError`.
- `output_prefix` is validated at construction and again before every
  connector-managed write or cleanup (source/density files, the window file, the
  saved `.smc`, stale-output removal and orbit staging), so an over-long prefix
  can no longer overwrite or delete existing `{prefix}.smc` / `{prefix}_tot_w1.hs`
  before it is rejected.
- Filesystem failures in connector-managed steps (writing the source, density and
  window inputs, staging an orbit file, clearing stale outputs, converting
  headers) raise `OutputError`. `OutputError` is also an `OSError`, so callers
  written against the pre-1.1.2 bare `OSError` raises keep catching these
  failures. Missing SPECT/PENETRATE headers still raise `FileNotFoundError` (the
  error is both), and missing inputs such as the configuration file or the orbit
  file still raise `FileNotFoundError`.

### Fixed
- `load_table()` reads both bundled NIST attenuation tables, skipping their
  headers. Its documentation now gives the table units correctly: energy in
  MeV and mass attenuation coefficient in cm²/g.
- Source distributions include the fixtures and runner/container files needed
  to run the bundled tests outside a Git checkout.
- Custom energy-window filenames respect SIMIND's shorter filename buffer:
  `output_prefix` is limited to 15 characters when an energy-window file is
  used, preventing silently ignored energy bounds. Example 02 uses safe names.
- Native source and attenuation images must share spatial geometry, including
  origins and axis directions, before simulation. Interfile byte offsets are
  parsed exactly and fractional offsets are rejected.
- `-` is rejected in `output_prefix`: SIMIND treats it as an argument delimiter,
  so the run silently produced no output.
- Orbit files are always staged as `{prefix}_orbit.cor`, so an orbit can no longer
  overwrite the source, density or window inputs when basenames collide.
- SIMIND error diagnostics are detected even when SIMIND exits 0 and raise
  `SimulationError`; previously the run failed later with a misleading
  missing-output error. The exit-0 scan also recognises
  `No input file found for the spectra:<x>.isd`, which used to surface as a
  misleading missing-header error, only before SIMIND's `OTHER INFORMATION`
  banner so an echoed configuration comment cannot be mistaken for the failure;
  a non-zero exit now quotes captured stdout/stderr excerpts in the
  `SimulationError` message.
- `find_penetrate_h00_file` matches only the exact prefix name or an
  underscore-boundary suffix, so overlapping prefixes such as `s` and `s2` no
  longer claim each other's `.h00` output.
- A failure while writing a PENETRATE component header now propagates as an
  `OutputError` instead of being swallowed, which used to drop that component
  with only a log line; the temporary `.hs` template is removed on every path,
  and a cleanup failure there logs a WARNING instead of masking the original
  error.
- Reflected and unary arithmetic on the STIR wrappers no longer raises
  `TypeError`: scalar-on-the-left operators (`2.0 + img`, `2.0 - img`,
  `2.0 * img`, `2.0 / img`), unary `-img`, and a native STIR object on the left
  of `+`, `-`, `*` or `/` (`native + image`, `native - image`, `native * image`,
  `native / image`, and likewise for acquisition data) all work, because STIR's
  own dunders return `NotImplemented` for the wrapper and Python falls back to
  the wrapper's reflected operator. NumPy arrays and other non-scalar,
  non-STIR operands on the left are not covered.
- `get_primary_output()` raises a clear `NotImplementedError` instead of
  `KeyError`, because SIMIND emits no standalone primary projection; before a
  successful run it still raises `RuntimeError`.
- `SimulationConfig` is robust to bad or short configurations: `save_file`
  formats and format-checks the whole fixed-width block before truncating the
  destination, rejects values that cannot fit a 12-character SMC field
  (NaN/Inf, oversized exponents) and canonicalises `-0.0`, `print_config` and
  `validate_parameters` handle configs shorter than the index tables, and YAML
  import validates that sections are mappings, accepts quoted integer indices
  and rejects duplicate indices after normalisation within each section it reads
  (`parameters`, `simulation_flags`, `data_files`, `text_variables`,
  `other_parameters`, `other_flags`).
- The scoring routine is read without rounding: a fractional or non-finite
  Index 84 raises `ValueError` instead of being rounded, a configuration too
  short to contain Index 84 reads as DUMMY, and a scoring value or
  `quantization_scale` that overflows float coercion raises `ValueError`
  instead of `OverflowError`.
- `stir.extra` is preferred over the deprecated `stirextra`; the legacy name is
  kept as a fallback for older STIR installs.
- `MP` (MPI process count) must be an integer >= 1; `MP=0` previously ran a
  single process silently.
- SIMIND reads the source and density integer maps as signed 16-bit, so the
  density cap of 65535 could wrap values above 32767 negative and silently
  corrupt the map; it is now capped at 32767. `MAX_SOURCE` becomes
  `np.iinfo(np.int16).max` (32767) and `DEFAULT_QUANTIZATION_SCALE`
  (`500 / 32767`) keeps the legacy 500-weight source default. Explicit
  `quantization_scale` values are not auto-migrated: rescale a scale calibrated
  for the former 500 maximum with `old_scale * DEFAULT_QUANTIZATION_SCALE`.

### Known Limitations
- A run that exceeds its `timeout` kills the launcher process but may not reap
  every `simind_mpi` rank; there is no process-tree teardown yet.
- Primary projections are unsupported by the adaptors, so
  `get_primary_output()` always raises `NotImplementedError`: SIMIND's standard
  SPECT output has no primary file, though lower-level `/CA:2` use can produce
  them.
- Scoring routines LIST_MODE (2) and FORCED_COLLIMATION (3) remain available
  only via lower-level use of the package.

## [1.1.1] - 2026-10-11

### Fixed
- A negative Index 1 (energies from the isotope file) turned the mu map into near-vacuum;
  it is now converted at `abs(Index 1)`.
- Acquisition arrays use `(tof, axial, view, bin)`, and DICOM matrix sizes and pixel
  spacing are no longer swapped, so non-square projections build correctly.
  `create_stir_acqdata` takes `(bins, axial)` and uses the same layout.
- The PyTomography adaptor reverses axial slices to match PyTomography's SIMIND reader;
  reconstructions were flipped along z.
- The STIR/SIRF adaptors raise a `ValueError` for non-cubic voxels instead of using the
  z spacing for every axis.
- The adaptors default `scoring_routine` to None, which keeps the configuration's
  Index 84; they used to overwrite it with SCATTWIN.
- The geometry docs give the projection axis orders correctly.
- PyTomography PENETRATE and fallback outputs use the reader's `(view, bin, axial)`
  order, and PyTomography scatter orders are validated like the core's.
- The STIR/SIRF adaptors reject a mu map whose voxel size differs from the source's.
- The legacy SIRF attenuation loader reverses the voxel sizes with the dimensions.
- The SIRF wrapper's `maximum()` and `sapyb()` update the wrapped object.
- Piecewise Schneider conversion gives HU outside the table the nearest segment's
  density, so metal above 4001 HU no longer becomes air.
- MPI runs use `$SIMIND_MPI_BIN`, else `simind_mpi` next to the serial binary, with
  `/MP` and without a stray `-p`.
- An output header that cannot be read raises instead of being skipped.
- YAML export keeps unnamed parameters and flags.
- New header keys are written before `!END OF INTERFILE`, and `set()` changes the
  last duplicate key, the one readers use.
- A header converted into another directory points at its data by absolute path, and
  an orbit file named like an ignored key keeps its radii.
- DICOM frames are split by `EnergyWindowVector` when present; each output declares
  one window and its share of the duration.
- `run()` sets Flags 5 and 14, which SIMIND needs to write readable headers, so
  config-defined phantom runs work.
- Example 07C no longer drops attenuation and PSF modelling silently, and the examples
  accept `SIMIND_BIN`.
- Importing the converter no longer configures root logging.
- setuptools 77.0.3 or newer is required to build.

### Changed
- `get_attenuation_coefficient` takes keV, as documented.
- `NN` must be an integer >= 1, switches reject None and bool values,
  `quantization_scale` must be in (0, 1], and `output_prefix` must match
  `[a-z0-9][a-z0-9_-]*` (SIMIND lowercases file names).
- Publishing waits for the tests and checks the tag against the version; tests also
  run on `release/**` branches.

## [1.1.0] - 2026-10-11

Replaces 2.0.0 and 3.0.0, which were published by mistake and have been withdrawn.

### Breaking Changes
- The projection bin size and matrix (Index 28, 76, 77) come from the configuration.
  `configure_voxel_phantom` no longer sets them from the phantom image, which made the
  projections square whenever the image slices were square and took the number of axial
  rows from the image height.
- `configure_voxel_phantom` keeps the configured scoring routine (Index 84) unless
  `scoring_routine` is given.
- `STIRSPECTAcquisitionDataBuilder` no longer flips projection data along the last axis.
- Runtime-operator switches passed to `run()` apply to that run only. They are no longer
  kept on the connector for later runs.
- Each run deletes earlier outputs that share the output prefix before SIMIND starts.
- The STIR backend needs `stirextra` as well as `stir`.
- STIR/SIRF adaptors take the z voxel spacing from a different axis (see Fixed). Results
  change for anisotropic voxels.
- Invalid inputs now raise `ValueError`: empty, non-finite or negative source and
  attenuation maps, array shapes that do not match the header, and `output_prefix`
  values that contain a path. Energy-window validation raises `ValueError` instead of
  `AssertionError`.

### Added
- `run()` warns when a voxel phantom is wider or longer than the projection field of
  view.

### Fixed
- The packaged `configs/input.smc` is now included in the wheel. SMC parsing handles
  fixed-width fields and checks section counts.
- STIR/SIRF adaptors read the z voxel spacing from the correct axis: `voxel_sizes()`
  is `(z, y, x)`, and raw `get_grid_spacing()` is `(unused, z, y, x)`.
- Voxel-map dimension indices 78, 79, 81 and 82 are set correctly.
- The executor resolves `SIMIND_BIN` and accepts executable paths that contain spaces.
  The connector no longer changes the process working directory.
- Runtime switches apply to one run only. Output prefixes cannot escape the output
  directory. Stale outputs are removed, and `.win`, `.smi` and `.dmi` inputs are kept.
- Energy-window validation raises `ValueError` instead of failing an `assert`.
- The Interfile loader honours data offsets, strips quoted values and rejects truncated
  or oversized payloads. The SIMIND-to-STIR converter keeps data-file lines and applies
  the radius scale factor.
- Attenuation utilities handle the `rho*1000` selector and STIR `#` header keys, and
  reject unknown attenuation types.
- Image and acquisition builders validate array shapes, split multi-energy data
  deterministically and no longer flip images. The DICOM builder handles missing timing
  information and 2D pixel data.
- Backend adaptors are imported lazily, so the core package imports without STIR, SIRF
  or PyTomography.

### Removed
- The legacy `scripts/simulation.py`.
- The "non-square matrix" warning from `SimulationConfig.validate_parameters`; SIMIND
  supports non-square projections.

## [1.0.1] - 2026-04-16

### Fixed
- Corrected installation documentation to lead with `pip install simind-python-connector`.
- Clarified that SIMIND is an external runtime dependency and must be available as `simind` on `PATH`.
- Updated contributing and testing documentation to match the current Ruff-based tooling.
- Documented the DICOM-driven adaptor examples.
- Fixed the GitHub Actions coverage target after the import package rename.
- Fixed README links and release metadata for the patch release.

## [1.0.0] - 2026-04-16

### Added
- Python SIMIND Monte Carlo connector for SPECT imaging simulations.
- STIR/SIRF adaptor for reading and writing SIMIND output data compatible with the STIR/SIRF reconstruction framework.
- PyTomography adaptor for integration with the PyTomography reconstruction library.
- Support for SIMIND `.atn` attenuation map data files.
- Helper utilities for configuring and running SIMIND Monte Carlo simulations from Python.
- Comprehensive test suite using `pytest`.
- Documentation hosted on ReadTheDocs.
- PyPI packaging (`simind-python-connector`) with optional `dev` and `examples` dependency groups.

### Changed
- Renamed the PyPI distribution to `simind-python-connector`.
- Renamed the import package to `simind_python_connector`.
- Moved to a connector-first public API centered on `SimindPythonConnector` and backend adaptors.
