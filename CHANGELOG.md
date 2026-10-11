# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
