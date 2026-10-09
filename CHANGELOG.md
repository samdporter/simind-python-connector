# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-10-09

### Fixed
- The packaged `configs/input.smc` is now included in the wheel. SMC parsing handles
  fixed-width fields and checks section counts.
- STIR/SIRF adaptors read the z voxel spacing from the correct axis: `voxel_sizes()`
  is `(z, y, x)`, and raw `get_grid_spacing()` is `(unused, z, y, x)`.
- Voxel-map dimension indices 76, 77, 78, 79, 81 and 82 are set correctly.
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
