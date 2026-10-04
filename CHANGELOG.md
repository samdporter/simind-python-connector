# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `simind_python_connector.utils.interfile`: one Interfile module (`InterfileHeader`,
  `read_header`, `load_interfile_array`, `ProjectionGeometry`, `read_projection_geometry`,
  `read_simind_density_image`, `normalise_key`, `parse_interfile_line`). Header key
  lookups ignore case, leading `!`/`#` and spacing.
- `utils.simind_utils.validate_energy_windows` (public).
- `converters.attenuation.density_to_attenuation`.
- Runtime switches from the SIMIND 8 manual: `CA`, `FW`, `OU`, `DP`, `BG`, `HO`, `CO`, `X1`–`X6`.

### Changed
- `SimindToStirConverter.convert_file` returns `None`;
  `create_penetrate_headers_from_template` returns component header paths.
- The STIR adaptor reads voxel sizes from `get_grid_spacing()`; the SIRF adaptor from
  `voxel_sizes()`.
- pytest configuration lives in `pyproject.toml`.
- `NN` must be an integer >= 1; `RR`/`SC` must be integers; `CA`/`DI` must be 0, 1 or 2.
  Setting a switch to `None` removes it, and `True` writes a switch without a value.
- MPI runs use `set_mpi(processes, split_projections=False)` and the `simind_mpi` binary
  (`SIMIND_MPI_BIN`); the `MP` switch is rejected.
- `quantization_scale` must be in (0, 1]. A warning reports activity that receives no
  photon histories.
- `output_prefix` must be lower case (SIMIND lowercases file names on Unix); orbit files
  are copied to `{prefix}_orbit.cor`.
- Attenuation helpers take photon energies in keV; packaged `.atn` tables are plain text.
- Unreadable SIMIND outputs raise `SimulationError` instead of being skipped.
- Python 3.10 or newer is required.
- SCATTWIN runs pass `/CA:1`; `pri_wN` is derived as `tot_wN - sca_wN` when SIMIND does
  not write it, and the synthetic extra scatter window is no longer written.
- Projection rows (Index 77) follow the axial dimension; columns (Index 76) use
  `max(dim_x, dim_y)`.

### Removed
- `simind_python_connector.backends` (`get_backend`, `set_backend`, `reset_backend`,
  `create_image_data`, `create_acquisition_data`, wrapper classes).
- `utils.backend_access`, `utils.import_helpers`, `utils.sirf_stir_utils`, `utils.io_utils`,
  `utils.to_projdata_in_memory`.
- `utils.stir_utils` (`parse_sinogram`, `parse_interfile`, `extract_attributes_from_stir*`,
  `get_sirf_attenuation_from_simind`, `create_stir_image`, `create_stir_acqdata`,
  `create_simple_phantom`, `create_attenuation_map`, `convert_value`,
  `harmonize_stir_attributes`).
- `utils.interfile_parser` and `utils.interfile_numpy` (replaced by `utils.interfile`).
- `converters.dicom_to_stir`, `connectors._spacing`.
- `SimindToStirConverter.read_parameter`, `edit_parameter`, `add_parameter`,
  `validate_and_fix_scaling_factors`, `validate_and_correct_radius`, `add_custom_rule`
  and the `return_object` arguments.
- The `NumpyConnector` alias.
- `core.types.ScatterType`, `OutputError`, `OUTPUT_EXTENSIONS`, `ORBIT_FILE_EXTENSION`.
- `utils.simind_utils.SimindError`, `SimindNotFoundError`, and the unused
  `create_window_file` arguments `energy_window`, `lower_ew`, `upper_ew`.
- `pytest.ini`, `pytest-ci.ini`, `requirements.txt`, `requirements-dev.txt`,
  `examples/run_all_examples.sh`, and the `requires_setr` marker.

### Fixed
- Importing the package no longer configures the root logger.
- A negative Index 1 no longer breaks the mu-map to density conversion.
- Stale `.res`, `.bis`, `.spe` and `.cor` files are removed before each run.
- For non-square projections, acquisition `pixel_array` is `(tof, axial, view, bin)`, and
  the DICOM matrix sizes and pixel spacing are no longer swapped.

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
