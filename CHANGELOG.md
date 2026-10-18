# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.1] - Unreleased

The supported successor to 1.1.1 and the replacement for the withdrawn 2.0.0 and
3.0.0. It carries the A0–E next-major work that was published by mistake under
2.0.0/3.0.0, together with the projection-geometry correction that followed.

If you installed the withdrawn 3.0.0, reinstall this release explicitly:
`pip install "simind-python-connector==2.0.1"` (a 2.x release does not
auto-supersede an installed 3.0.0, because pip does not downgrade an already
satisfied requirement).

### Added
- `simind_python_connector.utils.interfile`: one Interfile module
  (`InterfileHeader`, `read_header`, `load_interfile_array`, `ProjectionGeometry`,
  `read_projection_geometry`, `read_simind_density_image`, `normalise_key`,
  `parse_interfile_line`). Header key lookups ignore case, leading `!` and spacing.
- `simind_python_connector.normalisation.scale_to_reference` and the "Normalising
  SIMIND Output" documentation, for putting SIMIND output on a known scale when the
  activity is unknown.
- `set_activity(activity_mbq, time_per_projection_s=1.0)` on the connector and
  adaptors (SIMIND Index 25) for the known-activity route.
- Measured acquisition geometry: `configure_acquisition(geometry)` on the connector
  and `set_template(template)` on the SIRF/STIR adaptors simulate in a measured
  acquisition's geometry and return outputs in exactly that geometry after a
  tolerance check. `utils.interfile.check_geometry_match`,
  `write_in_template_geometry` and `ProjectionGeometry.time_per_projection_s`
  support it.
- `mu_map_type` ("attenuation", "density", "hu") and `mu_map_energy_kev`, plus
  `converters.attenuation.density_to_attenuation` and the public
  `utils.simind_utils.validate_energy_windows`.
- Anisotropic voxels: `voxel_size_mm` may be `(z, y, x)` with square in-plane voxels.
- Runtime switches from the SIMIND 8 manual: `CA`, `FW`, `BG`, `HO`,
  `CO`, `X1`–`X6`.
- Optional `simind_python_connector.recon` subpackage: a SIMIND-simulated additive
  term in SIRF and CIL reconstructions, refreshed at chosen intervals.
  - `UpdateSchedule`, `AdditiveUpdater` (damped, floored) and `UpdateRecord`.
  - `SimindProjector` and `reference_normaliser`.
  - `ScatterCorrection` (PENETRATE b01 − b02, or SCATTWIN scatter).
  - `measured_normaliser` and `ScatterCorrection(measured=..., smooth_across_views=...)`:
    Dewaraja et al. (2017) global and pixel-wise scaling of the SIMIND scatter estimate to
    measured data.
  - `recon.cil`: `build_subset_objectives`, `set_subset_additive`,
    `refresh_stochastic_state` and `CorrectionCallback`.
  - `recon.osem.run_osem_with_corrections`.
- Residual correction (Fu & Qi): `recon.ResidualCorrection` with SIMIND
  (`recon.SimindComponent`) or a SIRF model as the accurate forward model, and the
  diagnostics `recon.poisson_nll` and `recon.effective_objective`.
- `simind_python_connector.phantoms`: SIMIND's analytic phantoms (ellipsoid, box,
  cylinders, point, cardiac, multiple inserts), library phantoms (Zubal, NEMA IQ),
  `VoxelPhantom`, `voxelise` and `nema_iec_phantom`. `configure_phantom` on the
  connector and `set_phantom` / `get_ground_truth` on the SIRF/STIR adaptors.
- Examples 09–12 for the reconstruction and phantom workflows.
- `simind_python_connector.datasets.fetch_y90_phantom`: the Y-90 phantom data for the
  tutorials, from `SIMIND_CONNECTOR_Y90_DATA` or a pooch download once published.

### Changed
- Docs and examples 09-11 now rescale SIMIND inside the reconstruction loop with
  the ratio of sums (`reference_normaliser(full_model)`) and smooth the scatter
  estimate. At quick history counts the trimmed ratio scales low, because it
  drops bins where SIMIND's primary is zero.
- Examples 01-08 moved to `examples/getting_started/` and `examples/backends/` as
  `plot_*` sphinx-gallery files, with their rendered outputs in the documentation.
  Examples 08A/08B/08C no longer take a DICOM argument; they read the Y-90 dataset
  (`datasets.fetch_y90_phantom`).
- Python 3.10 or newer is required.
- Configuration behaviour: the projection bin size and matrix (Index 28, 76, 77)
  come from the configuration (or `configure_acquisition`/`set_template`), not from
  the phantom image; rows (Index 77) follow the axial dimension and columns
  (Index 76) use `max(dim_x, dim_y)`. `configure_voxel_phantom` keeps the configured
  scoring routine (Index 84) unless told otherwise, `VoxelPhantom` and library
  phantoms follow Flag 11 from the configuration, and configuring a phantom keeps
  source shifts (Index 16–18) set beforehand.
- `set_template` applies the template when called instead of on every `run()`, so
  projection settings changed afterwards with `add_config_value` are not overwritten.
  The outputs then come back in SIMIND's geometry, with a warning. The exception is
  Index 12 with a non-circular template: its per-view radii are stored and written to
  the acquisition orbit file, which overrides Index 12, and changing Index 12 does not
  clear them. To switch to a circular orbit, replace the acquisition with a circular
  `configure_acquisition()` geometry or supply a circular template via `set_template()`.
- Interfile key lookups ignore only `!` and spacing; a line whose first non-blank
  character is `#` is a comment, never a key. Bracket indices are canonicalised
  (`[ 1 ]`/`[01]` = `[1]`), lookup stops at the `!END OF INTERFILE` terminator, and
  the radius is taken from `Radius` for a circular orbit and `Radii` otherwise,
  matching STIR.
- `NN` must be an integer >= 1; `RR`/`SC` must be integers; `CA`/`DI` must be 0, 1 or
  2. Setting a switch to `None` removes it and `True` writes a switch without a
  value. `quantization_scale` must be in (0, 1] and `output_prefix` must be lower
  case.
- MPI runs use `set_mpi(processes, split_projections=False)` and the `simind_mpi`
  binary (`SIMIND_MPI_BIN`); the `MP` switch is rejected.
- `get_attenuation_coefficient` takes photon energies in keV; the packaged `.atn`
  tables are plain text.
- Backend adaptors are imported lazily, so the core imports without STIR, SIRF or
  PyTomography; the SIRF/STIR adaptors check that source and mu-map spacing, origins
  and voxel sizes match.
- The non-circular orbit file is written at run time so its centre bin follows the
  current Index 76; non-circular orbit radii keep 0.1 mm precision in converted
  headers.
- Documentation builds from the package's `dev` extra: Read the Docs points at it
  and `docs/requirements.txt` is gone. The sdist excludes local-only material
  (`AGENTS.md`, `CLAUDE.md`, `opencode.jsonc`, `simind_manual.pdf`, `simind/` and
  generated docs).
- The python container image's MPICH provisioning (`mpirun` plus the SONAME compat
  links) is asserted by the container-script tests, so rebuilding the image cannot
  silently drop the MPI runtime the MPI tests need.

### Removed
- `simind_python_connector.backends` (`get_backend`, `set_backend`, `reset_backend`,
  `create_image_data`, `create_acquisition_data`, wrapper classes) and
  `utils.backend_access`, `utils.import_helpers`, `utils.sirf_stir_utils`,
  `utils.io_utils`, `utils.to_projdata_in_memory`.
- `utils.stir_utils`, `utils.interfile_parser` and `utils.interfile_numpy`
  (replaced by `utils.interfile`), `converters.dicom_to_stir` and
  `connectors._spacing`.
- `SimindToStirConverter.read_parameter`, `edit_parameter`, `add_parameter`,
  `validate_and_fix_scaling_factors`, `validate_and_correct_radius`,
  `add_custom_rule` and the `return_object` arguments.
- The `NumpyConnector` alias, `core.types.ScatterType`, `OutputError`,
  `OUTPUT_EXTENSIONS`, `ORBIT_FILE_EXTENSION`, `utils.simind_utils.SimindError` and
  `SimindNotFoundError`, and the unused `create_window_file` energy-window
  arguments.
- `pytest.ini`, `pytest-ci.ini`, `requirements.txt`, `requirements-dev.txt`,
  `examples/run_all_examples.sh` and the `requires_setr` marker.

### Fixed
- `OU` (list-mode format) and `DP` (forced-collimation plane spacing) are rejected
  with a `ValueError`: they only act with the list-mode and forced-collimation
  scoring routines, which the connector refuses to run, so they could never take
  effect.
- `STIRSPECTAcquisitionDataBuilder.update_header_from_dicom` mirrored the bin
  (column) axis of the DICOM projections, so STIR and SIRF data built from DICOM
  disagreed with the vendor Interfile export and with PyTomography. Rows, frames and
  columns now keep their DICOM order. The start angle is also rounded to whole
  degrees, because STIR truncates a value such as 359.999997 to 359.
- The projection pixel size and matrix set from the phantom image overwrote the
  configured values and gave square projections for cubic phantoms (and cut off the
  axial field of view for slices thicker than the pixels). `configure_voxel_phantom`
  no longer sets Index 28, 76 or 77; they come from the configuration or from
  `configure_acquisition`/`set_template`.
- `run()` warns when a voxel phantom is wider or longer than the projection field of
  view, with or without a template.
- The adaptors default `scoring_routine` to None, keeping the configuration's
  Index 84; they used to overwrite it with SCATTWIN. `get_scoring_routine()` now
  follows the phantom-specific execution precedence of `run()`: an analytic or library
  phantom always uses Index 84, while a `VoxelPhantom` or source/mu-map inputs keep the
  constructor's override.
- Example 12's phantomgen branch configures a 128×128 projection grid of 4.42 mm
  pixels to match its NEMA phantom, instead of inheriting the profile's 32×32 grid.
- PyTomography image-axis conversions reverse axial slices to match its SIMIND
  reader, keeping source and attenuation maps aligned with native reconstruction
  tensors.
- Importing the package no longer configures the root logger; a negative Index 1 no
  longer breaks the mu-map to density conversion.
- Stale `.hct`/`.ict`, `.res`, `.bis`, `.spe` and `.cor` files are removed before
  each run, and unreadable SIMIND outputs raise `SimulationError` instead of being
  skipped.
- The "non-square matrix" warning is gone from
  `SimulationConfig.validate_parameters`; SIMIND supports non-square projection
  matrices.
- SIMIND reads the source and density integer maps as signed 16-bit, so the
  density cap of 65535 could wrap values above 32767 negative and silently corrupt
  the map; it is now capped at 32767. `MAX_SOURCE` becomes `np.iinfo(np.int16).max`
  (32767) and `DEFAULT_QUANTIZATION_SCALE` (`500 / 32767`) keeps the legacy
  500-weight source default. Explicit `quantization_scale` values are not
  auto-migrated: rescale a scale calibrated for the former 500 maximum with
  `old_scale * DEFAULT_QUANTIZATION_SCALE`.
- `-` is rejected in `output_prefix`, at the connector and in the executor: SIMIND
  V8 treats `-` as an argument delimiter, so a hyphenated prefix split the
  positional filename arguments and the run silently produced no output. One
  pattern, `[a-z0-9][a-z0-9_]*`, lives in `core.types` and is enforced by the
  connector (`ValueError`) and by `SimindExecutor.run_simulation` before any
  subprocess launch (`SimulationError`), together with a type check and a
  47-character cap that keeps SIMIND's output names inside its fixed 60-character
  buffer.
- SIMIND's own diagnostics are reported even when the process exits 0. The
  executor captures stdout and stderr, replays them, and raises `SimulationError`
  for an error number, an error divider, or
  `No input file found for the spectra:<x>.isd`. Only the echoed configuration
  comment on the single line after SIMIND's `OTHER INFORMATION` banner is blanked
  before scanning, and the missing-spectra diagnostic is only searched before that
  banner, so a user comment cannot fail a successful run; a non-zero exit quotes
  labelled stdout/stderr excerpts instead of a bare exit status.
- The MPI process count is validated at the executor boundary: `mpirun -np` takes
  a positive integer (Python or NumPy integers, integral reals and ASCII digit
  strings; `bool` is rejected) and a conversion failure raises `SimulationError`
  instead of launching a bad command.
- `run()` validates the output prefix, the effective scoring routine and the
  runtime operator — switch values and a regular-file `orbit_file` — before any
  managed write, generated acquisition orbit write, orbit staging, config save or
  stale-output cleanup, and the native and PyTomography adaptors validate before
  inputs are written. An invalid setup fails before prior outputs are removed.
- The scoring routine is read without rounding: an absent Index 84 reads as DUMMY,
  and fractional, boolean, non-finite and unsupported values raise `ValueError`
  instead of being rounded or crashing with `OverflowError` (a `quantization_scale`
  that overflows float coercion raises `ValueError` too). Only DUMMY (0),
  SCATTWIN (1) and PENETRATE (4) are supported by the Interfile connector;
  LIST_MODE (2) and FORCED_COLLIMATION (3) are rejected at preflight.
- A literal 1x1x1 voxel phantom is rejected, as is a non-zero source that
  `quantization_scale` rounds entirely to zero — both before phantom settings are
  reset or the source and density maps are written, so an invalid input cannot
  leave a half-written run behind. All-zero sources remain allowed.
- At most 99 energy windows are accepted, so the longest per-window output name
  still fits SIMIND's 60-character output buffer.
- `SimulationConfig.save_file` builds and format-checks the whole fixed-width
  block before opening the destination, so an invalid value in a late section can
  no longer truncate an existing `.smc` file. A basic-data value that cannot fit
  a 12-character SMC field (NaN/Inf, or a finite non-zero value below the float
  range such as `1e-400`) raises `ValueError` instead of being written as zero,
  and `-0.0` is written as canonical zero. `print_config` and
  `validate_parameters` handle configurations shorter than the index tables.
- YAML import validates that each section is a mapping, normalises quoted and
  float indices, and rejects duplicate keys — both identical keys and keys that
  collide after index normalisation — instead of silently keeping the last one,
  while merge keys (`<<`) keep `SafeLoader` precedence: an explicit entry still
  overrides an inherited merged key.
- A failure while writing a PENETRATE component header propagates instead of
  dropping that component with only a log line; the connector-owned
  `{prefix}_template.hs` scratch file is removed on every path, and a cleanup
  failure there logs a warning instead of masking the original error.
  `find_penetrate_h00_file` matches only the exact prefix name or an
  underscore-boundary suffix, so overlapping prefixes such as `s` and `s2` no
  longer claim each other's `.h00` output.
- The orbit source supplied through `RuntimeOperator` is protected by
  device/inode identity during staging and cleanup: the supplied path and its
  resolved referent are never unlinked, including symlink and hardlink aliases and
  differently-cased spellings on a case-insensitive filesystem. The
  connector-generated `{prefix}_acquisition.cor` is still removed.
- STIR detection and conversion prefer the modern `stir.extra` module and fall
  back to the legacy standalone `stirextra` only when `stir.extra` is genuinely
  absent; a broken dependency inside `stir.extra` propagates instead of being
  masked by the legacy module.
- MPI runs that could not be loaded or added no statistics are refused before
  launch. With `set_mpi(n, split_projections=True)` and the PENETRATE scoring
  routine, `simind_mpi` writes only the primary process's projections (12 of 24
  views with two processes), so loading failed. With replicate MPI (no split) and a
  fixed `RR` switch, every process repeated the same histories and the result was
  bin-for-bin the serial run. Leave `RR` unset for replicate MPI: `simind_mpi` then
  seeds process `r` (from 0) as `RR = r + 1`.
- `PyTomographySimindAdaptor.get_output_header_path` returned the converted STIR
  `.hs` header for PENETRATE components and derived `pri_wN`/`sca_wN` outputs. Its
  start angle is SIMIND's + 180 degrees, and PyTomography's SIMIND reader takes it as
  SIMIND's, so reconstructions came out rotated 180 degrees about z. These outputs
  now get a `{prefix}_{key}.h00` copy of SIMIND's own header (v8 `{prefix}.h00`, v9
  `{prefix}.hNN`; `tot_wN` for derived outputs) naming the output's data file.

### Known limitations
- The optional SIRF/CIL scatter and residual correction workflows ship with the
  documented [SPECTUB all-views-cache interference](docs/spectub_cache_interference.md):
  two separately set-up SPECTUB models can interfere through the shared all-views
  cache, which changes the residual correction and the reconstruction endpoint. No
  cache fix or workaround ships in this release. This is an environment-specific,
  reported limitation and is not a claim of deterministic reconstruction endpoints
  or of resolved residual interference.

## [3.0.0] - 2026-10-09

> **Withdrawn.** Published by mistake and yanked; do not use. See the 2.0.1 entry
> above for the supported release.

If you installed 3.0.0, reinstall the supported release with an exact pin:
`pip install "simind-python-connector==2.0.1"` (or
`pip install --force-reinstall "simind-python-connector==2.0.1"`). A 2.x release
does not auto-supersede an installed 3.0.0, because pip does not downgrade an
already satisfied requirement.

### Added
- `simind_python_connector.utils.interfile`: one Interfile module (`InterfileHeader`,
  `read_header`, `load_interfile_array`, `ProjectionGeometry`, `read_projection_geometry`,
  `read_simind_density_image`, `normalise_key`, `parse_interfile_line`). Header key
  lookups ignore case, leading `!` and spacing; a `#`-prefixed line is a comment.
- `utils.simind_utils.validate_energy_windows` (public).
- `converters.attenuation.density_to_attenuation`.
- Runtime switches from the SIMIND 8 manual: `CA`, `FW`, `OU`, `DP`, `BG`, `HO`, `CO`, `X1`–`X6`.
- `set_activity(activity_mbq, time_per_projection_s=1.0)` on the connector and adaptors
  (SIMIND Index 25).
- `simind_python_connector.normalisation.scale_to_reference` and the "Normalising SIMIND
  Output" docs page.
- `configure_acquisition(geometry)` on the connector and `set_template(template)` on the
  SIRF/STIR adaptors: simulate in a measured acquisition's geometry; outputs come back
  in exactly that geometry after a tolerance check.
- `utils.interfile.check_geometry_match`, `write_in_template_geometry` and
  `ProjectionGeometry.time_per_projection_s`.
- `mu_map_type` ("attenuation", "density", "hu") and `mu_map_energy_kev`.
- Anisotropic voxels: `voxel_size_mm` may be `(z, y, x)` with square in-plane voxels.
- Optional `simind_python_connector.recon` subpackage: a SIMIND-simulated additive
  term in SIRF and CIL reconstructions, refreshed at chosen intervals.
  - `UpdateSchedule`, `AdditiveUpdater` (damped, floored) and `UpdateRecord`.
  - `SimindProjector` and `reference_normaliser` (A2 scaling at every update).
  - `ScatterCorrection` (PENETRATE b01 − b02, or SCATTWIN scatter).
  - `recon.cil`: `build_subset_objectives`, `set_subset_additive`,
    `refresh_stochastic_state` and `CorrectionCallback`.
  - `recon.osem.run_osem_with_corrections`.
- Examples 09 (SIMIND scatter in SIRF OSEM) and 10 (in CIL ISTA).
- Residual correction (Fu & Qi): `recon.ResidualCorrection` with SIMIND
  (`recon.SimindComponent`) or a SIRF model as the accurate forward model, and the
  diagnostics `recon.poisson_nll` and `recon.effective_objective`.
- Example 11 (residual correction with SIMIND).
- `simind_python_connector.phantoms`: SIMIND's analytic phantoms (ellipsoid, box,
  cylinders, point, cardiac, multiple inserts), library phantoms (Zubal, NEMA IQ),
  `VoxelPhantom`, `voxelise` (analytic ground truth) and `nema_iec_phantom`
  (phantomgen NEMA/IEC body phantom).
- `SimindPythonConnector.configure_phantom`, and `set_phantom` / `get_ground_truth`
  on the SIRF/STIR adaptors.
- Example 12 (phantoms).

### Changed
- Interfile key lookups ignore only `!` and spacing; a line whose first non-blank
  character is `#` is a comment, never a key (STIR does not honour `#` settings).
  Migration: a nonstandard template that used `#` as an active key prefix now has
  those fields ignored (required fields fail loudly); remove the `#` or use `!`.
- Interfile key lookups canonicalise bracket indices (`[ 1 ]`/`[01]` = `[1]`), stop
  at the `!END OF INTERFILE` terminator, and take the radius from `Radius` for a
  circular `orbit` and `Radii` otherwise, matching STIR.
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
- SCATTWIN runs pass `/CA:1` only when the user has not set `CA`; `pri_wN` is derived as
  `tot_wN - sca_wN` only for order-0 windows when SIMIND does not write it, and the
  synthetic extra scatter window is no longer written.
- The SIRF/STIR adaptors reject mismatched source/mu-map voxel sizes, and the connector
  requires square in-plane voxels.
- Non-circular orbit radii keep 0.1 mm precision in converted headers.
- The SIRF/STIR adaptors check that source and mu-map spacing and origins match.
- Projection rows (Index 77) follow the axial dimension; columns (Index 76) use
  `max(dim_x, dim_y)`.
- Stale `.hct`/`.ict` files are removed before each run.
- `SimindPythonConnector.run()` logs a warning when no phantom was configured and runs
  the phantom defined by the configuration (Index 14/15).

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
- PyTomography image-axis conversions reverse axial slices to match its SIMIND
  reader, keeping source and attenuation maps aligned with native reconstruction
  tensors. Real-reader regressions and a sparse SIMIND smoke test cover the fix.
- Importing the package no longer configures the root logger.
- A negative Index 1 no longer breaks the mu-map to density conversion.
- Stale `.res`, `.bis`, `.spe` and `.cor` files are removed before each run.
- For non-square projections, acquisition `pixel_array` is `(tof, axial, view, bin)`, and
  the DICOM matrix sizes and pixel spacing are no longer swapped.

## [2.0.0] - 2026-10-09

> **Withdrawn.** Published by mistake and yanked; do not use. See the 2.0.1 entry
> above for the supported release.

### Breaking Changes
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
