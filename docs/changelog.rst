.. _changelog:

Changelog
=========

All notable changes to this project are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

2.0.1 - Unreleased
------------------

The supported successor to 1.1.1 and the replacement for the withdrawn 2.0.0
and 3.0.0. It carries the A0-E next-major work that was published by mistake
under 2.0.0/3.0.0, together with the projection-geometry correction that
followed.

If you installed the withdrawn 3.0.0, reinstall this release explicitly:
``pip install "simind-python-connector==2.0.1"`` (a 2.x release does not
auto-supersede an installed 3.0.0, because pip does not downgrade an already
satisfied requirement).

Added
~~~~~

- ``simind_python_connector.utils.interfile``: one Interfile module
  (``InterfileHeader``, ``read_header``, ``load_interfile_array``,
  ``ProjectionGeometry``, ``read_projection_geometry``,
  ``read_simind_density_image``, ``normalise_key``, ``parse_interfile_line``).
  Header key lookups ignore case, leading ``!`` and spacing.
- ``simind_python_connector.normalisation.scale_to_reference`` and the
  "Normalising SIMIND Output" documentation, for putting SIMIND output on a
  known scale when the activity is unknown.
- ``set_activity(activity_mbq, time_per_projection_s=1.0)`` on the connector
  and adaptors (SIMIND Index 25) for the known-activity route.
- Measured acquisition geometry: ``configure_acquisition(geometry)`` on the
  connector and ``set_template(template)`` on the SIRF/STIR adaptors simulate
  in a measured acquisition's geometry and return outputs in exactly that
  geometry after a tolerance check. ``check_geometry_match``,
  ``write_in_template_geometry`` and
  ``ProjectionGeometry.time_per_projection_s`` support it.
- ``mu_map_type`` ("attenuation", "density", "hu") and ``mu_map_energy_kev``,
  plus ``converters.attenuation.density_to_attenuation`` and the public
  ``utils.simind_utils.validate_energy_windows``.
- Anisotropic voxels: ``voxel_size_mm`` may be ``(z, y, x)`` with square
  in-plane voxels.
- Runtime switches from the SIMIND 8 manual: ``CA``, ``FW``, ``OU``, ``DP``,
  ``BG``, ``HO``, ``CO``, ``X1``-``X6``.
- Optional ``simind_python_connector.recon`` subpackage: a SIMIND-simulated
  additive term in SIRF and CIL reconstructions, refreshed at chosen intervals
  (``UpdateSchedule``, ``AdditiveUpdater``, ``UpdateRecord``,
  ``SimindProjector``, ``reference_normaliser``, ``ScatterCorrection``,
  ``recon.cil`` helpers and ``recon.osem.run_osem_with_corrections``).
  ``measured_normaliser`` and ``ScatterCorrection(measured=...,
  smooth_across_views=...)`` give Dewaraja et al. (2017) global and
  pixel-wise scaling of the SIMIND scatter estimate to measured data.
- Residual correction (Fu & Qi): ``recon.ResidualCorrection`` with SIMIND
  (``recon.SimindComponent``) or a SIRF model as the accurate forward model,
  and the diagnostics ``recon.poisson_nll`` and ``recon.effective_objective``.
- ``simind_python_connector.phantoms``: SIMIND's analytic phantoms, library
  phantoms (Zubal, NEMA IQ), ``VoxelPhantom``, ``voxelise`` and
  ``nema_iec_phantom``; ``configure_phantom`` on the connector and
  ``set_phantom`` / ``get_ground_truth`` on the SIRF/STIR adaptors.
- Examples 09-12 for the reconstruction and phantom workflows.

Changed
~~~~~~~

- Python 3.10 or newer is required.
- Configuration behaviour: the projection bin size and matrix (Index 28, 76,
  77) come from the configuration (or ``configure_acquisition`` /
  ``set_template``), not from the phantom image; rows (Index 77) follow the
  axial dimension and columns (Index 76) use ``max(dim_x, dim_y)``.
  ``configure_voxel_phantom`` keeps the configured scoring routine (Index 84)
  unless told otherwise, ``VoxelPhantom`` and library phantoms follow Flag 11
  from the configuration, and configuring a phantom keeps source shifts
  (Index 16-18) set beforehand.
- ``set_template`` applies the template when called instead of on every
  ``run()``, so projection settings changed afterwards with
  ``add_config_value`` are not overwritten. The outputs then come back in
  SIMIND's geometry, with a warning. The exception is Index 12 with a
  non-circular template: its per-view radii are stored and written to the
  acquisition orbit file, which overrides Index 12, and changing Index 12 does
  not clear them. To switch to a circular orbit, replace the acquisition with a
  circular ``configure_acquisition()`` geometry or supply a circular template
  via ``set_template()``.
- Interfile key lookups ignore only ``!`` and spacing; a line whose first
  non-blank character is ``#`` is a comment, never a key. Bracket indices are
  canonicalised, lookup stops at the ``!END OF INTERFILE`` terminator, and the
  radius is taken from ``Radius`` for a circular orbit and ``Radii`` otherwise,
  matching STIR.
- ``NN`` must be an integer >= 1; ``RR``/``SC`` must be integers; ``CA``/``DI``
  must be 0, 1 or 2. Setting a switch to ``None`` removes it and ``True``
  writes a switch without a value. ``quantization_scale`` must be in (0, 1]
  and ``output_prefix`` must be lower case.
- MPI runs use ``set_mpi(processes, split_projections=False)`` and the
  ``simind_mpi`` binary (``SIMIND_MPI_BIN``); the ``MP`` switch is rejected.
- ``get_attenuation_coefficient`` takes photon energies in keV; the packaged
  ``.atn`` tables are plain text.
- Backend adaptors are imported lazily, so the core imports without STIR, SIRF
  or PyTomography; the SIRF/STIR adaptors check that source and mu-map spacing,
  origins and voxel sizes match.
- The non-circular orbit file is written at run time so its centre bin follows
  the current Index 76; non-circular orbit radii keep 0.1 mm precision.
- Documentation builds from the package's ``dev`` extra: Read the Docs points
  at it and ``docs/requirements.txt`` is gone. The sdist excludes local-only
  material (``AGENTS.md``, ``CLAUDE.md``, ``opencode.jsonc``,
  ``simind_manual.pdf``, ``simind/`` and generated docs).
- The python container image's MPICH provisioning (``mpirun`` plus the SONAME
  compat links) is asserted by the container-script tests, so rebuilding the
  image cannot silently drop the MPI runtime the MPI tests need.

Removed
~~~~~~~

- ``simind_python_connector.backends`` and ``utils.backend_access``,
  ``utils.import_helpers``, ``utils.sirf_stir_utils``, ``utils.io_utils`` and
  ``utils.to_projdata_in_memory``.
- ``utils.stir_utils``, ``utils.interfile_parser`` and
  ``utils.interfile_numpy`` (replaced by ``utils.interfile``),
  ``converters.dicom_to_stir`` and ``connectors._spacing``.
- ``SimindToStirConverter`` editing helpers and the ``return_object``
  arguments; the ``NumpyConnector`` alias; ``core.types.ScatterType``,
  ``OutputError``, ``OUTPUT_EXTENSIONS``, ``ORBIT_FILE_EXTENSION`` and the
  unused ``utils.simind_utils`` errors.
- ``pytest.ini``, ``pytest-ci.ini``, ``requirements.txt``,
  ``requirements-dev.txt``, ``examples/run_all_examples.sh`` and the
  ``requires_setr`` marker.

Fixed
~~~~~

- The projection pixel size and matrix set from the phantom image overwrote
  the configured values and gave square projections for cubic phantoms (and
  cut off the axial field of view for slices thicker than the pixels).
  ``configure_voxel_phantom`` no longer sets Index 28, 76 or 77; they come
  from the configuration or from ``configure_acquisition``/``set_template``.
- ``run()`` warns when a voxel phantom is wider or longer than the projection
  field of view, with or without a template.
- The adaptors default ``scoring_routine`` to None, keeping the configuration's
  Index 84; they used to overwrite it with SCATTWIN.
  ``get_scoring_routine()`` now follows the phantom-specific execution
  precedence of ``run()``: an analytic or library phantom always uses Index 84,
  while a ``VoxelPhantom`` or source/mu-map inputs keep the constructor's
  override.
- Example 12's phantomgen branch configures a 128 x 128 projection grid of
  4.42 mm pixels to match its NEMA phantom, instead of inheriting the profile's
  32 x 32 grid.
- PyTomography image-axis conversions reverse axial slices to match its SIMIND
  reader, keeping source and attenuation maps aligned with native
  reconstruction tensors.
- Importing the package no longer configures the root logger; a negative
  Index 1 no longer breaks the mu-map to density conversion.
- Stale ``.hct``/``.ict``, ``.res``, ``.bis``, ``.spe`` and ``.cor`` files are
  removed before each run, and unreadable SIMIND outputs raise
  ``SimulationError`` instead of being skipped.
- The "non-square matrix" warning is gone from
  ``SimulationConfig.validate_parameters``; SIMIND supports non-square
  projection matrices.
- SIMIND reads the source and density integer maps as signed 16-bit, so the
  density cap of 65535 could wrap values above 32767 negative and silently
  corrupt the map; it is now capped at 32767. ``MAX_SOURCE`` becomes
  ``np.iinfo(np.int16).max`` (32767) and ``DEFAULT_QUANTIZATION_SCALE``
  (``500 / 32767``) keeps the legacy 500-weight source default. Explicit
  ``quantization_scale`` values are not auto-migrated: rescale a scale
  calibrated for the former 500 maximum with
  ``old_scale * DEFAULT_QUANTIZATION_SCALE``.
- ``-`` is rejected in ``output_prefix``, at the connector and in the executor:
  SIMIND V8 treats ``-`` as an argument delimiter, so a hyphenated prefix split
  the positional filename arguments and the run silently produced no output.
  One pattern, ``[a-z0-9][a-z0-9_]*``, lives in ``core.types`` and is enforced
  by the connector (``ValueError``) and by ``SimindExecutor.run_simulation``
  before any subprocess launch (``SimulationError``), together with a type
  check and a 47-character cap that keeps SIMIND's output names inside its
  fixed 60-character buffer.
- SIMIND's own diagnostics are reported even when the process exits 0. The
  executor captures stdout and stderr, replays them, and raises
  ``SimulationError`` for an error number, an error divider, or
  ``No input file found for the spectra:<x>.isd``. Only the echoed
  configuration comment on the single line after SIMIND's ``OTHER
  INFORMATION`` banner is blanked before scanning, and the missing-spectra
  diagnostic is only searched before that banner, so a user comment cannot
  fail a successful run; a non-zero exit quotes labelled stdout/stderr
  excerpts instead of a bare exit status.
- The MPI process count is validated at the executor boundary: ``mpirun -np``
  takes a positive integer (Python or NumPy integers, integral reals and
  ASCII digit strings; ``bool`` is rejected) and a conversion failure raises
  ``SimulationError`` instead of launching a bad command.
- ``run()`` validates the output prefix, the effective scoring routine and the
  runtime operator — switch values and a regular-file ``orbit_file`` — before
  any managed write, generated acquisition orbit write, orbit staging, config
  save or stale-output cleanup, and the native and PyTomography adaptors
  validate before inputs are written. An invalid setup fails before prior
  outputs are removed.
- The scoring routine is read without rounding: an absent Index 84 reads as
  DUMMY, and fractional, boolean, non-finite and unsupported values raise
  ``ValueError`` instead of being rounded or crashing with ``OverflowError``
  (a ``quantization_scale`` that overflows float coercion raises
  ``ValueError`` too). Only DUMMY (0), SCATTWIN (1) and PENETRATE (4) are
  supported by the Interfile connector; LIST_MODE (2) and FORCED_COLLIMATION
  (3) are rejected at preflight.
- A literal 1x1x1 voxel phantom is rejected, as is a non-zero source that
  ``quantization_scale`` rounds entirely to zero — both before phantom
  settings are reset or the source and density maps are written, so an
  invalid input cannot leave a half-written run behind. All-zero sources
  remain allowed.
- At most 99 energy windows are accepted, so the longest per-window output
  name still fits SIMIND's 60-character output buffer.
- ``SimulationConfig.save_file`` builds and format-checks the whole
  fixed-width block before opening the destination, so an invalid value in a
  late section can no longer truncate an existing ``.smc`` file. A basic-data
  value that cannot fit a 12-character SMC field (NaN/Inf, or a finite
  non-zero value below the float range such as ``1e-400``) raises
  ``ValueError`` instead of being written as zero, and ``-0.0`` is written as
  canonical zero. ``print_config`` and ``validate_parameters`` handle
  configurations shorter than the index tables.
- YAML import validates that each section is a mapping, normalises quoted and
  float indices, and rejects duplicate keys — both identical keys and keys
  that collide after index normalisation — instead of silently keeping the
  last one, while merge keys (``<<``) keep ``SafeLoader`` precedence: an
  explicit entry still overrides an inherited merged key.
- A failure while writing a PENETRATE component header propagates instead of
  dropping that component with only a log line; the connector-owned
  ``{prefix}_template.hs`` scratch file is removed on every path, and a
  cleanup failure there logs a warning instead of masking the original error.
  ``find_penetrate_h00_file`` matches only the exact prefix name or an
  underscore-boundary suffix, so overlapping prefixes such as ``s`` and ``s2``
  no longer claim each other's ``.h00`` output.
- The orbit source supplied through ``RuntimeOperator`` is protected by
  device/inode identity during staging and cleanup: the supplied path and its
  resolved referent are never unlinked, including symlink and hardlink aliases
  and differently-cased spellings on a case-insensitive filesystem. The
  connector-generated ``{prefix}_acquisition.cor`` is still removed.
- STIR detection and conversion prefer the modern ``stir.extra`` module and
  fall back to the legacy standalone ``stirextra`` only when ``stir.extra`` is
  genuinely absent; a broken dependency inside ``stir.extra`` propagates
  instead of being masked by the legacy module.

Known limitations
~~~~~~~~~~~~~~~~~

- The optional SIRF/CIL scatter and residual correction workflows ship with the
  documented :doc:`SPECTUB all-views-cache interference
  <spectub_cache_interference>`: two separately set-up SPECTUB models can
  interfere through the shared all-views cache, which changes the residual
  correction and the reconstruction endpoint. No cache fix or workaround ships
  in this release. This is an environment-specific, reported limitation and is
  not a claim of deterministic reconstruction endpoints or of resolved residual
  interference.

3.0.0 - 2026-10-09
------------------

**Withdrawn.** This release was published by mistake and yanked; do not use.
See the 2.0.1 entry above for the supported release.

If you installed 3.0.0, reinstall the supported release with an exact pin:
``pip install "simind-python-connector==2.0.1"`` (a 2.x release does not
auto-supersede an installed 3.0.0, because pip does not downgrade an already
satisfied requirement).

Breaking Changes
~~~~~~~~~~~~~~~~

- Python 3.10 or newer is required.
- Removed the backend wrappers and legacy Interfile and STIR utility modules.
  Use the native adaptors and ``simind_python_connector.utils.interfile``.
- Photon-history scaling and runtime switches now reject invalid values.

Added
~~~~~

- Measured acquisition geometry, activity scaling, anisotropic voxel maps and
  SIMIND MPI execution.
- Analytic and library phantoms, voxelisation and native adaptor ground truth.
- Optional SIMIND scatter and residual corrections for SIRF and CIL workflows.

Fixed
~~~~~

- PyTomography image-axis conversions reverse z slices to match its SIMIND
  reader. Source and attenuation maps now agree with native tensor slice order.
- Added asymmetric real-reader regressions and a sparse two-view SIMIND smoke
  test for axial source placement.
- Consolidated Interfile parsing and geometry handling to follow STIR semantics.

2.0.0 - 2026-10-09
------------------

**Withdrawn.** This release was published by mistake and yanked; do not use.
See the 2.0.1 entry above for the supported release.

Breaking Changes
~~~~~~~~~~~~~~~~

- ``STIRSPECTAcquisitionDataBuilder`` no longer flips projection data along the
  last axis.
- Runtime-operator switches passed to ``run()`` apply to that run only. They
  are no longer kept on the connector for later runs.
- Each run deletes earlier outputs that share the output prefix before SIMIND
  starts.
- The STIR backend needs ``stirextra`` as well as ``stir``.
- STIR/SIRF adaptors take the z voxel spacing from a different axis (see
  Fixed). Results change for anisotropic voxels.
- Invalid inputs now raise ``ValueError``: empty, non-finite or negative source
  and attenuation maps, array shapes that do not match the header, and
  ``output_prefix`` values that contain a path. Energy-window validation raises
  ``ValueError`` instead of ``AssertionError``.

Fixed
~~~~~

- The packaged ``configs/input.smc`` is now included in the wheel. SMC parsing
  handles fixed-width fields and checks section counts.
- STIR/SIRF adaptors read the z voxel spacing from the correct axis:
  ``voxel_sizes()`` is ``(z, y, x)``, and raw ``get_grid_spacing()`` is
  ``(unused, z, y, x)``.
- Voxel-map dimension indices 76, 77, 78, 79, 81 and 82 are set correctly.
- The executor resolves ``SIMIND_BIN`` and accepts executable paths that
  contain spaces. The connector no longer changes the process working directory.
- Runtime switches apply to one run only. Output prefixes cannot escape the
  output directory. Stale outputs are removed, and ``.win``, ``.smi`` and
  ``.dmi`` inputs are kept.
- Energy-window validation raises ``ValueError`` instead of failing an
  ``assert``.
- The Interfile loader honours data offsets, strips quoted values and rejects
  truncated or oversized payloads. The SIMIND-to-STIR converter keeps data-file
  lines and applies the radius scale factor.
- Attenuation utilities handle the ``rho*1000`` selector and STIR ``#`` header
  keys, and reject unknown attenuation types.
- Image and acquisition builders validate array shapes, split multi-energy data
  deterministically and no longer flip images. The DICOM builder handles missing
  timing information and 2D pixel data.
- Backend adaptors are imported lazily, so the core package imports without
  STIR, SIRF or PyTomography.

Removed
~~~~~~~

- The legacy ``scripts/simulation.py``.

1.1.1 - 2026-10-11
------------------

Fixed
~~~~~

- A negative Index 1 (energies from the isotope file) turned the mu map into
  near-vacuum; it is now converted at ``abs(Index 1)``.
- Acquisition arrays use ``(tof, axial, view, bin)``, and DICOM matrix sizes and
  pixel spacing are no longer swapped, so non-square projections build
  correctly. ``create_stir_acqdata`` takes ``(bins, axial)`` and uses the same
  layout.
- The PyTomography adaptor reverses axial slices to match PyTomography's SIMIND
  reader; reconstructions were flipped along z.
- The STIR/SIRF adaptors raise a ``ValueError`` for non-cubic voxels instead of
  using the z spacing for every axis.
- The adaptors default ``scoring_routine`` to None, which keeps the
  configuration's Index 84; they used to overwrite it with SCATTWIN.
- The geometry docs give the projection axis orders correctly.
- PyTomography PENETRATE and fallback outputs use the reader's
  ``(view, bin, axial)`` order, and PyTomography scatter orders are validated
  like the core's.
- The STIR/SIRF adaptors reject a mu map whose voxel size differs from the
  source's.
- The legacy SIRF attenuation loader reverses the voxel sizes with the
  dimensions.
- The SIRF wrapper's ``maximum()`` and ``sapyb()`` update the wrapped object.
- Piecewise Schneider conversion gives HU outside the table the nearest
  segment's density, so metal above 4001 HU no longer becomes air.
- MPI runs use ``$SIMIND_MPI_BIN``, else ``simind_mpi`` next to the serial
  binary, with ``/MP`` and without a stray ``-p``.
- An output header that cannot be read raises instead of being skipped.
- YAML export keeps unnamed parameters and flags.
- New header keys are written before ``!END OF INTERFILE``, and ``set()``
  changes the last duplicate key, the one readers use.
- A header converted into another directory points at its data by absolute
  path, and an orbit file named like an ignored key keeps its radii.
- DICOM frames are split by ``EnergyWindowVector`` when present; each output
  declares one window and its share of the duration.
- ``run()`` sets Flags 5 and 14, which SIMIND needs to write readable headers,
  so config-defined phantom runs work.
- Example 07C no longer drops attenuation and PSF modelling silently, and the
  examples accept ``SIMIND_BIN``.
- Importing the converter no longer configures root logging.
- setuptools 77.0.3 or newer is required to build.

Changed
~~~~~~~

- ``get_attenuation_coefficient`` takes keV, as documented.
- ``NN`` must be an integer >= 1, switches reject None and bool values,
  ``quantization_scale`` must be in (0, 1], and ``output_prefix`` must match
  ``[a-z0-9][a-z0-9_-]*`` (SIMIND lowercases file names).
- Publishing waits for the tests and checks the tag against the version; tests
  also run on ``release/**`` branches.

1.1.0 - 2026-10-11
------------------

Replaces 2.0.0 and 3.0.0, which were published by mistake and have been
withdrawn.

Breaking Changes
~~~~~~~~~~~~~~~~

- The projection bin size and matrix (Index 28, 76, 77) come from the
  configuration. ``configure_voxel_phantom`` no longer sets them from the
  phantom image, which made the projections square whenever the image slices
  were square and took the number of axial rows from the image height.
- ``configure_voxel_phantom`` keeps the configured scoring routine (Index 84)
  unless ``scoring_routine`` is given.
- ``STIRSPECTAcquisitionDataBuilder`` no longer flips projection data along the
  last axis.
- Runtime-operator switches passed to ``run()`` apply to that run only. They
  are no longer kept on the connector for later runs.
- Each run deletes earlier outputs that share the output prefix before SIMIND
  starts.
- The STIR backend needs ``stirextra`` as well as ``stir``.
- STIR/SIRF adaptors take the z voxel spacing from a different axis (see
  Fixed). Results change for anisotropic voxels.
- Invalid inputs now raise ``ValueError``: empty, non-finite or negative source
  and attenuation maps, array shapes that do not match the header, and
  ``output_prefix`` values that contain a path. Energy-window validation raises
  ``ValueError`` instead of ``AssertionError``.

Added
~~~~~

- ``run()`` warns when a voxel phantom is wider or longer than the projection
  field of view.

Fixed
~~~~~

- The packaged ``configs/input.smc`` is now included in the wheel. SMC parsing
  handles fixed-width fields and checks section counts.
- STIR/SIRF adaptors read the z voxel spacing from the correct axis:
  ``voxel_sizes()`` is ``(z, y, x)``, and raw ``get_grid_spacing()`` is
  ``(unused, z, y, x)``.
- Voxel-map dimension indices 78, 79, 81 and 82 are set correctly.
- The executor resolves ``SIMIND_BIN`` and accepts executable paths that
  contain spaces. The connector no longer changes the process working directory.
- Runtime switches apply to one run only. Output prefixes cannot escape the
  output directory. Stale outputs are removed, and ``.win``, ``.smi`` and
  ``.dmi`` inputs are kept.
- Energy-window validation raises ``ValueError`` instead of failing an
  ``assert``.
- The Interfile loader honours data offsets, strips quoted values and rejects
  truncated or oversized payloads. The SIMIND-to-STIR converter keeps data-file
  lines and applies the radius scale factor.
- Attenuation utilities handle the ``rho*1000`` selector and STIR ``#`` header
  keys, and reject unknown attenuation types.
- Image and acquisition builders validate array shapes, split multi-energy data
  deterministically and no longer flip images. The DICOM builder handles missing
  timing information and 2D pixel data.
- Backend adaptors are imported lazily, so the core package imports without
  STIR, SIRF or PyTomography.

Removed
~~~~~~~

- The legacy ``scripts/simulation.py``.
- The "non-square matrix" warning from
  ``SimulationConfig.validate_parameters``; SIMIND supports non-square
  projections.

1.0.1 - 2026-04-16
------------------

Fixed
~~~~~

- Corrected installation documentation to lead with the PyPI package install
  path instead of editable development installs.
- Clarified that SIMIND is an external runtime dependency and must be available
  as ``simind`` on ``PATH`` for local connector runs.
- Updated contributing and testing documentation to use Ruff for linting,
  import sorting, and formatting checks.
- Updated examples documentation to include the DICOM-driven adaptor examples.
- Fixed the GitHub Actions coverage target after the import package rename.
- Fixed README links and release metadata for the patch release.

1.0.0 - 2026-04-16
------------------

**Breaking Changes**
~~~~~~~~~~~~~~~~~~~~

**Connector-first public API**
  The legacy simulator-oriented surface has been removed in favor of
  connector/adaptor entry points:

  - Removed ``SimindSimulator`` from top-level exports.
  - Removed ``NativeBackendConnector`` from connector exports.
  - Removed legacy core runtime modules
    (``core.simulator``, ``core.backend_adapter``, ``core.components``,
    ``core.file_managers``, ``core.output_processor``).

**PyTomography adaptor boundary tightening**
  ``PyTomographySimindAdaptor`` no longer wraps reconstruction-package helper
  methods:

  - Removed ``build_system_matrix()``
  - Removed ``get_pytomography_metadata()``

  System-matrix creation is now performed directly with PyTomography APIs in
  user code/examples.

Added
~~~~~

- ``core.executor.SimindExecutor`` as a minimal subprocess runner for SIMIND.
- Expanded unit coverage for connector/adaptor behavior:
  - native STIR/SIRF adaptor validation and forwarding behavior
  - PyTomography adaptor connector wiring and axis-order forwarding
  - additional ``configure_voxel_phantom`` edge cases in Python connector tests
- Geometry documentation note for STIR/SIRF voxel-size extraction fallback
  behavior.

Changed
~~~~~~~

- Renamed the PyPI distribution to ``simind-python-connector``.
- Renamed the import package to ``simind_python_connector``.
- Updated user-facing branding to ``simind-python-connector`` and added explicit non-affiliation
  and external-SIMIND-install disclaimers in README/docs.
- ``SimindPythonConnector`` is now the central execution path for adaptor
  workflows.
- ``StirSimindAdaptor`` and ``SirfSimindAdaptor`` now delegate SIMIND setup/run
  through ``SimindPythonConnector`` and return native backend projection types.
- ``PyTomographySimindAdaptor`` remains focused on SIMIND I/O adaptation and
  tensor conversion only.
- README and docs pages were rewritten with user-facing guidance for connector
  and adaptor workflows.

Fixed
~~~~~

- STIR/SIRF voxel-size extraction fallback now supports both 4-element and
  3-element ``get_grid_spacing()`` outputs, plus non-iterable coordinate
  objects (for example, STIR ``Float3BasicCoordinate``) for z-spacing
  extraction.
- ``SimindExecutor`` command invocation hardened with explicit token validation
  and ``shell=False`` execution.

[0.4.0] - 2026-02-26
--------------------

**Breaking Changes**
~~~~~~~~~~~~~~~~~~~~

**Example suite renumbering and adaptor split**
  The old monolithic OSEM examples were replaced with backend-specific adaptor
  examples:

  - ``07A_stir_adaptor_osem.py``
  - ``07B_sirf_adaptor_osem.py``
  - ``07C_pytomography_adaptor_osem.py``
  - ``08A/08B/08C`` DICOM-driven adaptor examples

  Legacy numbering and naming references should be updated to the new files.

**Connector/adaptor naming normalization**
  Public docs and examples now consistently use:

  - ``SimindPythonConnector`` for backend-agnostic NumPy workflows
  - ``StirSimindAdaptor`` / ``SirfSimindAdaptor`` / ``PyTomographySimindAdaptor``
    for backend-native workflows

Added
~~~~~

- Geometry documentation for SIMIND/STIR/SIRF/PyTomography axis conventions
  (``docs/geometry.rst``).
- Container scripts for split validation and examples:
  ``scripts/run_container_validation.sh`` and
  ``scripts/run_container_examples.sh``.
- Backend-isolation and connector/adaptor unit tests (markers + import gating).
- Example profile config (``configs/Example.yaml``) and Python-connector helper
  flow for lightweight deterministic runs.

Changed
~~~~~~~

- Core examples 01-06 now use the pure Python connector path.
- OSEM examples now use dedicated adaptor paths and backend-specific outputs.
- Documentation updated for new example numbering, runtime flags, and testing
  entry points.
- Versioning aligned across package metadata and docs.

Fixed
~~~~~

- PENETRATE component header parsing via ``.bXX`` outputs in the Python
  connector.
- SIMIND optional-availability handling in container scripts (fail-fast option
  plus skip-by-default behavior when binary is absent).
- Quantization scaling docs and naming consistency (``quantization_scale``).

[0.3.0] - 2025-01-23
--------------------

**Breaking Changes**
~~~~~~~~~~~~~~~~~~~~

**Backend-Agnostic Public API**
  All public methods now return wrapped backend-agnostic objects instead of native SIRF/STIR objects:

  - ``SimindSimulator.get_total_output()`` → Returns ``AcquisitionDataInterface``
  - ``SimindSimulator.get_scatter_output()`` → Returns ``AcquisitionDataInterface``
  - ``SimindSimulator.get_primary_output()`` → Returns ``AcquisitionDataInterface``
  - ``SimindSimulator.get_penetrate_output()`` → Returns ``AcquisitionDataInterface``
  - ``SimindToStirConverter.convert(return_object=True)`` → Returns ``AcquisitionDataInterface``
  - ``convert_simind_to_stir()`` → Returns ``AcquisitionDataInterface``

  **Migration**: Use ``.native_object`` property to access underlying SIRF/STIR objects if needed for backend-specific operations. Wrapper objects support common operations (``.write()``, ``.sum()``, ``.as_array()``, etc.).

**Backend-Agnostic extract_attributes_from_stir()**
  Function now only accepts filepath (string), not acquisition data objects:

  - **Before**: ``extract_attributes_from_stir(acq_obj)`` or ``extract_attributes_from_stir(filepath)``
  - **After**: ``extract_attributes_from_stir(filepath)`` only
  - **Migration**: If you have an acquisition object, write it to a file first: ``acq_obj.write(temp_path); extract_attributes_from_stir(temp_path)``

Changed
~~~~~~~

**Enhanced Factory Functions**
  Factory functions now accept multiple input types:

  - String filepath → Loads and wraps
  - Already wrapped object → Returns as-is (idempotent)
  - Native SIRF/STIR object → Wraps it
  - ``None`` → Creates empty object (SIRF only)

**Internal Template Storage**
  ``SimindSimulator`` now stores both template filepath (backend-agnostic) and wrapped object for efficient operations.

Fixed
~~~~~

**Critical: Coordinator Update Timing Fix**
  Fixed off-by-one error in coordinator update scheduling:

  - **Issue**: Coordinator updates were occurring at iteration N, but callbacks (UpdateEtaCallback, ArmijoTriggerCallback) ran AFTER the step was taken, causing Armijo line search to run one iteration late with a mismatched objective function
  - **Fix**: Modified ``SimindCoordinator.should_update()`` and ``StirPsfCoordinator.should_update()`` to trigger updates ONE iteration early (at ``correction_update_interval - 1`` instead of ``correction_update_interval``)
  - **Impact**: Ensures proper callback ordering for convergent optimization
  - **Result**: Eliminates divergence caused by taking steps with wrong objective, ensures convexity is maintained

Added
~~~~~

**Backend Abstraction Layer**
  New backend system supporting both SIRF and STIR Python:

  - Factory functions: ``create_image_data()``, ``create_acquisition_data()``
  - Utility functions: ``get_backend()``, ``set_backend()``, ``is_sirf_backend()``
  - Wrapper classes: ``SirfImageData``, ``SirfAcquisitionData``, ``StirImageData``, ``StirAcquisitionData``
  - See :doc:`backends` for complete documentation

**Utility Functions for STIR Compatibility**
  - ``to_projdata_in_memory()``: Convert ProjData to ProjDataInMemory for arithmetic operations (STIR Python compatibility)
  - Fixes arithmetic operations (``+``, ``-``, ``*``) on STIR ProjData objects
  - Works transparently with both SIRF and STIR backends

**Example Updates**
  - Updated examples 01-06 to support both SIRF and STIR Python backends
  - Added ``--backend`` command-line option to force specific backend
  - Fixed arithmetic operations in example 03 (TEW correction)

**ShiftedKullbackLeibler Support**
  - Added conditional support for SETR's ``ShiftedKullbackLeibler`` in data partitioning
  - New ``use_shifted_kl`` parameter in ``partition_data_with_cil_objectives()`` function
  - Automatic fallback to standard ``KullbackLeibler`` when SETR is not available

**ResidualCorrectedKullbackLeibler Jordan Split**
  - Residual-aware KL now decomposes residuals into positive/negative parts
  - Keeps the objective convex and allows arbitrary residual signs

Changed
~~~~~~~

**Documentation Organization**
  - Moved BACKENDS.md to docs/backends.rst (reStructuredText format)
  - Updated docs/changelog.rst with comprehensive change history
  - Added backends page to documentation index

Version 0.2.1
-------------

New Features
~~~~~~~~~~~~

**Schneider2000 Density Conversion**
  Advanced 44-segment piecewise model for HU-to-density conversion:

  - ``hu_to_density_schneider()``: Interpolated conversion using all 44 tissue segments
  - ``hu_to_density_schneider_piecewise()``: Exact piecewise conversion matching lookup table
  - ``get_schneider_tissue_info()``: Lookup tissue information for specific HU values
  - ``compare_density_methods()``: Compare bilinear vs Schneider methods
  - Enhanced Accuracy: ~0.17-0.19 g/cm³ improved accuracy over bilinear model
  - Comprehensive Tissue Support: Covers air, lung variations, soft tissues, bones, and metal implants
  - New Example: ``06_schneider_density_conversion.py`` demonstrates advanced density conversion

**Coordinator Architecture**
  Efficient subset reconstruction with shared Monte Carlo corrections:

  - ``SimindCoordinator``: Manages single SIMIND simulation for all subsets
  - ``SimindSubsetProjector``: Projector for individual subsets
  - CIL partitioner utilities for subset-based reconstruction
  - 12× faster for 12-subset reconstruction
  - See coordinator-related API/docs sections for implementation details

**SimindProjector AcquisitionModel Interface**
  Drop-in replacement for SIRF AcquisitionModel with Monte Carlo corrections:

  - Complete AcquisitionModel API compatibility
  - Three correction modes (residual only, additive only, both)
  - Automatic iteration tracking
  - Intelligent scaling strategy
  - CIL framework compatibility

Improvements
~~~~~~~~~~~~

- Extended documentation with density conversion methods comparison
- Comprehensive test suite for Schneider functionality (16 new tests)
- Enhanced attenuation conversion utilities with clinical-grade accuracy
- MPI parallel execution support for SIMIND simulations
- STIR CUDARelativeDifferencePrior integration

Fixed
~~~~~

**Critical: Non-Circular Orbit Support**
  Multiple fixes for proper orbit file handling:

  - Issue 1: Missing orbit file creation (AcquisitionData.get_info() doesn't include orbit/radii)
  - Issue 2: Missing unit conversion (radii in mm instead of cm)
  - Issue 3: File naming collision (input orbit file overwritten)
  - Issue 4: Incorrect command argument order
  - Issue 5: Full path instead of filename

**Critical: Subset View Index Out of Range**
  - Fixed coordinator receiving subset acquisition models but attempting to extract using full-data indices
  - Coordinator now requires full-data acquisition models passed to ``__init__()``

**Module Import Issues**
  - Fixed RecursionError in lazy imports (``converters``, ``utils``, ``builders`` modules)
  - Wrapped SIRF imports in try/except blocks for better fallback behavior

Version 0.2.0
-------------

Breaking Changes
~~~~~~~~~~~~~~~~

- Modified config file loading mechanism in SimulationConfig class
- Updated API for configuration initialization

New Features
~~~~~~~~~~~~

- Comprehensive test suite with unit and integration tests
- Enhanced documentation with ReadTheDocs support
- GitHub Actions CI/CD pipeline
- Auto-generated API documentation
- Professional documentation structure

Improvements
~~~~~~~~~~~~

- Better test coverage for all components
- Improved code quality with automated checks
- Streamlined README for better user experience

Version 0.1.1
-------------

- Bug fixes and minor improvements from initial release

Version 0.1.0
-------------

- Initial release with core functionalities and examples
