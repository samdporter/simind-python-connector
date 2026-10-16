.. _changelog:

Changelog
=========

All notable changes to this project are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

1.2.0 - 2026-10-16
------------------

Changed
~~~~~~~

- **Breaking:** without the ``CA`` runtime switch, SCATTWIN runs return only
  the total projection. The energy-window file now holds exactly the requested
  windows; 1.1.x appended a scatter-order-1 line when every order was 0, which
  made SIMIND also write ``sca``, ``air`` and a duplicate ``sca_w2``. Code that
  reads ``outputs["sca_w1"]`` or ``get_scatter_output()`` needs
  ``add_runtime_switch("CA", 1)``.
- Internal tidy-up with no behaviour change: type hints use ``X | None`` and
  built-in generics, ``__all__`` lists are sorted, and redundant ``pass``
  statements are removed.

Added
~~~~~

- The ``CA`` runtime switch is accepted (``add_runtime_switch("CA", n)``). It
  selects the SCATTWIN projections SIMIND V8 writes: without it only ``tot``;
  ``1`` adds ``sca`` and ``air``; ``2`` adds ``pri``, ``sca`` and ``air``.
- A SCATTWIN run without the ``CA`` switch, and with every scatter order 0,
  issues a ``UserWarning`` saying that only total projections will be written.
- ``get_primary_output()`` returns the primary projection written with
  ``CA`` = 2.
- ``get_scatter_output()`` and ``get_primary_output()`` return the total minus
  the other component when only that one was written, and warn and return
  ``None`` when neither was.

Fixed
~~~~~

- ``number - x`` on SIRF image and acquisition wrappers returned wrong values.
- ``StirImageData`` arithmetic raised ``TypeError`` on STIR 6.2, and
  ``create_simple_phantom()`` and ``create_attenuation_map()`` failed there.
- ``StirAcquisitionData.get_info()`` returns STIR's parameter text instead of
  the SWIG object's address.
- ``from simind_python_connector.utils import *`` raised ``AttributeError``.
- Interfile data with a partial trailing pixel is rejected instead of the extra
  bytes being ignored.
- The DICOM acquisition builder no longer prints debug lines, and its warnings
  name the missing ``AcquisitionDate`` and ``AcquisitionTime`` fields.

Deprecated
~~~~~~~~~~

- STIR 6.2 and earlier, which have no image arithmetic. ``StirImageData``
  computes it in NumPy there and issues a ``DeprecationWarning``; support will
  be removed in a future release.

Notes
~~~~~

- This release supports SIMIND V8.0. SIMIND V9.0 is not supported yet: it
  changes several output formats and behaviours the connector relies on, so
  V9 support will follow in a later release.

1.1.2 - 2026-10-15
------------------

Added
~~~~~

- Voxel phantoms accept separate axial and in-plane spacing via
  ``voxel_size_mm=(z, y, x)`` in mm. SIRF/STIR adaptors read all three spacings
  from their images; PyTomography accepts the same tuple order. In-plane
  spacing must remain square (``y == x``), and scalar spacing remains supported.
- ``RuntimeOperator(timeout=...)`` takes a limit in seconds; a run that exceeds
  it stops and raises ``SimulationError``. The default remains unlimited.

Changed
~~~~~~~

- Python >=3.10 is now required (``requires-python`` and the
  metadata/classifiers are updated accordingly); previously the metadata
  claimed 3.9+.
- NumPy is capped at ``>=1.20.0,<2.0`` package-wide: this cap is a
  binary-ABI / backend-compatibility guard for the legacy 1.1.x line, because
  the STIR and SIRF bindings are compiled against NumPy 1.x and crash with
  NumPy 2. It is not a Python-version constraint.
- PENETRATE per-component fallback diagnostics distinguish a missing backend
  from a real failure: falling back because the backend is not installed is
  logged at DEBUG (quiet by default), while a load failure in an installed
  backend is logged at WARNING.
- The duplicate single-window scatter file was investigated on a real SIMIND
  V8.0 run: the extra ``.win`` line is REQUIRED (without it SIMIND emits no
  scatter/air outputs), so the behaviour is intentionally retained and is now
  documented and covered by a real-binary regression test.
- Input narrowing is deliberate, even in a patch release: the Interfile
  connector now rejects up front what it could never consume, instead of
  failing obscurely after SIMIND ran. Only the scoring routines DUMMY (0),
  SCATTWIN (1) and PENETRATE (4) are supported; LIST_MODE (2) and
  FORCED_COLLIMATION (3) are rejected at preflight. A literal 1x1x1 phantom is
  rejected (pad to at least two voxels in one dimension), as is a non-zero
  source that ``quantization_scale`` rounds to all zeros. ``output_prefix`` is
  capped at 47 characters and at most 99 energy windows are accepted: SIMIND
  V8 copies output names into a fixed 60-character buffer and silently
  truncates longer ones, dropping the final ``.h00``/``.a00`` character so no
  readable SPECT header remains. Measured on a real SIMIND V8.0 run, 49
  characters is the exact maximum prefix for ``{prefix}_tot_w1.a00`` (50
  truncates it); the 47-character cap is a conservative policy with a small
  margin below that boundary, not an engine-measured limit. The longest
  per-window name adds ``_tot_w{N}.a00``, and when every scatter order is
  below 1 the connector writes the required extra window line, which adds one
  more output set, so with the 99-window maximum the suffix is 13 characters
  and 47 + 13 = 60 fills the buffer exactly. ``MP`` takes positive integers
  (including NumPy integers), integral floats and ASCII digit strings; ``NN``
  takes positive integers and integral floats, and an integral-float ``NN`` is
  written to the command line as an integer (for example ``/NN:2``) while the
  stored switch value is unchanged. ``orbit_file`` must be a regular file.
  Index 26 (voxel history) and the ``/NN`` *simulation semantics* are
  unchanged — only validation was added.
- The direct executor validates every runtime switch value, not only ``MP``
  and ``NN``, before it builds the SIMIND command, so a caller driving
  ``SimindExecutor`` directly gets the same checks the connector applies;
  invalid values raise ``SimulationError``.
- ``output_prefix`` is validated at construction and again before every
  connector-managed write or cleanup (source/density files, the window file,
  the saved ``.smc``, stale-output removal and orbit staging), so an over-long
  prefix can no longer overwrite or delete existing ``{prefix}.smc`` /
  ``{prefix}_tot_w1.hs`` before it is rejected.
- Filesystem failures in connector-managed steps (writing the source, density
  and window inputs, staging an orbit file, clearing stale outputs,
  converting headers) raise ``OutputError``. ``OutputError`` is also an
  ``OSError``, so callers written against the pre-1.1.2 bare ``OSError``
  raises keep catching these failures. Missing SPECT/PENETRATE headers still
  raise ``FileNotFoundError`` (the error is both), and missing inputs such as
  the configuration file or the orbit file still raise ``FileNotFoundError``.

Fixed
~~~~~

- ``load_table()`` reads both bundled NIST attenuation tables, skipping their
  headers. Its documentation now gives the table units correctly: energy in
  MeV and mass attenuation coefficient in cm²/g.
- Source distributions include the fixtures and runner/container files needed
  to run the bundled tests outside a Git checkout.
- Custom energy-window filenames respect SIMIND's shorter filename buffer:
  ``output_prefix`` is limited to 15 characters when an energy-window file is
  used, preventing silently ignored energy bounds. Example 02 uses safe names.
- Native source and attenuation images must share spatial geometry, including
  origins and axis directions, before simulation. Interfile byte offsets are
  parsed exactly and fractional offsets are rejected.
- ``-`` is rejected in ``output_prefix``: SIMIND treats it as an argument
  delimiter, so the run silently produced no output.
- Orbit files are always staged as ``{prefix}_orbit.cor``, so an orbit can no
  longer overwrite the source, density or window inputs when basenames
  collide.
- SIMIND error diagnostics are detected even when SIMIND exits 0 and raise
  ``SimulationError``; previously the run failed later with a misleading
  missing-output error. The exit-0 scan also recognises
  ``No input file found for the spectra:<x>.isd``, which used to surface as a
  misleading missing-header error, only before SIMIND's ``OTHER INFORMATION``
  banner so an echoed configuration comment cannot be mistaken for the failure;
  a non-zero exit now quotes captured stdout/stderr excerpts in the
  ``SimulationError`` message.
- ``find_penetrate_h00_file`` matches only the exact prefix name or an
  underscore-boundary suffix, so overlapping prefixes such as ``s`` and ``s2``
  no longer claim each other's ``.h00`` output.
- A failure while writing a PENETRATE component header now propagates as an
  ``OutputError`` instead of being swallowed, which used to drop that
  component with only a log line; the temporary ``.hs`` template is removed
  on every path, and a cleanup failure there logs a WARNING instead of
  masking the original error.
- Reflected and unary arithmetic on the STIR wrappers no longer raises
  ``TypeError``: scalar-on-the-left operators (``2.0 + img``, ``2.0 - img``,
  ``2.0 * img``, ``2.0 / img``), unary ``-img``, and a native STIR object on
  the left of ``+``, ``-``, ``*`` or ``/`` (``native + image``,
  ``native - image``, ``native * image``, ``native / image``, and likewise for
  acquisition data) all work, because STIR's own dunders return
  ``NotImplemented`` for the wrapper and Python falls back to the wrapper's
  reflected operator. NumPy arrays and other non-scalar, non-STIR operands on
  the left are not covered.
- ``get_primary_output()`` raises a clear ``NotImplementedError`` instead of
  ``KeyError``, because SIMIND emits no standalone primary projection; before
  a successful run it still raises ``RuntimeError``.
- ``SimulationConfig`` is robust to bad or short configurations:
  ``save_file`` formats and format-checks the whole fixed-width block before
  truncating the destination, rejects values that cannot fit a 12-character
  SMC field (NaN/Inf, oversized exponents) and canonicalises ``-0.0``,
  ``print_config`` and ``validate_parameters`` handle configs shorter than the
  index tables, and YAML import validates that sections are mappings, accepts
  quoted integer indices and rejects duplicate indices after normalisation
  within each section it reads (``parameters``, ``simulation_flags``,
  ``data_files``, ``text_variables``, ``other_parameters``, ``other_flags``).
- The scoring routine is read without rounding: a fractional or non-finite
  Index 84 raises ``ValueError`` instead of being rounded, a configuration
  too short to contain Index 84 reads as DUMMY, and a scoring value or
  ``quantization_scale`` that overflows float coercion raises ``ValueError``
  instead of ``OverflowError``.
- ``stir.extra`` is preferred over the deprecated ``stirextra``; the legacy
  name is kept as a fallback for older STIR installs.
- ``MP`` (MPI process count) must be an integer >= 1; ``MP=0`` previously ran
  a single process silently.
- SIMIND reads the source and density integer maps as signed 16-bit, so the
  density cap of 65535 could wrap values above 32767 negative and silently
  corrupt the map; it is now capped at 32767. ``MAX_SOURCE`` becomes
  ``np.iinfo(np.int16).max`` (32767) and ``DEFAULT_QUANTIZATION_SCALE``
  (``500 / 32767``) keeps the legacy 500-weight source default. Explicit
  ``quantization_scale`` values are not auto-migrated: rescale a scale
  calibrated for the former 500 maximum with
  ``old_scale * DEFAULT_QUANTIZATION_SCALE``.

Known Limitations
~~~~~~~~~~~~~~~~~

- A run that exceeds its ``timeout`` kills the launcher process but may not
  reap every ``simind_mpi`` rank; there is no process-tree teardown yet.
- Primary projections are unsupported by the adaptors, so
  ``get_primary_output()`` always raises ``NotImplementedError``: SIMIND's
  standard SPECT output has no primary file, though lower-level ``/CA:2`` use
  can produce them.
- Scoring routines LIST_MODE (2) and FORCED_COLLIMATION (3) remain available
  only via lower-level use of the package.

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
