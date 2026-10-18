SIMIND workflow tutorial
========================

This tutorial follows one small workflow from a local SIMIND installation to
projection data and reconstruction. Run commands from the repository root.
SIMIND is licensed separately and is not included with this package.

Choose a matching SIMIND installation
-------------------------------------

Use the executable and ``smc_dir`` data files from the same SIMIND release.
Point ``SIMIND_BIN`` at the executable and ``SMC_DIR`` at that release's data
directory. The executor uses ``SIMIND_BIN`` when set and otherwise searches
``PATH``. For the repository's container scripts, the supported layout is
``./simind/simind`` with data in ``./simind/smc_dir/``.

.. code-block:: bash

    export SIMIND_BIN=/path/to/simind-v9/simind
    export SMC_DIR=/path/to/simind-v9/smc_dir/
    export PATH=/path/to/simind-v9:$PATH
    export SIMIND_VERSION=9
    python -m pip install -e '.[examples]'
    python -c 'import shutil; print(shutil.which("simind"))'

Set ``SIMIND_VERSION=8`` for v8. The default is 8. In Python, pass
``simind_version=8`` or ``simind_version=9`` to the connector or adaptor to
override the environment. Version selection also controls voxel-map origin
handling; using the wrong value can displace the source and attenuation map.

For SCATTWIN, use v8 or the macOS v9 build: the v9.0 Linux build can abort or
silently return corrupted projections (see :doc:`simind_validation`).

SIMIND releases may differ in their data files and output components. Check
your local release manual and the observations in :doc:`simind_validation`
before relying on a particular component name. Keep each binary paired with
its own data directory. Start with the core NumPy example:

.. code-block:: bash

    python examples/getting_started/plot_01_basic_simulation.py

It uses a small voxel phantom, one energy window, few projections, and
``NN=1``. Inspect the saved source, density, and total projection under
``output/basic_simulation``. The example requests SCATTWIN ``CA=0`` for its
126–154 keV window, which is available across the checked Linux builds. This is a
smoke run: its noisy counts are not a quantitative reference. For a quantitative
run, use the activity and
time for the acquisition, enough histories for stable statistics, the intended
number of views, and the matching collimator, windows, and detector settings.

Build a known-activity NEMA phantom
-----------------------------------

Example 12 uses `phantomgen <https://github.com/varzakis/phantomgen>`_ when
installed. It is optional and is currently installed from GitHub, not PyPI.

.. code-block:: bash

    python -m pip install 'phantomgen @ git+https://github.com/varzakis/phantomgen'
    python examples/12_phantoms.py

The example creates the NEMA/IEC body phantom with explicit activity in MBq per
voxel, density, and named masks. Inspect the printed activity summed within
each mask and the projection sum. Open ``output/phantoms`` to inspect the
written maps and projection. If phantomgen is absent, the example runs
SIMIND's built-in multiple-inserts phantom instead; that fallback is useful
for geometry, but it does not have phantomgen's known-activity ground truth.

For a quick smoke run, reduce ``quantization_scale`` and the number of views,
or threshold the source to a mostly-zero map. Use ``NN >= 1``. For quantitative
work, retain the full phantom and set histories and acquisition settings to
the study protocol. See :doc:`phantoms` for coordinate conventions and
:doc:`normalisation` for the distinction between activity and histories.

Activity, histories, time, energy windows, and scoring
------------------------------------------------------

The source map controls where histories start and their relative weights.
``quantization_scale`` and ``NN`` control the history budget and Monte Carlo
noise. They do not set the number of projection counts. The known NEMA activity
is passed through Index 25 as total MBq multiplied by seconds per projection.
For another known-activity source, use ``connector.set_activity(...)`` or
``configure_phantom(..., time_per_projection_s=...)``. For a source containing
only relative weights, normalise the simulation to a measured projection or
to an analytic primary projection; see :doc:`normalisation`.

Low quantisation budgets also round relative source weights coarsely. Use the
quick settings for execution checks; verify region activity ratios and increase
the budget before interpreting contrast or activity recovery.

Energy-window bounds and order select the output windows. SCATTWIN and
PENETRATE score different interaction components. Keep the raw SIMIND outputs
when comparing routines or releases. Example 05 requests SCATTWIN ``CA=2`` and
reads its raw total, scatter, and primary components; this mode works in the
checked v8 build and the macOS ARM v9 build. The checked Linux Intel v9 build
aborts with an engine error for that mode, so use its total-only ``CA=0`` example
or PENETRATE workflow there. The output component names depend on scoring
routine and SIMIND version; see :doc:`simind_validation` for the observed map.

Reconstruct the simulated acquisition
-------------------------------------

Run the same known-activity NEMA simulation through either backend:

.. code-block:: bash

    python examples/13_nema_reconstruction.py --backend stir
    python examples/13_nema_reconstruction.py --backend sirf

Install only the selected backend; see :doc:`installation`. The script uses
phantomgen, low source quantisation, and ``NN=1`` for a quick smoke run. Inspect
the activity and density arrays, projection header, and reconstruction under
``output/nema_reconstruction``. The short OSEM run confirms array shapes and
returns a finite image after two OSEM subiterations. Matching dimensions do not verify axis,
view-angle, or orbit alignment. The SIRF image reader also warns that STIR image
geometry information may be incorrect, so check the geometry fields
independently. For quantitative work, increase histories and iterations and
validate voxel spacing, view angles, center of rotation, axis orientation, and
absolute calibration. The observed SIMIND v8 Linux Intel smoke (activity
568.1 MBq, 4 views, ``NN=1``) returned projection shape ``(1, 96, 4, 96)`` with
66,190 counts. STIR and SIRF each returned a finite ``(96, 96, 96)`` image after
two OSEM subiterations. These values record that smoke run only; they are not
quantitative targets or evidence of absolute sensitivity.

The same entry point also completed with ``SIMIND_VERSION=9`` against the
Linux Intel v9 build: both backends returned 67,500 projection counts and a
finite reconstruction on the same grid. The counts are rounded run summaries;
they are not cross-version equality targets.

SIMIND scatter, residuals, and CIL
----------------------------------

Examples 09 and 11 use SIRF OSEM; Example 10 uses CIL. They provide a small
SIMIND-based scatter estimate and an iterative residual-correction workflow.

.. code-block:: bash

    python examples/09_mc_scatter_osem.py
    python examples/11_residual_correction.py
    python examples/10_mc_scatter_cil.py

Example 10 additionally needs CIL. Inspect the update records and reconstruction
metrics printed by each run. These examples use an installed fast-model
objective and small Monte Carlo runs. The objective, noise, and update cadence
are demonstration choices, not evidence of quantitative accuracy. See
:doc:`reconstruction` for the model, damping, floor, and calibration details.
For quantitative work, establish an accurate reference and validate the
scatter or residual estimate against it; a faster objective remains an
approximation, and the non-zero positivity floor can affect low-count regions.

Measured projections and DICOM calibration
------------------------------------------

Examples 08A, 08B and 08C read projection geometry from the Y-90 NM DICOM
dataset (``SIMIND_CONNECTOR_Y90_DATA``, or fetched on first use):

.. code-block:: bash

    python examples/backends/plot_08a_stir_adaptor_from_dicom.py
    python examples/backends/plot_08b_sirf_adaptor_from_dicom.py
    python examples/backends/plot_08c_pytomography_adaptor_from_dicom.py

The scripts build STIR, SIRF or PyTomography projection objects and plot the
projection.
Compare the DICOM-derived geometry with the simulated header before use. A
DICOM geometry match does not calibrate absolute sensitivity. For a calibrated
comparison, document the activity, acquisition time, decay reference, energy
window, collimator, camera, and any measured projection used to scale the
simulation. Treat any resulting factor as specific to those assumptions.
