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
its own data directory. Start with the :doc:`basic simulation tutorial
<auto_examples/getting_started/plot_01_basic_simulation>`, at
``examples/getting_started/plot_01_basic_simulation.py``:

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

Build the phantoms
------------------

The :doc:`phantom library tutorial
<auto_examples/phantoms/plot_09_phantom_library>` simulates SIMIND's analytic
multiple-inserts phantom and the NEMA/IEC body phantom. The NEMA phantom comes
from `phantomgen <https://github.com/varzakis/phantomgen>`_, which is optional
and is currently installed from GitHub, not PyPI:

.. code-block:: bash

    python -m pip install 'phantomgen @ git+https://github.com/varzakis/phantomgen'
    python examples/phantoms/plot_09_phantom_library.py

The NEMA/IEC body phantom has explicit activity in MBq per voxel, density, and
named masks. The example builds it before either simulation, so a missing
phantomgen stops there rather than after the analytic run. It writes both
projections under ``output/phantom_library`` and plots a central source slice
and the first projection view of each phantom. The 24 views and low history
count make the projections noisy.

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

The :doc:`NEMA simulation and OSEM tutorial
<auto_examples/phantoms/plot_10_nema_osem>` simulates the known-activity
NEMA/IEC phantom and reconstructs it with SIRF OSEM:

.. code-block:: bash

    python examples/phantoms/plot_10_nema_osem.py

Install SIRF and phantomgen; see :doc:`installation`. The tutorial simulates 60
views with 8 MPI processes, adds seeded Poisson noise, and reconstructs with
SPECTUB attenuation but no collimator blur or scatter. That model is close to
the one that made the data by construction, so the reported sphere recovery is
a best case. Inspect the measured counts, the reconstruction and truth slices,
and the contrast-recovery bars under ``output/nema_osem``. It scores the 17-37
mm hot spheres only; the 10 and 13 mm spheres are cold. The run takes a few
minutes, nearly all of it in SIMIND. Matching dimensions do not verify axis,
view-angle, or orbit alignment, and a finite image is not evidence of absolute
sensitivity. For quantitative work, increase histories and iterations and
validate voxel spacing, view angles, center of rotation, axis orientation, and
absolute calibration.

Monte Carlo scatter and residual correction
-------------------------------------------

The Monte Carlo tutorials simulate the NEMA/IEC phantom and reconstruct it with
SIRF or CIL. Each takes about 10–40 minutes, nearly all of it in SIMIND:

- :doc:`Monte Carlo scatter correction in OSEM
  <auto_examples/mc_scatter/plot_11_mc_scatter_osem>`
- :doc:`Normalising the Monte Carlo scatter
  <auto_examples/mc_scatter/plot_12_scatter_normalisation>`
- :doc:`Residual correction with SIMIND as the accurate model
  <auto_examples/mc_scatter/plot_13_residual_correction>`
- :doc:`Monte Carlo scatter correction with CIL
  <auto_examples/cil/plot_14_cil_scatter>`
- :doc:`RDP-regularised reconstruction with SVRG
  <auto_examples/cil/plot_15_cil_rdp_svrg>`

The CIL tutorials additionally need CIL. Inspect the update records and
reconstruction metrics printed by each run. These tutorials use a fast SPECTUB
model and small Monte Carlo runs. The objective, noise, and update cadence are
demonstration choices, not evidence of quantitative accuracy. See
:doc:`reconstruction` for the model, damping, floor, and calibration details.
For quantitative work, establish an accurate reference and validate the
scatter or residual estimate against it; a faster objective remains an
approximation, and the non-zero positivity floor can affect low-count regions.

Measured projections and DICOM calibration
------------------------------------------

The :doc:`STIR <auto_examples/backends/plot_08a_stir_adaptor_from_dicom>`,
:doc:`SIRF <auto_examples/backends/plot_08b_sirf_adaptor_from_dicom>` and
:doc:`PyTomography <auto_examples/backends/plot_08c_pytomography_adaptor_from_dicom>`
DICOM tutorials read projection geometry from the Y-90 NM dataset named by
``SIMIND_CONNECTOR_Y90_DATA``; no path argument is needed (see
:doc:`datasets`). They build STIR, SIRF or PyTomography projection objects and
plot the projection.
Compare the DICOM-derived geometry with the simulated header before use. A
DICOM geometry match does not calibrate absolute sensitivity. For a calibrated
comparison, document the activity, acquisition time, decay reference, energy
window, collimator, camera, and any measured projection used to scale the
simulation. Treat any resulting factor as specific to those assumptions.

Real Y-90 data
--------------

The rest of the measured Y-90 study is in the :doc:`Real Y-90 data
<auto_examples/real_data/index>` gallery section. These pages are rendered
case studies, not generally reproducible tutorials: the data are not
redistributed at this time, there is no public download or other access
route, and the pages cannot be reproduced without the restricted input.
Setting ``SIMIND_CONNECTOR_Y90_DATA`` to a local copy is the only current
route; see :doc:`datasets` for the availability statement. Every page below
reads the data through ``fetch_y90_phantom`` and takes no path argument.

Read them in this order:

1. The DICOM pages above (examples 08A, 08B, 08C) read ``peak.dcm`` and
   build the STIR, SIRF or PyTomography projection object.

2. The CT-to-SPECT conversion case studies read the CT series of the same
   phantom, register it onto the SPECT grid, and stop with diagnostics
   instead of publishing maps when the registration check fails:

   - :doc:`CT conversion with STIR
     <auto_examples/real_data/plot_16a_stir_ct_conversion>`
   - :doc:`CT conversion with SIRF
     <auto_examples/real_data/plot_16b_sirf_ct_conversion>`
   - :doc:`CT conversion with PyTomography
     <auto_examples/real_data/plot_16c_pytomography_ct_conversion>`

   Each page uses only its own backend. The SIRF route is the reusable
   one; the later case studies take their CT preparation from it.

3. The :doc:`Y-90 effective energy case study
   <auto_examples/real_data/plot_17_y90_effective_energy>` (SIRF) compares two
   reconstructions of the measured projections that differ only in the
   attenuation model: a single 150-keV map against a map weighted across the
   detected 75-225 keV spectrum. Neither reconstruction includes scatter, so
   the pair is a model comparison, not an accuracy ranking.

4. The :doc:`Monte Carlo scatter case study
   <auto_examples/real_data/plot_18_y90_mc_scatter>` (SIRF) is an audited
   diagnostic negative result, not a passing quantitative tutorial. It compares
   Monte Carlo scatter strategies on a common reconstruction schedule, but the
   global-scaling arm collapsed and is not quantitatively interpretable, the
   nominal sphere-to-liver ratios are centre-sensitive and do not support a
   strategy ranking, and statistical adequacy was not established. Read the
   page's quality warning and gate table before using any number from it.
