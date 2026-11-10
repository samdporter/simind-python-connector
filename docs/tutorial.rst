SIMIND workflow tutorial
========================

This tutorial follows one workflow from a local SIMIND installation to
projection data and reconstruction. Run commands from the repository root.

Choose a matching SIMIND installation
-------------------------------------

Use the executable and ``smc_dir`` data files from the same SIMIND release.
Point ``SIMIND_BIN`` at the executable and ``SMC_DIR`` at that release's data
directory:

.. code-block:: bash

    export SIMIND_BIN=/path/to/simind-v8/simind
    export SMC_DIR=/path/to/simind-v8/smc_dir/
    export PATH=/path/to/simind-v8:$PATH
    export SIMIND_VERSION=8
    python -m pip install -e '.[examples]'

SIMIND v8 is the supported engine and the default; pass ``simind_version=8`` to
a connector or adaptor to override the environment. Version selection controls
voxel-map origin handling, so a wrong value can displace the source and
attenuation map. SIMIND v9 selection is experimental; see
:doc:`simind_validation` and :doc:`installation`.

A core simulation
-----------------

Start with the :doc:`basic simulation tutorial <example_pages/basic_simulation>`:

.. code-block:: bash

    python examples/getting_started/plot_01_basic_simulation.py

It uses a small voxel phantom, one energy window, four views and ``NN=8``, and
writes its source, density and total projection under
``output/basic_simulation``. This is a teaching run; a quantitative study needs
the acquisition's activity and time, enough histories, and the intended views,
collimator, windows and detector settings.

Build the phantoms
------------------

The :doc:`phantom library tutorial <example_pages/phantom_library>` builds
SIMIND's analytic multiple-inserts phantom and the NEMA/IEC body phantom.
``phantomgen`` supplies the NEMA phantom and is installed from GitHub:

.. code-block:: bash

    python -m pip install 'phantomgen @ git+https://github.com/varzakis/phantomgen'
    python examples/phantoms/plot_09_phantom_library.py

The analytic phantom is voxelised to relative source weights; the NEMA object
has activity in MBq per voxel, density and named masks. See :doc:`phantoms` for
coordinate conventions.

Activity, histories, time, energy windows, and scoring
------------------------------------------------------

The source map controls where histories start and their relative weights.
``quantization_scale`` and ``NN`` control the history budget and Monte Carlo
noise; they do not set projection counts. Pass a known activity through Index
25 with ``connector.set_activity(...)`` or
``configure_phantom(..., time_per_projection_s=...)``. For relative weights,
normalise to a measured or analytic projection; see :doc:`normalisation`.

Low quantisation budgets round relative source weights coarsely. Use them for
execution checks and increase the budget before interpreting contrast or
activity recovery.

Energy-window bounds and order select the output windows. SCATTWIN and
PENETRATE score different interaction components, and component names depend on
the routine and SIMIND version; keep raw outputs and see :doc:`simind_validation`.
For a smoke run, reduce ``quantization_scale`` and the views, or threshold the
source. Keep ``NN >= 1``.

Reconstruct the simulated acquisition
-------------------------------------

The :doc:`Monte Carlo scatter correction tutorial
<example_pages/mc_scatter_osem>` simulates the known-activity NEMA/IEC phantom
and reconstructs it with SIRF OSEM, with and without a SIMIND scatter estimate:

.. code-block:: bash

    python examples/mc_scatter/plot_11_mc_scatter_osem.py

Install SIRF and phantomgen. The five correction tutorials share one 60-view
simulation; the first run takes about an hour and later tutorials reuse the
cached expectation. See :doc:`reconstruction` for the model and calibration
details.

- :doc:`Monte Carlo scatter correction in OSEM <example_pages/mc_scatter_osem>`
- :doc:`Normalising the Monte Carlo scatter <example_pages/scatter_normalisation>`
- :doc:`Residual correction with SIMIND <example_pages/residual_correction>`
- :doc:`CIL scatter correction <example_pages/cil_scatter>`
- :doc:`RDP regularisation with SVRG <example_pages/cil_rdp_svrg>`

The CIL tutorials additionally need CIL.

Measured projections and DICOM calibration
------------------------------------------

The :doc:`DICOM projection recipe <example_pages/dicom_recipe>` gives the STIR,
SIRF and PyTomography calls for a DICOM NM projection file. Compare the derived
geometry with the simulated header before use; a geometry match does not
calibrate absolute sensitivity. See :doc:`datasets` for the restricted data used
during development.

Real Y-90 data
--------------

The measured Y-90 material is in two pages:

- :doc:`Y-90 CT conversion <example_pages/y90_ct_conversion>`
- :doc:`Y-90 effective attenuation energy <example_pages/y90_effective_energy>`

The data are not redistributed; setting ``SIMIND_CONNECTOR_Y90_DATA`` to a
local copy is the only route. See :doc:`datasets`.
