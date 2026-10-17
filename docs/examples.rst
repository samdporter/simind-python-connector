Examples
========

Follow :doc:`tutorial` for the recommended order, prerequisites, smoke-run
settings, and what to inspect. Each script is a runnable entry point from the
repository root. The optional backend is imported only by examples that use
it.

Core Python examples
--------------------

Examples 01–06 run without STIR, SIRF, or PyTomography:

- ``01_basic_simulation.py`` runs a small Tc-99m voxel simulation with the
  SCATTWIN with ``CA=0`` and saves the total projection.
- ``02_runtime_switch_comparison.py`` compares runtime switch settings.
- ``03_multi_window.py`` builds a triple-energy-window scatter estimate.
- ``04_custom_config.py`` creates and exports edited configurations.
- ``05_scattwin_vs_penetrate_comparison.py`` compares scoring routines. It
  requests raw SCATTWIN components with ``CA=2``; see :doc:`simind_validation`
  for checked engine compatibility.
- ``06_schneider_density_conversion.py`` demonstrates HU-to-density
  conversion.
- ``12_phantoms.py`` uses phantomgen's known-activity NEMA/IEC phantom when
  available, otherwise SIMIND's built-in multiple-inserts phantom.
- ``13_nema_reconstruction.py --backend stir|sirf`` simulates and reconstructs
  the same known-activity phantom with the selected backend.

Backend examples
----------------

- ``07A_stir_adaptor_osem.py`` uses STIR.
- ``07B_sirf_adaptor_osem.py`` uses SIRF.
- ``07C_pytomography_adaptor_osem.py`` uses PyTomography.
- ``08A_stir_adaptor_from_dicom.py`` and ``08B_sirf_adaptor_from_dicom.py``
  read a projection DICOM; ``08C_pytomography_adaptor_from_dicom.py`` uses
  PyTomography. Pass a DICOM path on the command line.
- ``09_mc_scatter_osem.py`` and ``11_residual_correction.py`` use SIRF and
  SIMIND; ``10_mc_scatter_cil.py`` also requires CIL.

Run an example from the repository root, for example:

.. code-block:: bash

    python examples/01_basic_simulation.py
    python examples/12_phantoms.py
    python examples/13_nema_reconstruction.py --backend sirf
    python examples/07B_sirf_adaptor_osem.py
    python examples/08A_stir_adaptor_from_dicom.py /path/to/projection.dcm

Core examples need Python dependencies and a working SIMIND installation when
they simulate. Backend examples need that backend installed too. The
DICOM-driven examples need a compatible measured projection DICOM. Lower
``quantization_scale``, fewer views, or a mostly-zero source map can shorten a
smoke run; quantitative runs need appropriate statistics and calibration.
