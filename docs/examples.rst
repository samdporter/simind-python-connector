Examples
========

Examples 01–08 are in the :doc:`gallery <auto_examples/index>`: runnable
scripts with their figures and printed output. Follow :doc:`tutorial` for the
recommended order, prerequisites, and what to inspect.

Build the gallery locally with ``bash scripts/build_gallery.sh``. Read the Docs
shows the committed outputs and does not run the examples.

Examples 09–13 are runnable scripts at the repository root:

- ``09_mc_scatter_osem.py`` and ``11_residual_correction.py`` use SIRF and
  SIMIND; ``10_mc_scatter_cil.py`` also requires CIL.
- ``12_phantoms.py`` uses phantomgen's known-activity NEMA/IEC phantom when
  available, otherwise SIMIND's built-in multiple-inserts phantom.
- ``13_nema_reconstruction.py --backend stir|sirf`` simulates and reconstructs
  the same known-activity phantom with the selected backend.

Run them from the repository root, for example:

.. code-block:: bash

    python examples/12_phantoms.py
    python examples/13_nema_reconstruction.py --backend sirf

Examples that simulate need a working SIMIND installation, and backend examples
need that backend installed. Gallery outputs are written under
``$SIMIND_EXAMPLES_OUTPUT`` (default ``./output``). Lower ``quantization_scale``,
fewer views, or a mostly-zero source map can shorten a smoke run; quantitative
runs need appropriate statistics and calibration.
