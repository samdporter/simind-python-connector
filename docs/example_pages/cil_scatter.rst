CIL scatter correction
======================

Add an updated SIMIND scatter estimate to a CIL ISTA reconstruction. Requires
SIMIND, SIRF, CIL, NumPy, Matplotlib and ``phantomgen``. See the
:doc:`shared setup <index>`.

Each reconstruction runs 48 iterations, refreshing scatter at iterations 12, 24
and 36.

Download the script: :download:`plot_14_cil_scatter.py
</examples/cil/plot_14_cil_scatter.py>`.

.. figure:: /_static/examples/cil_scatter/reconstructions.png
   :alt: Truth, uncorrected ISTA and scatter-corrected ISTA slices
   :width: 100%

   Figure 1. Truth and ISTA reconstructions without and with scatter correction.

.. figure:: /_static/examples/cil_scatter/recovery.png
   :alt: Contrast recovery bars for uncorrected and scatter-corrected ISTA
   :width: 100%

   Figure 2. Hot-sphere recovery; error bars show standard error over three
   Poisson realisations.

Correction raises recovery from 0.32 to 0.36 for the 17 mm sphere and from 0.54
to 0.61 for the 37 mm sphere. Background coefficient of variation rises from
0.62 to 0.68.
