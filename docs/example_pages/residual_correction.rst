Residual correction with SIMIND
===============================

Compare OSEM with no correction, a SIMIND scatter term, and a full residual:
SIMIND's total projection minus the fast model's projection.

Requires SIMIND, SIRF, NumPy, Matplotlib and ``phantomgen``. See the
:doc:`shared setup <index>`. Each corrected reconstruction makes three estimates
per Poisson realisation.

Download the script: :download:`plot_13_residual_correction.py
</examples/mc_scatter/plot_13_residual_correction.py>`.

.. figure:: /_static/examples/residual_correction/reconstructions.png
   :alt: Truth, fast-only, scatter-corrected and residual-corrected slices
   :width: 100%

   Figure 1. Truth, uncorrected OSEM, scatter correction and full residual
   correction.

.. figure:: /_static/examples/residual_correction/recovery.png
   :alt: Contrast recovery bars for the three reconstruction arms
   :width: 100%

   Figure 2. Hot-sphere recovery; error bars show standard error over three
   Poisson realisations.

Residual correction gives recovery 0.37 for the 17 mm sphere, compared with
0.36 for scatter correction. Background coefficient of variation is much
higher: 0.91 versus 0.69.

Only the scatter estimate is smoothed, so this comparison does not isolate the
effect of the correction model.
