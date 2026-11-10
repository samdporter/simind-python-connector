Normalising the Monte Carlo scatter
===================================

Compare five ways to put a Monte Carlo scatter estimate on the projection
count scale:

- known activity and acquisition time;
- reference-primary scaling by a ratio of sums;
- reference-primary scaling by trimmed bin ratios;
- global scaling to measured totals;
- pixel-wise scaling by a smoothed scatter-to-total fraction.

Requires SIMIND, SIRF, NumPy, Matplotlib and ``phantomgen``. See the
:doc:`shared setup <index>`; allow about two hours after the shared
expectation is available.

Download the script: :download:`plot_12_scatter_normalisation.py
</examples/mc_scatter/plot_12_scatter_normalisation.py>`.

.. figure:: /_static/examples/scatter_normalisation/scale_scatter.png
   :alt: Scale factor and raw scatter sum for the five normalisations
   :width: 100%

   Figure 1. Scale factors and raw scatter sums at each update. Known activity
   needs no scale factor.

.. figure:: /_static/examples/scatter_normalisation/recovery.png
   :alt: Contrast recovery bars for the five normalisations and no correction
   :width: 100%

   Figure 2. Hot-sphere recovery; error bars show standard error over three
   Poisson realisations.

Raw scatter sums are 5.686e5 (known activity), 5.915e5 (reference sum), 5.968e5
(reference trimmed), 5.730e5 (global) and 5.695e5 (pixel-wise), against a true
5.703e5.

All five methods improve recovery over no correction. Agreement in scatter
total alone does not establish spatial accuracy.
