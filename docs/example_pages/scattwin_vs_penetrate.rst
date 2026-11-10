SCATTWIN vs PENETRATE
=====================

Compare SCATTWIN's primary/scatter split with PENETRATE's interaction
components. Requires SIMIND, NumPy and Matplotlib.

The script runs each routine for one view with ``NN=120``.

Download the script: :download:`plot_05_scattwin_vs_penetrate_comparison.py
</examples/getting_started/plot_05_scattwin_vs_penetrate_comparison.py>`.

.. figure:: /_static/examples/scattwin_vs_penetrate/decomposition.png
   :alt: Stacked SCATTWIN bars beside the PENETRATE component table
   :width: 100%

   Figure 1. SCATTWIN primary and scatter beside PENETRATE's component
   breakdown.

PENETRATE distinguishes geometric primary, penetration and collimator
interactions. The printed checks compare each routine's component sum with its
own total, and each matches.
