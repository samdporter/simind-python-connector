SCATTWIN vs PENETRATE
=====================

Prerequisites
-------------

A working SIMIND installation on ``PATH`` or via ``SIMIND_BIN``, NumPy and
Matplotlib.

Cost and runtime
----------------

Two SIMIND runs of one 32 x 32 view at 4 mm with ``NN=120``: one SCATTWIN run
and one PENETRATE run. The outputs are written under
``$SIMIND_EXAMPLES_OUTPUT/routine_comparison``.

.. literalinclude:: /examples/getting_started/plot_05_scattwin_vs_penetrate_comparison.py
   :language: python
   :linenos:

Download the script: :download:`plot_05_scattwin_vs_penetrate_comparison.py
</examples/getting_started/plot_05_scattwin_vs_penetrate_comparison.py>`.

Results
-------

.. figure:: /_static/examples/scattwin_vs_penetrate/decomposition.png
   :alt: Stacked SCATTWIN bars beside the PENETRATE component table
   :width: 100%

   Figure 1. SCATTWIN primary and scatter beside the PENETRATE component
   decomposition and its share of the PENETRATE total.

Both routines score the same simulated histories, so the total projection
counts agree and the printed count-preservation check is the meaningful
consistency test. The decomposition shows how PENETRATE separates geometric
collimation, septal penetration, collimator scatter, collimator X-rays and
their backscatter partners; SCATTWIN reports only primary and scatter.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
