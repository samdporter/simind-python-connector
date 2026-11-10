CIL scatter correction
======================

Prerequisites
-------------

SIMIND, SIRF (with STIR), CIL, NumPy, Matplotlib and phantomgen.

Cost and runtime
----------------

The shared expectation takes about an hour with 8 MPI processes. The final run
makes nine scatter estimates, so the example takes about an hour. Outputs are
written under ``$SIMIND_EXAMPLES_OUTPUT/cil_scatter``.

.. literalinclude:: /examples/cil/plot_14_cil_scatter.py
   :language: python
   :linenos:

Download the script: :download:`plot_14_cil_scatter.py
</examples/cil/plot_14_cil_scatter.py>`.

Results
-------

.. figure:: /_static/examples/cil_scatter/reconstructions.png
   :alt: Truth, uncorrected ISTA and scatter-corrected ISTA slices
   :width: 100%

   Figure 1. Truth, ISTA without scatter correction and ISTA with the SIMIND
   scatter estimate, relative to background.

.. figure:: /_static/examples/cil_scatter/recovery.png
   :alt: Contrast recovery bars for uncorrected and scatter-corrected ISTA
   :width: 100%

   Figure 2. Contrast recovery with standard error over the three Poisson
   realisations.

The final 48-iteration run raises recovery from 0.32/0.39/0.48/0.54 without
correction to 0.36/0.43/0.55/0.61 with the SIMIND estimate. Background CoV
rises from 0.62 to 0.68, so the correction trades noise for recovery. The
subset objectives, ``CorrectionCallback`` and refresh every 12 iterations are
the whole correction path; the final iterations use the estimate from the
previous callback.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
