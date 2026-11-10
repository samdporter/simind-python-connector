Residual correction with SIMIND
===============================

Prerequisites
-------------

SIMIND, SIRF (with STIR), NumPy, Matplotlib and phantomgen.

Cost and runtime
----------------

The shared expectation takes about an hour with 8 MPI processes. The two
corrected arms each make three estimates per realisation, so this example takes
about an hour in all. Outputs are written under
``$SIMIND_EXAMPLES_OUTPUT/residual_correction``.

.. literalinclude:: /examples/mc_scatter/plot_13_residual_correction.py
   :language: python
   :linenos:

Download the script: :download:`plot_13_residual_correction.py
</examples/mc_scatter/plot_13_residual_correction.py>`.

Results
-------

.. figure:: /_static/examples/residual_correction/reconstructions.png
   :alt: Truth, fast-only, scatter-corrected and residual-corrected slices
   :width: 100%

   Figure 1. Truth, fast-only OSEM, SIMIND scatter and full residual
   reconstructions, relative to background.

.. figure:: /_static/examples/residual_correction/recovery.png
   :alt: Contrast recovery bars for the three reconstruction arms
   :width: 100%

   Figure 2. Contrast recovery with standard error over the three Poisson
   realisations.

The residual arm recovers 0.37/0.41/0.54/0.60 against 0.36/0.43/0.56/0.61 for
the scatter arm and 0.33/0.38/0.49/0.54 for fast-only OSEM. Its background CoV
is 0.91, against 0.69 for scatter and 0.64 for fast-only, so the small 17 mm
gain comes with much higher background variability. Only the scatter arm
smooths its estimate over 3 bins and views; the residual arm is unsmoothed, so
the CoV difference is not a like-for-like smoothing comparison. The recorded effective
objective becomes less negative after each residual update (for example
-3.945e6, -3.729e6, -3.672e6 for seed 12345), so the accurate-model fit
worsens rather than improving. This is not an unqualified improvement over
scatter subtraction.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
