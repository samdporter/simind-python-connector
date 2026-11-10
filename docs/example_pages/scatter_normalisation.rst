Normalising the Monte Carlo scatter
===================================

Prerequisites
-------------

SIMIND, SIRF (with STIR), NumPy, Matplotlib and phantomgen.

Cost and runtime
----------------

The shared expectation takes about an hour with 8 MPI processes. Five
strategies make three updated estimates on each of three realisations, so the
example takes about two hours. Outputs are written under
``$SIMIND_EXAMPLES_OUTPUT/scatter_normalisation``.

.. literalinclude:: /examples/mc_scatter/plot_12_scatter_normalisation.py
   :language: python
   :linenos:

Download the script: :download:`plot_12_scatter_normalisation.py
</examples/mc_scatter/plot_12_scatter_normalisation.py>`.

Results
-------

.. figure:: /_static/examples/scatter_normalisation/scale_scatter.png
   :alt: Scale factor and raw scatter sum for the five normalisations
   :width: 100%

   Figure 1. Scale factor (applied normalisations only) and raw scatter
   estimate sum per update. Known activity applies no scale factor and appears
   only in the sum panel.

.. figure:: /_static/examples/scatter_normalisation/recovery.png
   :alt: Contrast recovery bars for the five normalisations and no correction
   :width: 100%

   Figure 2. Contrast recovery with standard error over the three Poisson
   realisations.

All five strategies raise sphere recovery above the uncorrected arm and stay
below the truth's masked score. The recorded raw scatter sums are 5.686e5
(known activity), 5.915e5 (reference sum), 5.968e5 (reference trimmed), 5.730e5
(Dewaraja global) and 5.695e5 (Dewaraja pixel-wise), against a true 5.703e5.
Reference normalisations put the reconstruction total at 1.01 and 1.00 of the
true total; known activity and both Dewaraja variants are 1.05. Background CoV
is 0.64 uncorrected and 0.68-0.69 for the corrected arms.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
