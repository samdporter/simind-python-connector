Monte Carlo scatter correction in OSEM
======================================

Prerequisites
-------------

SIMIND, SIRF (with STIR), NumPy, Matplotlib and phantomgen:
``pip install "phantomgen @ git+https://github.com/varzakis/phantomgen"``.

Cost and runtime
----------------

The shared NEMA/IEC expectation takes about an hour with 8 MPI processes.
Each of the three realisations needs three SIMIND scatter estimates, each with a
fortieth of the expectation's history budget, so the corrected arm adds about
half an hour. Outputs are written under
``$SIMIND_EXAMPLES_OUTPUT/mc_scatter_osem``.

.. literalinclude:: /examples/mc_scatter/plot_11_mc_scatter_osem.py
   :language: python
   :linenos:

Download the script: :download:`plot_11_mc_scatter_osem.py
</examples/mc_scatter/plot_11_mc_scatter_osem.py>`.

Results
-------

.. figure:: /_static/examples/mc_scatter_osem/reconstructions.png
   :alt: Truth, uncorrected OSEM and SIMIND-scatter OSEM slices
   :width: 100%

   Figure 1. Mean reconstruction without and with the SIMIND scatter estimate,
   beside the truth, all relative to background.

.. figure:: /_static/examples/mc_scatter_osem/recovery.png
   :alt: Contrast recovery bars for truth, uncorrected and scatter-corrected OSEM
   :width: 100%

   Figure 2. Contrast recovery with standard error over three independent
   Poisson realisations.

The phantom is phantomgen's ``pet`` preset: 8:1 sphere-to-background activity
in MBq per voxel, four hot spheres (17, 22, 28 and 37 mm), two cold spheres
(10 and 13 mm), and named masks. SIMIND simulates one expectation with the
known total activity and 2 s per view, so the projections are in counts. Three
seeded Poisson draws (seeds 12345, 12346 and 12347) share that expectation
and both arms. Recovery is the
hot-sphere mask mean over the background mean, divided by the true ratio; the
masks include edge voxels, so the truth's own score is the ceiling. The
recorded recovery rises from 0.33 to 0.36 at 17 mm, 0.38 to 0.43 at 22 mm,
0.49 to 0.56 at 28 mm and 0.54 to 0.61 at 37 mm. Background CoV rises from
0.64 to 0.69, so the correction buys recovery at a cost in noise.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
