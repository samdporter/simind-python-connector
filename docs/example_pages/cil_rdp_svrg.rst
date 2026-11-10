RDP regularisation with SVRG
============================

Prerequisites
-------------

SIMIND, SIRF (with STIR), CIL, NumPy, Matplotlib and phantomgen.

Cost and runtime
----------------

The shared expectation takes about an hour with 8 MPI processes. OSEM plus the
RDP/SVRG run make repeated scatter estimates and subset-gradient setups, so the
example takes about two to three hours. Outputs are written under
``$SIMIND_EXAMPLES_OUTPUT/cil_rdp_svrg``.

.. literalinclude:: /examples/cil/plot_15_cil_rdp_svrg.py
   :language: python
   :linenos:

Download the script: :download:`plot_15_cil_rdp_svrg.py
</examples/cil/plot_15_cil_rdp_svrg.py>`.

Results
-------

.. figure:: /_static/examples/cil_rdp_svrg/reconstructions.png
   :alt: Truth, OSEM and RDP-SVRG reconstruction slices
   :width: 100%

   Figure 1. Truth, OSEM and RDP+SVRG reconstructions, relative to background.

.. figure:: /_static/examples/cil_rdp_svrg/recovery.png
   :alt: Contrast recovery bars for OSEM and RDP-SVRG
   :width: 100%

   Figure 2. Contrast recovery with standard error over the three Poisson
   realisations.

The recorded demonstration uses beta 0.20 and 24 epochs. OSEM recovers
0.36/0.43/0.56/0.61 with background CoV 0.69; RDP+SVRG recovers
0.24/0.29/0.42/0.50 with CoV 0.35. The prior and SVRG smooth the image and
lower background noise at the cost of sphere recovery. The comparison also
changes the algorithm and step size, not only the prior, so it does not isolate
the prior's effect. The recorded step size is 0.1: a 0.5 step grew edge voxels
without bound with this prior and refreshed scatter.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
