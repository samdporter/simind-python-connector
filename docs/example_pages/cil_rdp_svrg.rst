RDP regularisation with SVRG
============================

Compare scatter-corrected OSEM with reconstruction using a relative difference
prior (RDP) and stochastic variance reduced gradient (SVRG). Requires SIMIND,
SIRF, CIL, NumPy, Matplotlib and ``phantomgen``. See the :doc:`shared setup
<index>`; allow about two to three hours.

The RDP+SVRG reconstruction uses prior strength 0.20, 24 epochs and step size
0.1.

Download the script: :download:`plot_15_cil_rdp_svrg.py
</examples/cil/plot_15_cil_rdp_svrg.py>`.

.. figure:: /_static/examples/cil_rdp_svrg/reconstructions.png
   :alt: Truth, OSEM and RDP-SVRG reconstruction slices
   :width: 100%

   Figure 1. Truth, OSEM and RDP+SVRG reconstructions.

.. figure:: /_static/examples/cil_rdp_svrg/recovery.png
   :alt: Contrast recovery bars for OSEM and RDP-SVRG
   :width: 100%

   Figure 2. Hot-sphere recovery; error bars show standard error over three
   Poisson realisations.

RDP+SVRG lowers background coefficient of variation from 0.69 to 0.35, but
lowers 37 mm sphere recovery from 0.61 to 0.50. The comparison changes the
algorithm and step size as well as adding the prior; it does not isolate the
prior's effect.
