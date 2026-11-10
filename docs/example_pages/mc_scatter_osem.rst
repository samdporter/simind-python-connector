Monte Carlo scatter correction in OSEM
======================================

Compare OSEM without scatter correction against OSEM with a SIMIND scatter
estimate updated from the reconstruction.

Requires SIMIND, SIRF, NumPy, Matplotlib and ``phantomgen``. See the
:doc:`shared setup <index>` for runtime and acquisition settings.

Download the script: :download:`plot_11_mc_scatter_osem.py
</examples/mc_scatter/plot_11_mc_scatter_osem.py>`.

.. figure:: /_static/examples/mc_scatter_osem/reconstructions.png
   :alt: Truth, uncorrected OSEM and SIMIND-scatter OSEM slices
   :width: 100%

   Figure 1. Mean uncorrected and scatter-corrected reconstructions beside the
   truth.

.. figure:: /_static/examples/mc_scatter_osem/recovery.png
   :alt: Contrast recovery bars for truth, uncorrected and scatter-corrected OSEM
   :width: 100%

   Figure 2. Hot-sphere recovery; error bars show standard error over three
   Poisson realisations.

Scatter correction raises recovery from 0.33 to 0.36 for the 17 mm sphere and
from 0.54 to 0.61 for the 37 mm sphere. Background coefficient of variation
rises from 0.64 to 0.69: improved recovery comes with more noise.
