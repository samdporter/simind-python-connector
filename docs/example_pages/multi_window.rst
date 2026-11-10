Multi-window simulation
=======================

Estimate scatter with the triple-energy-window method (TEW). Requires SIMIND,
NumPy and Matplotlib.

One run produces three Lu-177 energy windows over four views.

Download the script: :download:`plot_03_multi_window.py
</examples/getting_started/plot_03_multi_window.py>`.

.. figure:: /_static/examples/multi_window/multi_window.png
   :alt: Source and five TEW panels from the lower window to the corrected image
   :width: 100%

   Figure 1. Source, energy-window projections, TEW scatter estimate and
   corrected projection.

Subtracting the side-window estimate reduces the photopeak background. Negative
corrected bins are clipped to zero. TEW assumes a linear scatter spectrum across
the windows.
