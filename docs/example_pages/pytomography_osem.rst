PyTomography adaptor OSEM
=========================

Simulate a hot-sphere phantom and reconstruct it with PyTomography OSEM,
without and with a fixed SIMIND scatter term. Requires SIMIND, PyTomography,
PyTorch, NumPy and Matplotlib.

One low-history simulation feeds both reconstructions.

Download the script: :download:`plot_07c_pytomography_adaptor_osem.py
</examples/backends/plot_07c_pytomography_adaptor_osem.py>`.

.. figure:: /_static/examples/pytomography_osem/pytomography_osem.png
   :alt: Source, projection, two PyTomography OSEM reconstructions and a profile
   :width: 100%

   Figure 1. Source, projection, reconstructions and hot-sphere profile.

The scatter term lowers the reconstruction level but changes the
hot-to-background ratio little. Both images remain blurred because no
resolution model is used.
