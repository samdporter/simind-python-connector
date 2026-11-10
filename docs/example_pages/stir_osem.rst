STIR adaptor OSEM
=================

Simulate a hot-sphere phantom and reconstruct it with STIR OSEM, without and
with a fixed SIMIND scatter term. Requires SIMIND, STIR, NumPy and Matplotlib.

One low-history simulation feeds both reconstructions.

Download the script: :download:`plot_07a_stir_adaptor_osem.py
</examples/backends/plot_07a_stir_adaptor_osem.py>`.

.. figure:: /_static/examples/stir_osem/stir_osem.png
   :alt: Source, projection, two STIR OSEM reconstructions and a profile
   :width: 100%

   Figure 1. Source, projection, reconstructions and hot-sphere profile.

The scatter term lowers the reconstruction level but changes the
hot-to-background ratio little. Both images remain blurred because no
resolution model is used. The script applies the SPECTUB attenuation-map flip
described in :doc:`/backends`.
