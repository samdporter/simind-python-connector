STIR adaptor OSEM
=================

Prerequisites
-------------

SIMIND and STIR Python bindings. NumPy and Matplotlib.

Cost and runtime
----------------

One short SIMIND run with a low history budget plus two STIR OSEM
reconstructions. Outputs are written under
``$SIMIND_EXAMPLES_OUTPUT/stir_adaptor_osem``.

.. literalinclude:: /examples/backends/plot_07a_stir_adaptor_osem.py
   :language: python
   :linenos:

Download the script: :download:`plot_07a_stir_adaptor_osem.py
</examples/backends/plot_07a_stir_adaptor_osem.py>`.

Results
-------

.. figure:: /_static/examples/stir_osem/stir_osem.png
   :alt: Source, projection, two STIR OSEM reconstructions and a profile
   :width: 100%

   Figure 1. Input source, projection view 0, OSEM without and with the SIMIND
   scatter term, and a profile through the hot sphere.

The adaptor takes STIR images directly. STIR geometry is in mm while SIMIND
geometry is in cm, and the adaptor converts between them. SPECTUB mirrors x
relative to the attenuation image, so the script reverses the mu map for the
projector. The scatter term lowers the absolute reconstruction level but barely
changes the hot-to-background ratio; no PSF model is used, so both images
remain blurred.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
