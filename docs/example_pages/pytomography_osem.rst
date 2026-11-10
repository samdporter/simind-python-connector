PyTomography adaptor OSEM
=========================

Prerequisites
-------------

SIMIND, PyTomography and PyTorch. NumPy and Matplotlib.

Cost and runtime
----------------

One short SIMIND run with a low history budget plus two PyTomography OSEM
reconstructions. Outputs are written under
``$SIMIND_EXAMPLES_OUTPUT/pytomography_adaptor_osem``.

.. literalinclude:: scripts/plot_07c_pytomography_adaptor_osem.py
   :language: python
   :linenos:

Download the script: :download:`plot_07c_pytomography_adaptor_osem.py
<scripts/plot_07c_pytomography_adaptor_osem.py>`.

Results
-------

.. figure:: /_static/examples/pytomography_osem/pytomography_osem.png
   :alt: Source, projection, two PyTomography OSEM reconstructions and a profile
   :width: 100%

   Figure 1. Input source, projection view 0, OSEM without and with the SIMIND
   scatter term, and a profile through the hot sphere.

Object tensors are ordered (x, y, z) and projections (view, radial, axial); the
script transposes the displayed planes accordingly. The adaptor passes torch
tensors directly, and the attenuation transform uses the mu tensor. The scatter
term lowers the absolute reconstruction level but barely changes the
hot-to-background ratio; no PSF model is used, so both images remain blurred.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
