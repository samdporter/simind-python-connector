Multi-window simulation
=======================

Prerequisites
-------------

A working SIMIND installation on ``PATH`` or via ``SIMIND_BIN``, NumPy and
Matplotlib.

Cost and runtime
----------------

One SIMIND run of four 360-degree views, 32 x 32 bins of 4 mm, with ``NN=10``
and three Lu-177 energy windows. Outputs are written under
``$SIMIND_EXAMPLES_OUTPUT/multi_window``.

.. literalinclude:: scripts/plot_03_multi_window.py
   :language: python
   :linenos:

Download the script: :download:`plot_03_multi_window.py
<scripts/plot_03_multi_window.py>`.

Results
-------

.. figure:: /_static/examples/multi_window/multi_window.png
   :alt: Source and five TEW panels from the lower window to the corrected image
   :width: 100%

   Figure 1. Input source, the three windows, the TEW scatter estimate and the
   clipped corrected projection.

The photopeak keeps the hot sphere while the side windows contain a smoother
scatter background. The estimate removes counts from the photopeak and is
clipped at zero; the printed clipping fraction shows how much the noisy side
windows affected the correction. TEW assumes a linear scatter spectrum across
the windows, which is an approximation.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
