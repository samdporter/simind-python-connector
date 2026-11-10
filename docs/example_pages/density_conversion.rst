Density conversion
==================

Prerequisites
-------------

NumPy and Matplotlib. No SIMIND run is needed.

Cost and runtime
----------------

Seconds of Python work. The figure is written under
``$SIMIND_EXAMPLES_OUTPUT/schneider_density``.

.. literalinclude:: /examples/getting_started/plot_06_schneider_density_conversion.py
   :language: python
   :linenos:

Download the script: :download:`plot_06_schneider_density_conversion.py
</examples/getting_started/plot_06_schneider_density_conversion.py>`.

Results
-------

.. figure:: /_static/examples/density_conversion/density_conversion.png
   :alt: Bilinear and Schneider density curves and their differences
   :width: 100%

   Figure 1. Bilinear and interpolated Schneider density over the full and
   clinical HU ranges, with their differences.

The two models agree over soft tissue and diverge in lung, bone and metal. The
printed maximum differences are model differences over the sampled HU grid;
they are not an accuracy claim for either conversion.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
