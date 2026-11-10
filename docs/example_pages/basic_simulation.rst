Basic simulation
================

Prerequisites
-------------

A working SIMIND installation on ``PATH`` or via ``SIMIND_BIN``, NumPy and
Matplotlib.

Cost and runtime
----------------

Two short SIMIND runs of four 360-degree views, 32 x 32 bins of 4 mm, with
``NN=8``. Each run writes its projection under
``$SIMIND_EXAMPLES_OUTPUT/basic_simulation``.

.. literalinclude:: /examples/getting_started/plot_01_basic_simulation.py
   :language: python
   :linenos:

Download the script: :download:`plot_01_basic_simulation.py
</examples/getting_started/plot_01_basic_simulation.py>`.

Results
-------

.. figure:: /_static/examples/basic_simulation/basic_simulation.png
   :alt: Input source, first total projection and a centre-row profile
   :width: 100%

   Figure 1. Input source, total projection in view 0 and its centre-row
   profile.

The hot sphere appears as the bright disc in the source and as the peak in the
profile. Running the same ``RR`` seed reproduces the projection exactly; the
two runs use different seeds, so their mean absolute difference is Monte Carlo
noise, not a change in geometry or physics.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
