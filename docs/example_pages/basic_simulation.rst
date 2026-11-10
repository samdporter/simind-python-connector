Basic simulation
================

Simulate a hot sphere in a cylindrical background and compare two random
seeds. Requires SIMIND, NumPy and Matplotlib.

The script makes two short runs with four views and ``NN=8``.

Download the script: :download:`plot_01_basic_simulation.py
</examples/getting_started/plot_01_basic_simulation.py>`.

.. figure:: /_static/examples/basic_simulation/basic_simulation.png
   :alt: Input source, first total projection and a centre-row profile
   :width: 100%

   Figure 1. Source, first projection and centre-row profile.

The sphere produces a peak in the projection profile. Differences between the
two runs are Monte Carlo noise; repeating a seed reproduces its projection.
