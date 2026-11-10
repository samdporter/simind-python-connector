Phantom library
===============

Prerequisites
-------------

The connector and ``phantomgen`` for the NEMA/IEC phantom:
``pip install "phantomgen @ git+https://github.com/varzakis/phantomgen"``.
No SIMIND run is needed.

Cost and runtime
----------------

Seconds to build both phantoms; the figure is written under
``$SIMIND_EXAMPLES_OUTPUT/phantom_library``.

.. literalinclude:: /examples/phantoms/plot_09_phantom_library.py
   :language: python
   :linenos:

Download the script: :download:`plot_09_phantom_library.py
</examples/phantoms/plot_09_phantom_library.py>`.

Results
-------

.. figure:: /_static/examples/phantom_library/phantom_library.png
   :alt: Central Jaszczak source slice and hottest NEMA activity slice
   :width: 100%

   Figure 1. Central slice of the analytic Jaszczak source and the hottest
   axial slice of the NEMA/IEC activity map.

The analytic phantom is voxelised to relative source weights and a per-voxel
attenuator fraction. The NEMA/IEC phantom carries MBq per voxel, a
water-equivalent density and named masks for the spheres and lung insert. The
printed ranges confirm what each construction returns; no projections are
simulated here.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
