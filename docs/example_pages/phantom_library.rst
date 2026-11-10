Phantom library
===============

Build an analytic Jaszczak phantom and a NEMA/IEC body phantom. Requires the
connector and ``phantomgen``; see :doc:`/phantoms`. No simulation runs.

Download the script: :download:`plot_09_phantom_library.py
</examples/phantoms/plot_09_phantom_library.py>`.

.. figure:: /_static/examples/phantom_library/phantom_library.png
   :alt: Central Jaszczak source slice and hottest NEMA activity slice
   :width: 100%

   Figure 1. Jaszczak source weights and NEMA/IEC activity. Each panel has its
   own scale.

The Jaszczak source has relative weights; the NEMA/IEC source has activity in
MBq per voxel. Do not treat their displayed intensities as equivalent.
