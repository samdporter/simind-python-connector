Y-90 effective attenuation energy
=================================

Prerequisites
-------------

The restricted Y-90 dataset from :doc:`/datasets`, SIMIND, SIRF, NumPy,
Matplotlib, pydicom and nibabel.

Cost and runtime
----------------

The list-mode spectrum acquisition takes about 11 minutes with the recorded
settings; CT reading and resampling are seconds. Outputs are written under
``$SIMIND_EXAMPLES_OUTPUT/real_data/effective_energy``.

.. literalinclude:: /examples/real_data/plot_17_y90_effective_energy.py
   :language: python
   :linenos:

Download the script: :download:`plot_17_y90_effective_energy.py
</examples/real_data/plot_17_y90_effective_energy.py>`.

Results
-------

.. figure:: /_static/examples/y90_effective_energy/detected_spectrum.png
   :alt: Simulated list-mode detected Y-90 spectrum with the imaging window
   :width: 100%

   Figure 1. Simulated LIST_MODE detected spectrum from 24 views, including
   detector energy broadening; the shaded band is the 75-225 keV imaging
   window.

.. figure:: /_static/examples/y90_effective_energy/attenuation_maps.png
   :alt: Attenuation at 150 keV, spectrum-weighted attenuation and difference
   :width: 100%

   Figure 2. Attenuation at 150 keV, the spectrum-weighted attenuation and
   their difference. The weighted map is a global effective-map
   approximation.

The recorded list-mode run has 21,076,461 records and a window weight of
227,115.8, within 0.1% of both the ``.a00`` projection total and the ``.res``
count rate. The weighted mean energy is 125.6 keV; over the body the weighted
map differs from the 150 keV map by 0.01373 cm^-1 on average and 0.04273 cm^-1
at most. The detected spectrum is simulated, not measured, and the weighted map
is an approximation; neither is an accuracy reference.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
