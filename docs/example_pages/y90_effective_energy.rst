Y-90 effective attenuation energy
=================================

Compare attenuation at 150 keV with attenuation weighted by a simulated
detected Y-90 spectrum.

Requires the restricted data from :doc:`/datasets`, SIMIND, SIRF, NumPy,
Matplotlib, pydicom and nibabel. The spectrum simulation takes about 11 minutes.

Download the script: :download:`plot_17_y90_effective_energy.py
</examples/real_data/plot_17_y90_effective_energy.py>`.

.. figure:: /_static/examples/y90_effective_energy/detected_spectrum.png
   :alt: Simulated list-mode detected Y-90 spectrum with the imaging window
   :width: 100%

   Figure 1. Simulated detected spectrum; shading marks the 75-225 keV imaging
   window.

.. figure:: /_static/examples/y90_effective_energy/attenuation_maps.png
   :alt: Attenuation at 150 keV, spectrum-weighted attenuation and difference
   :width: 100%

   Figure 2. Attenuation at 150 keV, spectrum-weighted attenuation and their
   difference.

The window-weighted mean energy is 125.6 keV. The weighted attenuation map uses
one simulated spectrum for the whole object; it does not model spatial variation
in the detected spectrum.
