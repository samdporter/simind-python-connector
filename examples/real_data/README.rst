Real Y-90 data
--------------

These authored pages use restricted-data case studies, not generally
reproducible tutorials. They use a measured Y-90 phantom acquisition: a Mediso
AnyScan bremsstrahlung study with 120 views over 360 degrees, 128 x 128 bins of
4.2578 mm and a 75-225 keV window, together with the CT series of the same
phantom. The reconstruction grid is the vendor ``umap_zoomed`` grid of 128
voxels at 4.2578 mm.

The data are not redistributed. Setting ``SIMIND_CONNECTOR_Y90_DATA`` to a
local copy is the only current route; see :doc:`/datasets` for the availability
statement. Provide a directory containing ``peak.dcm``, ``peak.hs``, ``peak.s``,
``umap_zoomed.hv``, ``umap_zoomed.v`` and the ``ctac`` directory of DICOM
slices.

The pages are:

- :doc:`/example_pages/y90_ct_conversion` documents the shared CT preparation,
  gives separate STIR, SIRF and PyTomography resampling recipes, and converts
  HU to density and attenuation. The vendor attenuation map has no recorded
  photon energy, so it is a registration reference only and is never converted
  to density. All routes use the same one-voxel registration gate and stop
  before producing maps when it fails.
- :doc:`/example_pages/y90_effective_energy` uses the retained
  ``plot_17_y90_effective_energy.py`` script. It compares a 150 keV map with
  one weighted by the simulated detected spectrum, checks the list-mode window
  weight against the engine counts, and keeps the detected-spectrum and
  attenuation-map figures. The spectrum is simulated and the weighted map is an
  approximation; neither is an accuracy reference.

The CT page needs NumPy, pydicom, matplotlib and SciPy plus the matching
backend (SIRF also needs nibabel). The effective-energy page needs SIMIND,
SIRF, NumPy, Matplotlib, pydicom and nibabel. No paired reconstructions,
profiles or Monte Carlo case study remain.
