Real Y-90 data
--------------

The real-data pages use a restricted measured Y-90 phantom acquisition and its
CT series: a Mediso AnyScan bremsstrahlung study with 120 views over 360
degrees, 128 x 128 bins of 4.2578 mm and a 75-225 keV window. The
reconstruction grid is the vendor ``umap_zoomed`` grid of 128 voxels at
4.2578 mm.

The data are not redistributed. Set ``SIMIND_CONNECTOR_Y90_DATA`` to a local
directory containing ``peak.dcm``, ``peak.hs``, ``peak.s``,
``umap_zoomed.hv``, ``umap_zoomed.v`` and the ``ctac`` directory of DICOM
slices; see :doc:`/datasets` for the availability statement.

Pages:

- :doc:`/example_pages/y90_ct_conversion` — CT preparation, STIR, SIRF and
  PyTomography resampling recipes, and HU-to-density and attenuation
  conversion.
- :doc:`/example_pages/y90_effective_energy` — attenuation at 150 keV compared
  with a spectrum-weighted map from a simulated detected spectrum.
