Real Y-90 data
--------------

These tutorials use a measured Y-90 phantom acquisition: a Mediso AnyScan
bremsstrahlung study with 120 views over 360 degrees, 128 x 128 bins of
4.2578 mm and a 75-225 keV window, together with the CT series of the same
phantom. The reconstruction grid is the vendor ``umap_zoomed`` grid of 128
voxels at 4.2578 mm.

TODO: A dataset download is coming, but is not yet available.

For now, set ``SIMIND_CONNECTOR_Y90_DATA`` to a directory containing ``peak.dcm``,
``peak.hs``, ``peak.s``, ``umap_zoomed.hv``, ``umap_zoomed.v`` and the
``ctac`` directory of DICOM slices. The tutorials read the CT series from that
directory and write their figures and summaries under
``$SIMIND_EXAMPLES_OUTPUT/real_data/ct_conversion`` (or ``output`` when the
variable is unset).

The three CT conversion tutorials show the same registration chain with a
different backend:

- ``plot_16a_stir_ct_conversion.py`` builds the CT in the STIR frame and
  resamples it with ``zoom_image``.
- ``plot_16b_sirf_ct_conversion.py`` writes the CT as a NIfTI image and
  resamples it with SIRF's NiftyResample.
- ``plot_16c_pytomography_ct_conversion.py`` uses PyTomography's CT reader and
  attenuation table with a corrected patient-to-SPECT affine.

All three sort the CT slices by ``ImagePositionPatient``, validate the series,
register the CT-derived attenuation map to ``umap_zoomed``, and stop before
writing accepted maps if the registration gate fails. The vendor attenuation
map has no recorded photon energy, so it is used only as a registration
reference and is never converted to density.

The tutorials need NumPy, pydicom, matplotlib and SciPy. The SIRF tutorial
also needs nibabel. All three need the matching backend.
