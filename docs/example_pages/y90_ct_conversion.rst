Y-90 CT conversion
==================

Convert a restricted Y-90 study's CT to density and 150 keV attenuation maps on
the SPECT grid. The data are not redistributed; see :doc:`/datasets`. These are
conversion recipes, not runnable scripts.

Requires NumPy, pydicom and SciPy, plus the chosen backend. The SIRF route also
needs ``nibabel``.

Preparation
-----------

Read the DICOM series, sort by ``ImagePositionPatient`` and validate the series,
frame of reference, orientation and spacing. Apply each slice's rescaling to
obtain HU. Resample onto the vendor ``umap_zoomed`` grid of 128 voxels of
4.2578 mm, preserving patient coordinates. Pad outside the CT with -1000 HU.

- STIR: write the CT in the SPECT frame and call ``zoom_image`` with
  ``preserve_values``.
- SIRF: resample the RAS NIfTI with ``NiftyResample``; store ``HU + 1000``
  during zero-padded resampling, then subtract it.
- PyTomography: build the patient-to-SPECT affine, including the Mediso
  table-height correction.

Check registration against the vendor map before conversion. The illustrated
STIR result is offset by 0.409 voxels (1.74 mm). The vendor map's photon energy
is unknown, so use it for registration, not density conversion.

Convert with ``hu_to_density_schneider(hu)`` and
``hu_to_attenuation(hu, 150.0)``.

.. figure:: /_static/examples/y90_ct_conversion/registration.png
   :alt: CT and vendor attenuation-map overlay after registration
   :width: 100%

   Figure 1. STIR-route CT and vendor-map overlay after registration.

.. figure:: /_static/examples/y90_ct_conversion/maps.png
   :alt: Resampled CT HU, Schneider density and 150 keV attenuation maps
   :width: 100%

   Figure 2. Resampled CT, Schneider density and 150 keV attenuation.

Backend interpolation and coordinate conventions can produce different maps.
These figures illustrate conversion, not a calibrated accuracy test.
