Y-90 CT conversion
==================

This is a restricted-data case study, not a reproducible tutorial. The DICOM
and vendor attenuation files are not redistributed, so no runnable conversion
script is included here; the recipes are prose descriptions of the earlier
reviewed workflow.

Prerequisites
-------------

The restricted Y-90 dataset from :doc:`/datasets`, NumPy, pydicom and SciPy,
plus STIR, SIRF or PyTomography for the matching recipe. The SIRF route also
requires ``nibabel`` for its NIfTI resampling step.

Cost and runtime
----------------

Seconds for CT reading and resampling per route. The CT and vendor
``umap_zoomed`` files are read locally; no projection data are reconstructed.

Data and common preparation
---------------------------

The measured CT is a 656-slice Mediso acquisition on the vendor
``umap_zoomed`` grid: 128 voxels of 4.2578 mm. The vendor attenuation map has no
recorded photon energy, so it is a registration reference only and is never
converted to density.

``examples/_y90.py`` provides only the shared preparation helpers:
``read_ct_series`` (DICOM reading, sorting by ``ImagePositionPatient``,
series/frame/orientation/spacing validation, rescaling to HU and the LPS
affine), ``patient_to_spect_affine``, ``read_umap``, ``ct_coverage`` and the
SIRF ``resample_ct``. It contains no registration gate and no executable
conversion route.

The preparation used by each route is:

- read the DICOM slices and sort them by ``ImagePositionPatient``;
- require one series, one frame of reference, unique SOP Instance UIDs, one
  orientation and one pixel spacing;
- accept only axial, axis-aligned geometry with uniform slice spacing and a
  common slice axis;
- apply each slice's ``RescaleSlope`` and ``RescaleIntercept`` to obtain HU;
- build a voxel-centre LPS affine from the row, column and normal directions;
- place the SPECT grid from the NM detector centre and the CT ``TableHeight``
  of -140 mm, without recentring.

SIRF padding is air: the CT is written as ``HU + 1000`` and 1000 is subtracted
after resampling, so outside the CT becomes -1000 HU. STIR and PyTomography
need the same out-of-volume handling.

Registration gate
-----------------

This is the protocol each route must implement; it is not a function exported
by ``examples/_y90.py``. After resampling, compute normalized cross-correlation
between the CT-derived attenuation map and ``umap_zoomed`` on the once-eroded
intersection of CT coverage, HU > -400, positive vendor attenuation and finite
values. The gate is a one-voxel Euclidean limit on that eroded CT/vendor
intersection. On failure write ``summary.json`` and ``UNACCEPTED.txt`` and stop
before producing density or attenuation maps.

STIR recipe
-----------

Apply the registration gate before conversion; on failure, stop before writing
maps.

Write the CT as an Interfile image in the SPECT frame and call ``zoom_image``
with ``preserve_values``; HU are not count-scaled. Convert HU to attenuation at
150 keV and to Schneider density after resampling.

SIRF recipe
-----------

Apply the registration gate before conversion; on failure, stop before writing
maps.

Write the CT and an empty SPECT reference as NIfTI images in RAS coordinates,
then use ``sirf.Reg.NiftyResample`` with linear interpolation and zero padding.
The CT is ``HU + 1000`` as above. Read the resampled array back into a STIR
image and verify the values survived the transfer before conversion.

PyTomography recipe
-------------------

Apply the registration gate before conversion; on failure, stop before writing
maps.

Read the CT with PyTomography's DICOM reader and its attenuation table. Its
Mediso projection affine uses the detector x position and needs the
dataset-local correction: shift the CT y origin by ``-TableHeight`` and express
the target affine in the same convention before applying the affine transform.
Replace the reader's projection affine with the patient-to-SPECT affine
validated on the landmarks, then convert HU to density and attenuation.

Conversion
----------

After resampling, use ``hu_to_attenuation(hu, 150.0)`` for the 150 keV map and
``hu_to_density_schneider(hu)`` for density. The three backend routes are not
claimed to be equivalent: their interpolation, padding and affine conventions
differ.

Results
-------

.. figure:: /_static/examples/y90_ct_conversion/registration.png
   :alt: CT and vendor attenuation-map overlay after registration
   :width: 100%

   Figure 1. Registration illustration produced by the STIR route after the
   one-voxel gate (0.409 voxels, 1.74 mm). The vendor map (green) fills the
   body and largely covers the CT (red); the red/orange edges and table show
   the residual alignment. The vendor map is a registration reference, not a
   known-energy attenuation map.

.. figure:: /_static/examples/y90_ct_conversion/maps.png
   :alt: Resampled CT HU, Schneider density and 150 keV attenuation maps
   :width: 100%

   Figure 2. CT HU, Schneider density and 150 keV attenuation maps produced by
   the STIR route on the SPECT grid. These are conversion illustrations, not an
   accuracy reference.

.. note::

   Retained figures come from the earlier reviewed run and were not
   re-executed for this editorial revision.
