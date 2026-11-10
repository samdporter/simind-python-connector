.. _datasets:

Datasets
========

Y-90 phantom data
-----------------

The real-data pages use one measured Y-90 bremsstrahlung study of a phantom and
the CT series from the same scan. ``fetch_y90_phantom`` from
``simind_python_connector.datasets`` locates the files.

The Y-90 data are not redistributed and have no public download. The
CT-conversion and effective-energy case studies require a local copy. Set
``SIMIND_CONNECTOR_Y90_DATA`` to the directory containing the files below. The
DICOM projection recipe can instead use your own DICOM NM file.

- ``peak.dcm`` — the projection DICOM file;
- ``peak.hs`` and ``peak.s`` — the measured projections as Interfile;
- ``umap_zoomed.hv`` and ``umap_zoomed.v`` — the vendor attenuation map,
  used only as a registration reference and as the reconstruction grid;
- ``ctac`` — the CT series, as a directory of DICOM slices.

Scanner and acquisition
-----------------------

- System: Mediso AnyScan SPECT/CT.
- Collimator: MLEGP.
- Energy window: 75–225 keV.
- 120 views of 40 s over 360 degrees, counter-clockwise, on a non-circular
  orbit with bed radii mostly near 300 mm.
- Matrix: 128 x 128 bins at 4.2578 mm.
- Reconstruction and registration grid: the vendor ``umap_zoomed`` grid of
  128 voxels at 4.2578 mm. Its recorded photon energy is unknown, so it is
  never converted to density.

Truth table
-----------

The known compartment contents at the scan, with a +-2% activity uncertainty:

.. list-table::
   :header-rows: 1
   :widths: 30 25 20 25

   * - Compartment
     - Activity (MBq)
     - Mass (g)
     - Concentration (kBq/g)
   * - Liver
     - 156.4 (+-2%)
     - 1149.49
     - 136.1
   * - Hot sphere
     - 4.40 (+-2%)
     - 8.082
     - 544.4

The total activity is 160.8 MBq and the sphere-to-liver
activity-concentration ratio is 4.0. The activities are the compartment
contents at the scan; they are not decay corrected, and the +-2% applies to
the activity only.
