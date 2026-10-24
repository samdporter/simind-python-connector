.. _datasets:

Datasets
========

Y-90 phantom data
-----------------

The real-data gallery uses one measured Y-90 bremsstrahlung study of a
phantom, together with the CT series of the same phantom. The examples
locate the files with ``fetch_y90_phantom``, from
``simind_python_connector.datasets``.

.. _y90-data-availability:

Availability
~~~~~~~~~~~~

The data are not redistributed at this time. There is no public download,
DOI, or other access route. Setting ``SIMIND_CONNECTOR_Y90_DATA`` to a local
copy is the only current route, and readers without a copy cannot reproduce
the real-data pages.

The real-data gallery (examples 08A/B/C, 16A/B/C, 17 and 18) consists of
rendered case studies produced from data available to the authors. The pages
are not generally reproducible tutorials: without the restricted input they
cannot be run or regenerated.

The examples fail fast with a clear message when the data are absent. This
statement makes no privacy or redistribution claim about the data.

If you have a local copy, point ``SIMIND_CONNECTOR_Y90_DATA`` at a directory
containing:

- ``peak.dcm`` — the projection DICOM file;
- ``peak.hs`` and ``peak.s`` — the measured projections as Interfile;
- ``umap_zoomed.hv`` and ``umap_zoomed.v`` — the vendor attenuation map,
  used only as a registration reference and as the reconstruction grid;
- ``ctac`` — the CT series, as a directory of DICOM slices.

``fetch_y90_phantom()`` returns that directory unchanged, so every example
works with the same layout. The fetcher reports what is wrong when the data
are not usable:

- if ``SIMIND_CONNECTOR_Y90_DATA`` is not set, it raises ``RuntimeError``
  naming the variable;
- if the directory is missing files, it raises ``FileNotFoundError`` listing
  the missing names.

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

Where the data are used
-----------------------

- The :doc:`workflow tutorial <tutorial>` lists the real-data case studies in
  their recommended order.
- :doc:`examples` points at the gallery sections that contain them.
