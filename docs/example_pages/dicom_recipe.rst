DICOM projection recipe
=======================

Load your own DICOM NM projection file with STIR, SIRF or PyTomography. No
SIMIND run is needed.

STIR and SIRF
-------------

.. code-block:: python

   from simind_python_connector.builders import STIRSPECTAcquisitionDataBuilder
   from simind_python_connector.utils import get_array

   for backend in ("stir", "sirf"):
       builder = STIRSPECTAcquisitionDataBuilder(backend=backend)
       builder.update_header_from_dicom("/path/to/peak.dcm")
       acquisition = builder.build(output_path=f"projection_{backend}")
       projection = get_array(acquisition)

Both return projection arrays ordered ``(1, axial, view, bin)``, preserving
DICOM frame order.

PyTomography
------------

.. code-block:: python

   from pytomography.io.SPECT import dicom
   projections = dicom.get_projections("/path/to/peak.dcm", index_peak=0)

PyTomography returns ``(view, bin, axial)`` and reorders views by detector
angle. For a full-turn acquisition, its view order differs from DICOM by half a
turn. Compare views by angle, not index.

See :doc:`/geometry` for display orientation.
