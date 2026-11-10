DICOM projection recipe
=======================

Prerequisites
-------------

A DICOM NM projection file supplied by the reader, plus STIR, SIRF or
PyTomography for the matching snippet.

Cost and runtime
----------------

Seconds per snippet. No simulation is run and no figure is retained.

Dataset note
------------

The dataset used during development is a restricted Y-90 bremsstrahlung study
(Mediso AnyScan, 120 views over 360 degrees, 128 x 128 bins of 4.2578 mm, 75-225
keV). It is not redistributed, so the path below is a placeholder. The snippets
show the call sequence only.

STIR
----

.. code-block:: python

   from pathlib import Path
   import stir
   from simind_python_connector.builders import STIRSPECTAcquisitionDataBuilder

   dicom_file = Path("/path/to/peak.dcm")
   builder = STIRSPECTAcquisitionDataBuilder(backend="stir")
   builder.update_header_from_dicom(str(dicom_file))
   acq = builder.build(output_path=Path("output/stir_from_dicom"))
   # STIR ProjData order is (view, radial, axial); views follow the DICOM
   # detector angles.

SIRF
----

.. code-block:: python

   from pathlib import Path
   import sirf.STIR as sirf
   from simind_python_connector.builders import STIRSPECTAcquisitionDataBuilder

   dicom_file = Path("/path/to/peak.dcm")
   builder = STIRSPECTAcquisitionDataBuilder(backend="sirf")
   builder.update_header_from_dicom(str(dicom_file))
   acq = builder.build(output_path=Path("output/sirf_from_dicom"))
   # SIRF AcquisitionData shares STIR's accounting; convert with get_array().

PyTomography
------------

.. code-block:: python

   from pathlib import Path
   from pytomography.io.SPECT import dicom as pytomo_dicom

   dicom_file = Path("/path/to/peak.dcm")
   projections = pytomo_dicom.get_projections(str(dicom_file), index_peak=0)
   # PyTomography returns (theta, r, z). Its view order follows
   # (gantry angle + 180) mod 360, so view k is DICOM frame (k + n / 2) mod n;
   # transpose and use origin="lower" to display axial rows with the head up.

Orientation qualifications
--------------------------

The three backends do not expose the same view order. STIR and SIRF keep the
DICOM frame order, while PyTomography rotates it by half a turn. Display code
must therefore match the backend to the array order rather than assuming a
shared convention. The examples above do not include a display step.
