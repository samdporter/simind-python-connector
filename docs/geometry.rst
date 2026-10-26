Geometry Considerations
=======================

This page summarizes the geometry conventions used across SIMIND, STIR/SIRF,
and PyTomography in this package.

At-a-Glance Axis Conventions
----------------------------

- STIR/SIRF image arrays are handled as ``(z, y, x)``.
- STIR/SIRF projection arrays are often 4D with a singleton TOF axis:
  ``(tof, axial, view, bin)``. For these, **view 0** is ``arr[0, :, 0, :]``.
- Raw Interfile projection loading in ``utils.interfile`` is exposed as
  ``(view, axial, bin)`` when no singleton TOF axis is present.
- ``PyTomographySimindAdaptor`` public object-space tensors are ``(x, y, z)``.
- PyTomography object space is ``(x, y, z)`` (``Lx, Ly, Lz``).
- PyTomography SIMIND projections from ``pytomography.io.SPECT.simind`` are
  ``(theta, r, z)``, which is ``(view, bin, axial)``.
- ``PyTomographySimindAdaptor`` uses that same ``(view, bin, axial)`` layout
  even on its fallback path (without PyTomography installed): the core
  ``(view, axial, bin)`` array is transposed and its axial direction reversed
  to match the reader.
- ``PyTomographySimindAdaptor.get_output_header_path`` is a SIMIND header
  (start angle in SIMIND's convention) when PyTomography is installed, which is
  what ``pytomography.io.SPECT.simind.get_metadata`` expects. Without
  PyTomography it is the converted ``.hs``. SIMIND writes no
  per-key ``.h00`` for PENETRATE components or derived ``pri_wN``/``sca_wN``,
  so the adaptor writes ``{prefix}_{key}.h00`` from SIMIND's own header (v8
  ``{prefix}.h00``, v9 ``{prefix}.hNN``, ``tot_wN`` for derived outputs). Do not
  give PyTomography the converted ``.hs`` headers: their start angle is STIR's,
  SIMIND's + 180 degrees, and the reconstruction comes out rotated 180 degrees
  about z. ``get_output_metadata`` is read from the ``.hs`` and so also reports
  STIR's start angle.
- PyTomography DICOM projections (``pytomography.io.SPECT.dicom.get_projections``)
  are also ``(theta, r, z)``, but the views are sorted by PyTomography's
  detector angle, ``(gantry angle + 180) mod 360``, and ``z`` runs from the
  bottom DICOM row upward. For a 360-degree acquisition of ``N`` views,
  PyTomography view ``k`` is DICOM frame ``(k + N/2) mod N``. STIR/SIRF
  projections built from the same DICOM (``update_header_from_dicom``) keep
  the DICOM frame, row and column order: ``arr[0, row, frame, col]``. To
  display a PyTomography view with the head end up, use
  ``imshow(proj[k].T, origin="lower")``.

Practical Implication
---------------------

When comparing STIR/SIRF and PyTomography outputs:

- Use PyTomography projections directly for reconstruction (no extra manual flip).
- Keep PyTomography object tensors in ``(x, y, z)`` end-to-end
  (input -> system matrix -> reconstruction).
- Connector internals convert object tensors to SIMIND image file order
  ``(z, y, x)`` when writing ``.smi/.dmi`` inputs.
- PyTomography view 0 and STIR view 0 are generally different physical views;
  compare by angle, not by index.

Units
-----

- SIMIND geometry parameters are configured in **cm**.
- STIR/SIRF image geometry is typically expressed in **mm**.
- Connectors handle conversion internally via voxel-size settings (for example,
  runtime switch ``PX`` is set in cm for SIMIND).
- STIR/SIRF adaptors read voxel sizes in mm in ``(z, y, x)`` order: SIRF from
  ``voxel_sizes()``, STIR from the 1-based ``get_grid_spacing()`` coordinate
  (``[1]`` is z).

Example Configuration Guardrails
--------------------------------

The OSEM examples intentionally pin key simulation parameters so geometry checks
are reproducible:

- ``NN=1`` (runtime switch) for faster, deterministic iteration.
- ``config[29]=24`` for projection count.
- ``config[53]=0`` to keep collimator modeling geometric-only in these tests.
- ``config[19]=2`` to keep a consistent mapping used by current examples.

Debug Checklist
---------------

If reconstruction geometry looks wrong:

1. Confirm array shape and axis order before plotting/reconstruction.
2. Confirm you are extracting projection **view 0** from the correct axis.
3. Confirm PyTomography object-space tensors are in ``(x, y, z)``.
4. Confirm attenuation-map orientation matches the reconstruction backend.
5. Compare hotspot center-of-mass between source and recon in a common axis convention.

Simulating a Measured Acquisition
---------------------------------

Give the SIRF or STIR adaptor the measured projection data (or its ``.hs``
header) and SIMIND runs in exactly that geometry:

.. code-block:: python

    adaptor.set_template(measured)        # sirf.STIR.AcquisitionData or path
    outputs = adaptor.run()
    difference = outputs["tot_w1"] - measured   # same geometry

The geometry maps onto SIMIND as follows:

.. list-table::
   :header-rows: 1

   * - SIMIND setting
     - Value from the template
   * - Index 29
     - number of projections
   * - Index 30
     - extent of rotation, positive for CW, negative for CCW
   * - Index 41
     - (start angle + 180) mod 360
   * - Index 12
     - ``radius_mm / 10`` (cm) for a circular orbit; for a non-circular orbit
       ``mean(radii_mm) / 10``, and the per-projection radii are written to
       ``{prefix}_acquisition.cor``, which the orbit file then overrides
   * - Index 28
     - bin size / 10 (cm); bin and axial sizes must be equal
   * - Index 76, 77
     - number of bins, number of axial positions

Without a template, the projection pixel size and matrix (Index 28, 76 and 77)
come from the configuration. Configuring a phantom never changes them, so set
them to cover the phantom; ``run()`` warns when the phantom is wider or longer
than the projection field of view.

``set_template()`` writes these settings when it is called. Values set with
``add_config_value()`` afterwards usually win; the adaptor then warns and
returns the outputs in SIMIND's geometry rather than the template's. The
exception is Index 12 when the template has non-circular radii: the per-view
radii are stored and written to the acquisition orbit file, which overrides
Index 12, and changing Index 12 does not clear them. Editing Index 12 to
``15.0`` therefore does not turn a non-circular template into a 150 mm circular
orbit. To switch to a circular orbit, replace the acquisition with a circular
``configure_acquisition()`` geometry, or replace the template with a circular
one via ``set_template()``. There is no API to clear a configured acquisition
outright; replace it as above, or build a new adaptor or connector for a
different geometry.

``ProjectionGeometry.time_per_projection_s`` is the frame time the header
reports. Nothing sets it automatically: pass it to an explicit
``set_activity(activity_mbq, time_per_projection_s=...)`` call if you want
SIMIND Index 25 to use it.

After the run, and only when no tracked projection setting has changed, the
output geometry is checked against the template before the data are written
under a copy of the template header. A change to any tracked setting (Index 12,
28, 29, 30, 41, 76 or 77) returns the raw converted SIMIND headers instead, with
no copying and no checking, even when the change is an Index 12 edit that the
stored non-circular orbit makes ineffective:

.. list-table::
   :header-rows: 1

   * - Field
     - Tolerance
   * - projections, bins, axial positions, direction
     - equal
   * - extent of rotation
     - 0.5 degrees; up to 360.5 is accepted and clamped to 360 for SIMIND
   * - start angle
     - 0.5 degrees (modulo 360)
   * - bin and axial size
     - 0.001 mm
   * - radius, or each radius
     - 0.5 mm

SIMIND centres the phantom on the rotation axis, so source and mu-map images
must be centred on the axis, as SIRF/STIR SPECT reconstruction images are.
Voxels may have a different slice thickness from their in-plane size; the
in-plane size must be square, because SIMIND takes a single voxel side (Index 31
and the ``PX`` switch). The slices themselves do not have to be square: the
columns (x, Index 78 and 79) and rows (y, Index 81 and 82) of the maps are set
separately, and a SIMIND test checks that non-square slices give the same
projections as the same slices padded to square.

SIMIND v9 uses a different transverse origin for image-based source and density
maps. Select the matching version explicitly with ``simind_version=9`` or set
``SIMIND_VERSION=9``; v8 remains the default for existing callers. For v9
voxel maps, the connector adds the validated source shifts to Indexes 17/18 and
density-map origin offsets to Indexes 36/37 in the saved run configuration.
The shifts use the source map's effective ``PX`` and the density pixel size in
Index 31, preserving the input arrays and the caller's in-memory origin
offsets. This compensation is validated for the
default orientation only: ``DI=0``, density Index 32 = 0, and no ``OR`` switch.
Other v9 voxel-map orientations are rejected before engine launch. Analytic
phantoms and v8 voxel maps do not receive these offsets.

Mu-maps can be given as attenuation (``mu_map_type="attenuation"``, cm^-1 at
``mu_map_energy_kev``, which defaults to ``abs(Index 1)``), density
(``"density"``, g/cm^3) or CT Hounsfield units (``"hu"``, converted to density
with the Schneider2000 model of ``hu_to_density_schneider``).
