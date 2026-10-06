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
  ``(theta, r, z)``.

Practical Implication
---------------------

When comparing STIR/SIRF and PyTomography outputs:

- Use PyTomography projections directly for reconstruction (no extra manual flip).
- Keep PyTomography object tensors in ``(x, y, z)`` end-to-end
  (input -> system matrix -> reconstruction).
- Connector internals convert object tensors to SIMIND image file order
  ``(z, y, x)`` when writing ``.smi/.dmi`` inputs.

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

A configured acquisition wins: once ``set_template()`` or
``configure_acquisition()`` has run, the projection geometry of the phantom
no longer changes it, whichever is configured first. There is no API to clear
a configured acquisition, so build a new adaptor or connector for a different
geometry.

``ProjectionGeometry.time_per_projection_s`` is the frame time the header
reports. Nothing sets it automatically: pass it to an explicit
``set_activity(activity_mbq, time_per_projection_s=...)`` call if you want
SIMIND Index 25 to use it.

After the run, the output geometry is checked against the template before
the data are written under a copy of the template header:

.. list-table::
   :header-rows: 1

   * - Field
     - Tolerance
   * - projections, bins, axial positions, direction
     - equal
   * - extent of rotation
     - 0.5 degrees
   * - start angle
     - 0.5 degrees (modulo 360)
   * - bin and axial size
     - 0.001 mm
   * - radius, or each radius
     - 0.5 mm

SIMIND centres the phantom on the rotation axis, so source and mu-map images
must be centred on the axis, as SIRF/STIR SPECT reconstruction images are.
Voxels may have a different slice thickness from their in-plane size; the
in-plane size must be square.

Mu-maps can be given as attenuation (``mu_map_type="attenuation"``, cm^-1 at
``mu_map_energy_kev``, which defaults to ``abs(Index 1)``), density
(``"density"``, g/cm^3) or CT Hounsfield units (``"hu"``).
