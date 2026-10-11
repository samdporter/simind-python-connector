Geometry Considerations
=======================

This page summarizes the geometry conventions used across SIMIND, STIR/SIRF,
and PyTomography in this package.

At-a-Glance Axis Conventions
----------------------------

- STIR/SIRF image arrays are handled as ``(z, y, x)``.
- STIR/SIRF projection arrays are often 4D with a singleton TOF axis:
  ``(tof, axial, view, bin)``. For these, **view 0** is ``arr[0, :, 0, :]``.
- Raw Interfile projection loading in ``interfile_numpy`` is exposed as
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
- STIR/SIRF adaptor voxel size extraction expects z-spacing in mm. Images
  report ``voxel_sizes()`` as ``(z, y, x)``, so the **first** element is the z
  spacing. Raw ``get_grid_spacing()`` sequences follow the STIR coordinate
  layout ``(unused, z, y, x)`` and use index 1; coordinate objects with a
  ``z`` accessor use that accessor directly.

Projection Size
---------------

The projection bin size and matrix come from the configuration: Index 28 (bin
size in cm), Index 76 (bins) and Index 77 (axial rows). Configuring a voxel
phantom does not change them, so set them to cover the phantom; ``run()`` warns
when the phantom is wider or longer than the projection field of view. The
matrix does not have to be square.

Some projectors need the projections sampled like the image. STIR's SPECTUB
needs equal axial sampling, so the OSEM examples set Index 28 to the voxel size.

Voxel Phantom Slices
--------------------

Image slices do not have to be square. The columns (x, Index 78 and 79) and
rows (y, Index 81 and 82) of the source and density maps are set separately,
and a SIMIND test checks that non-square slices give the same projections as
the same slices padded to square. The voxels themselves must be square in-plane:
SIMIND takes a single voxel side (Index 31 and the ``PX`` switch).

Example Configuration Guardrails
--------------------------------

The OSEM examples intentionally pin key simulation parameters so geometry checks
are reproducible:

- ``NN=1`` (runtime switch) for faster, deterministic iteration.
- ``config[29]=24`` for projection count.
- ``config[53]=0`` to keep collimator modeling geometric-only in these tests.
- ``config[19]=2`` to keep a consistent mapping used by current examples.
- ``config[28]=0.4``, ``config[76]=32`` and ``config[77]=32`` to put the
  projections on the 4 mm, 32-voxel reconstruction grid.

Debug Checklist
---------------

If reconstruction geometry looks wrong:

1. Confirm array shape and axis order before plotting/reconstruction.
2. Confirm you are extracting projection **view 0** from the correct axis.
3. Confirm PyTomography object-space tensors are in ``(x, y, z)``.
4. Confirm attenuation-map orientation matches the reconstruction backend.
5. Compare hotspot center-of-mass between source and recon in a common axis convention.
