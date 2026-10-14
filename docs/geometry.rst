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
- Connectors handle conversion internally. ``PX`` and SIMIND's image dimensions
  use the in-plane spacing in cm; ``TH`` carries distinct axial spacing when
  needed.
- STIR/SIRF adaptors read spacing in mm. Images report ``voxel_sizes()`` as
  ``(z, y, x)``, and raw ``get_grid_spacing()`` sequences follow the STIR
  coordinate layout ``(unused, z, y, x)``.

Source and attenuation images must describe the same physical grid. The native
adaptors compare SIRF offsets, spacing and direction matrices, or STIR physical
array origins (including minimum indices). This also applies to backend-wrapped
images. Spacing must be finite and positive. The adaptors reject mismatches;
they do not resample or register images. Array-only objects without geometry
metadata still require callers to supply aligned arrays.

Energy-window filenames
-----------------------

With custom energy windows, use an output prefix of at most 15 characters.
SIMIND v8 truncates its derived window filename to 19 characters including
``.win`` and can silently use its default energy window instead. The connector
rejects longer prefixes before writing inputs. If a Python/native connector's
prefix changes after setting windows, call ``set_energy_windows()`` again.

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
the same slices padded to square. Voxels must be square in-plane because SIMIND
takes one side for x/y (Index 31 and the ``PX`` switch); axial spacing may
differ. The connector accepts a scalar (backward compatible) or a finite,
positive ``(z, y, x)`` spacing tuple, with y and x matching within 0.001 mm.
For distinct axial spacing it sets ``TH`` to the z spacing in cm, while ``PX``
and Index 31 retain the in-plane spacing. A later cubic configuration clears a
previously set ``TH``. PyTomography tensors remain in ``(x, y, z)`` order, but
the ``voxel_size_mm`` tuple still uses canonical ``(z, y, x)`` order.

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

Asymmetric geometry examples
----------------------------

The three OSEM examples now share an asymmetric phantom with three unequal hot
markers, an L-shaped central structure and an asymmetric body. Its builder is
``simind_python_connector.utils.geometry_phantom.build_asymmetric_phantom``.

The higher-statistics experiments, forward projections and comparison reports
used to investigate backend geometry were local diagnostic artifacts under
``tmp/geometry_sweep``. That directory is ignored by Git, so those scripts,
outputs and machine-specific measurements are not part of the distributed
package. The checked-in examples and shared phantom builder are the reproducible
starting point for running the backends with the same asymmetric input.
