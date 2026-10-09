Phantoms
========

``simind_python_connector.phantoms`` describes phantoms as Python objects, and
``SimindPythonConnector.configure_phantom`` (or ``set_phantom`` on the SIRF and
STIR adaptors) simulates them. ``configure_voxel_phantom`` remains the array API.
The last of these calls wins.

Phantom configuration enables Flag 14 so SIMIND writes the Interfile headers
required by the connector. Flag 14 controls output formatting; Index 14 selects
the attenuator shape or voxel-map routine.

Coordinates
-----------

Analytic shapes use SIMIND's coordinates in cm: x is the patient axis (feet to
head), y points to the patient's right seen from the feet, and z points towards
the camera at angle 0 (anterior). In the connector's ``(z, y, x)`` arrays, the
array z axis is SIMIND -x, array y is -z and array x is +y (manual, voxel-map
orientation).

Analytic phantoms
-----------------

.. code-block:: python

    from simind_python_connector.phantoms import (
        AnalyticPhantom, Ellipsoid, HorizontalCylinder,
    )

    phantom = AnalyticPhantom(
        source=Ellipsoid((4.0, 3.0, 2.0)),
        attenuator=HorizontalCylinder(12.0, (10.0, 10.0)),   # water
        source_shift_cm=(2.0, 1.0, -1.0),
    )
    connector.configure_phantom(phantom)

.. list-table::
   :header-rows: 1

   * - Class
     - SIMIND setting
   * - ``Ellipsoid``
     - Index 15 = 1 (source only); Index 2-4 = half-axes
   * - ``Box``
     - Index 15/14 = 2; half-sizes in Index 2-4 (source) or 5-7 (attenuator)
   * - ``VerticalCylinder``
     - Index 15/14 = 3; axis along z, Index 4 (or 7) = half-length
   * - ``HorizontalCylinder``
     - Index 15/14 = 4; axis along x, Index 2 (or 5) = half-length
   * - ``PointSource``
     - Index 15 = 5, moved by ``source_shift_cm`` (Index 16-18)
   * - ``CardiacSource``
     - Index 15 = 6, with the ``/A``, ``/L`` and ``/M`` switches
   * - ``MultipleInserts``
     - Index 15 = 7, ``{prefix}.inp`` rows selected explicitly by
       ``/IF:{prefix}.inp``, with ``/BG``, ``/HO`` or ``/CO``

An attenuator turns Flag 11 (interactions in the phantom) on; without one it
is off.

Library phantoms
----------------

``LibraryPhantom.ZUBAL_TORSO``, ``ZUBAL_BRAIN``, ``ZUBAL_WHOLE_BODY`` and
``NEMA_IQ`` are the voxel phantoms in SIMIND's ``smc_dir``. All four select a
section of the shipped ``phantom.zub`` code table (Index 45: torso/vox_man = 1,
brain/vox_brn = 2, whole body/vox_man3 = 1, NEMA/nema = 3) and use
``/FZ:phantom``, with the table copied into the run directory because SIMIND
resolves ``/FZ`` against its working directory. The NEMA IQ phantom also uses
the manual's settings (Index 31 = 0.1 cm, Index 2 = Index 5 = 11 cm, a
364 x 364 x 110 map). Flag 15 makes SIMIND write the aligned density, which is
the only ground truth available for these phantoms.

phantomgen and voxel phantoms
-----------------------------

``nema_iec_phantom`` builds the NEMA/IEC body phantom with
`phantomgen <https://github.com/varzakis/phantomgen>`_, which is not on PyPI
yet::

    pip install "phantomgen @ git+https://github.com/varzakis/phantomgen"

It returns a ``VoxelPhantom``: activity in MBq per voxel, water-equivalent
density (phantomgen's mu divided by the mu of the background fill, so the
result does not depend on the energy the mu values were defined at) and
phantomgen's region masks. Its activity is known, so ``configure_phantom``
gives SIMIND the total activity through Index 25 (see :doc:`normalisation`)
and the projections come out on that scale.

phantomgen's ``(Z, Y, X)`` arrays, with the cylinder axis along Z, are already
in the connector's order. Its flat side (+Y) maps to SIMIND -z, i.e.
posterior: the phantom lies on its flat side.

Ground truth
------------

``voxelise(phantom, shape_zyx, voxel_size_mm, supersample)`` returns the
activity and the attenuator fraction of an analytic phantom on a centred grid.
It supports ellipsoids, boxes, both cylinders, point sources (the nearest
voxel) and inserts that are spheres, horizontal, rectangular or vertical rods.
Insert sizes are taken as half-dimensions along x, y and z. Hexagonal rods,
cones and the cardiac phantom raise ``NotImplementedError``.

The adaptors' ``get_ground_truth(image_template)`` returns native images:

.. list-table::
   :header-rows: 1

   * - Phantom
     - Activity
     - mu (cm^-1 at abs(Index 1))
   * - ``VoxelPhantom``
     - its own array (the template grid must match)
     - density x water mu
   * - ``AnalyticPhantom``
     - ``voxelise`` on the template grid
     - water mu inside the attenuator
   * - ``LibraryPhantom``
     - not available
     - SIMIND's aligned density x water mu, on SIMIND's density grid, after
       ``run()``
