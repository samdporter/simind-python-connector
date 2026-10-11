Backend and Adaptor Dependencies
================================

Use ``SimindPythonConnector`` for direct SIMIND execution from Python. Use an
adaptor when you want outputs returned as native STIR/SIRF/PyTomography types.

Dependency Matrix
-----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Component
     - Required dependency
   * - ``SimindPythonConnector``
     - SIMIND executable on ``PATH`` (no SIRF/STIR/PyTomography requirement)
   * - ``StirSimindAdaptor``
     - STIR Python (``stir`` and ``stirextra``)
   * - ``SirfSimindAdaptor``
     - SIRF (``sirf.STIR``)
   * - ``PyTomographySimindAdaptor``
     - PyTomography + torch

The adaptors are responsible for converting input/output object types across
package boundaries. Reconstruction-system objects (for example, a PyTomography
system matrix) should be created directly in the target package.

Returned Layouts
----------------

Each adaptor returns that package's native objects, in that package's native
layout. See :doc:`geometry` for the axis conventions in full.

- ``SimindPythonConnector`` returns NumPy arrays: image maps as ``(z, y, x)``
  and projections as ``(view, axial, bin)``.
- ``StirSimindAdaptor`` and ``SirfSimindAdaptor`` return ``(z, y, x)`` image
  objects. STIR acquisitions are ``(tof, axial, view, bin)``.
- ``PyTomographySimindAdaptor`` returns torch tensors: object space as
  ``(x, y, z)`` and projections as ``(view, bin, axial)``, matching
  ``pytomography.io.SPECT.simind.get_projections``. The fallback used for
  outputs without a per-key ``.h00`` (for example PENETRATE) applies the same
  layout, so downstream reconstruction code does not need to special-case it.

Replacing Removed Helpers
-------------------------

The backend wrapper layer was removed. ``simind_python_connector.backends``
(``get_backend``, ``set_backend``, ``reset_backend``, ``create_image_data``,
``create_acquisition_data``, ``unwrap`` and the wrapper classes) and the legacy
``utils`` helpers no longer exist. Migration:

- Use ``StirSimindAdaptor``, ``SirfSimindAdaptor`` or
  ``PyTomographySimindAdaptor`` for SIMIND execution and output conversion.
- Build STIR/SIRF images and acquisitions directly with those packages' own
  constructors instead of the removed factory helpers.
- Read and write Interfile headers and arrays with
  ``simind_python_connector.utils.interfile``.

