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
     - STIR Python (``stir`` plus ``stir.extra``, with the legacy ``stirextra``
       module as a fallback)
   * - ``SirfSimindAdaptor``
     - SIRF (``sirf.STIR``)
   * - ``PyTomographySimindAdaptor``
     - PyTomography + torch

The adaptors are responsible for converting input/output object types across
package boundaries. Reconstruction-system objects (for example, a PyTomography
system matrix) should be created directly in the target package.

Input and Output Layouts
------------------------

Image maps are inputs, not outputs: they are supplied in the orientation each
package uses. What comes back from ``run()`` is the projection outputs, in that
package's native type and layout. See :doc:`geometry` for the axis conventions
in full.

Image inputs
   - ``SimindPythonConnector.configure_voxel_phantom`` takes NumPy source and
     mu-map arrays as ``(z, y, x)``.
   - ``StirSimindAdaptor`` and ``SirfSimindAdaptor`` take those packages' own
     image objects (``set_source`` / ``set_mu_map``), in ``(z, y, x)``.
   - ``PyTomographySimindAdaptor`` takes torch tensors in object space
     ``(x, y, z)``.

Returned projections
   - ``SimindPythonConnector.run`` returns ``dict[str, ProjectionResult]``;
     each ``ProjectionResult.projection`` is a NumPy ``(view, axial, bin)``
     array.
   - ``StirSimindAdaptor`` returns STIR ``ProjData`` objects and
     ``SirfSimindAdaptor`` returns SIRF ``AcquisitionData`` objects. STIR
     acquisitions are ``(tof, axial, view, bin)``.
   - ``PyTomographySimindAdaptor`` returns torch tensors in
     ``(view, bin, axial)``, matching
     ``pytomography.io.SPECT.simind.get_projections``. The fallback used for
     outputs without a per-key ``.h00`` (for example PENETRATE) applies the
     same layout, so downstream reconstruction code does not need to
     special-case it.

Replacing Removed Helpers
-------------------------

The backend wrapper layer was removed. ``simind_python_connector.backends`` and
its helpers (``get_backend``, ``set_backend``, ``reset_backend``,
``create_image_data``, ``create_acquisition_data``, ``unwrap`` and the wrapper
classes) no longer exist. The legacy utility modules
``utils.backend_access``, ``utils.import_helpers``, ``utils.sirf_stir_utils``,
``utils.io_utils``, ``utils.to_projdata_in_memory``, ``utils.stir_utils``,
``utils.interfile_parser`` and ``utils.interfile_numpy`` were removed too,
along with ``converters.dicom_to_stir``. Migration:

- Use ``StirSimindAdaptor``, ``SirfSimindAdaptor`` or
  ``PyTomographySimindAdaptor`` for SIMIND execution and output conversion.
- Build STIR/SIRF images and acquisitions directly with those packages' own
  constructors instead of the removed factory helpers.
- Read and write Interfile headers and arrays with
  ``simind_python_connector.utils.interfile``, which replaces
  ``utils.interfile_parser`` and ``utils.interfile_numpy``.

These utilities remain: ``utils.get_array`` for pulling a NumPy array from a
SIRF or STIR object, ``utils.interfile``, and ``utils.simind_utils``.

