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
     - STIR Python (``stir``)
   * - ``SirfSimindAdaptor``
     - SIRF (``sirf.STIR``)
   * - ``PyTomographySimindAdaptor``
     - PyTomography + torch

The adaptors are responsible for converting input/output object types across
package boundaries. Reconstruction-system objects (for example, a PyTomography
system matrix) should be created directly in the target package.

