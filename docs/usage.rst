.. _usage:

Usage Guide
===========

Quick Start
-----------

Use ``SimindPythonConnector`` when you want direct Python control of SIMIND
inputs/outputs without any reconstruction-package dependency.

The Python package does not bundle SIMIND. Before calling ``run()``, make sure
the SIMIND executable is available as ``simind`` on ``PATH``; see
:doc:`installation`.

.. code-block:: python

    import numpy as np
    from simind_python_connector import SimindPythonConnector
    from simind_python_connector.configs import get
    from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE

    source = np.zeros((32, 32, 32), dtype=np.float32)  # z, y, x
    source[12:20, 12:20, 12:20] = 1.0
    mu_map = np.zeros_like(source)
    mu_map[source > 0] = 0.15

    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir="output/basic",
        output_prefix="case01",
        quantization_scale=0.05 * DEFAULT_QUANTIZATION_SCALE,
    )

    connector.configure_voxel_phantom(
        source=source,
        mu_map=mu_map,
        voxel_size_mm=4.0,
    )
    connector.set_energy_windows([126], [154], [0])
    connector.add_runtime_switch("FI", "tc99m")
    connector.add_runtime_switch("CC", "ma-lehr")
    connector.add_runtime_switch("NN", 1)
    connector.add_runtime_switch("RR", 12345)

    outputs = connector.run()
    total = outputs["tot_w1"].projection
    print(total.shape)

Runtime Control
---------------

``output_prefix`` must start with a lowercase letter or digit and contain only
lowercase letters, digits and ``_``. SIMIND lowercases file names, a dot would
be read as a file extension, and a hyphen is an argument delimiter: a prefix
containing ``-`` splits SIMIND's filename arguments and the run silently
produces no output. The prefix is also capped at 47 characters, so the longest
name SIMIND writes (``{prefix}_tot_w{N}.a00``, up to 12 characters at the
99-window maximum) stays inside its fixed 60-character output buffer; a longer
name is truncated by SIMIND, dropping the final ``.h00``/``.a00`` character so
no readable SPECT header remains. The connector raises ``ValueError`` at
construction and re-checks the prefix in ``run()``; the executor raises
``SimulationError`` before it launches SIMIND.

``RuntimeOperator`` carries per-run switches and an optional orbit file. The
orbit file must point at an existing regular file: a missing path raises
``FileNotFoundError`` and a directory raises ``ValueError``. Switch values are
validated when they are set and again by the executor before the command is
built.

.. code-block:: python

    from simind_python_connector import RuntimeOperator, SimindPythonConnector

    outputs = connector.run(
        RuntimeOperator(switches={"NN": 1, "RR": 12345})
    )

``set_mpi(processes, split_projections=False)`` runs SIMIND under ``mpirun``
with the ``simind_mpi`` binary; ``processes`` must be an integer >= 1 (Python
or NumPy integers, never ``bool``). The executor accepts Python and NumPy
integers, integral reals and ASCII digit strings for the process count and
raises ``SimulationError`` for anything else before launch. ``NN`` must be an
integer >= 1.

.. code-block:: python

    connector.set_mpi(4)

Input Validation
----------------

Validation runs where it can still do good. ``output_prefix`` is checked when
the connector is constructed, and ``run()`` re-checks the prefix and the
effective scoring routine before it saves the configuration or clears stale
outputs. An invalid ``RuntimeOperator`` (switch values, ``orbit_file``) is
rejected before the orbit file is staged and before prior outputs are removed;
the native and PyTomography adaptors apply the same checks before inputs are
written. ``configure_voxel_phantom`` checks the arrays, the scoring routine and
``quantization_scale`` before the configuration is updated or the source and
density inputs are written, and the energy-window bounds are validated before
the window file is written (at most 99 windows). The executor then re-validates
the output prefix, the MPI process count and every runtime switch value
immediately before it builds the SIMIND command, raising ``SimulationError``
for an invalid value. An unsupported setup therefore fails with a clear message
instead of an obscure one after the run; failures that only SIMIND can detect
(a missing isotope or spectra input, for example) still surface during the run
as ``SimulationError``. Missing inputs such as the configuration file, the
orbit file or the output headers raise ``FileNotFoundError``.

Only the scoring routines DUMMY (0), SCATTWIN (1) and PENETRATE (4) are
supported by the Interfile connector. LIST_MODE (2) and FORCED_COLLIMATION (3)
write outputs the connector cannot read, so they are rejected at preflight;
they remain available through lower-level use of the package.

A literal 1x1x1 voxel phantom is rejected (SIMIND does not support it), so pad
the source and mu map to at least two voxels in one dimension. A non-zero
source that ``quantization_scale`` rounds to all zeros is also rejected;
increase ``quantization_scale`` so the source survives quantization. An
all-zero source is accepted unchanged.

Adaptor Workflows
-----------------

Use adaptors when you want connector-managed SIMIND execution plus native
objects for a target reconstruction package.

``get_primary_output()`` returns the window's primary projection: SIMIND's own
``pri_wN`` when it wrote one, otherwise the derived ``tot_wN - sca_wN`` the
connector writes for order-0 windows. It raises ``RuntimeError`` before a
successful run and ``KeyError`` when that window has no primary output.

STIR adaptor
~~~~~~~~~~~~

.. code-block:: python

    from simind_python_connector import StirSimindAdaptor
    from simind_python_connector.configs import get

    adaptor = StirSimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir="output/stir_adaptor",
        output_prefix="stir_case01",
    )
    adaptor.set_source(stir_source)
    adaptor.set_mu_map(stir_mu_map)
    adaptor.set_energy_windows([75], [225], [0])
    adaptor.add_runtime_switch("FI", "y90_tissue")
    adaptor.add_runtime_switch("CC", "ma-megp")
    adaptor.add_runtime_switch("RR", 12345)

    outputs = adaptor.run()
    stir_total = outputs["tot_w1"]

SIRF adaptor
~~~~~~~~~~~~

.. code-block:: python

    from simind_python_connector import SirfSimindAdaptor
    from simind_python_connector.configs import get

    adaptor = SirfSimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir="output/sirf_adaptor",
        output_prefix="sirf_case01",
    )
    adaptor.set_source(sirf_source)
    adaptor.set_mu_map(sirf_mu_map)
    adaptor.set_energy_windows([75], [225], [0])
    adaptor.add_runtime_switch("FI", "y90_tissue")
    adaptor.add_runtime_switch("CC", "ma-megp")
    adaptor.add_runtime_switch("RR", 12345)

    outputs = adaptor.run()
    sirf_total = outputs["tot_w1"]

PyTomography adaptor
~~~~~~~~~~~~~~~~~~~~

The adaptor returns PyTomography-compatible tensors and output headers.
Build the system matrix directly with PyTomography APIs.

.. code-block:: python

    import torch
    from pytomography.io.SPECT import simind as pytomo_simind
    from pytomography.projectors.SPECT import SPECTSystemMatrix

    from simind_python_connector import PyTomographySimindAdaptor
    from simind_python_connector.configs import get

    adaptor = PyTomographySimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir="output/pytomo_adaptor",
        output_prefix="pytomo_case01",
    )
    adaptor.set_source(source_tensor_xyz)
    adaptor.set_mu_map(mu_tensor_xyz)
    adaptor.set_energy_windows([75], [225], [0])
    adaptor.add_runtime_switch("FI", "y90_tissue")
    adaptor.add_runtime_switch("CC", "ma-megp")
    adaptor.add_runtime_switch("RR", 12345)
    adaptor.run()

    projections = adaptor.get_total_output(window=1).to(dtype=torch.float32)
    header = adaptor.get_output_header_path("tot_w1")
    object_meta, proj_meta = pytomo_simind.get_metadata(str(header))

    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms=[],
        proj2proj_transforms=[],
        object_meta=object_meta,
        proj_meta=proj_meta,
    )

Density Conversion
------------------

The package includes HU-to-density conversion utilities, including the
Schneider2000 model.

.. code-block:: python

    import numpy as np
    from simind_python_connector.converters.attenuation import hu_to_density_schneider

    hu_image = np.array([[-1000, 0, 500], [800, 1200, 2000]])
    density_map = hu_to_density_schneider(hu_image)
