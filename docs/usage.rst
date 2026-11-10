.. _usage:

Usage Guide
===========

Quick Start
-----------

Use ``SimindPythonConnector`` when you want direct Python control of SIMIND
inputs/outputs without any reconstruction-package dependency.

The Python package does not bundle SIMIND. Pair the executable and ``smc_dir``
from the same release. The executor uses ``SIMIND_BIN`` when set and otherwise
searches ``PATH``; see :doc:`installation`.

SIMIND v8 is the supported engine for this release. Set ``SIMIND_VERSION=8``,
or pass ``simind_version=8`` to the connector/adaptor constructor. An explicit
constructor value takes precedence over the environment and the default is 8.

``simind_version=9`` remains available for experimental use, but v9 is
unvalidated in this release. Its voxel-map compensation requires the default
source/density orientations (``DI=0``, Index 32 = 0, and no ``OR`` override);
other map orientations fail before SIMIND starts.

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
        simind_version=8,
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

``output_prefix`` must match ``[a-z0-9][a-z0-9_]*`` and is capped at 47
characters. The connector checks it at construction and again in ``run()``; the
executor rejects an invalid prefix before launching SIMIND.

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

By default every process simulates all projections and SIMIND sums them.
Without an ``RR`` switch, ``simind_mpi`` seeds process ``r`` (counting from 0)
as ``RR = r + 1``, so do not combine replicate output with serial runs that use
those small ``RR`` values. A fixed ``RR`` would repeat the same histories in
every process, so ``run()`` refuses it with more than one replicate process.
``split_projections=True`` (``/MP``) shares the projections out instead; Index
29 must be a multiple of ``processes``, a fixed ``RR`` is allowed, and
PENETRATE is refused because ``simind_mpi`` writes only the primary process's
PENETRATE projections.

.. code-block:: python

    connector.set_mpi(4)

Input Validation
----------------

Inputs are checked before managed files are written or previous outputs are
removed. Invalid values raise ``ValueError``; missing files raise
``FileNotFoundError``. Failures reported by SIMIND raise ``SimulationError``.

Only the scoring routines DUMMY (0), SCATTWIN (1) and PENETRATE (4) are
supported by the Interfile connector. LIST_MODE (2) and FORCED_COLLIMATION (3)
write outputs the connector cannot read, so they are rejected at preflight;
they remain available through lower-level use of the package.

``set_energy_windows`` writes a SCATTWIN window file. DUMMY and PENETRATE use
the global energy thresholds in Indexes 20 (upper) and 21 (lower) instead.

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
