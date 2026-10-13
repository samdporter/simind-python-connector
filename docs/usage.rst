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

    source = np.zeros((32, 32, 32), dtype=np.float32)  # z, y, x
    source[12:20, 12:20, 12:20] = 1.0
    mu_map = np.zeros_like(source)
    mu_map[source > 0] = 0.15

    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir="output/basic",
        output_prefix="case01",
        quantization_scale=0.05,
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
produces no output. The prefix is also capped at 47 characters: SIMIND V8
copies output names into a fixed 60-character buffer and silently truncates
longer ones, dropping the final ``.h00``/``.a00`` character so no readable
SPECT header remains. Measured on a real SIMIND V8.0 run, 49 characters is the
exact maximum prefix for ``{prefix}_tot_w1.a00``; a 50-character prefix
truncates it. The 47-character cap is a conservative policy with a small margin
below that measured boundary, not an engine-measured limit. The longest
engine-written name adds a ``_tot_w{N}.a00`` suffix (10 characters plus the
digits of the window number), and when every scatter order is below 1 the
window file gains the required extra line, which adds one more output set, so
the suffix can reach 13 characters and a 47-character prefix plus that suffix
uses the 60-character buffer exactly. To stay inside that budget, at most 99
energy windows are accepted; a larger set raises ``ValueError``.

``RuntimeOperator`` carries per-run switches, an orbit file and an optional
timeout in seconds. A run that exceeds the timeout stops and raises
``SimulationError``; by default there is no timeout. ``orbit_file`` must point
at an existing regular file; a directory is rejected with ``ValueError``.

.. code-block:: python

    from simind_python_connector import RuntimeOperator, SimindPythonConnector

    outputs = connector.run(
        RuntimeOperator(switches={"NN": 1, "RR": 12345}, timeout=600)
    )

``MP`` (the MPI process count) must be an integer >= 1; ``MP=0`` is rejected
instead of silently running a single process. ``MP`` accepts positive integers
(including NumPy integers), integral floats and ASCII digit strings. ``NN``
accepts positive integers and integral floats; an integral-float ``NN`` value
is written to the command line as an integer (for example ``/NN:2``) while the
switch value you stored is unchanged.

.. code-block:: python

    connector.add_runtime_switch("MP", 4)

Input Validation
----------------

Validation runs where it can still do good. ``output_prefix`` is checked when
the connector is constructed, and ``run()`` re-checks the output directory, the
prefix and the scoring routine before it saves the configuration or clears
stale outputs. An invalid ``RuntimeOperator`` (timeout, switch values,
``orbit_file``) is rejected before the orbit file is staged and before prior
outputs are removed. ``configure_voxel_phantom`` checks the arrays, the scoring
routine and ``quantization_scale`` before the configuration is updated or the
source and density inputs are written, and energy-window bounds and scatter
orders are validated before the window file is written. The executor then
re-validates the output prefix and every runtime switch value — not only ``MP``
and ``NN`` — immediately before it builds the SIMIND command, raising
``SimulationError`` for an invalid value. An unsupported setup therefore fails
with a clear message instead of an obscure one after the run; failures that
only SIMIND can detect (a missing isotope or spectra input, for example) still
surface during the run as ``SimulationError``.

Only the scoring routines DUMMY (0), SCATTWIN (1) and PENETRATE (4) are
supported by the Interfile connector. LIST_MODE (2) and FORCED_COLLIMATION (3)
write outputs the connector cannot read, so they are rejected at preflight;
they remain available through lower-level use of the package.

A literal 1x1x1 voxel phantom is rejected (SIMIND does not support it), so pad
the source and mu map to at least two voxels in one dimension. A non-zero
source that ``quantization_scale`` rounds to all zeros is also rejected;
increase ``quantization_scale`` so the source survives quantization.

Filesystem failures in connector-managed steps (writing the source, density
and window inputs, staging an orbit file, clearing stale outputs, converting
headers) raise ``OutputError`` rather than a bare ``OSError``. ``OutputError``
is itself an ``OSError``, so existing ``except OSError`` handlers keep
catching these failures. Missing SPECT or PENETRATE output headers raise a
``FileNotFoundError`` (an ``OutputError`` too), as they did before, and
missing inputs such as the configuration file or the orbit file still raise
``FileNotFoundError``.

Adaptor Workflows
-----------------

Use adaptors when you want connector-managed SIMIND execution plus native
objects for a target reconstruction package.

Primary projections are unsupported by these adaptors. ``get_primary_output()``
raises ``NotImplementedError`` once outputs exist and ``RuntimeError`` before a
successful run. The adaptors consume SIMIND's standard SPECT output, which has
no standalone primary file; primary projections can still be produced through
lower-level SIMIND use with ``/CA:2``. Use ``get_total_output()`` and
``get_scatter_output()`` instead.

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
