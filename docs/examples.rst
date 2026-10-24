Examples
========

Examples 01–18 are in the :doc:`gallery <auto_examples/index>`: runnable
scripts with their figures and printed output. Follow :doc:`tutorial` for the
recommended order, prerequisites, and what to inspect.

Build the gallery locally with ``bash scripts/build_gallery.sh``. Read the Docs
shows the committed outputs and does not run the examples.

The gallery has six sections:

- :doc:`Getting started <auto_examples/getting_started/index>`: core NumPy
  simulations, energy windows, configurations, scoring, and density conversion.
- :doc:`Backends <auto_examples/backends/index>`: STIR, SIRF and PyTomography
  reconstruction, and projection geometry from DICOM.
- :doc:`Phantoms and reconstruction <auto_examples/phantoms/index>`: SIMIND's
  analytic phantom library, and a NEMA/IEC simulation reconstructed with SIRF
  OSEM.
- :doc:`MC scatter and normalisation <auto_examples/mc_scatter/index>`: Monte
  Carlo scatter in SIRF OSEM, the scale of the scatter estimate, and residual
  correction.
- :doc:`Regularised reconstruction with CIL <auto_examples/cil/index>`: scatter
  correction with CIL ISTA and a relative-difference prior with SVRG.
- :doc:`Real Y-90 data <auto_examples/real_data/index>`: CT-to-SPECT
  conversion in STIR, SIRF and PyTomography, the Y-90 effective attenuation
  energy, and a Monte Carlo scatter comparison published as a diagnostic
  negative result. These are rendered case studies on restricted input data,
  not reproducible tutorials; see :doc:`datasets`.

Examples that simulate need a working SIMIND installation, and backend examples
need that backend installed. The NEMA examples also need phantomgen; its
install command is in the tutorial. Gallery outputs are written under
``$SIMIND_EXAMPLES_OUTPUT`` (default ``./output``). Lower ``quantization_scale``,
fewer views, or a mostly-zero source map can shorten a smoke run; quantitative
runs need appropriate statistics and calibration.
