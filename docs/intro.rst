.. _introduction:

Introduction
============

simind-python-connector is a Python toolkit for SIMIND SPECT workflows.

Disclaimer
----------

This project is independent and is **not affiliated with, endorsed by, or
maintained by** the SIMIND project or Lund University.

Entry points:

- ``SimindPythonConnector`` runs SIMIND and returns NumPy-first outputs.
- ``StirSimindAdaptor`` bridges SIMIND outputs into STIR-native objects.
- ``SirfSimindAdaptor`` bridges SIMIND outputs into SIRF-native objects.
- ``PyTomographySimindAdaptor`` bridges SIMIND outputs into torch/PyTomography
  workflows.

Reconstruct the generated projections with STIR, SIRF or PyTomography's own
tools. See :doc:`geometry` for the axis conventions.
