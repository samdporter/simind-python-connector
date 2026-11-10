Custom configuration
====================

Prerequisites
-------------

The connector and its bundled ``input.smc`` template. No SIMIND run is needed.

Cost and runtime
----------------

Seconds of Python work. The exported YAML files are written under
``$SIMIND_EXAMPLES_OUTPUT/custom_config``.

.. literalinclude:: scripts/plot_04_custom_config.py
   :language: python
   :linenos:

Download the script: :download:`plot_04_custom_config.py
<scripts/plot_04_custom_config.py>`.

Workflow
--------

The script starts from the bundled template and sets collimator geometry,
intrinsic and energy resolution, photon energy and the energy-window
thresholds. It exports the configuration to YAML, reloads that YAML into a
fresh configuration, changes the history count and pulse-height flag, validates
the result and exports it again. The printed hole size confirms that the
reloaded values survived the round trip.

.. note::

   This page has no retained figure; it illustrates a configuration workflow,
   not a simulation result.
