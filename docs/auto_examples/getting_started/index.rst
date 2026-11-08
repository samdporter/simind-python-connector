

.. _sphx_glr_auto_examples_getting_started:

Getting started
---------------


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example runs SIMIND with voxelized NumPy inputs and reads projections back as NumPy arrays via SimindPythonConnector. It uses the default source quantization and a history multiplier large enough for the body and the hot sphere to be resolved in the projection, rather than a smoke-test run.">

.. only:: html

  .. image:: /auto_examples/getting_started/images/thumb/sphx_glr_plot_01_basic_simulation_thumb.png
    :alt:

  :doc:`/auto_examples/getting_started/plot_01_basic_simulation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Basic simulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example runs the same simulation twice with two independent random seeds and compares the projections. The only runtime switch that differs is RR, so the difference isolates Monte Carlo noise, not a change of physics.">

.. only:: html

  .. image:: /auto_examples/getting_started/images/thumb/sphx_glr_plot_02_runtime_switch_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/getting_started/plot_02_runtime_switch_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Seed reproducibility</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example runs a multi-window (TEW-style) simulation of Lu-177 with the pure Python connector and builds a triple-energy-window scatter correction. It uses the default source quantization, a history multiplier of 10 and four 360-degree views of 32 x 32 bins at 4 mm.">

.. only:: html

  .. image:: /auto_examples/getting_started/images/thumb/sphx_glr_plot_03_multi_window_thumb.png
    :alt:

  :doc:`/auto_examples/getting_started/plot_03_multi_window`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-window simulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create and modify SIMIND configurations, including exporting them to YAML for easy editing and version control.">

.. only:: html

  .. image:: /auto_examples/getting_started/images/thumb/sphx_glr_plot_04_custom_config_thumb.png
    :alt:

  :doc:`/auto_examples/getting_started/plot_04_custom_config`

.. raw:: html

      <div class="sphx-glr-thumbnail-title"># sphinx_gallery_thumbnail_path = '_static/custom_config_summary.png'</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example compares the SCATTWIN and PENETRATE scoring routines of SIMIND on the same phantom, with the same geometry, histories and detector acceptance. It replaces the earlier 16 x 16 preview with a 32 x 32 grid of 4 mm bins and a history budget large enough for the component decomposition.">

.. only:: html

  .. image:: /auto_examples/getting_started/images/thumb/sphx_glr_plot_05_scattwin_vs_penetrate_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/getting_started/plot_05_scattwin_vs_penetrate_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SCATTWIN vs PENETRATE comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the Schneider2000-based density conversion, which is more accurate than the traditional bilinear model.">

.. only:: html

  .. image:: /auto_examples/getting_started/images/thumb/sphx_glr_plot_06_schneider_density_conversion_thumb.png
    :alt:

  :doc:`/auto_examples/getting_started/plot_06_schneider_density_conversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Schneider density conversion</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/getting_started/plot_01_basic_simulation
   /auto_examples/getting_started/plot_02_runtime_switch_comparison
   /auto_examples/getting_started/plot_03_multi_window
   /auto_examples/getting_started/plot_04_custom_config
   /auto_examples/getting_started/plot_05_scattwin_vs_penetrate_comparison
   /auto_examples/getting_started/plot_06_schneider_density_conversion

