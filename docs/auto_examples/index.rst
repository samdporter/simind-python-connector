:orphan:

Tutorials and examples
======================

Runnable tutorials for the SIMIND Python connector. Each example is a plain
Python script that can also be run from the repository root.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Getting started
---------------


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example runs SIMIND with voxelized NumPy inputs and reads projections back as NumPy arrays via SimindPythonConnector. It is a quick low-history run, so the projection is noisy.">

.. only:: html

  .. image:: /auto_examples/getting_started/images/thumb/sphx_glr_plot_01_basic_simulation_thumb.png
    :alt:

  :doc:`/auto_examples/getting_started/plot_01_basic_simulation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Basic simulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example runs two SIMIND cases with different runtime switch settings and compares the resulting projection view. It is a quick low-history run, so the seed-to-seed noise is large.">

.. only:: html

  .. image:: /auto_examples/getting_started/images/thumb/sphx_glr_plot_02_runtime_switch_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/getting_started/plot_02_runtime_switch_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Runtime-switch comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example runs a multi-window (TEW-style) simulation of Lu-177 with the pure Python connector and builds a triple-energy-window scatter correction. It is a quick low-history run, so the projections are noisy.">

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

      <div class="sphx-glr-thumbnail-title">Custom configuration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example compares the SCATTWIN and PENETRATE scoring routines of SIMIND on the same phantom using the pure Python connector. It is a quick low-history run, so the component counts are noisy.">

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

Backend adaptors
----------------


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="SIMIND simulation followed by STIR OSEM reconstruction. This is a quick low-history Tc-99m run on a tiny phantom with few projection views, so the projection and the reconstruction are mildly noisy.">

.. only:: html

  .. image:: /auto_examples/backends/images/thumb/sphx_glr_plot_07a_stir_adaptor_osem_thumb.png
    :alt:

  :doc:`/auto_examples/backends/plot_07a_stir_adaptor_osem`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">STIR adaptor OSEM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="SIMIND simulation followed by SIRF OSEM reconstruction. This is a quick low-history Tc-99m run on a tiny phantom with few projection views, so the projection and the reconstruction are mildly noisy.">

.. only:: html

  .. image:: /auto_examples/backends/images/thumb/sphx_glr_plot_07b_sirf_adaptor_osem_thumb.png
    :alt:

  :doc:`/auto_examples/backends/plot_07b_sirf_adaptor_osem`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SIRF adaptor OSEM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="SIMIND simulation followed by PyTomography OSEM reconstruction. This is a quick low-history Tc-99m run on a tiny phantom, so the projection and the reconstruction are mildly noisy. Only PyTomography and torch data objects are used in the adaptor and reconstruction path.">

.. only:: html

  .. image:: /auto_examples/backends/images/thumb/sphx_glr_plot_07c_pytomography_adaptor_osem_thumb.png
    :alt:

  :doc:`/auto_examples/backends/plot_07c_pytomography_adaptor_osem`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PyTomography adaptor OSEM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Create a STIR projection object from a DICOM NM projection file. The data is a Y-90 bremsstrahlung study: a Mediso AnyScan acquisition of 120 views over 360 degrees, 128 x 128 bins of 4.2578 mm and a 75-225 keV window. It is fetched with fetch_y90_phantom.">

.. only:: html

  .. image:: /auto_examples/backends/images/thumb/sphx_glr_plot_08a_stir_adaptor_from_dicom_thumb.png
    :alt:

  :doc:`/auto_examples/backends/plot_08a_stir_adaptor_from_dicom`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">STIR projection from DICOM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Create a SIRF projection object from a DICOM NM projection file. The data is a Y-90 bremsstrahlung study: a Mediso AnyScan acquisition of 120 views over 360 degrees, 128 x 128 bins of 4.2578 mm and a 75-225 keV window. It is fetched with fetch_y90_phantom.">

.. only:: html

  .. image:: /auto_examples/backends/images/thumb/sphx_glr_plot_08b_sirf_adaptor_from_dicom_thumb.png
    :alt:

  :doc:`/auto_examples/backends/plot_08b_sirf_adaptor_from_dicom`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SIRF projection from DICOM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Create a PyTomography projection tensor from a DICOM NM projection file, following PyTomography&#x27;s DICOM IO API. The data is a Y-90 bremsstrahlung study: a Mediso AnyScan acquisition of 120 views over 360 degrees, 128 x 128 bins of 4.2578 mm and a 75-225 keV window. It is fetched with fetch_y90_phantom.">

.. only:: html

  .. image:: /auto_examples/backends/images/thumb/sphx_glr_plot_08c_pytomography_adaptor_from_dicom_thumb.png
    :alt:

  :doc:`/auto_examples/backends/plot_08c_pytomography_adaptor_from_dicom`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PyTomography projection from DICOM</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Phantoms and reconstruction
---------------------------


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Simulate phantoms that need no hand-made voxel maps: SIMIND&#x27;s analytic multiple-inserts (Jaszczak-like) phantom and the NEMA/IEC body phantom from phantomgen. Both are quick Tc-99m runs with 24 views and few photon histories, so the projections are noisy.">

.. only:: html

  .. image:: /auto_examples/phantoms/images/thumb/sphx_glr_plot_09_phantom_library_thumb.png
    :alt:

  :doc:`/auto_examples/phantoms/plot_09_phantom_library`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Phantom library</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Simulate &quot;measured&quot; data from the NEMA/IEC body phantom with a known activity, reconstruct it with SIRF OSEM and measure the contrast recovery of the hot spheres.">

.. only:: html

  .. image:: /auto_examples/phantoms/images/thumb/sphx_glr_plot_10_nema_osem_thumb.png
    :alt:

  :doc:`/auto_examples/phantoms/plot_10_nema_osem`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">NEMA simulation and OSEM</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

MC scatter and normalisation
----------------------------


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Regularised reconstruction with CIL
-----------------------------------


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /auto_examples/getting_started/index.rst
   /auto_examples/backends/index.rst
   /auto_examples/phantoms/index.rst
   /auto_examples/mc_scatter/index.rst
   /auto_examples/cil/index.rst



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
