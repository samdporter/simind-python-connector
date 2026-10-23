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

    <div class="sphx-glr-thumbcontainer" tooltip="Create a STIR projection object from a DICOM NM projection file. The data is a Y-90 bremsstrahlung study: a Mediso AnyScan acquisition of 120 views over 360 degrees, 128 x 128 bins of 4.2578 mm and a 75-225 keV window. The script takes no path argument: fetch_y90_phantom reads the directory named by SIMIND_CONNECTOR_Y90_DATA.">

.. only:: html

  .. image:: /auto_examples/backends/images/thumb/sphx_glr_plot_08a_stir_adaptor_from_dicom_thumb.png
    :alt:

  :doc:`/auto_examples/backends/plot_08a_stir_adaptor_from_dicom`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">STIR projection from DICOM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Create a SIRF projection object from a DICOM NM projection file. The data is a Y-90 bremsstrahlung study: a Mediso AnyScan acquisition of 120 views over 360 degrees, 128 x 128 bins of 4.2578 mm and a 75-225 keV window. The script takes no path argument: fetch_y90_phantom reads the directory named by SIMIND_CONNECTOR_Y90_DATA.">

.. only:: html

  .. image:: /auto_examples/backends/images/thumb/sphx_glr_plot_08b_sirf_adaptor_from_dicom_thumb.png
    :alt:

  :doc:`/auto_examples/backends/plot_08b_sirf_adaptor_from_dicom`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SIRF projection from DICOM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Create a PyTomography projection tensor from a DICOM NM projection file, following PyTomography&#x27;s DICOM IO API. The data is a Y-90 bremsstrahlung study: a Mediso AnyScan acquisition of 120 views over 360 degrees, 128 x 128 bins of 4.2578 mm and a 75-225 keV window. The script takes no path argument: fetch_y90_phantom reads the directory named by SIMIND_CONNECTOR_Y90_DATA.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Simulate phantoms that need no hand-made voxel maps: SIMIND&#x27;s analytic multiple-inserts (Jaszczak-like) phantom and the NEMA/IEC body phantom from phantomgen. Both are Tc-99m runs with 24 views, so the projections are noisy.">

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

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reconstruct a simulated NEMA/IEC phantom acquisition twice with SIRF OSEM: once without scatter correction, and once with a SIMIND scatter estimate that is refreshed from the current image between OSEM passes.">

.. only:: html

  .. image:: /auto_examples/mc_scatter/images/thumb/sphx_glr_plot_11_mc_scatter_osem_thumb.png
    :alt:

  :doc:`/auto_examples/mc_scatter/plot_11_mc_scatter_osem`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Monte Carlo scatter correction in OSEM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A SIMIND run does not know the scale of the data. This example reconstructs one simulated NEMA/IEC phantom acquisition with SIRF OSEM and a SIMIND scatter estimate that is refreshed from the current image, and compares five ways of putting that estimate on the scale of the data:">

.. only:: html

  .. image:: /auto_examples/mc_scatter/images/thumb/sphx_glr_plot_12_scatter_normalisation_thumb.png
    :alt:

  :doc:`/auto_examples/mc_scatter/plot_12_scatter_normalisation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Normalising the Monte Carlo scatter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reconstruct a simulated NEMA/IEC phantom acquisition with SIRF OSEM three times: with the fast SPECTUB model only, with a SIMIND scatter estimate, and with SIMIND as the accurate forward model in a Fu &amp; Qi residual correction.">

.. only:: html

  .. image:: /auto_examples/mc_scatter/images/thumb/sphx_glr_plot_13_residual_correction_thumb.png
    :alt:

  :doc:`/auto_examples/mc_scatter/plot_13_residual_correction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Residual correction with SIMIND as the accurate model</div>
    </div>


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

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reconstruct a simulated NEMA/IEC phantom acquisition twice with CIL&#x27;s ISTA: once without scatter correction, and once with a SIMIND scatter estimate that is refreshed from the current image during the run.">

.. only:: html

  .. image:: /auto_examples/cil/images/thumb/sphx_glr_plot_14_cil_scatter_thumb.png
    :alt:

  :doc:`/auto_examples/cil/plot_14_cil_scatter`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Monte Carlo scatter correction with CIL</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reconstruct a simulated NEMA/IEC phantom acquisition with a SIRF relative difference prior (RDP) and CIL&#x27;s stochastic variance reduced gradient (SVRG), and compare it with OSEM. Both reconstructions use a SIMIND scatter estimate that is refreshed from the current image during the run.">

.. only:: html

  .. image:: /auto_examples/cil/images/thumb/sphx_glr_plot_15_cil_rdp_svrg_thumb.png
    :alt:

  :doc:`/auto_examples/cil/plot_15_cil_rdp_svrg`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">RDP-regularised reconstruction with SVRG</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Real Y-90 data
--------------

These tutorials use a measured Y-90 phantom acquisition: a Mediso AnyScan
bremsstrahlung study with 120 views over 360 degrees, 128 x 128 bins of
4.2578 mm and a 75-225 keV window, together with the CT series of the same
phantom. The reconstruction grid is the vendor ``umap_zoomed`` grid of 128
voxels at 4.2578 mm.

TODO: A dataset download is coming, but is not yet available.

For now, set ``SIMIND_CONNECTOR_Y90_DATA`` to a directory containing ``peak.dcm``,
``peak.hs``, ``peak.s``, ``umap_zoomed.hv``, ``umap_zoomed.v`` and the
``ctac`` directory of DICOM slices. The tutorials read the CT series from that
directory and write their figures and summaries under
``$SIMIND_EXAMPLES_OUTPUT/real_data/ct_conversion`` (or ``output`` when the
variable is unset).

The three CT conversion tutorials show the same registration chain with a
different backend:

- ``plot_16a_stir_ct_conversion.py`` builds the CT in the STIR frame and
  resamples it with ``zoom_image``.
- ``plot_16b_sirf_ct_conversion.py`` writes the CT as a NIfTI image and
  resamples it with SIRF's NiftyResample.
- ``plot_16c_pytomography_ct_conversion.py`` uses PyTomography's CT reader and
  attenuation table with a corrected patient-to-SPECT affine.

All three sort the CT slices by ``ImagePositionPatient``, validate the series,
register the CT-derived attenuation map to ``umap_zoomed``, and stop before
writing accepted maps if the registration gate fails. The vendor attenuation
map has no recorded photon energy, so it is used only as a registration
reference and is never converted to density.

The tutorials need NumPy, pydicom, matplotlib and SciPy. The SIRF tutorial
also needs nibabel. All three need the matching backend.

The remaining two tutorials use the same dataset:

- ``plot_17_y90_effective_energy.py`` (SIRF) compares a single 150-keV
  attenuation map with one weighted by the detected 75-225 keV spectrum, using
  two otherwise identical reconstructions that both omit scatter.
- ``plot_18_y90_mc_scatter.py`` (SIRF) compares Monte Carlo scatter strategies
  on a common reconstruction schedule. It is an audited diagnostic negative
  result, not a passing quantitative tutorial: the global-scaling arm collapsed
  and is not quantitatively interpretable, the nominal sphere-to-liver ratios
  are sensitive to the frozen sphere centre and do not support a strategy
  ranking, and statistical adequacy was not established. The page reports
  every failed or inconclusive quality gate.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Convert the CT series of a measured Y-90 phantom acquisition into attenuation and density maps on the SPECT reconstruction grid with STIR.">

.. only:: html

  .. image:: /auto_examples/real_data/images/thumb/sphx_glr_plot_16a_stir_ct_conversion_thumb.png
    :alt:

  :doc:`/auto_examples/real_data/plot_16a_stir_ct_conversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">STIR CT conversion for Y-90</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Convert the CT series of a measured Y-90 phantom acquisition into attenuation and density maps on the SPECT reconstruction grid with SIRF.">

.. only:: html

  .. image:: /auto_examples/real_data/images/thumb/sphx_glr_plot_16b_sirf_ct_conversion_thumb.png
    :alt:

  :doc:`/auto_examples/real_data/plot_16b_sirf_ct_conversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SIRF CT conversion for Y-90</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Convert the CT series of a measured Y-90 phantom acquisition into attenuation and density maps on the SPECT reconstruction grid with PyTomography.">

.. only:: html

  .. image:: /auto_examples/real_data/images/thumb/sphx_glr_plot_16c_pytomography_ct_conversion_thumb.png
    :alt:

  :doc:`/auto_examples/real_data/plot_16c_pytomography_ct_conversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PyTomography CT conversion for Y-90</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Y-90 is a pure beta emitter: the 75-225 keV imaging window contains a broad bremsstrahlung and scatter continuum, not a photopeak. A single &quot;150 keV&quot; attenuation map is therefore an approximation. This tutorial compares it with a map weighted by the simulated detected spectrum.">

.. only:: html

  .. image:: /auto_examples/real_data/images/thumb/sphx_glr_plot_17_y90_effective_energy_thumb.png
    :alt:

  :doc:`/auto_examples/real_data/plot_17_y90_effective_energy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Y-90 effective attenuation energy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The history-doubling probe uses a reduced 60-view geometry to keep its cost down. The reference-sum and reference-trimmed arms collapsed in that probe (NN=1/NN=2 ratios 0.107/0.218 and 0.008/0.023), so the 179.48% largest change is a probe artefact rather than native-geometry evidence. The known-activity (+10.2%) and pixel-wise (+1.6%) changes are usable. A native-geometry history check was not established.">

.. only:: html

  .. image:: /auto_examples/real_data/images/thumb/sphx_glr_plot_18_y90_mc_scatter_thumb.png
    :alt:

  :doc:`/auto_examples/real_data/plot_18_y90_mc_scatter`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Y-90 reconstruction with Monte Carlo scatter: an audited negative result</div>
    </div>


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
   /auto_examples/real_data/index.rst



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
