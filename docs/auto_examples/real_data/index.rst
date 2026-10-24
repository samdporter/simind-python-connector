

.. _sphx_glr_auto_examples_real_data:

Real Y-90 data
--------------

These pages are restricted-data case studies, not generally reproducible
tutorials. They use a measured Y-90 phantom acquisition: a Mediso AnyScan
bremsstrahlung study with 120 views over 360 degrees, 128 x 128 bins of
4.2578 mm and a 75-225 keV window, together with the CT series of the same
phantom. The reconstruction grid is the vendor ``umap_zoomed`` grid of 128
voxels at 4.2578 mm.

The data are not redistributed at this time. There is no public download,
DOI, or other access route, and the pages cannot be reproduced without the
restricted input. Setting ``SIMIND_CONNECTOR_Y90_DATA`` to a local copy is
the only current route; see :doc:`/datasets` for the availability statement.

For that local copy, provide a directory containing ``peak.dcm``, ``peak.hs``,
``peak.s``, ``umap_zoomed.hv``, ``umap_zoomed.v`` and the ``ctac`` directory
of DICOM slices. The pages read the CT series from that directory and write
their figures and summaries under
``$SIMIND_EXAMPLES_OUTPUT/real_data/ct_conversion`` (or ``output`` when the
variable is unset).

The three CT conversion case studies show the same registration chain with a
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

The pages need NumPy, pydicom, matplotlib and SciPy. The SIRF case study
also needs nibabel. All three need the matching backend.

The remaining two case studies use the same restricted data:

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

   /auto_examples/real_data/plot_16a_stir_ct_conversion
   /auto_examples/real_data/plot_16b_sirf_ct_conversion
   /auto_examples/real_data/plot_16c_pytomography_ct_conversion
   /auto_examples/real_data/plot_17_y90_effective_energy
   /auto_examples/real_data/plot_18_y90_mc_scatter

