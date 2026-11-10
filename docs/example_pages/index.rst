Examples
========

Download each runnable example from its page. Run scripts from the repository
root; outputs use ``SIMIND_EXAMPLES_OUTPUT`` or default to ``output``.

Figures are from recorded example runs and were not regenerated for these
pages.

Monte Carlo reconstruction examples
-----------------------------------

The five correction tutorials share a 60-view NEMA/IEC simulation with known
activity, an 8:1 hot-to-background ratio and three paired Poisson realisations.
Generating the shared expectation takes about an hour with eight MPI processes;
later tutorials reuse it.

Recovery is the sphere-to-background mean ratio divided by the true ratio. Mask
edges reduce even the truth image's score, shown in the recovery plots. Error
bars describe Poisson variation, not Monte Carlo uncertainty.

Install SIRF and phantomgen as described in :doc:`/installation` and
:doc:`/phantoms`; the CIL examples also require CIL.

.. toctree::
   :maxdepth: 1

   basic_simulation
   multi_window
   custom_configuration
   scattwin_vs_penetrate
   density_conversion
   phantom_library
   mc_scatter_osem
   scatter_normalisation
   residual_correction
   cil_scatter
   cil_rdp_svrg
   stir_osem
   sirf_osem
   pytomography_osem
   dicom_recipe
   y90_ct_conversion
   y90_effective_energy
