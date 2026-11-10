Reconstruction with SIMIND Corrections
======================================

The optional ``simind_python_connector.recon`` subpackage lets SIRF and CIL
reconstructions use a SIMIND-simulated additive term, refreshed from the
current image at chosen intervals. The core package never imports it.

Model
-----

The forward model for the measured data :math:`y` is

.. math::

   \mu(x) = A x + \eta_k, \qquad \eta_k = \max(D_k, \varepsilon), \qquad
   D_k = (1 - \alpha) D_{k-1} + \alpha\, \mathrm{estimate}(x_k)

- :math:`A` is the fast SIRF linear model (e.g. SPECTUB with attenuation).
- ``estimate`` is a correction model evaluated at the image of update
  :math:`k`, here SIMIND scatter.
- :math:`\alpha \in (0, 1]` (``damping``) averages out Monte Carlo noise, but
  damping below 1 needs a non-zero starting term: starting from zero, the
  term lags the estimate, and with roughly constant statistics this biases
  the scatter low. At damping 0.5 a single update gives
  :math:`D_1 = 0.5 \times \mathrm{estimate}`; after three roughly constant
  updates the term is at 87.5% of the estimate. Start non-zero with
  ``CorrectionCallback.prime`` or a first estimate as ``initial_additive``.
- :math:`\varepsilon` (``floor``) keeps the Poisson mean positive. Only
  :math:`\eta_k` is floored, so that later damped updates combine correctly.

Correction behaviour
--------------------

- The correction goes into the data fidelity, KL(y; A x + η), so ``A`` stays
  linear and CIL algorithms, operator norms and preconditioners keep working.
- After an additive refresh, stored stochastic gradients refer to the old
  objective. ``refresh_stochastic_state`` recomputes them for SVRG and SAGA.
- CIL's ``run(N)`` calls callbacks at iterations ``0..N``. A refresh at
  iteration :math:`k` takes effect from iteration :math:`k + 1`, and a refresh
  in the final callback cannot affect the returned reconstruction.
- With PENETRATE, the scatter estimate is all events minus the geometrically
  collimated primary (``b01 - b02``): phantom scatter, septal penetration,
  collimator scatter and X-rays, i.e. everything ``A`` does not model. With
  SCATTWIN it is the window's scatter output, which holds phantom scatter only;
  use PENETRATE for high-energy isotopes.
- SIMIND's output does not follow the image magnitude (the source map is
  always scaled to the same number of photon histories), so every update is
  rescaled: by matching ``b02`` to ``A x``
  (``reference_normaliser(full_model)``, the ratio of sums, see
  :doc:`normalisation`), or by giving SIMIND the activity when ``x`` is in
  known activity units.

CIL acquisition storage
-----------------------

In SIRF 3.10.1 / STIR 6.4, native acquisition subsetting crashes on file-backed
data, and selecting memory storage after an object is created does not convert
that object. ``build_subset_objectives`` and ``set_subset_additive`` therefore
select process-wide SIRF memory storage before cloning their acquisition
inputs; caller-owned objects are unchanged.

For data creation and subsetting outside these drivers, select memory storage
before creating or loading acquisition data:

.. code-block:: python

    import sirf.STIR as sirf

    sirf.AcquisitionData.set_storage_scheme("memory")
    # Create or load acquisition data after selecting the scheme.

Do not change the storage scheme concurrently with driver calls or
acquisition creation.

SIRF OSEM
---------

.. code-block:: python

    from simind_python_connector.recon import (
        AdditiveUpdater, ScatterCorrection, SimindProjector, UpdateSchedule,
        reference_normaliser,
    )
    from simind_python_connector.recon.osem import run_osem_with_corrections

    projector = SimindProjector(adaptor, template, mu_map,
                                normalise=reference_normaliser(full_model),
                                seed=1)
    correction = ScatterCorrection(projector, smoothing_fwhm_bins=3.0,
                                   smooth_across_views=True)
    zero_additive = measured.get_uniform_copy(0)
    updater = AdditiveUpdater(correction, UpdateSchedule(every=10),
                              zero_additive, damping=0.5)
    image = run_osem_with_corrections(measured, make_model, initial, updater,
                                      num_subsets=5, subiterations_per_update=10,
                                      num_updates=3)

``template`` is the projection-geometry template passed to
``SimindProjector``, not the measured acquisition. ``measured`` enters only as
the OSEM data and the zero additive image. ``full_model`` is the full-data
linear model (no additive term), set up with the measured data and kept
separate for reference normalisation; to scale the scatter estimate to the
measured data instead, see :ref:`scaling-scatter-to-measured-data`.
``make_model`` returns a fresh, configured, not-yet-set-up acquisition model on
every call, with independent state and the same linear operator settings; for a
matrix-based model, build a fresh matrix too. The driver builds a fresh
objective and reconstructor on each pass.

CIL
---

.. code-block:: python

    from cil.optimisation.algorithms import ISTA
    from cil.optimisation.functions import IndicatorBox, SumFunction
    from simind_python_connector.recon.cil import (
        CorrectionCallback, build_subset_objectives,
    )

    objectives = build_subset_objectives(measured, zero_additive, make_model,
                                         num_subsets=5, initial_image=initial)
    callback = CorrectionCallback(updater, objectives)
    callback.prime(initial)      # optional: start from a SIMIND estimate
    algorithm = ISTA(initial=initial, f=SumFunction(*objectives.functions),
                     g=IndicatorBox(lower=0), step_size=step)
    algorithm.run(100, callbacks=[callback])

The same objectives work inside ``SVRGFunction`` or ``SAGAFunction``; during
the run, the callback refreshes their stored gradients after each scheduled
additive refresh.

Update intervals
----------------

``UpdateSchedule(every, first=None, stop_at=None)`` counts the CIL algorithm's
iterations, i.e. subiterations for algorithms over subsets. ``first=0``
refreshes in the initial callback, after initial objective evaluation.
``stop_at`` is exclusive: ``run(48)`` with ``every=12`` and ``stop_at=48``
refreshes at iterations 12, 24 and 36, each consumed by a later iteration.

The OSEM outer loop calls ``update_now`` directly, with cadence
``subiterations_per_update``; ``num_updates`` controls how many correction
updates are made.

Each update costs one SIMIND run: update every 1-2 epochs early on, when the
image changes quickly, and use ``stop_at`` to freeze the estimate for the final
CIL iterations.

Residual correction (Fu & Qi)
-----------------------------

Fu and Qi (Med Phys 37:704, 2010) reconstruct with a fast model while matching
the forward projection of an accurate one, by adding the residual to the
background:

.. math::

   x^{(n+1)} = \arg\max_{x \ge 0} \Psi_\mathrm{fast}\big(x \mid y,\;
   r + (A_\mathrm{acc} - A_\mathrm{fast})\, x^{(n)}\big)

Only forward projections use the accurate model, and only at updates. This
is the same engine as above with ``ResidualCorrection`` as the correction
model; using SIMIND as the forward operator is the special case
``A_acc = SIMIND``.

.. code-block:: python

    from simind_python_connector.recon import (
        AdditiveUpdater, ResidualCorrection, SimindComponent, UpdateSchedule,
    )

    accurate = SimindComponent(projector, "all_interactions")
    correction = ResidualCorrection(accurate, fast_model, base_additive=zero)
    updater = AdditiveUpdater(correction, UpdateSchedule(every=10), zero, damping=1.0)

``fast_model`` is the full-data linear model without an additive term. Give
the reconstruction a separate model object, because the drivers set additive
terms on theirs. For a SIRF accurate model, pass ``accurate_model.direct``.

.. list-table:: Recipes
   :header-rows: 1

   * - Recipe
     - Accurate model
     - SIMIND settings
     - ``base_additive``
     - Corrects
   * - Geometric residual
     - ``SimindComponent(p, "geom_coll_primary")``
     - PENETRATE, Index 53 = 0, Index 19 = 2
     - a scatter estimate (fixed, or from ``ScatterCorrection``)
     - collimator and detector response only
   * - Full residual
     - ``SimindComponent(p, "all_interactions")``
     - PENETRATE, Index 53 = 1, Index 19 = 3
     - background only (usually zero)
     - resolution, scatter, penetration and collimator scatter in one run
   * - PSF residual (no SIMIND)
     - ``psf_model.direct`` (SPECTUB with a resolution model, or a
       ``SeparableGaussianImageFilter`` image processor)
     - none
     - a scatter estimate
     - resolution modelling at fast-model cost

In both SIMIND recipes the projector uses
``reference_normaliser(fast_model, "geom_coll_primary")``.

The theorem behind the method converges to the accurate-model MAP solution
when that solution is unique, each inner correction step converges, and
:math:`A_\mathrm{acc} = A_\mathrm{fast} B` with :math:`B` having positive
eigenvalues (e.g. an image blur). Scatter and septal penetration do not
obviously fit, and this engine floors the additive term, which can discard a
negative residual: the theorem therefore does not directly apply. Monitor
convergence instead: ``effective_objective(measured, correction)`` is the
accurate model's Poisson data term at the image of the last update. Log it
from ``CorrectionCallback(on_update=...)``. A rising trend across updates,
beyond Monte Carlo noise, means the update interval or the damping needs
changing. The PSF-residual container test is a fixed-budget data-domain
regression check, not a convergence proof.

The :doc:`residual correction tutorial
<example_pages/residual_correction>` compares the fast model alone, the SIMIND
scatter estimate and the full residual. The :doc:`SIRF OSEM
<example_pages/mc_scatter_osem>` and :doc:`CIL ISTA <example_pages/cil_scatter>`
tutorials show complete workflows; see :doc:`the example pages
<example_pages/index>`.
