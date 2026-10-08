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
- :math:`\alpha \in (0, 1]` (``damping``) averages out Monte Carlo noise.
- :math:`\varepsilon` (``floor``) keeps the Poisson mean positive. Only
  :math:`\eta_k` is floored, so that later damped updates combine correctly.

Why it is built this way
------------------------

- The correction goes into the data fidelity, KL(y; A x + η), so ``A`` stays
  linear and CIL algorithms, operator norms and preconditioners keep working.
- CIL's ``KullbackLeibler`` copies η when it is created, so a refresh gives each
  subset objective a new ``KullbackLeibler`` rather than editing the old one.
- After a refresh, SVRG's snapshot gradient and SAGA's stored gradients belong
  to the old η and are recomputed (``refresh_stochastic_state``). SAGA uses
  CIL's public ``warm_start_approximate_gradients``. In the inspected
  CIL 26.0.1.dev0 / SIRF 3.10.1 installation, its NumPy sum loses the image
  container, so the driver then sums the stored images explicitly using the
  private ``_list_stored_gradients`` and ``_full_gradient_at_iterate`` fields.
  SVRG uses the private ``SVRGFunction._update_full_gradient_and_return``,
  because CIL has no public way to retake a snapshot.
- In the inspected CIL installation, ``run(N)`` calls callbacks at iterations
  ``0..N``. The initial iteration-0 callback follows initial objective
  evaluation and precedes the first algorithm update; later callbacks follow
  updates and iteration increments. A refresh at iteration :math:`k` takes
  effect from iteration :math:`k + 1`. A refresh in the final callback cannot
  affect the returned reconstruction.
- With PENETRATE, the scatter estimate is all events minus the geometrically
  collimated primary (``b01 - b02``): phantom scatter, septal penetration,
  collimator scatter and X-rays, i.e. everything ``A`` does not model. With
  SCATTWIN it is the window's scatter output, which holds phantom scatter only;
  use PENETRATE for high-energy isotopes.
- SIMIND's output does not follow the image magnitude (the source map is
  always scaled to the same number of photon histories), so every update is
  rescaled: by matching ``b02`` to ``A x`` (``reference_normaliser``, see
  :doc:`normalisation`), or by giving SIMIND the activity when ``x`` is in
  known activity units.

CIL acquisition storage
-----------------------

In the inspected SIRF 3.10.1 / STIR 6.4 installation, native acquisition
subsetting crashes on file-backed data. Selecting memory storage after an
object has been created does not convert that object.

``build_subset_objectives`` and ``set_subset_additive`` therefore select
process-wide SIRF memory storage before cloning their acquisition inputs.
Native subsetting uses only those memory-created objects. The original
caller-owned objects are not converted or modified, and the scheme is left
at ``"memory"``. Building objectives copies both full acquisition inputs;
each additive refresh copies the full additive term before subsetting.

For data creation and any subsetting outside these driver functions, select
memory storage before creating or loading acquisition data:

.. code-block:: python

    import sirf.STIR as sirf

    sirf.AcquisitionData.set_storage_scheme("memory")
    # Create or load acquisition data after selecting the scheme.

Do not change the storage scheme concurrently with driver calls or
acquisition creation. Examples 09 and 10 select memory storage in their
shared template setup. The OSEM outer-loop function itself does not change
the storage scheme.

SIRF OSEM
---------

.. code-block:: python

    from simind_python_connector.recon import (
        AdditiveUpdater, ScatterCorrection, SimindProjector, UpdateSchedule,
        reference_normaliser,
    )
    from simind_python_connector.recon.osem import run_osem_with_corrections

    projector = SimindProjector(adaptor, measured, mu_map,
                                normalise=reference_normaliser(full_model), seed=1)
    updater = AdditiveUpdater(ScatterCorrection(projector), UpdateSchedule(every=10),
                              initial_additive=measured.get_uniform_copy(0))
    image = run_osem_with_corrections(measured, make_model, initial, updater,
                                      num_subsets=5, subiterations_per_update=10,
                                      num_updates=3)

``full_model`` is the full-data linear model (no additive term), set up with
the measured data and kept separate for reference normalisation.
``make_model`` is a zero-argument factory returning a fresh, configured,
not-yet-set-up acquisition model on every call, with independent model-owned
state and the same linear operator settings. For a matrix-based model,
construct a fresh matrix as well.

The OSEM driver calls the factory once per pass, installs the current
additive term before setup, and builds a fresh objective and reconstructor.
On the inspected SIRF 3.10.1 / STIR 6.4 installation, reusing a model for a
second reconstruction produced an all-zero image; rebuilding only the
objective and reconstructor or repeating model setup did not repair it.

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
iterations, i.e. subiterations for algorithms over subsets. ``first=0`` can
trigger a refresh in the initial callback before the first algorithm update,
but after initial objective evaluation.

Example 10 uses ``run(40)`` and ``every=10``, producing records at
``[10, 20, 30, 40]``. Only the first three affect later iterations. To avoid
a refresh in a final callback at iteration N, callers can use ``stop_at=N``;
that bound is exclusive.

The OSEM outer loop calls ``update_now`` directly and does not use
``updater.schedule``. Its cadence is ``subiterations_per_update``, and
``num_updates`` controls how many correction updates are made.

Each update costs one SIMIND run, so:

- update every 1-2 epochs early on, when the image changes quickly;
- use ``damping`` below 1 (e.g. 0.5) when the SIMIND statistics are low;
- for CIL, use ``stop_at`` to freeze the estimate for the final iterations.

Examples 09 (SIRF OSEM) and 10 (CIL ISTA) show complete workflows; see
:doc:`examples`.
