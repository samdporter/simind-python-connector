Normalising SIMIND Output
=========================

SIMIND output needs an explicit scale before it can be compared with or
added to other projection data.

Photon histories are not counts
-------------------------------

The source map holds integer weights: each voxel's value sets how many photon
histories start there. SIMIND reads the source and density integer maps as
**signed** 16-bit values, so a voxel value of 32768 or more wraps negative and
silently corrupts the map. The connector therefore caps both maps at 32767.

The connector scales the source maximum to the effective budget
``MAX_SOURCE x quantization_scale`` (= ``32767 x quantization_scale``) with a
single rounding step, then rounds and clips once:

.. code-block:: python

    source_scaled = source / source.max() * (32767 * quantization_scale)
    source_written = np.clip(np.round(source_scaled), 0, 32767)

The default ``quantization_scale`` is ``500 / 32767``, which reproduces the
legacy 500-weight budget at the full signed-range resolution. An explicit
``quantization_scale=1.0`` scales the maximum to the whole signed range.
Explicit values are used exactly as given; the connector does **not**
auto-migrate scales written for releases whose ``MAX_SOURCE`` was 500. To keep
an old explicit scale's budget, pass ``old_scale * DEFAULT_QUANTIZATION_SCALE``.

Increasing ``quantization_scale`` or ``NN`` runs more histories, improving
statistics without setting the projection count scale. Multiplying the NumPy
source by a constant leaves the exported map unchanged because its maximum is
normalised. Coarse integer rounding can also distort relative voxel weights;
check region ratios before using a low quantisation budget quantitatively.

Route 1: the activity is known
------------------------------

Tell SIMIND the activity. Index 25 holds activity [MBq] times the acquisition
time per projection [s]:

.. code-block:: python

    connector.set_activity(activity_mbq=source.sum(), time_per_projection_s=20.0)
    outputs = connector.run()  # projections on the scale of that acquisition

``source.sum()`` is the activity only when the source values are in MBq per
voxel. If the source map holds relative weights, or the activity is already
written into the source file, pass the activity in MBq directly instead.

Route 2: the activity is not known
----------------------------------

Scale the output to a projection that is already on the wanted scale, using
:func:`simind_python_connector.normalisation.scale_to_reference`. Apply the
factor to every output of the same run.

.. list-table::
   :header-rows: 1

   * - Situation
     - ``component`` (SIMIND)
     - ``reference``
     - ``mask``
   * - Analytic projector available (e.g. SIRF/STIR)
     - geometric primary: PENETRATE ``geom_coll_primary``
     - linear forward projection of the same (masked) image
     - optional: forward-projected body mask, thresholded at a fraction of its
       maximum
   * - Only measured data available
     - total: ``tot_wN`` or ``all_interactions``
     - measured data
     - optional

With an analytic projector as the reference:

.. code-block:: python

    from simind_python_connector.normalisation import scale_to_reference

    factor = scale_to_reference(
        outputs["geom_coll_primary"].projection,
        analytic_projection,
        method="sum",
    )
    scaled = {key: value.projection * factor for key, value in outputs.items()}

``method="trimmed"`` drops the ``trim_fraction`` most extreme bin ratios at
each end, which makes it robust against outlier bins. It also keeps only bins
where the component is non-zero, so with few photon histories, where SIMIND's
primary is zero in many bins, it drops reference counts but no SIMIND counts
and scales low: 0.55 (SIMIND v8) and 0.51 (v9) of the true scale on a 32³ OSEM
iterate at a tenth of the default quantization, 0.96 at 64 times those
histories. Use ``method="sum"`` at quick history counts (including inside a
reconstruction loop, see :doc:`reconstruction`) and keep trimmed for
references with outlier bins at high histories.

Against measured data:

.. code-block:: python

    from simind_python_connector.normalisation import scale_to_reference

    factor = scale_to_reference(
        outputs["tot_w1"].projection,
        measured_projection,
        method="sum",
        mask=body_mask_projection,
    )
    scaled = {key: value.projection * factor for key, value in outputs.items()}

SCATTWIN ``pri`` excludes phantom scatter but can include septal penetration,
collimator scatter and detector effects. It is a different physical component
from PENETRATE ``geom_coll_primary``, even when those extra contributions are
small. An uncalibrated analytic projection can check geometry and define a
relative scale; it does not independently establish detector sensitivity.

.. _scaling-scatter-to-measured-data:

Scaling a scatter estimate to measured data
-------------------------------------------

Dewaraja et al. scale the SIMIND scatter estimate to the measured projections,
either pixel by pixel,

.. math::

    s_i = s^{\mathrm{SIM}}_i \, \frac{Y_i}{Y^{\mathrm{SIM}}_i},

or with one global factor,

.. math::

    s_i = b \, s^{\mathrm{SIM}}_i, \qquad b = \frac{\sum_i Y_i}{\sum_i Y^{\mathrm{SIM}}_i},

where :math:`Y` is the measured projection and :math:`Y^{\mathrm{SIM}}` the
simulated all-interactions projection. The global factor is a normaliser:

.. code-block:: python

    from simind_python_connector.recon import SimindProjector, measured_normaliser

    projector = SimindProjector(..., normalise=measured_normaliser(measured))

The pixel-wise scaling is done by the correction:

.. code-block:: python

    from simind_python_connector.recon import ScatterCorrection

    correction = ScatterCorrection(
        projector, measured=measured, smoothing_fwhm_bins=3.0, smooth_across_views=True
    )

The ``projector`` still needs a normaliser, since ``SimindProjector`` requires
one; the scale cancels in the pixel-wise ratio. With SCATTWIN, :math:`Y^{\mathrm{SIM}}`
is the ``tot_wN`` output, so pass ``component="tot_wN"`` to ``measured_normaliser``.

Dewaraja Y.K. et al., "Improved quantitative 90Y bremsstrahlung SPECT/CT
reconstruction with Monte Carlo scatter modeling", Med Phys 44(12):6364-6376,
2017, doi:10.1002/mp.12597. The authors found the pixel-wise scaling better in
their simulations. It multiplies the measured noise into the scatter estimate,
hence the smoothing.
