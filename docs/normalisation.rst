Normalising SIMIND Output
=========================

SIMIND output needs an explicit scale before it can be compared with or
added to other projection data.

Photon histories are not counts
-------------------------------

The source map holds integer weights: each voxel's value sets how many photon
histories start there. The connector scales the source so its maximum maps to
``500 x quantization_scale``. Scaling the source up, or raising ``NN``, runs
more histories. That gives better statistics and a longer run, but does **not**
give more counts in the projections.

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
     - geometric primary: PENETRATE ``geom_coll_primary``; SCATTWIN ``pri_wN``
       for low-energy isotopes
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
        method="trimmed",
        trim_fraction=0.1,
    )
    scaled = {key: value.projection * factor for key, value in outputs.items()}

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

For high-energy isotopes (Lu-177, Y-90, I-131), SCATTWIN ``pri`` includes
septal penetration and collimator scatter, which analytic projectors do not
model. Use PENETRATE ``geom_coll_primary`` for those.