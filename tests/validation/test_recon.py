from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from simind_python_connector import SirfSimindAdaptor, configs
from simind_python_connector.builders import (
    STIRSPECTAcquisitionDataBuilder,
    STIRSPECTImageDataBuilder,
)
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
)
from simind_python_connector.normalisation import scale_to_reference
from simind_python_connector.recon import (
    AdditiveUpdater,
    ResidualCorrection,
    ScatterCorrection,
    SimindComponent,
    SimindProjector,
    UpdateSchedule,
    reference_normaliser,
)


pytestmark = [
    pytest.mark.validation,
    pytest.mark.requires_simind,
    pytest.mark.requires_sirf,
]

N = 32
VOXEL_MM = 4.42
VIEWS = 30
RADIUS_MM = 200
QUANTIZATION_SCALE = 0.1 * DEFAULT_QUANTIZATION_SCALE
# At 0.1x the per-bin MC noise of both scatter maps caps their correlation near
# 0.47; at 3.2x the noise-limited ceiling is about 0.95.
SHAPE_QUANTIZATION_SCALE = 3.2 * DEFAULT_QUANTIZATION_SCALE
HOT_OFFSET = 5
SUBSETS = 6


def _image(array):
    builder = STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": str(N),
            "!matrix size [2]": str(N),
            "!matrix size [3]": str(N),
            "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [3]": str(VOXEL_MM),
        },
        backend="sirf",
    )
    builder.set_pixel_array(array.astype(np.float32))
    return builder.build()


def _phantom():
    z, y, x = np.indices((N, N, N))
    c = N / 2
    body = ((x - c) ** 2 + (y - c) ** 2 <= (0.35 * N) ** 2) & (np.abs(z - c) <= 0.3 * N)
    hot = (x - c - HOT_OFFSET) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= 3**2
    return np.where(body, 1.0, 0.0) + np.where(hot, 4.0, 0.0), np.where(body, 0.15, 0.0)


def _template(path):
    import sirf.STIR as sirf

    sirf.AcquisitionData.set_storage_scheme("memory")
    return STIRSPECTAcquisitionDataBuilder(
        header_overrides={
            "!matrix size [1]": str(N),
            "!matrix size [2]": str(N),
            "!number of projections": str(VIEWS),
            "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
            "!direction of rotation": "CW",
            "start angle": "180",
            "Radius": str(RADIUS_MM),
        },
        backend="sirf",
    ).build(output_path=path / "template")


def _adaptor(root, name, quantization_scale=QUANTIZATION_SCALE):
    adaptor = SirfSimindAdaptor(
        configs.get("Example.yaml"),
        str(root / name),
        name,
        quantization_scale=quantization_scale,
        scoring_routine=ScoringRoutine.PENETRATE,
    )
    adaptor.add_runtime_switch("CC", "ma-lehr")
    return adaptor


class _Replay:
    """Hands out previously computed estimates, one per update."""

    def __init__(self, estimates):
        self.estimates = iter(estimates)

    def estimate(self, image):
        return next(self.estimates)


def _route1(root, name, template, mu_image, quantization_scale=QUANTIZATION_SCALE):
    route1 = _adaptor(root, name, quantization_scale)
    route1.set_template(template)
    route1.set_source(_image(_phantom()[0]))
    route1.set_mu_map(mu_image)
    route1.set_activity(100.0, 10.0)
    route1.add_runtime_switch("RR", 1)
    return route1.run()


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    import sirf.STIR as sirf

    root = tmp_path_factory.mktemp("recon")
    template = _template(root)
    activity, mu = _phantom()
    mu_image = _image(mu)
    # STIR SPECTUB mirrors x relative to the mu map (UCL/STIR#452)
    spectub_mu_image = _image(mu[..., ::-1])

    def make_model():
        matrix = sirf.SPECTUBMatrix()
        matrix.set_attenuation_image(spectub_mu_image)
        matrix.set_keep_all_views_in_cache(False)
        matrix.set_resolution_model(0.0, 0.0, False)
        return sirf.AcquisitionModelUsingMatrix(matrix)

    outputs = _route1(root, "measured", template, mu_image)
    expected = outputs["all_interactions"]
    primary = outputs["geom_coll_primary"].as_array()
    true_scatter = expected.as_array() - primary
    measured = expected.clone()
    measured.fill(
        np.random.default_rng(1)
        .poisson(np.maximum(expected.as_array(), 0))
        .astype(np.float32)
    )

    # The fast model is unitless, so the image is put on the data scale by
    # matching its primary projection to the route-1 primary.
    full_model = make_model()
    full_model.set_up(measured, _image(activity))
    to_data = 1 / scale_to_reference(
        primary, full_model.direct(_image(activity)).as_array(), method="trimmed"
    )
    true_image = _image(activity * to_data)

    return SimpleNamespace(
        root=root,
        template=template,
        mu_image=mu_image,
        make_model=make_model,
        full_model=full_model,
        measured=measured,
        true_image=true_image,
        true_scatter=true_scatter,
        scatter_fraction=true_scatter.sum() / measured.sum(),
    )


def _projector(setup, name, seed, quantization_scale=QUANTIZATION_SCALE):
    return SimindProjector(
        _adaptor(setup.root, name, quantization_scale),
        setup.template,
        setup.mu_image,
        normalise=reference_normaliser(setup.full_model, method="trimmed"),
        seed=seed,
    )


@pytest.fixture(scope="module")
def estimates(setup):
    correction = ScatterCorrection(_projector(setup, "estimate", seed=100))
    return [correction.estimate(setup.true_image) for _ in range(2)]


def test_route2_scatter_matches_route1_sum(setup, estimates, validate):
    """to_data divides by the same trimmed scale, so this validates the
    scatter/primary ratio and chain consistency, not absolute scale."""
    validate(
        "route2_scatter_matches_route1_sum",
        float(estimates[0].sum() / setup.true_scatter.sum()),
        expected=1.0,
        rel=0.10,
        smoke=True,
    )


def test_route2_scatter_matches_route1_shape(setup, validate):
    outputs = _route1(
        setup.root,
        "shape_measured",
        setup.template,
        setup.mu_image,
        SHAPE_QUANTIZATION_SCALE,
    )
    true_scatter = (
        outputs["all_interactions"].as_array() - outputs["geom_coll_primary"].as_array()
    )
    estimate = ScatterCorrection(
        _projector(setup, "shape_estimate", 100, SHAPE_QUANTIZATION_SCALE)
    ).estimate(setup.true_image)
    sigma = 3 / (2 * np.sqrt(2 * np.log(2)))
    array = estimate.as_array().astype(np.float64)
    sigmas = [0.0] * array.ndim
    sigmas[-3] = sigmas[-1] = sigma
    smoothed = gaussian_filter(array, sigmas)
    correlation = np.corrcoef(smoothed.ravel(), true_scatter.ravel())[0, 1]
    validate(
        "route2_scatter_matches_route1_shape",
        float(correlation),
        lower=0.9,
        smoke=True,
    )


def test_update_schedule_is_due(validate):
    schedule = UpdateSchedule(every=2, stop_at=6)
    due = [i for i in range(1, 7) if schedule.is_due(i)]
    validate(
        "update_schedule_is_due",
        1.0 if due == [2, 4] else 0.0,
        expected=1.0,
        abs_tol=0,
    )


def test_osem_refreshes_at_update_boundaries(setup, estimates, validate):
    zero = setup.measured.get_uniform_copy(0)
    updater = AdditiveUpdater(
        _Replay(estimates), UpdateSchedule(every=2, stop_at=6), zero
    )
    from simind_python_connector.recon.osem import run_osem_with_corrections

    run_osem_with_corrections(
        setup.measured,
        setup.make_model,
        setup.true_image.get_uniform_copy(setup.true_image.as_array().mean()),
        updater,
        num_subsets=2,
        subiterations_per_update=2,
        num_updates=2,
    )
    validate(
        "osem_refreshes_at_update_boundaries",
        1.0 if [r.iteration for r in updater.history] == [2, 4] else 0.0,
        expected=1.0,
        abs_tol=0,
    )


def test_damping_combines_exactly(setup, estimates, validate):
    e1, e2 = (estimate.as_array().astype(np.float64) for estimate in estimates)
    zero = setup.measured.get_uniform_copy(0)
    updater = AdditiveUpdater(_Replay(estimates), UpdateSchedule(every=1), zero, 0.5)
    updater.update_now(1, setup.true_image)
    updater.update_now(2, setup.true_image)
    difference = np.abs(updater.raw.as_array() - (0.25 * e1 + 0.5 * e2)).max()
    validate(
        "damping_combines_exactly",
        float(difference / e2.max()),
        expected=0,
        abs_tol=1e-6,
    )


@pytest.fixture(scope="module")
def scatter_loop(setup):
    """OSEM without and with the SIMIND scatter, from the same uniform start.

    The settings are those that let the loop converge at the module's 0.1x
    histories: unsmoothed, about half of the spiky estimate lies in bins
    above the data, which OSEM cannot remove; the trimmed normaliser scales
    low on the sparse MC primary; undamped refreshes from a zero start
    oscillate (1.29x, then 0.64x the true scatter). Each refresh also records,
    on that iterate, the trimmed / sum scale ratio, the same ratio without
    end trimming, and the reference fraction in bins where the primary is 0.
    """
    from simind_python_connector.recon.osem import run_osem_with_corrections

    initial = setup.true_image.get_uniform_copy(setup.true_image.as_array().mean())
    zero = setup.measured.get_uniform_copy(0)
    num_updates = 4
    trimmed_to_sum = []
    untrimmed_to_sum = []
    reference_at_zero = []
    iterates = []

    def normalise(outputs, image):
        primary = outputs["geom_coll_primary"].as_array()
        reference = setup.full_model.direct(image).as_array()
        scale = scale_to_reference(primary, reference, method="sum")
        trimmed = scale_to_reference(primary, reference, method="trimmed")
        untrimmed = scale_to_reference(
            primary, reference, method="trimmed", trim_fraction=0.0
        )
        trimmed_to_sum.append(trimmed / scale)
        untrimmed_to_sum.append(untrimmed / scale)
        reference_at_zero.append(reference[primary == 0].sum() / reference.sum())
        iterates.append(image)
        return scale

    projector = SimindProjector(
        _adaptor(setup.root, "osem"),
        setup.template,
        setup.mu_image,
        normalise=normalise,
        seed=300,
    )
    correction = ScatterCorrection(
        projector, smoothing_fwhm_bins=3.0, smooth_across_views=True
    )

    def osem(updater, subiterations_per_update, updates):
        return run_osem_with_corrections(
            setup.measured,
            setup.make_model,
            initial,
            updater,
            SUBSETS,
            subiterations_per_update,
            updates,
        )

    without = osem(
        AdditiveUpdater(None, UpdateSchedule(every=1), zero),
        SUBSETS * (num_updates + 1),
        0,
    )
    with_scatter = osem(
        AdditiveUpdater(correction, UpdateSchedule(every=SUBSETS), zero, 0.5),
        SUBSETS,
        num_updates,
    )
    return SimpleNamespace(
        without=without,
        with_scatter=with_scatter,
        trimmed_to_sum=trimmed_to_sum,
        untrimmed_to_sum=untrimmed_to_sum,
        reference_at_zero=reference_at_zero,
        iterates=iterates,
    )


def test_scatter_correction_removes_the_scatter_fraction(setup, scatter_loop, validate):
    without = scatter_loop.without.sum()
    validate(
        "scatter_correction_removes_the_scatter_fraction",
        float((without - scatter_loop.with_scatter.sum()) / without),
        expected=float(setup.scatter_fraction),
        abs_tol=0.05,
        smoke=True,
    )


def test_trimmed_scale_bias_low_histories(scatter_loop, validate):
    def rounded(values):
        return [round(float(v), 3) for v in values]

    validate(
        "trimmed_scale_bias_low_histories",
        float(np.mean(scatter_loop.trimmed_to_sum)),
        expected=1.0,
        rel=0.10,
        smoke=True,
        note=(
            f"per refresh trimmed/sum {rounded(scatter_loop.trimmed_to_sum)}; "
            "trim_fraction=0 / sum "
            f"{rounded(scatter_loop.untrimmed_to_sum)}; reference fraction "
            "in bins where the MC primary is 0 "
            f"{rounded(scatter_loop.reference_at_zero)}. Without end trimming "
            "the ratio is about 1 minus that fraction, so the bias comes from "
            "excluding bins where the sparse MC primary is 0; the 10% end "
            "trimming raises it slightly"
        ),
    )


SWEEP_MULTIPLIERS = (1, 4, 16, 64)
SWEEP_SEEDS = (500, 501)
SWEEP_TRUTH_SEEDS = (900, 901)


@pytest.fixture(scope="module")
def history_sweep(setup, scatter_loop):
    """Trimmed and sum scales / true scale for one OSEM iterate, by history level.

    The iterate is the image at the scatter loop's 2nd refresh, reused at every
    level so only the SIMIND histories change. The true scale is the ratio of
    expected totals, sum(A x) / E[sum(b02)], with E[sum(b02)] the mean over two
    extra runs at the highest level whose seeds no method sees. The sum
    method estimates that same quantity, so its highest-level row agrees by
    construction and shows only seed noise; the informative rows are the
    lower levels.
    """
    image = scatter_loop.iterates[1]
    reference = setup.full_model.direct(image).as_array()

    def primary(multiplier, seed):
        adaptor = _adaptor(
            setup.root,
            f"sweep_{multiplier}x_{seed}",
            multiplier * QUANTIZATION_SCALE,
        )
        adaptor.set_template(setup.template)
        adaptor.set_mu_map(setup.mu_image)
        adaptor.set_source(image)
        adaptor.add_runtime_switch("RR", seed)
        return adaptor.run()["geom_coll_primary"].as_array()

    top = SWEEP_MULTIPLIERS[-1]
    true_scale = reference.sum() / np.mean(
        [primary(top, seed).sum() for seed in SWEEP_TRUTH_SEEDS]
    )
    ratios = {}
    for multiplier in SWEEP_MULTIPLIERS:
        runs = [primary(multiplier, seed) for seed in SWEEP_SEEDS]
        for method in ("trimmed", "sum"):
            ratios[method, multiplier] = [
                scale_to_reference(run, reference, method=method) / true_scale
                for run in runs
            ]
    return ratios


def test_normaliser_history_sweep(history_sweep, validate):
    for (method, multiplier), ratios in history_sweep.items():
        validate(
            f"normaliser_history_sweep_{method}_{multiplier}x",
            float(np.mean(ratios)),
            lower=0,
            smoke=True,
            note=(
                f"scale / true scale at {multiplier}x the module's 0.1x histories, "
                f"seeds {list(SWEEP_SEEDS)}: {[round(float(r), 3) for r in ratios]}"
            ),
        )


def test_normalisers_agree_at_high_histories(history_sweep, validate):
    top = SWEEP_MULTIPLIERS[-1]
    means = {
        method: float(np.mean(history_sweep[method, top]))
        for method in ("trimmed", "sum")
    }
    validate(
        "normalisers_agree_at_high_histories",
        max(means.values(), key=lambda value: abs(value - 1)),
        expected=1.0,
        rel=0.05,
        smoke=True,
        note=(
            f"the method farthest from 1 at {top}x (trimmed, about 0.96, on v8 "
            f"and v9); sum agrees there by construction; means {means}"
        ),
    )


def test_residual_identity(setup, validate):
    projector = _projector(setup, "residual", seed=200)
    base_additive = setup.measured.get_uniform_copy(2.5)
    correction = ResidualCorrection(
        SimindComponent(projector, "all_interactions"),
        setup.full_model,
        base_additive,
    )
    x = setup.true_image
    rebuilt = correction.estimate(x) + setup.full_model.direct(x) - base_additive
    accurate = projector.last_outputs["all_interactions"].as_array()
    difference = np.abs(rebuilt.as_array() - accurate).max()
    validate(
        "residual_identity",
        float(difference / np.abs(accurate).max()),
        expected=0,
        abs_tol=1e-5,
    )
