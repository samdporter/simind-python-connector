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


def _adaptor(root, name):
    adaptor = SirfSimindAdaptor(
        configs.get("Example.yaml"),
        str(root / name),
        name,
        quantization_scale=QUANTIZATION_SCALE,
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


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    import sirf.STIR as sirf

    root = tmp_path_factory.mktemp("recon")
    template = _template(root)
    activity, mu = _phantom()
    mu_image = _image(mu)

    def make_model():
        matrix = sirf.SPECTUBMatrix()
        matrix.set_attenuation_image(mu_image)
        matrix.set_keep_all_views_in_cache(False)
        matrix.set_resolution_model(0.0, 0.0, False)
        return sirf.AcquisitionModelUsingMatrix(matrix)

    route1 = _adaptor(root, "measured")
    route1.set_template(template)
    route1.set_source(_image(activity))
    route1.set_mu_map(mu_image)
    route1.set_activity(100.0, 10.0)
    route1.add_runtime_switch("RR", 1)
    outputs = route1.run()
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


def _projector(setup, name, seed):
    return SimindProjector(
        _adaptor(setup.root, name),
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
    validate(
        "route2_scatter_matches_route1_sum",
        float(estimates[0].sum() / setup.true_scatter.sum()),
        expected=1.0,
        rel=0.10,
        smoke=True,
    )


def test_route2_scatter_matches_route1_shape(setup, estimates, validate):
    sigma = 3 / (2 * np.sqrt(2 * np.log(2)))
    array = estimates[0].as_array().astype(np.float64)
    sigmas = [0.0] * array.ndim
    sigmas[-3] = sigmas[-1] = sigma
    smoothed = gaussian_filter(array, sigmas)
    correlation = np.corrcoef(smoothed.ravel(), setup.true_scatter.ravel())[0, 1]
    validate(
        "route2_scatter_matches_route1_shape",
        float(correlation),
        lower=0.9,
        smoke=True,
    )


def test_refreshes_follow_the_schedule(setup, estimates, validate):
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
        "refreshes_follow_the_schedule",
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


def test_scatter_correction_removes_the_scatter_fraction(setup, validate):
    from simind_python_connector.recon.osem import run_osem_with_corrections

    initial = setup.true_image.get_uniform_copy(setup.true_image.as_array().mean())
    zero = setup.measured.get_uniform_copy(0)
    num_updates = 2

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
        AdditiveUpdater(
            ScatterCorrection(_projector(setup, "osem", seed=300)),
            UpdateSchedule(every=SUBSETS),
            zero,
        ),
        SUBSETS,
        num_updates,
    )
    validate(
        "scatter_correction_removes_the_scatter_fraction",
        float((without.sum() - with_scatter.sum()) / without.sum()),
        expected=float(setup.scatter_fraction),
        abs_tol=0.05,
        smoke=True,
    )


def test_residual_identity(setup, validate):
    projector = _projector(setup, "residual", seed=200)
    correction = ResidualCorrection(
        SimindComponent(projector, "all_interactions"),
        setup.full_model,
        setup.measured.get_uniform_copy(0),
    )
    x = setup.true_image
    rebuilt = correction.estimate(x) + setup.full_model.direct(x)
    accurate = projector.last_outputs["all_interactions"].as_array()
    difference = np.abs(rebuilt.as_array() - accurate).max()
    validate(
        "residual_identity",
        float(difference / np.abs(accurate).max()),
        expected=0,
        abs_tol=1e-5,
    )
