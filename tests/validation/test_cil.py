from types import SimpleNamespace

import numpy as np
import pytest

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
    ScatterCorrection,
    SimindProjector,
    UpdateSchedule,
    reference_normaliser,
)


pytest.importorskip("cil")

from simind_python_connector.recon.cil import (  # noqa: E402
    CorrectionCallback,
    build_subset_objectives,
    refresh_stochastic_state,
    set_subset_additive,
)


pytestmark = [
    pytest.mark.validation,
    pytest.mark.requires_simind,
    pytest.mark.requires_sirf,
    pytest.mark.requires_cil,
]

N = 16
VOXEL_MM = 8.84
VIEWS = 12
RADIUS_MM = 200
QUANTIZATION_SCALE = 0.1 * DEFAULT_QUANTIZATION_SCALE
HOT_OFFSET = 2
SUBSETS = 4
ISTA_STEPS = 3
STEP_SIZE = 1e-4


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
    hot = (x - c - HOT_OFFSET) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= 2**2
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


class _Fixed:
    def __init__(self, estimate):
        self.value = estimate

    def estimate(self, image):
        return self.value


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    import sirf.STIR as sirf

    root = tmp_path_factory.mktemp("cil")
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

    route1 = _adaptor(root, "measured")
    route1.set_template(template)
    route1.set_source(_image(activity))
    route1.set_mu_map(mu_image)
    route1.set_activity(100.0, 10.0)
    route1.add_runtime_switch("RR", 1)
    outputs = route1.run()
    expected = outputs["all_interactions"]
    primary = outputs["geom_coll_primary"].as_array()
    measured = expected.clone()
    measured.fill(
        np.random.default_rng(1)
        .poisson(np.maximum(expected.as_array(), 0))
        .astype(np.float32)
    )

    full_model = make_model()
    full_model.set_up(measured, _image(activity))
    to_data = 1 / scale_to_reference(
        primary, full_model.direct(_image(activity)).as_array(), method="trimmed"
    )
    image = _image(activity * to_data)

    projector = SimindProjector(
        _adaptor(root, "estimate"),
        template,
        mu_image,
        normalise=reference_normaliser(full_model, method="trimmed"),
        seed=100,
    )
    scatter = ScatterCorrection(projector).estimate(image).maximum(1e-5)

    return SimpleNamespace(
        make_model=make_model,
        measured=measured,
        image=image,
        scatter=scatter,
    )


def _objectives(setup):
    return build_subset_objectives(
        setup.measured,
        setup.measured.get_uniform_copy(0),
        setup.make_model,
        SUBSETS,
        setup.image,
    )


def _ista(setup, f):
    from cil.optimisation.algorithms import ISTA
    from cil.optimisation.functions import IndicatorBox

    return ISTA(
        initial=setup.image.get_uniform_copy(1.0),
        f=f,
        g=IndicatorBox(lower=0),
        step_size=STEP_SIZE,
    )


def _relative_difference(stored, fresh):
    difference = sum(float(np.linalg.norm(s - f)) ** 2 for s, f in zip(stored, fresh))
    scale = sum(float(np.linalg.norm(f)) ** 2 for f in fresh)
    return float(np.sqrt(difference / scale))


def _fresh_gradients(objectives, x):
    gradients = [function.gradient(x).as_array() for function in objectives.functions]
    return gradients, sum(gradients)


def test_svrg_snapshot_after_refresh(setup, validate):
    from cil.optimisation.functions import SVRGFunction
    from cil.optimisation.utilities import Sampler

    objectives = _objectives(setup)
    f = SVRGFunction(
        objectives.functions,
        sampler=Sampler.sequential(SUBSETS),
        snapshot_update_interval=100,
    )
    algorithm = _ista(setup, f)
    algorithm.run(ISTA_STEPS)
    x = algorithm.solution

    set_subset_additive(objectives, setup.scatter)

    _, full = _fresh_gradients(objectives, x)
    validate(
        "svrg_stale_without_refresh",
        _relative_difference([f._full_gradient_at_snapshot.as_array()], [full]),
        lower=0.1,
    )
    refresh_stochastic_state(f, x)
    validate(
        "svrg_snapshot_after_refresh",
        _relative_difference([f._full_gradient_at_snapshot.as_array()], [full]),
        expected=0,
        abs_tol=1e-5,
    )


def test_sag_gradients_after_refresh(setup, validate):
    from cil.optimisation.functions import SAGFunction
    from cil.optimisation.utilities import Sampler

    objectives = _objectives(setup)
    f = SAGFunction(objectives.functions, sampler=Sampler.sequential(SUBSETS))
    algorithm = _ista(setup, f)
    algorithm.run(ISTA_STEPS)
    x = algorithm.solution

    set_subset_additive(objectives, setup.scatter)

    gradients, full = _fresh_gradients(objectives, x)

    def stored():
        return [
            *(g.as_array() for g in f._list_stored_gradients),
            f._full_gradient_at_iterate.as_array(),
        ]

    validate(
        "sag_stale_without_refresh",
        _relative_difference(stored(), [*gradients, full]),
        lower=0.1,
    )
    refresh_stochastic_state(f, x)
    validate(
        "sag_gradients_after_refresh",
        _relative_difference(stored(), [*gradients, full]),
        expected=0,
        abs_tol=1e-5,
    )


def test_callback_refresh_iterations(setup, validate):
    from cil.optimisation.functions import SumFunction

    objectives = _objectives(setup)
    updater = AdditiveUpdater(
        _Fixed(setup.scatter),
        UpdateSchedule(every=3, stop_at=9),
        setup.measured.get_uniform_copy(0),
    )
    algorithm = _ista(setup, SumFunction(*objectives.functions))
    algorithm.run(9, callbacks=[CorrectionCallback(updater, objectives)])
    validate(
        "callback_refresh_iterations",
        1.0 if [r.iteration for r in updater.history] == [3, 6] else 0.0,
        expected=1.0,
        abs_tol=0,
    )
