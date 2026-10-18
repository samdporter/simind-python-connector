"""Route-2 scaling: scale_to_reference against a SPECTUB forward projection.

The reference is Setup.forward, which reproduces the fast_model.direct(image)
that recon.reference_normaliser computes; reference_normaliser is not called.
"""

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
SEED = 12345
OTHER_SEED = 54321
HOT_OFFSET = 5


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


def _masks():
    z, y, x = np.indices((N, N, N))
    c = N / 2
    body = ((x - c) ** 2 + (y - c) ** 2 <= (0.35 * N) ** 2) & (np.abs(z - c) <= 0.3 * N)
    hot = (x - c - HOT_OFFSET) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= 3**2
    return body, hot


def _activity():
    body, hot = _masks()
    return np.where(body, 1.0, 0.0) + np.where(hot, 4.0, 0.0)


def _template(path):
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


class _Setup:
    def __init__(self, root):
        import sirf.STIR as sirf

        sirf.AcquisitionData.set_storage_scheme("memory")
        self.root = root
        self.template = _template(root)
        body, _ = _masks()
        self.mu = _image(np.where(body, 0.15, 0.0))
        matrix = sirf.SPECTUBMatrix()
        matrix.set_attenuation_image(self.mu)
        matrix.set_keep_all_views_in_cache(False)
        matrix.set_resolution_model(0.0, 0.0, False)
        self.fast_model = sirf.AcquisitionModelUsingMatrix(matrix)
        self.fast_model.set_up(self.template, _image(_activity()))

    def simulate(self, name, activity, seed):
        adaptor = SirfSimindAdaptor(
            configs.get("Example.yaml"),
            str(self.root / name),
            name,
            quantization_scale=QUANTIZATION_SCALE,
            scoring_routine=ScoringRoutine.PENETRATE,
        )
        adaptor.set_template(self.template)
        adaptor.set_source(_image(activity))
        adaptor.set_mu_map(self.mu)
        adaptor.add_runtime_switch("CC", "ma-lehr")
        adaptor.add_runtime_switch("RR", seed)
        return adaptor.run()["geom_coll_primary"].as_array()

    def forward(self, activity):
        return self.fast_model.direct(_image(activity)).as_array()

    def scale(self, simulated, activity, method="trimmed"):
        return scale_to_reference(simulated, self.forward(activity), method=method)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return _Setup(tmp_path_factory.mktemp("scale_to_reference"))


@pytest.fixture(scope="module")
def reference_run(setup):
    activity = _activity()
    return activity, setup.simulate("x", activity, SEED)


def test_scale_homogeneous_in_reference_magnitude(setup, reference_run, validate):
    """Passes by construction; it is not a physics check.

    The connector max-normalises the source map, so x and 3x reach SIMIND as
    the identical map and give the identical projection. The ratio therefore
    only shows that scale_to_reference is homogeneous in the reference and
    that the connector is magnitude-invariant.
    """
    activity, simulated = reference_run
    tripled = setup.simulate("x3", 3 * activity, SEED)
    source_maps = [
        (setup.root / name / f"{name}_src.smi").read_bytes() for name in ("x", "x3")
    ]
    assert source_maps[0] == source_maps[1]
    ratio = setup.scale(tripled, 3 * activity) / setup.scale(simulated, activity)
    validate(
        "scale_homogeneous_in_reference_magnitude",
        ratio,
        expected=3.0,
        rel=1e-4,
        note="by construction: the connector max-normalises the source map, so "
        "x and 3x are the identical SIMIND input",
    )


def test_scale_stable_across_seeds(setup, reference_run, validate):
    activity, simulated = reference_run
    other = setup.simulate("x_seed2", activity, OTHER_SEED)
    ratio = setup.scale(simulated, activity) / setup.scale(other, activity)
    validate("scale_stable_across_seeds", ratio, expected=1.0, rel=0.03, smoke=True)


def test_scale_per_unit_activity_image_independent(setup, validate):
    """Scale per unit activity for a uniform cylinder and an off-centre sphere.

    This measures SPECTUB-vs-SIMIND model consistency between the two
    objects. The observed deviation (4.6 % on v8, 7.8 % on v9) is systematic,
    since the seed-to-seed spread is about 1 %. It is likely a position-
    dependent collimator response that SPECTUB without PSF does not model;
    the cause is not established.
    """
    body, hot = _masks()
    uniform = np.where(body, 1.0, 0.0)
    sphere = np.where(hot, 1.0, 0.0)
    per_unit = [
        setup.scale(setup.simulate(name, image, SEED), image) / image.sum()
        for name, image in (("uniform", uniform), ("sphere", sphere))
    ]
    validate(
        "scale_per_unit_activity_image_independent",
        float(per_unit[0] / per_unit[1]),
        expected=1.0,
        rel=0.10,
        smoke=True,
        note="systematic SPECTUB-vs-SIMIND model difference (seed spread ~1 %), "
        "likely position-dependent collimator response; cause not established",
    )


def test_trimmed_ignores_outlier_bins(setup, reference_run, validate):
    activity, simulated = reference_run
    reference = setup.forward(activity)
    rng = np.random.default_rng(0)
    dirty = simulated.copy()
    outliers = rng.choice(dirty.size, dirty.size // 100, replace=False)
    dirty.flat[outliers] *= 50

    changes = {}
    for method in ("trimmed", "sum"):
        clean = scale_to_reference(simulated, reference, method=method)
        changes[method] = abs(
            1 - scale_to_reference(dirty, reference, method=method) / clean
        )
    validate(
        "trimmed_ignores_outlier_bins",
        changes["trimmed"],
        upper=0.02,
        note="thin margin: about 0.015 against the 0.02 limit",
    )
    validate("sum_changes_with_outlier_bins", changes["sum"], lower=0.10)
