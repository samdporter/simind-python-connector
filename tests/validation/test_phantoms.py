import math
from contextlib import contextmanager

import numpy as np
import pytest

from simind_python_connector.core.types import SimulationError
from simind_python_connector.phantoms import (
    AnalyticPhantom,
    Box,
    CardiacDefect,
    CardiacSource,
    Ellipsoid,
    HorizontalCylinder,
    Insert,
    LibraryPhantom,
    MultipleInserts,
    VerticalCylinder,
    nema_iec_phantom,
    voxelise,
)
from simind_python_connector.utils.interfile import read_simind_density_image
from tests.test_phantoms_simind import (
    _N,
    _VOXEL,
    _WATER,
    _compare,
    _connector,
)


pytestmark = [pytest.mark.validation, pytest.mark.requires_simind]

V9_LINUX_DEFECT = "v9 Linux SCATTWIN engine defect (heap corruption)"
SOURCE_SHIFT_CM = (2.0, 1.0, -1.0)
SHAPES = {
    "ellipsoid": Ellipsoid((4.0, 3.0, 2.0)),
    "box": Box((4.0, 3.0, 2.0)),
    "horizontal_cylinder": HorizontalCylinder(4.0, (3.0, 2.0)),
    "vertical_cylinder": VerticalCylinder(2.0, (4.0, 3.0)),
}
JASZCZAK_RADII_CM = (0.5, 0.65, 0.85, 1.1, 1.4, 1.85)
INSERT_MODES = {
    "background_1": {"background": 1.0, "mode": "background"},
    "background_3": {"background": 3.0, "mode": "background"},
    "hot": {"background": None, "mode": "hot"},
    "cold": {"background": 1.0, "mode": "cold"},
}
HOT_SPHERE_DIAMETERS_MM = (17, 22, 28, 37)
VIEWS = 16
# shorter than the example's 10 cm: source histories scale with the container volume
CONTAINER_HALF_LENGTH_CM = 3.0
ANALYTIC_HISTORIES = 0.25
VOXEL_QUANTIZATION_SCALE = 0.03
# a uniform source needs no quantisation levels; histories scale with voxel count
COLD_VOXEL_QUANTIZATION_SCALE = 0.002


def _analytic_connector(tmp_path, prefix):
    connector = _connector(tmp_path, prefix, num_projections=VIEWS)
    connector.add_config_value(26, ANALYTIC_HISTORIES)
    return connector


def _voxel_projection(
    tmp_path, prefix, phantom, quantization_scale=VOXEL_QUANTIZATION_SCALE
):
    activity, attenuator = voxelise(phantom, (_N, _N, _N), (_VOXEL,) * 3, supersample=2)
    connector = _connector(tmp_path, prefix, num_projections=VIEWS)
    connector.quantization_scale = quantization_scale
    connector.get_config().set_flag(11, True)
    connector.configure_voxel_phantom(
        activity, attenuator, (_VOXEL,) * 3, mu_map_type="density"
    )
    return connector.run()["tot_w1"].projection


@pytest.fixture
def v9_linux_guard(engine, not_run):
    @contextmanager
    def guard(case: str):
        try:
            yield
        except SimulationError:
            if engine.label != "v9-linux":
                raise
            not_run(case, V9_LINUX_DEFECT)

    return guard


def _validate_all(validate, checks):
    """Record every check before failing on the first one that missed."""
    failures = []
    for case, measured, kwargs in checks:
        try:
            validate(case, float(measured), **kwargs)
        except AssertionError as error:
            failures.append(error)
    if failures:
        raise failures[0]


def _projection_checks(case, correlation, centre_gap):
    return [
        (f"{case}_correlation", correlation, {"lower": 0.95, "smoke": True}),
        (f"{case}_centre_gap", centre_gap, {"upper": 1.0, "smoke": True}),
    ]


@pytest.mark.macos_v9
@pytest.mark.parametrize("shape", list(SHAPES))
def test_analytic_vs_voxel_shape(validate, v9_linux_guard, tmp_path, shape):
    case = f"analytic_vs_voxel_{shape}"
    phantom = AnalyticPhantom(SHAPES[shape], _WATER, source_shift_cm=SOURCE_SHIFT_CM)
    with v9_linux_guard(case):
        connector = _analytic_connector(tmp_path, "analytic")
        connector.configure_phantom(phantom)
        analytic = connector.run()["tot_w1"].projection
        voxel = _voxel_projection(tmp_path, "voxel", phantom)

    _validate_all(validate, _projection_checks(case, *_compare(case, analytic, voxel)))


def _jaszczak_like(**insert_mode) -> AnalyticPhantom:
    water = HorizontalCylinder(CONTAINER_HALF_LENGTH_CM, (10.0, 10.0))
    inserts = tuple(
        Insert(
            (r, r, r),
            (0.0, 5.7 * math.cos(math.radians(a)), 5.7 * math.sin(math.radians(a))),
            8.0,
        )
        for r, a in zip(JASZCZAK_RADII_CM, range(30, 360, 60))
    )
    return AnalyticPhantom(MultipleInserts(water, inserts, **insert_mode), water)


@pytest.mark.macos_v9
@pytest.mark.parametrize("mode", list(INSERT_MODES))
def test_inserts_mode(validate, v9_linux_guard, tmp_path, mode):
    case = f"inserts_{mode}"
    phantom = _jaszczak_like(**INSERT_MODES[mode])
    with v9_linux_guard(case):
        connector = _analytic_connector(tmp_path, "inserts")
        connector.configure_phantom(phantom)
        analytic = connector.run()["tot_w1"].projection
        voxel = _voxel_projection(
            tmp_path,
            "inserts_voxel",
            phantom,
            COLD_VOXEL_QUANTIZATION_SCALE
            if mode == "cold"
            else VOXEL_QUANTIZATION_SCALE,
        )

    checks = _projection_checks(case, *_compare(case, analytic, voxel))
    checks.append(
        (
            f"{case}_total_ratio",
            analytic.sum() / voxel.sum(),
            {"expected": 1.0, "rel": 0.05, "smoke": True},
        )
    )
    _validate_all(validate, checks)


@pytest.mark.macos_v9
def test_cardiac_defect_lowers_counts(validate, v9_linux_guard, tmp_path):
    """Fractional drop in the pixels the defect empties, net of a no-op defect.

    SIMIND histories are fixed, so a defect redistributes counts: the projections
    are normalised to equal totals and the region is where the healthy view-sum
    exceeds the defect one. That region is chosen from the data, so noise alone
    gives a positive drop; the same measure for a defect with activity ratio 1
    (which changes nothing but the random stream) is subtracted.

    The defect footprint is not projected geometrically: voxelise does not
    support CardiacSource, SIMIND's cardiac rotation conventions are not
    documented, and activity ratios above 1 behave as 1 (so no defect-only run).
    """

    def projection(prefix, defect):
        connector = _analytic_connector(tmp_path, prefix)
        heart = CardiacSource((4.0, 4.0, 4.0), defect=defect)
        connector.configure_phantom(AnalyticPhantom(heart, _WATER))
        return connector.run()["tot_w1"].projection.sum(axis=0)

    def defect_at(activity_ratio):
        return CardiacDefect(
            location_deg=0.0, angular_size_deg=45.0, activity_ratio=activity_ratio
        )

    def drop_in_emptied_pixels(healthy, other):
        healthy, other = healthy / healthy.sum(), other / other.sum()
        emptied = healthy > other
        return (healthy[emptied].sum() - other[emptied].sum()) / healthy[emptied].sum()

    case = "cardiac_defect_lowers_counts"
    with v9_linux_guard(case):
        healthy = projection("heart", None)
        defect = projection("defect", defect_at(0.0))
        no_op_defect = projection("no_op", defect_at(1.0))

    validate(
        case,
        float(
            drop_in_emptied_pixels(healthy, defect)
            - drop_in_emptied_pixels(healthy, no_op_defect)
        ),
        lower=0.01,
        smoke=True,
    )


@pytest.mark.macos_v9
@pytest.mark.parametrize("library", list(LibraryPhantom))
def test_library_phantom(validate, v9_linux_guard, tmp_path, library):
    case = f"library_{library.name.lower()}"
    prefix = "library"
    with v9_linux_guard(case):
        connector = _connector(tmp_path, prefix, num_projections=8)
        connector.configure_phantom(library)
        projection = connector.run()["tot_w1"].projection
    density, _ = read_simind_density_image(tmp_path / prefix / f"{prefix}.hct")

    validate(
        case,
        float(projection.sum() > 0 and density.max() > 0),
        expected=1.0,
        abs_tol=0,
    )


@pytest.fixture(scope="module")
def nema():
    import phantomgen

    phantom = nema_iec_phantom((96,) * 3, (4.0,) * 3, preset="pet", supersample=4)
    return phantom, phantomgen.pet_nema_dict


def _hot_sphere(preset, diameter_mm):
    sphere_dict = preset["sphere_dict"]
    spheres = sphere_dict["spheres"]
    index = spheres["diametre_mm"].index(diameter_mm)
    angle = math.radians(spheres["angle_loc"][index])
    centre_zyx_mm = (
        sphere_dict["ring_z"],
        -sphere_dict["ring_R"] * math.sin(angle),
        sphere_dict["ring_R"] * math.cos(angle),
    )
    return centre_zyx_mm, spheres["act_conc_MBq_ml"][index]


@pytest.mark.requires_phantomgen
@pytest.mark.parametrize("diameter_mm", HOT_SPHERE_DIAMETERS_MM)
def test_nema_hot_sphere_activity(validate, nema, diameter_mm):
    phantom, preset = nema
    voxel_mm = phantom.voxel_size_mm[0]
    centre_mm, sphere_concentration = _hot_sphere(preset, diameter_mm)
    background_concentration = preset["activity_concentration_background"]
    ml_per_voxel = voxel_mm**3 / 1000.0

    axes = [(np.arange(n) - (n - 1) / 2) * voxel_mm for n in phantom.activity_mbq.shape]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    radius_mm = diameter_mm / 2
    ball = (
        (z - centre_mm[0]) ** 2 + (y - centre_mm[1]) ** 2 + (x - centre_mm[2]) ** 2
    ) <= (radius_mm + 2 * voxel_mm) ** 2
    excess_mbq = (
        phantom.activity_mbq[ball].sum()
        - background_concentration * ml_per_voxel * ball.sum()
    )
    sphere_volume_ml = 4 / 3 * math.pi * radius_mm**3 / 1000.0

    validate(
        f"nema_hot_sphere_activity_{diameter_mm}mm",
        float(
            excess_mbq
            / ((sphere_concentration - background_concentration) * sphere_volume_ml)
        ),
        expected=1.0,
        rel=0.03,
    )


@pytest.mark.requires_phantomgen
def test_nema_background_density_is_water(validate, nema):
    phantom, _ = nema

    validate(
        "nema_background_density_is_water",
        float(np.median(phantom.density_g_cm3[phantom.masks["background"] > 0])),
        expected=1.0,
        abs_tol=1e-6,
    )
