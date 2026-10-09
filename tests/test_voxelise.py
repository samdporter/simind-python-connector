import math

import numpy as np
import pytest

from simind_python_connector.phantoms import (
    AnalyticPhantom,
    Box,
    CardiacSource,
    Ellipsoid,
    HorizontalCylinder,
    Insert,
    MultipleInserts,
    PointSource,
    VerticalCylinder,
    voxelise,
)


pytestmark = pytest.mark.unit


def _simind_centroid(activity, voxel_size_mm):
    """Activity centroid in SIMIND (X, Y, Z) cm, using the manual's orientation."""
    nz, ny, nx = activity.shape
    dz, dy, dx = (v / 10.0 for v in voxel_size_mm)
    k, j, i = np.indices(activity.shape)
    total = activity.sum()
    kc, jc, ic = ((index * activity).sum() / total for index in (k, j, i))
    return (
        -(kc - (nz - 1) / 2) * dz,
        (ic - (nx - 1) / 2) * dx,
        -(jc - (ny - 1) / 2) * dy,
    )


@pytest.mark.parametrize(
    "source, volume_cm3",
    [
        (Ellipsoid((3.0, 2.0, 1.5)), 4.0 / 3.0 * math.pi * 3.0 * 2.0 * 1.5),
        (HorizontalCylinder(2.5, (2.0, 1.5)), math.pi * 2.0 * 1.5 * 5.0),
        (VerticalCylinder(1.5, (2.0, 2.5)), math.pi * 2.0 * 2.5 * 3.0),
    ],
)
def test_volumes_within_three_percent(source, volume_cm3):
    activity, attenuator = voxelise(
        AnalyticPhantom(source), (70, 70, 70), (1.0, 1.0, 1.0), supersample=3
    )
    assert attenuator is None
    assert activity.dtype == np.float32
    assert activity.sum() * 0.1**3 == pytest.approx(volume_cm3, rel=0.03)


def test_centroid_follows_the_source_shift():
    shift = (2.0, 1.0, -1.5)
    phantom = AnalyticPhantom(Box((1.0, 1.0, 1.0)), source_shift_cm=shift)
    activity, _ = voxelise(phantom, (40, 50, 60), (2.0, 2.0, 2.0), supersample=2)
    np.testing.assert_allclose(
        _simind_centroid(activity, (2.0, 2.0, 2.0)), shift, atol=0.02
    )


def test_point_source_is_one_voxel():
    phantom = AnalyticPhantom(PointSource(), source_shift_cm=(1.0, -0.5, 2.0))
    activity, _ = voxelise(phantom, (21, 21, 21), (5.0, 5.0, 5.0))
    assert activity.sum() == 1.0
    assert (
        activity[8, 6, 9] == 1.0
    )  # k = 10 - 1.0/0.5, j = 10 - 2.0/0.5, i = 10 - 0.5/0.5


def test_point_source_outside_the_grid_is_rejected():
    phantom = AnalyticPhantom(PointSource(), source_shift_cm=(50.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="outside the grid"):
        voxelise(phantom, (11, 11, 11), (5.0, 5.0, 5.0))


def test_attenuator_fraction_is_returned():
    phantom = AnalyticPhantom(PointSource(), attenuator=Box((1.0, 1.0, 1.0)))
    _, attenuator = voxelise(phantom, (20, 20, 20), (2.0, 2.0, 2.0))
    assert attenuator.dtype == np.float32
    assert attenuator.sum() * 0.2**3 == pytest.approx(8.0, rel=0.01)


@pytest.mark.parametrize(
    "mode, background, centre, edge",
    [("background", 1.0, 4.0, 1.0), ("hot", None, 4.0, 0.0), ("cold", None, 0.0, 1.0)],
)
def test_inserts_are_painted_over_the_container(mode, background, centre, edge):
    source = MultipleInserts(
        HorizontalCylinder(6.0, (6.0, 6.0)),
        (Insert((2.0, 2.0, 2.0), (0.0, 0.0, 0.0), 4.0),),
        background=background,
        mode=mode,
    )
    activity, _ = voxelise(AnalyticPhantom(source), (31, 31, 31), (4.0, 4.0, 4.0))
    assert activity[15, 15, 15] == pytest.approx(centre)
    assert activity[15, 15, 25] == pytest.approx(
        edge
    )  # 4 cm along +Y, inside the container
    assert activity[15, 0, 0] == 0.0  # corner (Y = -6, Z = 6 cm), outside the container


def test_absolute_concentrations_are_painted_as_magnitudes():
    source = MultipleInserts(
        HorizontalCylinder(6.0, (6.0, 6.0)),
        (Insert((2.0, 2.0, 2.0), (0.0, 0.0, 0.0), -0.3),),
        mode="hot",
    )
    activity, _ = voxelise(AnalyticPhantom(source), (31, 31, 31), (4.0, 4.0, 4.0))
    assert activity[15, 15, 15] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "source",
    [
        CardiacSource((4.0, 4.0, 4.0)),
        MultipleInserts(
            HorizontalCylinder(6.0, (6.0, 6.0)),
            (Insert((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 1.0, shape="cone"),),
        ),
        MultipleInserts(
            HorizontalCylinder(6.0, (6.0, 6.0)),
            (Insert((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 1.0, shape="hexagonal_rod"),),
        ),
    ],
)
def test_unsupported_shapes_raise(source):
    with pytest.raises(NotImplementedError):
        voxelise(AnalyticPhantom(source), (8, 8, 8), (4.0, 4.0, 4.0))
