import numpy as np
import pytest

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
    PointSource,
    VerticalCylinder,
    VoxelPhantom,
    _cardiac_switches,
    _code_and_half_dims,
    _insert_row,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "shape, code, dims",
    [
        (Ellipsoid((3.0, 2.0, 1.0)), 1, (3.0, 2.0, 1.0)),
        (Box((1.0, 2.0, 3.0)), 2, (1.0, 2.0, 3.0)),
        (VerticalCylinder(4.0, (1.0, 2.0)), 3, (1.0, 2.0, 4.0)),
        (HorizontalCylinder(4.0, (1.0, 2.0)), 4, (4.0, 1.0, 2.0)),
        (PointSource(), 5, (0.0, 0.0, 0.0)),
        (CardiacSource((4.0, 3.0, 2.0)), 6, (4.0, 3.0, 2.0)),
        (MultipleInserts(HorizontalCylinder(9.0, (8.0, 7.0)), ()), 7, (9.0, 8.0, 7.0)),
    ],
)
def test_code_and_half_dims(shape, code, dims):
    assert _code_and_half_dims(shape) == (code, dims)


def test_insert_rows_use_the_simind_shape_codes():
    assert (
        _insert_row(Insert((2.0, 2.0, 2.0), (3.0, 0.0, 0.0), 4.0)) == "2,2,2,3,0,0,4,0"
    )
    rod = Insert((1.0, 1.0, 5.0), (-3.0, 1.5, 0.0), -0.5, shape="vertical_rod")
    assert _insert_row(rod) == "1,1,5,-3,1.5,0,-0.5,4"


def test_insert_rejects_an_unknown_shape():
    with pytest.raises(ValueError, match="shape must be one of"):
        Insert((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 1.0, shape="star")


def test_multiple_inserts_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        MultipleInserts(HorizontalCylinder(5.0, (5.0, 5.0)), (), mode="warm")


def test_attenuator_must_be_a_phantom_shape():
    with pytest.raises(ValueError, match="attenuator must be"):
        AnalyticPhantom(PointSource(), attenuator=Ellipsoid((1.0, 1.0, 1.0)))


def test_cardiac_switches_without_a_defect_use_the_manual_defaults():
    assert _cardiac_switches(CardiacSource((4.0, 4.0, 4.0))) == {
        "A1": 122.0,
        "A2": 52.0,
        "A3": 0.0,
        "M1": 1.0,
        "M2": 0.2,
        "M3": 8.0,
        "M4": 6.1,
        "L1": -999,
    }


def test_cardiac_switches_with_a_defect():
    defect = CardiacDefect(
        location_deg=180.0, angular_size_deg=45.0, activity_ratio=0.2
    )
    switches = _cardiac_switches(CardiacSource((4.0, 4.0, 4.0), defect=defect))
    assert {key: switches[key] for key in ("L1", "L2", "L3", "L4", "L5", "L6")} == {
        "L1": 180.0,
        "L2": 45.0,
        "L3": 0.6,
        "L4": 2.0,
        "L5": 1.0,
        "L6": 0.2,
    }


def test_library_phantom_values():
    assert LibraryPhantom.NEMA_IQ.value == (-5, "nema", 3)
    assert LibraryPhantom.ZUBAL_TORSO.value == (-2, "vox_man", 1)
    assert LibraryPhantom.ZUBAL_BRAIN.value == (-3, "vox_brn", 2)
    assert LibraryPhantom.ZUBAL_WHOLE_BODY.value == (-4, "vox_man3", 1)


def test_voxel_phantom_masks_default_to_empty():
    phantom = VoxelPhantom(np.ones((2, 2, 2)), np.ones((2, 2, 2)), (4.0, 4.0, 4.0))
    assert phantom.masks == {}
