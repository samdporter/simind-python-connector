"""Phantoms: SIMIND's built-in shapes and library phantoms, voxel phantoms, and
phantomgen's NEMA/IEC body phantom.

Analytic shapes use SIMIND's coordinates in cm: x is the patient axis (feet to
head), y points to the patient's right seen from the feet, and z points towards
the camera at angle 0 (anterior).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class Ellipsoid:
    """Source only (Index 15 = 1)."""

    half_axes_cm: tuple[float, float, float]  # (x, y, z)


@dataclass(frozen=True)
class Box:
    """Index 15 = 2 as a source, Index 14 = 2 as an attenuator."""

    half_sizes_cm: tuple[float, float, float]  # (x, y, z)


@dataclass(frozen=True)
class HorizontalCylinder:
    """Index 15/14 = 4, with its axis along x (the patient axis)."""

    half_length_cm: float
    semi_axes_cm: tuple[float, float]  # (y, z); elliptical if different


@dataclass(frozen=True)
class VerticalCylinder:
    """Index 15/14 = 3, with its axis along z (towards the camera)."""

    half_length_cm: float
    semi_axes_cm: tuple[float, float]  # (x, y)


@dataclass(frozen=True)
class PointSource:
    """Index 15 = 5: a point at the origin, moved by source_shift_cm."""


_INSERT_SHAPES = {
    "sphere": 0,
    "horizontal_rod": 1,
    "rectangular_rod": 2,
    "hexagonal_rod": 3,
    "vertical_rod": 4,
    "cone": 5,
}


@dataclass(frozen=True)
class Insert:
    """One row of SIMIND's multiple-spheres input file."""

    size_cm: tuple[float, float, float]
    position_cm: tuple[float, float, float]
    concentration: float  # > 0 relative, < 0 absolute MBq/cc
    shape: str = "sphere"

    def __post_init__(self) -> None:
        if self.shape not in _INSERT_SHAPES:
            raise ValueError(
                f"shape must be one of {', '.join(_INSERT_SHAPES)}, got {self.shape!r}"
            )


@dataclass(frozen=True)
class MultipleInserts:
    """Index 15 = 7: inserts in a cylindrical container (the Jaszczak routine)."""

    container: HorizontalCylinder  # Index 2-4
    inserts: tuple[Insert, ...]
    background: Optional[float] = None  # /BG
    mode: str = "background"  # "background", "hot" (/HO) or "cold" (/CO)

    def __post_init__(self) -> None:
        if self.mode not in ("background", "hot", "cold"):
            raise ValueError(
                f"mode must be 'background', 'hot' or 'cold', got {self.mode!r}"
            )


@dataclass(frozen=True)
class CardiacDefect:
    location_deg: float  # /L1: 0 anterior, 180 inferior
    angular_size_deg: float = 30.0  # /L2
    start_from_base_cm: float = 0.6  # /L3
    axial_extent_cm: float = 2.0  # /L4
    transgression: float = 1.0  # /L5
    activity_ratio: float = 1.0  # /L6


@dataclass(frozen=True)
class CardiacSource:
    """Index 15 = 6, SIMIND's myocardial routine; the defaults are the manual's."""

    half_dims_cm: tuple[float, float, float]  # Index 2-4
    orientation_deg: tuple[float, float, float] = (122.0, 52.0, 0.0)  # /A1 /A2 /A3
    defect: Optional[CardiacDefect] = None  # None -> /L1:-999
    myocardium_thickness_cm: float = 1.0  # /M1
    plastic_wall_cm: float = 0.2  # /M2
    chamber_length_cm: float = 8.0  # /M3
    chamber_diameter_cm: float = 6.1  # /M4


Source = Union[
    Ellipsoid,
    Box,
    HorizontalCylinder,
    VerticalCylinder,
    PointSource,
    MultipleInserts,
    CardiacSource,
]
Attenuator = Union[Box, HorizontalCylinder, VerticalCylinder]


@dataclass(frozen=True)
class AnalyticPhantom:
    source: Source
    attenuator: Optional[Attenuator] = None
    source_shift_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Index 16-18

    def __post_init__(self) -> None:
        if self.attenuator is not None and not isinstance(
            self.attenuator, (Box, HorizontalCylinder, VerticalCylinder)
        ):
            raise ValueError(
                "attenuator must be a Box, HorizontalCylinder or VerticalCylinder"
            )


class LibraryPhantom(Enum):
    """Voxel phantoms shipped in SIMIND's smc_dir.

    value = (Index 14/15 code, base name of the file in smc_dir,
             Index 45 .zub table or None)
    """

    ZUBAL_TORSO = (-2, "vox_man1", None)
    ZUBAL_BRAIN = (-3, "vox_brn", None)
    ZUBAL_WHOLE_BODY = (-4, "vox_man3", None)
    NEMA_IQ = (-5, "nema", 4)


@dataclass(frozen=True, eq=False)
class VoxelPhantom:
    activity_mbq: np.ndarray  # (z, y, x), MBq per voxel
    density_g_cm3: np.ndarray  # (z, y, x), water-equivalent density
    voxel_size_mm: tuple[float, float, float]  # (z, y, x)
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # name -> uint8 mask


def _code_and_half_dims(shape) -> tuple[int, tuple[float, float, float]]:
    """Index 14/15 code and the half-dimensions for Index 2-4 (or 5-7), in x, y, z."""
    if isinstance(shape, Ellipsoid):
        return 1, tuple(shape.half_axes_cm)
    if isinstance(shape, Box):
        return 2, tuple(shape.half_sizes_cm)
    if isinstance(shape, VerticalCylinder):
        rx, ry = shape.semi_axes_cm
        return 3, (rx, ry, shape.half_length_cm)
    if isinstance(shape, HorizontalCylinder):
        ry, rz = shape.semi_axes_cm
        return 4, (shape.half_length_cm, ry, rz)
    if isinstance(shape, PointSource):
        return 5, (0.0, 0.0, 0.0)
    if isinstance(shape, CardiacSource):
        return 6, tuple(shape.half_dims_cm)
    if isinstance(shape, MultipleInserts):
        return 7, _code_and_half_dims(shape.container)[1]
    raise TypeError(f"Unsupported phantom shape {type(shape).__name__}")


def _insert_row(insert: Insert) -> str:
    values = (
        *insert.size_cm,
        *insert.position_cm,
        insert.concentration,
        _INSERT_SHAPES[insert.shape],
    )
    return ",".join(f"{value:g}" for value in values)


def _cardiac_switches(source: CardiacSource) -> dict[str, float]:
    a1, a2, a3 = source.orientation_deg
    switches = {
        "A1": a1,
        "A2": a2,
        "A3": a3,
        "M1": source.myocardium_thickness_cm,
        "M2": source.plastic_wall_cm,
        "M3": source.chamber_length_cm,
        "M4": source.chamber_diameter_cm,
    }
    defect = source.defect
    if defect is None:
        switches["L1"] = -999
    else:
        switches.update(
            L1=defect.location_deg,
            L2=defect.angular_size_deg,
            L3=defect.start_from_base_cm,
            L4=defect.axial_extent_cm,
            L5=defect.transgression,
            L6=defect.activity_ratio,
        )
    return switches
