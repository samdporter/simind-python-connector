"""Phantoms: SIMIND's built-in shapes and library phantoms, voxel phantoms, and
phantomgen's NEMA/IEC body phantom.

Analytic shapes use SIMIND's coordinates in cm:

- x is the patient axis (feet to head);
- y points to the patient's right seen from the feet;
- z points towards the camera at angle 0 (anterior).
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
             Index 45 section of the phantom.zub code table)
    """

    ZUBAL_TORSO = (-2, "vox_man", 1)
    ZUBAL_BRAIN = (-3, "vox_brn", 2)
    ZUBAL_WHOLE_BODY = (-4, "vox_man3", 1)
    NEMA_IQ = (-5, "nema", 3)


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


def _inside(shape, X, Y, Z) -> np.ndarray:
    """Boolean mask of the points (X, Y, Z), in cm, that lie inside shape."""
    if isinstance(shape, Ellipsoid):
        a, b, c = shape.half_axes_cm
        return (X / a) ** 2 + (Y / b) ** 2 + (Z / c) ** 2 <= 1.0
    if isinstance(shape, Box):
        hx, hy, hz = shape.half_sizes_cm
        return (np.abs(X) <= hx) & (np.abs(Y) <= hy) & (np.abs(Z) <= hz)
    if isinstance(shape, HorizontalCylinder):
        ry, rz = shape.semi_axes_cm
        return (np.abs(X) <= shape.half_length_cm) & (
            (Y / ry) ** 2 + (Z / rz) ** 2 <= 1.0
        )
    if isinstance(shape, VerticalCylinder):
        rx, ry = shape.semi_axes_cm
        return (np.abs(Z) <= shape.half_length_cm) & (
            (X / rx) ** 2 + (Y / ry) ** 2 <= 1.0
        )
    raise NotImplementedError(f"voxelise does not support {type(shape).__name__}")


def _insert_shape(insert: Insert):
    # Insert sizes are taken as half-dimensions along x, y and z, like Index 2-4.
    sx, sy, sz = insert.size_cm
    if insert.shape == "sphere":
        return Ellipsoid((sx, sy, sz))
    if insert.shape == "horizontal_rod":
        return HorizontalCylinder(sx, (sy, sz))
    if insert.shape == "rectangular_rod":
        return Box((sx, sy, sz))
    if insert.shape == "vertical_rod":
        return VerticalCylinder(sz, (sx, sy))
    raise NotImplementedError(f"voxelise does not support {insert.shape} inserts")


def _source_values(source, X, Y, Z) -> np.ndarray:
    if isinstance(source, CardiacSource):
        raise NotImplementedError("voxelise does not support CardiacSource")
    if not isinstance(source, MultipleInserts):
        return _inside(source, X, Y, Z).astype(np.float64)
    if source.mode == "cold":
        base = 1.0
    elif source.mode == "hot" or source.background is None:
        base = 0.0
    else:
        base = source.background
    values = np.where(_inside(source.container, X, Y, Z), base, 0.0)
    for insert in source.inserts:
        px, py, pz = insert.position_cm
        inside = _inside(_insert_shape(insert), X - px, Y - py, Z - pz)
        value = 0.0 if source.mode == "cold" else abs(insert.concentration)
        values = np.where(inside, value, values)
    return values


def _block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    nz, ny, nx = (n // factor for n in values.shape)
    blocks = values.reshape(nz, factor, ny, factor, nx, factor)
    return blocks.mean(axis=(1, 3, 5)).astype(np.float32)


def voxelise(
    phantom: AnalyticPhantom,
    shape_zyx: tuple[int, int, int],
    voxel_size_mm: tuple[float, float, float],
    supersample: int = 1,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (activity, attenuator fraction) on a centred grid in (z, y, x) order.

    Voxel [k, j, i] has its centre at SIMIND coordinates (cm), from the
    manual's voxel-map orientation::

        X = -(k - (nz - 1) / 2) * dz,  Y = (i - (nx - 1) / 2) * dx,
        Z = -(j - (ny - 1) / 2) * dy,

    The source is moved by source_shift_cm. A voxel holds the fraction of its
    supersample**3 sub-voxel centres inside the shape, times the concentration
    (1 for plain sources). A point source is the single nearest voxel. The
    attenuator fraction is None without an attenuator.
    """
    nz, ny, nx = (int(n) for n in shape_zyx)
    dz, dy, dx = (float(v) / 10.0 for v in voxel_size_mm)
    sx, sy, sz = phantom.source_shift_cm
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5

    def centres(n: int, size: float) -> np.ndarray:
        return ((np.arange(n)[:, None] + offsets).ravel() - (n - 1) / 2) * size

    X = -centres(nz, dz)[:, None, None]
    Z = -centres(ny, dy)[None, :, None]
    Y = centres(nx, dx)[None, None, :]
    fine_shape = (X.shape[0], Z.shape[1], Y.shape[2])

    if isinstance(phantom.source, PointSource):
        k = round((nz - 1) / 2 - sx / dz)
        j = round((ny - 1) / 2 - sz / dy)
        i = round((nx - 1) / 2 + sy / dx)
        if not (0 <= k < nz and 0 <= j < ny and 0 <= i < nx):
            raise ValueError(
                f"point source at {phantom.source_shift_cm} cm is outside the grid"
            )
        activity = np.zeros((nz, ny, nx), dtype=np.float32)
        activity[k, j, i] = 1.0
    else:
        values = _source_values(phantom.source, X - sx, Y - sy, Z - sz)
        activity = _block_mean(np.broadcast_to(values, fine_shape), supersample)

    attenuator = None
    if phantom.attenuator is not None:
        inside = np.broadcast_to(_inside(phantom.attenuator, X, Y, Z), fine_shape)
        attenuator = _block_mean(inside.astype(np.float64), supersample)
    return activity, attenuator


def nema_iec_phantom(
    matrix_size_zyx: tuple[int, int, int],
    voxel_size_mm: tuple[float, float, float],
    preset: Union[str, dict] = "pet",
    supersample: int = 1,
    center_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> VoxelPhantom:
    """Build the NEMA/IEC body phantom with phantomgen.

    preset "pet" or "earl" selects phantomgen's dict; a dict is passed through.
    The activity (MBq per voxel) and the masks are used unchanged. The density
    is mu divided by the median mu of the background fill, i.e. water
    equivalent whatever energy phantomgen's mu values were defined at.
    phantomgen's (Z, Y, X) arrays, with the cylinder axis along Z, are already
    in the connector's (z, y, x) order.
    """
    try:
        import phantomgen
    except ImportError as exc:
        raise ImportError(
            "nema_iec_phantom needs phantomgen: pip install "
            '"phantomgen @ git+https://github.com/varzakis/phantomgen"'
        ) from exc

    if isinstance(preset, str):
        presets = {"pet": phantomgen.pet_nema_dict, "earl": phantomgen.earl_nema_dict}
        if preset not in presets:
            raise ValueError(
                f"preset must be 'pet' or 'earl', or a dict; got {preset!r}"
            )
        nema_dict = presets[preset]
    else:
        nema_dict = preset

    activity, mu, masks = phantomgen.create_nema(
        matrix_size=tuple(int(n) for n in matrix_size_zyx),
        voxel_size_mm=tuple(float(v) for v in voxel_size_mm),
        nema_dict=nema_dict,
        center_offset_mm=tuple(float(v) for v in center_offset_mm),
        supersample=supersample,
    )
    fill_mu = float(np.median(mu[masks["background"] > 0]))
    return VoxelPhantom(
        activity_mbq=activity,
        density_g_cm3=(mu / fill_mu).astype(np.float32),
        voxel_size_mm=tuple(float(v) for v in voxel_size_mm),
        masks=dict(masks),
    )
