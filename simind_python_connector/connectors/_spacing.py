"""Helpers for extracting voxel spacing from SIRF/STIR image objects."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_voxel_size_mm(voxel_size_mm: Any) -> tuple[float, float, float]:
    """Return positive voxel spacing in canonical ``(z, y, x)`` order."""
    try:
        values = np.asarray(voxel_size_mm, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "voxel_size_mm must be a scalar or a (z, y, x) sequence"
        ) from exc
    if values.ndim == 0:
        z = y = x = float(values)
    elif values.ndim == 1 and values.size == 3:
        z, y, x = (float(value) for value in values)
    else:
        raise ValueError("voxel_size_mm must be a scalar or a (z, y, x) sequence")
    if not np.isfinite((z, y, x)).all() or min(z, y, x) <= 0:
        raise ValueError(
            "voxel_size_mm must be > 0 and all spacing values must be finite"
        )
    if abs(y - x) > 1e-3:
        raise ValueError(
            f"SIMIND requires square in-plane voxels; got y={y} mm and x={x} mm"
        )
    return z, y, x


def _call_or_value(value: Any) -> Any:
    return value() if callable(value) else value


def _spacing_zyx(spacing: Any) -> tuple[float, float, float] | None:
    try:
        values = tuple(float(v) for v in spacing)
    except Exception:
        values = ()

    if len(values) >= 4:
        # Raw STIR-style coordinate sequence: (unused, z, y, x)
        return values[1], values[2], values[3]
    if len(values) == 3:
        # Public (z, y, x) convention
        return values[0], values[1], values[2]

    if all(hasattr(spacing, axis) for axis in ("z", "y", "x")):
        try:
            z, y, x = (float(_call_or_value(getattr(spacing, axis))) for axis in "zyx")
            return z, y, x
        except Exception:
            pass

    # Non-iterable STIR coordinates index (z, y, x) from 1.
    for accessor in (
        getattr(spacing, "at", None),
        getattr(spacing, "__getitem__", None),
    ):
        if accessor is None:
            continue
        try:
            z, y, x = (float(accessor(index)) for index in (1, 2, 3))
            return z, y, x
        except Exception:
            continue

    return None


def extract_voxel_spacing_mm(
    image: Any, backend_name: str
) -> tuple[float, float, float]:
    """Return voxel spacing in mm from backend image metadata, in (z, y, x).

    SIMIND supports axial spacing distinct from the in-plane image spacing;
    the in-plane dimensions must remain square.
    """
    zyx = None
    if hasattr(image, "voxel_sizes"):
        try:
            voxel_sizes = tuple(float(v) for v in image.voxel_sizes())
            if len(voxel_sizes) >= 3:
                zyx = voxel_sizes[:3]
        except Exception:
            pass

    if zyx is None and hasattr(image, "get_grid_spacing"):
        zyx = _spacing_zyx(image.get_grid_spacing())
        if zyx is None:
            raise ValueError(
                f"Unable to read voxel spacing from {backend_name} get_grid_spacing()."
            )

    if zyx is None:
        raise ValueError(
            f"{backend_name} source object must expose voxel_sizes() or "
            "get_grid_spacing()."
        )
    try:
        return normalize_voxel_size_mm(zyx)
    except ValueError as exc:
        raise ValueError(
            f"{backend_name} voxel spacing is invalid (z, y, x) = {tuple(zyx)} mm: "
            f"{exc}"
        ) from exc


def extract_voxel_size_mm(image: Any, backend_name: str) -> float:
    """Return a scalar for legacy callers when backend voxels are cubic."""
    z, y, x = extract_voxel_spacing_mm(image, backend_name)
    if max(z, y, x) - min(z, y, x) > 1e-3:
        raise ValueError(
            f"{backend_name} images must have cubic voxels; got (z, y, x) = "
            f"{(z, y, x)} mm"
        )
    return z


def _numeric_array(value: Any) -> np.ndarray | None:
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None


def _spatial_geometry(image: Any) -> tuple[Any, ...] | None:
    """Read native geometry where the backend exposes a documented API.

    SIRF's ImageData returns a GeometricalInfo from
    ``get_geometrical_info()``, with offset, spacing, and direction-matrix getters.
    STIR's VoxelsOnCartesianGrid exposes origin and signed grid spacing.
    """
    native_object = getattr(image, "native_object", image)
    image = native_object() if callable(native_object) else native_object

    get_geometrical_info = getattr(image, "get_geometrical_info", None)
    if callable(get_geometrical_info):
        info = get_geometrical_info()
        getters = ("get_offset", "get_spacing", "get_direction_matrix")
        if not all(callable(getattr(info, name, None)) for name in getters):
            raise ValueError(
                "Unable to read SIRF geometry from get_geometrical_info()."
            )
        arrays = [_numeric_array(getattr(info, name)()) for name in getters]
        if any(array is None or not array.size for array in arrays):
            raise ValueError(
                "Unable to read SIRF geometry from get_geometrical_info()."
            )
        if any(not np.isfinite(array).all() for array in arrays if array is not None):
            raise ValueError("SIRF geometry values must be finite.")
        return ("sirf", np.concatenate(arrays))

    get_origin = getattr(image, "get_origin", None)
    get_grid_spacing = getattr(image, "get_grid_spacing", None)
    if callable(get_origin) and callable(get_grid_spacing):
        origin = _spacing_zyx(get_origin())
        spacing = _spacing_zyx(get_grid_spacing())
        if origin is None or spacing is None:
            return None
        minimum_indices = []
        for axis in "zyx":
            getter = getattr(image, f"get_min_{axis}", None)
            if not callable(getter):
                return None
            minimum_indices.append(float(getter()))
        physical_start = np.asarray(origin) + np.asarray(minimum_indices) * np.asarray(
            spacing
        )
        return (
            "stir",
            physical_start,
        )
    return None


def same_spatial_geometry(source: Any, mu_map: Any) -> bool:
    """Compare origins, spacing, and axis directions when both APIs expose them."""
    source_geometry = _spatial_geometry(source)
    mu_geometry = _spatial_geometry(mu_map)
    if source_geometry is None and mu_geometry is None:
        return True
    if source_geometry is None or mu_geometry is None:
        return False
    return source_geometry[0] == mu_geometry[0] and np.allclose(
        source_geometry[1], mu_geometry[1], rtol=0.0, atol=1e-3
    )


__all__ = [
    "extract_voxel_size_mm",
    "extract_voxel_spacing_mm",
    "normalize_voxel_size_mm",
    "same_spatial_geometry",
]
