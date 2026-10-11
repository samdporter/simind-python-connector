"""Helpers for extracting voxel spacing from SIRF/STIR image objects."""

from __future__ import annotations

from typing import Any, Optional


def _call_or_value(value: Any) -> Any:
    return value() if callable(value) else value


def _spacing_zyx(spacing: Any) -> Optional[tuple[float, float, float]]:
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


def extract_voxel_size_mm(image: Any, backend_name: str) -> float:
    """Return the voxel size in mm from backend image metadata.

    Backend images report spacing in ``(z, y, x)`` order. The connector takes a
    single voxel size, so the voxels must be cubic.
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
    if max(zyx) - min(zyx) > 1e-3:
        raise ValueError(
            f"{backend_name} images must have cubic voxels; got (z, y, x) = "
            f"{tuple(zyx)} mm"
        )
    return float(zyx[0])


__all__ = ["extract_voxel_size_mm"]
