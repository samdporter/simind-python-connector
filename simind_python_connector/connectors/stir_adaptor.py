"""STIR adaptor: STIR images in, STIR ProjData out."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simind_python_connector.connectors._native_adaptor import _NativeSimindAdaptor


try:
    import stir
except ImportError:  # pragma: no cover - optional dependency
    stir = None  # type: ignore[assignment]


class StirSimindAdaptor(_NativeSimindAdaptor):
    """Adaptor consuming/returning STIR-native objects."""

    def _require_backend(self) -> None:
        if stir is None:
            raise ImportError("StirSimindAdaptor requires the STIR Python package.")

    def _load_output(self, header_path: Path) -> Any:
        return stir.ProjData.read_from_file(str(header_path))

    def _voxel_sizes_mm(self, image: Any) -> tuple[float, float, float]:
        # STIR coordinates are 1-based: [1] is z, [2] is y, [3] is x.
        spacing = image.get_grid_spacing()
        return (float(spacing[1]), float(spacing[2]), float(spacing[3]))


__all__ = ["StirSimindAdaptor"]
