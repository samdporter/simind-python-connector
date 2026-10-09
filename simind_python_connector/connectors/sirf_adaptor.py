"""SIRF adaptor: SIRF images in, SIRF AcquisitionData out."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simind_python_connector.connectors._native_adaptor import _NativeSimindAdaptor


try:
    import sirf.STIR as sirf
except ImportError:  # pragma: no cover - optional dependency
    sirf = None  # type: ignore[assignment]


class SirfSimindAdaptor(_NativeSimindAdaptor):
    """Adaptor consuming/returning SIRF-native objects."""

    _backend = "sirf"

    def _image_like(self, template: Any, array: Any) -> Any:
        image = template.clone()
        image.fill(array)
        return image

    def _require_backend(self) -> None:
        if sirf is None:
            raise ImportError("SirfSimindAdaptor requires the SIRF Python package.")

    def _load_output(self, header_path: Path) -> Any:
        return sirf.AcquisitionData(str(header_path))

    def _voxel_sizes_mm(self, image: Any) -> tuple[float, float, float]:
        z, y, x = image.voxel_sizes()
        return (float(z), float(y), float(x))

    def _origin_mm(self, image: Any) -> tuple[float, float, float]:
        z, y, x = image.get_geometrical_info().get_offset()
        return (float(z), float(y), float(x))


__all__ = ["SirfSimindAdaptor"]
