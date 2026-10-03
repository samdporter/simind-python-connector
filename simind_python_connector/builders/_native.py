"""Load SIRF or STIR objects for the builders.

Backends are imported inside each function so importing the builders never
imports SIRF or STIR.
"""

from __future__ import annotations

import importlib
from typing import Optional

import numpy as np


def resolve_backend(backend: Optional[str]) -> str:
    """Return "sirf" or "stir". None picks SIRF if importable, else STIR."""
    if backend is None:
        try:
            importlib.import_module("sirf.STIR")
            return "sirf"
        except ImportError:
            pass
        try:
            importlib.import_module("stir")
            importlib.import_module("stirextra")
            return "stir"
        except ImportError:
            raise ImportError(
                "Neither SIRF nor STIR Python found. Install one of: "
                "SIRF (sirf.STIR) or STIR Python (stir + stirextra)."
            ) from None
    if backend == "sirf":
        try:
            importlib.import_module("sirf.STIR")
        except ImportError as exc:
            raise ImportError("SIRF is not available") from exc
        return "sirf"
    if backend == "stir":
        try:
            importlib.import_module("stir")
            importlib.import_module("stirextra")
        except ImportError as exc:
            raise ImportError("STIR Python is not available") from exc
        return "stir"
    raise ValueError("backend must be one of: 'sirf', 'stir', or None")


def load_image(header_path: str, backend: Optional[str]):
    """Read an image header as a native SIRF or STIR image."""
    if resolve_backend(backend) == "sirf":
        import sirf.STIR

        return sirf.STIR.ImageData(header_path)
    import stir

    return stir.FloatVoxelsOnCartesianGrid.read_from_file(header_path)


def load_acquisition(header_path: str, backend: Optional[str]):
    """Read projection data as native SIRF or in-memory STIR data.

    STIR's ProjData reads lazily from disk, so it is copied into memory;
    the builders may delete the files afterwards.
    """
    if resolve_backend(backend) == "sirf":
        import sirf.STIR

        return sirf.STIR.AcquisitionData(header_path)
    import stir

    return stir.ProjDataInMemory(stir.ProjData.read_from_file(header_path))


def refill_acquisition(acquisition, array: np.ndarray, header_path: str, backend: str):
    """Fill a copy of acquisition with array, write it to header_path, return it."""
    if backend == "sirf":
        filled = acquisition.clone()
        filled.fill(array)
        filled.write(header_path)
        return filled
    import stir

    filled = stir.ProjDataInMemory(acquisition)
    filled.fill(array.flat)
    filled.write_to_file(header_path)
    return filled
