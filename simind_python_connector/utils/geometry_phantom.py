"""Asymmetric SIMIND image used by the SIRF, STIR, and PyTomography demos."""

from __future__ import annotations

import numpy as np


def build_asymmetric_phantom() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return canonical ``(z, y, x)`` activity, attenuation, and cut indices."""
    z, y, x = np.indices((32, 32, 32))
    body = (
        ((x - 15.0) / 13.5) ** 2
        + ((y - 15.0) / 12.0) ** 2
        + ((z - 15.0) / 13.0) ** 2
        <= 1.0
    )
    body &= ~((x < 11) & (y > 21) & (z > 18))

    source = np.zeros((32, 32, 32), dtype=np.float32)
    source[body] = 0.25

    # A three-axis central bar forms an L in each intersecting central plane.
    bar_x = (np.abs(y - 15) <= 1) & (np.abs(z - 15) <= 1) & (x >= 15) & (x <= 24)
    bar_y = (np.abs(x - 15) <= 1) & (np.abs(z - 15) <= 1) & (y >= 15) & (y <= 23)
    bar_z = (np.abs(x - 15) <= 1) & (np.abs(y - 15) <= 1) & (z >= 15) & (z <= 24)
    source[(bar_x | bar_y | bar_z) & body] = 0.7

    # Unequal off-centre markers encode x, y, and z orientation distinctly.
    marker_a = (x - 23) ** 2 + (y - 11) ** 2 + (z - 15) ** 2 <= 3**2
    marker_b = (x - 15) ** 2 + (y - 21) ** 2 + (z - 22) ** 2 <= 3**2
    marker_c = (x - 10) ** 2 + (y - 16) ** 2 + (z - 9) ** 2 <= 2.5**2
    source[marker_a & body] = 3.0
    source[marker_b & body] = 2.0
    source[marker_c & body] = 1.35

    cold = (x - 11) ** 2 + (y - 14) ** 2 + (z - 23) ** 2 <= 1**2
    source[cold & body] = 0.02

    mu_map = np.zeros_like(source)
    mu_map[body] = 0.12
    mu_map[marker_a & body] = 0.16
    return source, mu_map, {"axial": 15, "coronal": 15, "sagittal": 15}


__all__ = ["build_asymmetric_phantom"]
