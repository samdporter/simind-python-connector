#!/usr/bin/env python
"""
Phantoms without hand-made voxel maps.

With phantomgen installed, simulate its NEMA/IEC body phantom (pet preset: 8:1
hot spheres, two cold spheres, lung insert) by normalisation route 1, since its
activity is known. Without phantomgen, simulate SIMIND's own multiple-inserts
(Jaszczak-like) phantom.

Install phantomgen with:
    pip install "phantomgen @ git+https://github.com/varzakis/phantomgen"
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np

from simind_python_connector import SimindPythonConnector, configs
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE
from simind_python_connector.phantoms import (
    AnalyticPhantom,
    HorizontalCylinder,
    Insert,
    MultipleInserts,
    nema_iec_phantom,
)


def require_simind() -> None:
    if shutil.which(os.environ.get("SIMIND_BIN") or "simind") is None:
        raise RuntimeError("SIMIND executable not found ($SIMIND_BIN or PATH).")


def nema_phantom_or_none():
    try:
        return nema_iec_phantom((128, 128, 128), (4.42, 4.42, 4.42), preset="pet")
    except ImportError as exc:
        print(f"{exc}\nFalling back to SIMIND's multiple-inserts phantom.")
        return None


def jaszczak_like() -> AnalyticPhantom:
    water = HorizontalCylinder(10.0, (10.0, 10.0))
    radii = (0.5, 0.65, 0.85, 1.1, 1.4, 1.85)
    inserts = tuple(
        Insert(
            (r, r, r),
            (0.0, 5.7 * math.cos(math.radians(a)), 5.7 * math.sin(math.radians(a))),
            8.0,
        )
        for r, a in zip(radii, range(30, 360, 60))
    )
    return AnalyticPhantom(MultipleInserts(water, inserts, background=1.0), water)


def main() -> None:
    require_simind()
    output_dir = Path("output/phantoms")
    output_dir.mkdir(parents=True, exist_ok=True)
    connector = SimindPythonConnector(
        configs.get("Example.yaml"),
        output_dir,
        "phantom",
        quantization_scale=0.05 * DEFAULT_QUANTIZATION_SCALE,
    )
    connector.set_energy_windows([126], [154], [0])
    connector.add_runtime_switch("CC", "ma-lehr")
    connector.add_runtime_switch("RR", 12345)

    phantom = nema_phantom_or_none()
    if phantom is None:
        connector.add_config_value(28, 0.4)  # 64 x 64 projections of 4 mm
        connector.add_config_value(76, 64)
        connector.add_config_value(77, 64)
        connector.configure_phantom(jaszczak_like())
    else:
        # The projection grid is no longer derived from the phantom, so match it
        # to the phantom instead of inheriting Example.yaml's 32 x 32 grid.
        nz, _, nx = phantom.activity_mbq.shape
        connector.add_config_value(28, phantom.voxel_size_mm[2] / 10.0)
        connector.add_config_value(76, nx)
        connector.add_config_value(77, nz)
        connector.configure_phantom(phantom, time_per_projection_s=10.0)
        np.save(output_dir / "activity_mbq_per_voxel.npy", phantom.activity_mbq)
        np.save(output_dir / "density_g_cm3.npy", phantom.density_g_cm3)
        for name, mask in sorted(phantom.masks.items()):
            np.save(output_dir / f"mask_{name}.npy", mask)
            print(f"{name:12s} {phantom.activity_mbq[mask > 0].sum():9.3f} MBq")

    projection = connector.run()["tot_w1"].projection
    print(f"projections {projection.shape}, total counts {projection.sum():.4g}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
