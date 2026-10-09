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
import shutil
from pathlib import Path

from simind_python_connector import SimindPythonConnector, configs
from simind_python_connector.phantoms import (
    AnalyticPhantom,
    HorizontalCylinder,
    Insert,
    MultipleInserts,
    nema_iec_phantom,
)


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
    if shutil.which("simind") is None:
        raise RuntimeError("SIMIND executable not found in PATH.")
    output_dir = Path("output/phantoms")
    connector = SimindPythonConnector(
        configs.get("Example.yaml"), output_dir, "phantom"
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
        connector.configure_phantom(phantom, time_per_projection_s=10.0)
        for name, mask in sorted(phantom.masks.items()):
            print(f"{name:12s} {phantom.activity_mbq[mask > 0].sum():9.3f} MBq")

    projection = connector.run()["tot_w1"].projection
    print(f"projections {projection.shape}, total counts {projection.sum():.4g}")


if __name__ == "__main__":
    main()
