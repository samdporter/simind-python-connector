#!/usr/bin/env python
"""
Basic pure Python connector simulation.

This example runs SIMIND with voxelized NumPy inputs and reads projections
back as NumPy arrays via ``SimindPythonConnector``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _python_connector_helpers import (
    add_standard_runtime,
    build_small_phantom_zyx,
    projection_view0,
    require_simind,
)

from simind_python_connector import SimindPythonConnector, configs
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE


def main() -> None:
    require_simind()

    output_dir = Path("output/basic_simulation")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = "basic_case01"
    connector = SimindPythonConnector(
        config_source=configs.get("Example.yaml"),
        output_dir=output_dir,
        output_prefix=output_prefix,
        quantization_scale=0.01 * DEFAULT_QUANTIZATION_SCALE,
    )

    source, mu_map = build_small_phantom_zyx()
    source_path, density_path = connector.configure_voxel_phantom(
        source=source,
        mu_map=mu_map,
        voxel_size_mm=4.0,
        scoring_routine=1,
    )
    connector.add_config_value(1, 140.0)
    connector.set_energy_windows([126.0], [154.0], [0])
    connector.add_runtime_switch("CA", 0)
    add_standard_runtime(connector, photon_multiplier=1, seed=12345, nuclide="tc99m")

    outputs = connector.run()
    total = outputs["tot_w1"].projection
    np.save(output_dir / "total.npy", total)

    source_slice = source[source.shape[0] // 2, :, :]
    total_view0 = projection_view0(total)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(source_slice, cmap="viridis")
    axes[0].set_title("Input source")
    axes[1].imshow(total_view0, cmap="magma")
    axes[1].set_title("Total (view 0)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plot_path = output_dir / "basic_simulation_summary.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    print(f"Output directory: {output_dir}")
    print(f"Source file: {source_path}")
    print(f"Density file: {density_path}")
    print(f"Total counts: {float(total.sum()):.3f}")
    print(f"Summary plot: {plot_path}")


if __name__ == "__main__":
    main()
