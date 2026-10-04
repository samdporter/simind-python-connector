"""Serial vs MPI SIMIND check for A1 item 9 (Task 2 Step 9).

Runs the same phantom serially, with two replicate ranks and with two split
ranks. Prints each run's projection totals and fails if the output keys differ
or the totals do not agree within 5%.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from simind_python_connector import SimindPythonConnector
from simind_python_connector.configs import get


TOTAL_AGREEMENT = 0.05


def run(prefix, output_dir, processes=None, split=False):
    connector = SimindPythonConnector(
        get("Example.yaml"), output_dir, prefix, quantization_scale=0.05
    )
    source = np.zeros((32, 32, 32), np.float32)
    source[12:20, 12:20, 12:20] = 1.0
    connector.configure_voxel_phantom(source, np.where(source > 0, 0.15, 0.0), 4.0)
    connector.set_energy_windows([126], [154], [0])
    connector.add_runtime_switch("FI", "tc99m")
    connector.add_runtime_switch("CC", "ma-lehr")
    connector.add_runtime_switch("RR", 12345)
    connector.add_config_value(29, 24)
    if processes:
        connector.set_mpi(processes, split)
    outputs = connector.run()
    return {key: float(value.projection.sum()) for key, value in outputs.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/a1_mpi")
    args = parser.parse_args(argv)

    results = {
        "serial": run("serial", args.output_dir),
        "replicate": run("replicate", args.output_dir, 2),
        "split": run("split", args.output_dir, 2, True),
    }
    for label, totals in results.items():
        print(f"{label:<9}", totals)

    reference = results["serial"]
    if not reference.get("tot_w1"):
        raise SystemExit("serial run produced no tot_w1")

    failures = []
    for label in ("replicate", "split"):
        totals = results[label]
        if set(totals) != set(reference):
            failures.append(f"{label} keys differ from serial: {sorted(totals)}")
            continue
        relative = abs(totals["tot_w1"] - reference["tot_w1"]) / reference["tot_w1"]
        print(f"{label} tot_w1 relative difference: {relative:.3%}")
        if relative > TOTAL_AGREEMENT:
            failures.append(
                f"{label} tot_w1 differs by {relative:.3%} (> {TOTAL_AGREEMENT:.0%})"
            )

    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
