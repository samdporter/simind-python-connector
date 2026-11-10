"""Edit, export and reload a SIMIND configuration.

Start from the bundled ``input.smc`` template, change collimator and acquisition
values by name, export the result to YAML, then load that YAML into a fresh
configuration and change two more values. No simulation or plotting is needed.
"""

import os
from pathlib import Path

from simind_python_connector import SimulationConfig, configs


output_dir = Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "custom_config"
output_dir.mkdir(parents=True, exist_ok=True)

config = SimulationConfig(str(configs.get("input.smc")))
config.set_value("hole_size_x", 0.111)  # 1.11 mm diameter
config.set_value("hole_size_y", 0.111)
config.set_value("collimator_thickness", 2.405)  # 24.05 mm
config.set_value("distance_between_holes_x", 0.16)
config.set_value("distance_between_holes_y", 0.16)
config.set_value("intrinsic_resolution", 0.32)  # 3.2 mm FWHM
config.set_value("energy_resolution", 9.5)  # percent at 140 keV
config.set_value("photon_energy", 140.0)
config.set_value("upper_window_threshold", 154)
config.set_value("lower_window_threshold", 126)

yaml_path = output_dir / "lehr_collimator.yaml"
config.export_yaml(str(yaml_path))
print(f"Exported {yaml_path}")

reloaded = SimulationConfig(str(configs.get("input.smc")))
reloaded.import_yaml(str(yaml_path))
reloaded.set_value("number_photon_histories", 1e7)
reloaded.set_flag("write_pulse_height_distribution_to_file", True)

final_path = output_dir / "lehr_high_stats.yaml"
reloaded.export_yaml(str(final_path))
print(f"Exported {final_path}")
print(f"Validation passed: {reloaded.validate_parameters()}")
print(f"Reloaded hole size: {reloaded.get_value('hole_size_x')} cm")
