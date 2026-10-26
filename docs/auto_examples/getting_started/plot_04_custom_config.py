"""
Custom configuration
====================

This example shows how to create and modify SIMIND configurations, including
exporting them to YAML for easy editing and version control.
"""

# %%
# Setup
# -----
# Every configuration starts from the bundled ``input.smc`` template and is
# exported to YAML in the output directory.

import os
from pathlib import Path

from simind_python_connector import SimulationConfig, configs


template_path = str(configs.get("input.smc"))

output_dir = Path(os.environ.get("SIMIND_EXAMPLES_OUTPUT", "output")) / "custom_configs"
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Low-energy high-resolution (LEHR) collimator
# --------------------------------------------
# Collimator geometry, intrinsic resolution and energy resolution are set by
# parameter name.

lehr_config = SimulationConfig(template_path)

lehr_config.set_value("hole_size_x", 0.111)  # 1.11 mm diameter
lehr_config.set_value("hole_size_y", 0.111)  # Round holes
lehr_config.set_value("collimator_thickness", 2.405)  # 24.05 mm
lehr_config.set_value("distance_between_holes_x", 0.16)  # 1.6 mm
lehr_config.set_value("distance_between_holes_y", 0.16)
lehr_config.set_value("intrinsic_resolution", 0.32)  # 3.2 mm FWHM
lehr_config.set_value("energy_resolution", 9.5)  # 9.5% at 140 keV

lehr_yaml = output_dir / "lehr_collimator.yaml"
lehr_config.export_yaml(str(lehr_yaml))

# %%
# High-energy general-purpose (HEGP) collimator
# ---------------------------------------------
# A thicker collimator with wider holes, set up for I-131 at 364 keV.

hegp_config = SimulationConfig(template_path)

hegp_config.set_value("hole_size_x", 0.40)  # 4.0 mm diameter
hegp_config.set_value("hole_size_y", 0.40)
hegp_config.set_value("collimator_thickness", 5.9)  # 59 mm
hegp_config.set_value("distance_between_holes_x", 0.50)  # 5.0 mm
hegp_config.set_value("distance_between_holes_y", 0.50)
hegp_config.set_value("photon_energy", 364.0)
hegp_config.set_value("emitted_photons_per_decay", 0.812)
hegp_config.set_value("upper_window_threshold", 400)
hegp_config.set_value("lower_window_threshold", 328)

hegp_yaml = output_dir / "hegp_collimator.yaml"
hegp_config.export_yaml(str(hegp_yaml))

# %%
# Cardiac SPECT
# -------------
# A 180 degree acquisition from RAO to LPO with 32 views and a camera offset.

cardiac_config = SimulationConfig(template_path)

cardiac_config.set_value("pixel_size_simulated_image", 0.64)  # 6.4 mm pixels
cardiac_config.set_value("matrix_size_image_i", 64)
cardiac_config.set_value("matrix_size_image_j", 64)
cardiac_config.set_value("spect_no_projections", 32)  # 32 views
cardiac_config.set_value("spect_rotation", 3)  # 180 degrees CW
cardiac_config.set_value("spect_starting_angle", 45)  # Start at RAO
cardiac_config.set_value("camera_offset_x", 10.0)  # 10 cm offset

cardiac_yaml = output_dir / "cardiac_spect.yaml"
cardiac_config.export_yaml(str(cardiac_yaml))

# %%
# Modify a configuration through YAML
# -----------------------------------
# Load the exported LEHR YAML back into a fresh configuration, change more
# values, validate it and save it under a new name.

modified_config = SimulationConfig(template_path)
modified_config.import_yaml(str(lehr_yaml))

modified_config.set_value("number_photon_histories", 1e7)  # 10 million
modified_config.set_flag("write_pulse_height_distribution_to_file", True)

modified_yaml = output_dir / "lehr_high_stats.yaml"
modified_config.export_yaml(str(modified_yaml))
print(f"Saved modified configuration to {modified_yaml}")

if modified_config.validate_parameters():
    print("Configuration validation passed!")

# %%
# Use a configuration with the connector
# --------------------------------------
# The exported YAML files can be edited by hand and passed to
# ``SimindPythonConnector`` as ``config_source``:
#
# .. code-block:: python
#
#     from simind_python_connector import SimindPythonConnector
#
#     connector = SimindPythonConnector(
#         config_source="lehr_collimator.yaml",
#         output_dir="output",
#         output_prefix="lehr_sim",
#     )
#     connector.add_runtime_switch("NN", 10)
#     connector.configure_phantom(phantom)  # or configure_voxel_phantom(...)
#     outputs = connector.run()
#
# A configuration-defined phantom also works: without ``configure_phantom()``
# the connector warns and runs the phantom from Index 14/15.
