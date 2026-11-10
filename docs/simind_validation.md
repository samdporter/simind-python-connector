# SIMIND validation

SIMIND v8 is supported. V9 is experimental: the Linux build can abort or return
corrupted projections. Use v8 for supported workflows.

The tables report measured values and acceptance thresholds. Smoke cases check
execution and coarse consistency, not quantitative accuracy. Absolute detector
sensitivity and measured-data calibration are not established.

Measurements are from 2026-10-26, except `normalisers_agree_at_high_histories`
and `residual_identity`, measured on 2026-10-18.

Practical limitations:

- Trimmed scaling is biased low with sparse Monte Carlo projections; use sum
  scaling at low histories.
- SPECTUB scaling varies with source distribution; use a matching reference.
- SPECTUB requires the attenuation-map orientation correction described in
  Backends.

Run or update these checks as described in Testing.

## Validation results

<!-- validation:begin -->
### activity

| case | expected | v8-linux | smoke |
|---|---|---|---|
| activity_doubling | 2 | pass 2 | no |
| activity_doubling_binwise | ≤ 1e-05 | pass 0 | no |
| time_doubling | 2 | pass 2 | no |
| source_magnitude_does_not_change_counts | 0 | pass 0 | no |
| histories_do_not_change_counts | ≤ 0.705 | pass 0.019 | yes |

### anisotropic_voxels

| case | expected | v8-linux | smoke |
|---|---|---|---|
| axial_point_spacing_4_12 | 16 | pass 15.9 | no |
| axial_point_spacing_12_20 | 16 | pass 16 | no |
| axial_point_centring | ≤ 0.25 | pass 0.0472 | no |
| axial_extent | 48 | pass 48 | yes |

### backend_agreement

| case | expected | v8-linux | smoke |
|---|---|---|---|
| identical_simind_data | 0 | pass 0 | no |
| hot_to_background_ratio_agrees | 1 | pass 1 | no |
| image_correlation | ≥ 0.95 | pass 0.999 | no |
| total_activity_agrees | 1 | pass 1 | no |

### cil

| case | expected | v8-linux | smoke |
|---|---|---|---|
| svrg_stale_without_refresh | ≥ 0.1 | pass 1.21 | no |
| svrg_snapshot_after_refresh | 0 | pass 0 | no |
| sag_stale_without_refresh | ≥ 0.1 | pass 0.84 | no |
| sag_gradients_after_refresh | 0 | pass 0 | no |
| callback_refresh_iterations | 1 | pass 1 | no |

### interfile

| case | expected | v8-linux | smoke |
|---|---|---|---|
| outputs_parse_dummy | 1 | pass 1 | no |
| outputs_parse_scattwin_ca0 | 1 | pass 1 | no |
| outputs_parse_scattwin_ca1 | 1 | pass 1 | no |
| outputs_parse_penetrate | 1 | pass 1 | no |
| truncated_payload_rejected | 1 | pass 1 | no |
| projection_geometry_from_header | 0 | pass 0 | no |
| penetrate_components_labelled | 1 | pass 1 | no |

### measured_geometry

| case | expected | v8-linux | smoke |
|---|---|---|---|
| cw_correlation | ≥ 0.5 | pass 0.68 | yes |
| cw_centre_gap | ≤ 1 | pass 0.485 | yes |
| ccw_correlation | ≥ 0.5 | pass 0.659 | yes |
| ccw_centre_gap | ≤ 1 | pass 0.488 | yes |
| non_circular_correlation | ≥ 0.5 | pass 0.66 | yes |
| non_circular_centre_gap | ≤ 1 | pass 0.461 | yes |
| anisotropic_correlation | ≥ 0.5 | pass 0.547 | yes |
| anisotropic_centre_gap | ≤ 1 | pass 0.561 | yes |

### mpi

| case | expected | v8-linux | smoke |
|---|---|---|---|
| serial_vs_replicate | 1 | pass 1.01 | yes |
| serial_vs_split | 1 | pass 0.999 | yes |
| replicate_halves_variance | ≤ 0.75 | pass 0.537 | yes |
| replicate_seeds_processes_as_rr_1_to_n | 0 | pass 2.38e-07 | no |
| replicate_three_processes_runs | 1 | pass 1 | yes |
| penetrate_split_refused | 1 | pass 1 | no |
| replicate_fixed_rr_refused | 1 | pass 1 | no |

### mu_map

| case | expected | v8-linux | smoke |
|---|---|---|---|
| attenuation_and_density_inputs_agree | 0 | pass 0 | no |
| hu_input_uses_schneider | 0 | pass 0.000449 | no |
| water_transmission | 0.3956 | pass 0.402 | no |
| bone_transmission | 0.1783 | pass 0.185 | no |

### phantoms

| case | expected | v8-linux | smoke |
|---|---|---|---|
| analytic_vs_voxel_shape_ellipsoid_correlation | ≥ 0.95 | pass 0.997 | yes |
| analytic_vs_voxel_shape_ellipsoid_centre_gap | ≤ 1 | pass 0.102 | yes |
| analytic_vs_voxel_shape_box_correlation | ≥ 0.95 | pass 0.997 | yes |
| analytic_vs_voxel_shape_box_centre_gap | ≤ 1 | pass 0.0754 | yes |
| analytic_vs_voxel_shape_horizontal_cylinder_correlation | ≥ 0.95 | pass 0.997 | yes |
| analytic_vs_voxel_shape_horizontal_cylinder_centre_gap | ≤ 1 | pass 0.0769 | yes |
| analytic_vs_voxel_shape_vertical_cylinder_correlation | ≥ 0.95 | pass 0.997 | yes |
| analytic_vs_voxel_shape_vertical_cylinder_centre_gap | ≤ 1 | pass 0.0596 | yes |
| inserts_mode_background_1_correlation | ≥ 0.95 | pass 0.986 | yes |
| inserts_mode_background_1_centre_gap | ≤ 1 | pass 0.0891 | yes |
| inserts_mode_background_1_total_ratio | 1 | pass 0.99 | yes |
| inserts_mode_background_3_correlation | ≥ 0.95 | pass 0.985 | yes |
| inserts_mode_background_3_centre_gap | ≤ 1 | pass 0.119 | yes |
| inserts_mode_background_3_total_ratio | 1 | pass 0.988 | yes |
| inserts_mode_hot_correlation | ≥ 0.95 | pass 0.998 | yes |
| inserts_mode_hot_centre_gap | ≤ 1 | pass 0.109 | yes |
| inserts_mode_hot_total_ratio | 1 | pass 0.992 | yes |
| inserts_mode_cold_correlation | ≥ 0.95 | pass 0.985 | yes |
| inserts_mode_cold_centre_gap | ≤ 1 | pass 0.112 | yes |
| inserts_mode_cold_total_ratio | 1 | pass 0.988 | yes |
| cardiac_defect_lowers_counts | ≥ 0.01 | pass 0.0865 | yes |
| library_phantom_ZUBAL_TORSO | 1 | pass 1 | no |
| library_phantom_ZUBAL_BRAIN | 1 | pass 1 | no |
| library_phantom_ZUBAL_WHOLE_BODY | 1 | pass 1 | no |
| library_phantom_NEMA_IQ | 1 | pass 1 | no |
| nema_hot_sphere_activity_17mm | 1 | pass 0.998 | no |
| nema_hot_sphere_activity_22mm | 1 | pass 0.998 | no |
| nema_hot_sphere_activity_28mm | 1 | pass 1 | no |
| nema_hot_sphere_activity_37mm | 1 | pass 1 | no |

### recon

| case | expected | v8-linux | smoke |
|---|---|---|---|
| route2_scatter_matches_route1_sum | 1 | pass 0.995 | yes |
| route2_scatter_matches_route1_shape | ≥ 0.9 | pass 0.95 | yes |
| osem_refreshes_at_update_boundaries | 1 | pass 1 | no |
| damping_combines_exactly | 0 | pass 1.59e-08 | no |
| scatter_correction_removes_the_scatter_fraction | 0.4366 | pass 0.403 | yes |
| trimmed_scale_bias_low_histories | 0.3–0.8 | pass 0.548 | yes |
| normalisers_agree_at_high_histories | 1 | pass 0.959 | yes |
| residual_identity | 0 | pass 1.87e-08 | no |

### runtime_switches

| case | expected | v8-linux | smoke |
|---|---|---|---|
| ca_raw_files_0 | 1 | pass 1 | no |
| ca_raw_files_1 | 1 | pass 1 | no |
| ca_raw_files_2 | 1 | pass 1 | no |
| ca1_derived_primary | 0 | pass 0 | no |
| fw_two_windows | 0.05–1 | pass 0.678 | yes |
| fw_sixteen_character_name | 2 | pass 2 | no |
| x1_default_is_identity | 0 | pass 0 | no |
| x1_copper_transmission | 0.7–0.8 | pass 0.76 | no |
| x2_default_is_identity | 0 | pass 0 | no |
| x3_default_is_identity | 0 | pass 0 | no |
| x4_default_is_identity | 0 | pass 0 | no |
| x6_default_is_identity | 0 | pass 0 | no |
| scattwin_default_ca | ≥ 1 | pass 1 | no |

### scale_to_reference

| case | expected | v8-linux | smoke |
|---|---|---|---|
| scale_stable_across_seeds | 1 | pass 0.991 | yes |
| scale_per_unit_activity_image_independent | 1 | pass 1.07 | yes |
| trimmed_ignores_outlier_bins | ≤ 0.02 | pass 0.015 | no |
| sum_changes_with_outlier_bins | ≥ 0.1 | pass 0.355 | no |

### versions

| case | expected | v8-linux | smoke |
|---|---|---|---|
| point_pair_symmetric_about_centre_row | ≤ 0.25 | pass 0.106 | no |
| point_pair_symmetric_about_centre_bin | ≤ 0.25 | pass 0.0737 | no |
| timeout_enforced | ≤ 10 | pass 1.01 | no |
<!-- validation:end -->
