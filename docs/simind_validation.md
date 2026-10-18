# SIMIND v8 and v9 validation

This page lists what the connector checks against SIMIND v8 and v9, the defects
and biases found in the engines, and the numbers recorded by the validation
suite for each engine (see the validation results at the end).

- **Input and output checks:** engine output prefixes are capped at 47 characters; up to 99 numbered windows and exact `tot`/`sca`/`pri`/PENETRATE component names are loaded from matching headers; byte offsets and full-pixel payload sizes are checked; measured geometry and source/mu-map grids are checked; DICOM diagnostics, container library isolation, and validation scripts have focused coverage.
- **Window files:** generated `.win` prefixes are capped at 15 characters, and a window file cannot silently follow a changed output prefix. This cap does not apply to runtime `FW`: a v8 raw-engine check read two requested windows from a 16-character `FW` filename. The connector preserves the caller's `/CA` selection and derives the missing order-0 scatter or primary image only when compatible output headers exist. Executor timeouts are exposed through core, native, and PyTomography runs.
- **Interfile parsing:** rejects non-finite or fractional byte offsets and payloads ending in a partial pixel, and complete extra or short pixels.
- **v9 PENETRATE headers:** SIMIND v9 PENETRATE `.hNN` headers are converted alongside their matching `.bNN` payloads. SIMIND v8's shared `.h00` template remains supported. Unknown component 20 is exposed as `b20` without assigning it an inferred physical label.
- **Version selection and centring:** `simind_version=8|9` is available on core/native/PyTomography constructors; `SIMIND_VERSION` supplies the default when omitted, and v8 is the fallback. SIMIND v9's manual records its image-map origin correction. Matching v8/v9 runs used byte-identical source and density arrays, while the `.res` center changed from I/J=9 to 9.5 for a 16³ map. Analytic point-source probes established the corresponding transverse source shift and density-map origin offsets. The connector applies the measured shifts only to a copied v9 voxel-run `.smc`: source Indexes 17/18 use effective `PX`, density Indexes 36/37 use Index 31. It rejects non-default `DI`, density Index 32, or `OR` orientations because those correction mappings have not been validated.
- **Measured-geometry smoke:** the four 16³/eight-view geometry cases (clockwise, counter-clockwise, non-circular, anisotropic) are recorded in the `measured_geometry` table below. Profile: smoke (16³ voxels, 8 views, 10 photon histories per voxel, `quantization_scale` 0.02×default). Each case must reach a correlation above 0.5 with its reference and a centre gap of at most 1 pixel; the full-profile gate (64³, 60 views, 500 histories per voxel, correlation above 0.9) has not been run.
- **MPI normalization:** serial, two-rank replicate and two-rank split totals are compared per build in the `mpi` table below. These are per-build normalization checks, not a cross-version accuracy comparison.
- **Smoke evidence:** which raw files each SCATTWIN `/CA` value writes is recorded in the `runtime_switches` table below (`ca_raw_files_*`). The v8 manual lists `air`, `tot` and `pri` for `/CA:2`, but the engine also writes `sca`. The native macOS ARM64 v9.0 build completes `/CA:2` and returns `ai2/air/pri/sca/tot`.
- **SIMIND v9.0 Linux SCATTWIN defect:** the v9.0 Linux Intel build's SCATTWIN routine either aborts with heap corruption or exits normally with silently corrupted projections (scrambled per-view bin and axial centroids, correlation 0.07-0.42 with the v8 reference). Which happens depends on the number of views, the matrix size, the source content and the working-directory length, so there is no safe setting. v8 Linux, the macOS ARM v9 build and v9 Linux PENETRATE are correct. For SCATTWIN use SIMIND v8 or the macOS v9 build. The validation suite records v9-linux SCATTWIN cases that check projection data as not run.
- **Reviewed math:** the attenuation tables use MeV and cm²/g. The loader divides input keV by 1000 for interpolation, then multiplies the mass attenuation coefficient by density to obtain cm⁻¹. At 140 keV the checked values are water 0.15454 cm⁻¹ and bone 0.287342 cm⁻¹ at densities 1.0 and 1.85 g/cm³. The bilinear `hu_to_attenuation` maps HU values −1000/0/+1000 to air/water/bone; `mu_map_type="hu"` in `configure_voxel_phantom` uses the Schneider2000 density conversion instead (about 1.02 g/cm³ at 0 HU and 1.61 g/cm³ at 1000 HU). Index 25 carries MBq×seconds per projection; source-map values control histories and remain separate from absolute activity. Activity and acquisition-time doubling are checked in the `activity` table below. An exploratory v8 water/air PENETRATE ratio was 0.6081 versus center-path Beer–Lambert 0.6188 for a 16³, 4 mm point source. Monte Carlo uncertainty was not quantified for that check; it does not establish quantitative calibration.
- **Reviewed geometry/reconstruction:** source and attenuation images must share shape, finite positive voxel sizes, origin, and a native direction matrix where the backend exposes one. Measured-geometry SIRF/reconstruction regressions pass. In the SIRF 3.10.1/STIR 6.4 cache regression, disabling all-views caching reduced first-update residual discrepancy from 0.0222821 to 1.55e-6.
- **Complete workflow smokes:** the same 96³ NEMA phantom at 4 mm, 568.1 MBq, four views, `NN=1`, and `quantization_scale=0.01×DEFAULT` completed with both STIR and SIRF on v8 and v9 Linux. Both backends produced finite 96³ reconstructions; projection totals were approximately 66,190 (v8) and 67,500 (v9). A reduced v8 SIRF scatter-correction run completed one refresh with scale 2019 and scatter sum 2.321×10⁵; its contrast error was 7.8% both with and without correction. These smoke runs do not establish quantitative accuracy or an improvement in reconstruction quality.
- **Other backend evidence:** SIRF 3.10.1 uses STIR 6.4.0. PyTomography 3.4.0 with torch 2.14.1+cpu completed reduced Example 07C on both Linux engines: four views, 32³ images, `NN=1`, and `quantization_scale=0.01×DEFAULT`, producing finite nonzero reconstructions. The selected backend tests passed (SIRF geometry/reconstruction 66; STIR 3; PyTomography 31). Both supported real-engine regression runs passed 13 tests, on v8 Linux and v9 macOS ARM respectively.
- **Core and packaging evidence:** 1,148 local core tests passed; the rebuilt core container passed 1,146 tests plus three import-isolation checks. Ruff, strict offline Sphinx, wheel/sdist resource checks, and `uv lock --check --offline` passed. Build tools are declared in the dev extra, and Compose forwards explicit version selection to all services.
- **Remaining quantitative gates:** calibrated absolute detector sensitivity, sufficient-history attenuation and reconstruction comparisons with uncertainty estimates, and measured-data calibration remain outside the evidence above. Full release validation must establish those references and tolerances before making quantitative claims.

## Findings

- **SIMIND v9 Linux SCATTWIN:** see the defect description above. v8 and the macOS v9 build are correct on identical inputs; the `v9_linux_scattwin_defect` case in `runtime_switches` records the failure.
- **SIMIND v9 `simind_mpi` with three processes:** a run was observed to segfault; the suite does not run it on v9. `simind_mpi` seeds rank r with `RR=r+1`, and a replicate run with a fixed `RR` is refused because every process would then simulate the same histories.
- **STIR SPECTUB mirrors x relative to the mu map** (UCL/STIR#452). With the flip that the examples apply, SIRF and PyTomography agree: the recovered hot-to-background ratios differ by 0.1 % on v8 (2.194 against 2.197) and 0.2 % on v9 (2.109 against 2.113). Without the flip the ratio between the two was 1.08.
- **STIR SPECTUB `set_up`:** calling it twice on the same acquisition model gives zero back-projections. This is known STIR behaviour; set up the model once.
- **Per-unit-activity scale:** the scale that matches SPECTUB to SIMIND differs between source distributions by 7 % on v8 and 10 % on v9 (`scale_per_unit_activity_image_independent`). This is a known mismatch between the SIMIND and SPECTUB models; scale to a reference that has the same source distribution as the object being reconstructed. The v9 value is outside the 10 % limit and is an expected failure.
- **Analytic against voxel inserts:** the analytic Jaszczak-like phantom gives 1-2 % fewer counts than its voxelised equivalent in the background and cold modes (total ratio 0.981-0.990 on v8 and macOS v9; hot mode agrees within 1 %). The cause is not established.
- **`trimmed` scaling is biased low at low histories:** the scale is 0.548 (v8) and 0.512 (v9) of the true value on the recon OSEM iterate at a tenth of the default quantization, and 0.959 (v8) and 0.965 (v9) at 64 times those histories (`normalisers_agree_at_high_histories`). The examples use `sum` with smoothing. `trimmed_scale_bias_low_histories` is an expected failure on both engines.

## Validation results

<!-- validation:begin -->
### activity

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| activity_doubling | 2 | pass 2 | — | not run | no |
| activity_doubling_binwise | ≤ 1e-05 | pass 0 | — | — | no |
| time_doubling | 2 | pass 2 | — | not run | no |
| source_magnitude_does_not_change_counts | 0 | pass 0 | — | not run | no |
| histories_do_not_change_counts | ≤ 0.705 | pass 0.019 | — | not run | yes |

### anisotropic_voxels

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| axial_point_spacing_4_12 | 16 | pass 15.9 | — | pass 16 | no |
| axial_point_spacing_12_20 | 16 | pass 16 | — | pass 16 | no |
| axial_point_centring | ≤ 0.25 | pass 0.0472 | — | pass 0.000301 | no |
| axial_extent | 48 | pass 48 | — | pass 48 | yes |

### backend_agreement

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| identical_simind_data | 0 | pass 0 | — | pass 0 | no |
| hot_to_background_ratio_agrees | 1 | pass 1 | — | pass 1 | no |
| image_correlation | ≥ 0.95 | pass 0.999 | — | pass 0.999 | no |
| total_activity_agrees | 1 | pass 1 | — | pass 1 | no |

### cil

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| svrg_stale_without_refresh | ≥ 0.1 | pass 1.21 | — | pass 1.25 | no |
| svrg_snapshot_after_refresh | 0 | pass 0 | — | pass 0 | no |
| sag_stale_without_refresh | ≥ 0.1 | pass 0.84 | — | pass 0.887 | no |
| sag_gradients_after_refresh | 0 | pass 0 | — | pass 0 | no |
| callback_refresh_iterations | 1 | pass 1 | — | pass 1 | no |

### interfile

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| outputs_parse_dummy | 1 | pass 1 | — | pass 1 | no |
| outputs_parse_scattwin_ca0 | 1 | pass 1 | — | not run | no |
| outputs_parse_scattwin_ca1 | 1 | pass 1 | — | not run | no |
| outputs_parse_penetrate | 1 | pass 1 | — | pass 1 | no |
| truncated_payload_rejected | 1 | pass 1 | — | pass 1 | no |
| projection_geometry_from_header | 0 | pass 0 | — | not run | no |
| penetrate_components_labelled | 1 | pass 1 | — | pass 1 | no |

### measured_geometry

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| cw_correlation | ≥ 0.5 | pass 0.68 | — | pass 0.664 | yes |
| cw_centre_gap | ≤ 1 | pass 0.485 | — | pass 0.391 | yes |
| ccw_correlation | ≥ 0.5 | pass 0.659 | — | pass 0.675 | yes |
| ccw_centre_gap | ≤ 1 | pass 0.488 | — | pass 0.482 | yes |
| non_circular_correlation | ≥ 0.5 | pass 0.66 | — | pass 0.676 | yes |
| non_circular_centre_gap | ≤ 1 | pass 0.461 | — | pass 0.328 | yes |
| anisotropic_correlation | ≥ 0.5 | pass 0.547 | — | pass 0.54 | yes |
| anisotropic_centre_gap | ≤ 1 | pass 0.561 | — | pass 0.638 | yes |

### mpi

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| serial_vs_replicate | 1 | pass 1.01 | — | not run | yes |
| serial_vs_split | 1 | pass 0.999 | — | not run | yes |
| replicate_halves_variance | ≤ 0.75 | pass 0.537 | — | not run | yes |
| replicate_seeds_processes_as_rr_1_to_n | 0 | pass 2.38e-07 | — | not run | no |
| replicate_three_processes_runs | 1 | pass 1 | — | not run | yes |
| penetrate_split_refused | 1 | pass 1 | — | pass 1 | no |
| replicate_fixed_rr_refused | 1 | pass 1 | — | pass 1 | no |

### mu_map

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| attenuation_and_density_inputs_agree | 0 | pass 0 | — | pass 0 | no |
| hu_input_uses_schneider | 0 | pass 0.000449 | — | pass 0.000449 | no |
| water_transmission | 0.3956 | pass 0.402 | — | pass 0.4 | no |
| bone_transmission | 0.1783 | pass 0.185 | — | pass 0.185 | no |

### phantoms

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| analytic_vs_voxel_shape_ellipsoid_correlation | ≥ 0.95 | pass 0.997 | pass 0.997 | not run | yes |
| analytic_vs_voxel_shape_ellipsoid_centre_gap | ≤ 1 | pass 0.102 | pass 0.0568 | not run | yes |
| analytic_vs_voxel_shape_box_correlation | ≥ 0.95 | pass 0.997 | pass 0.996 | not run | yes |
| analytic_vs_voxel_shape_box_centre_gap | ≤ 1 | pass 0.0754 | pass 0.137 | not run | yes |
| analytic_vs_voxel_shape_horizontal_cylinder_correlation | ≥ 0.95 | pass 0.997 | pass 0.997 | not run | yes |
| analytic_vs_voxel_shape_horizontal_cylinder_centre_gap | ≤ 1 | pass 0.0769 | pass 0.0715 | not run | yes |
| analytic_vs_voxel_shape_vertical_cylinder_correlation | ≥ 0.95 | pass 0.997 | pass 0.997 | not run | yes |
| analytic_vs_voxel_shape_vertical_cylinder_centre_gap | ≤ 1 | pass 0.0596 | pass 0.0948 | not run | yes |
| inserts_mode_background_1_correlation | ≥ 0.95 | pass 0.986 | pass 0.985 | not run | yes |
| inserts_mode_background_1_centre_gap | ≤ 1 | pass 0.0891 | pass 0.552 | not run | yes |
| inserts_mode_background_1_total_ratio | 1 | pass 0.99 | pass 0.983 | not run | yes |
| inserts_mode_background_3_correlation | ≥ 0.95 | pass 0.985 | pass 0.985 | not run | yes |
| inserts_mode_background_3_centre_gap | ≤ 1 | pass 0.119 | pass 0.592 | not run | yes |
| inserts_mode_background_3_total_ratio | 1 | pass 0.988 | pass 0.984 | not run | yes |
| inserts_mode_hot_correlation | ≥ 0.95 | pass 0.998 | pass 0.997 | not run | yes |
| inserts_mode_hot_centre_gap | ≤ 1 | pass 0.109 | pass 0.197 | not run | yes |
| inserts_mode_hot_total_ratio | 1 | pass 0.992 | pass 1 | not run | yes |
| inserts_mode_cold_correlation | ≥ 0.95 | pass 0.985 | pass 0.984 | not run | yes |
| inserts_mode_cold_centre_gap | ≤ 1 | pass 0.112 | pass 0.583 | not run | yes |
| inserts_mode_cold_total_ratio | 1 | pass 0.988 | pass 0.981 | not run | yes |
| cardiac_defect_lowers_counts | ≥ 0.01 | pass 0.0865 | pass 0.0828 | not run | yes |
| library_phantom_ZUBAL_TORSO | 1 | pass 1 | pass 1 | not run | no |
| library_phantom_ZUBAL_BRAIN | 1 | pass 1 | pass 1 | not run | no |
| library_phantom_ZUBAL_WHOLE_BODY | 1 | pass 1 | pass 1 | not run | no |
| library_phantom_NEMA_IQ | 1 | pass 1 | pass 1 | not run | no |
| nema_hot_sphere_activity_17mm | 1 | pass 0.998 | — | pass 0.998 | no |
| nema_hot_sphere_activity_22mm | 1 | pass 0.998 | — | pass 0.998 | no |
| nema_hot_sphere_activity_28mm | 1 | pass 1 | — | pass 1 | no |
| nema_hot_sphere_activity_37mm | 1 | pass 1 | — | pass 1 | no |
| nema_background_density_is_water | 1 | pass 1 | — | pass 1 | no |

### recon

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| route2_scatter_matches_route1_sum | 1 | pass 0.995 | — | pass 1.02 | yes |
| route2_scatter_matches_route1_shape | ≥ 0.9 | pass 0.95 | — | pass 0.952 | yes |
| osem_refreshes_at_update_boundaries | 1 | pass 1 | — | pass 1 | no |
| damping_combines_exactly | 0 | pass 1.59e-08 | — | pass 1.26e-08 | no |
| scatter_correction_removes_the_scatter_fraction | 0.4312 | pass 0.403 | — | pass 0.402 | yes |
| trimmed_scale_bias_low_histories | 1 | FAIL 0.548 | — | FAIL 0.512 | yes |
| normalisers_agree_at_high_histories | 1 | pass 0.959 | — | pass 0.965 | yes |
| residual_identity | 0 | pass 1.87e-08 | — | pass 4.39e-08 | no |

### runtime_switches

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| ca_raw_files_0 | 1 | pass 1 | pass 1 | not run | no |
| ca_raw_files_1 | 1 | pass 1 | pass 1 | not run | no |
| ca_raw_files_2 | 1 | pass 1 | pass 1 | not run | no |
| ca1_derived_primary | 0 | pass 0 | not run | not run | no |
| fw_two_windows | 0.05–1 | pass 0.678 | pass 0.6 | not run | yes |
| fw_sixteen_character_name | 2 | pass 2 | pass 2 | not run | no |
| x1_default_is_identity | 0 | pass 0 | — | pass 0 | no |
| x1_copper_transmission | 0.7–0.8 | pass 0.758 | — | pass 0.762 | no |
| x2_default_is_identity | 0 | pass 0 | — | pass 0 | no |
| x3_default_is_identity | 0 | pass 0 | — | pass 0 | no |
| x4_default_is_identity | 0 | pass 0 | — | pass 0 | no |
| x6_default_is_identity | 0 | pass 0 | — | pass 0 | no |
| v9_linux_scattwin_defect | 1 | not run | not run | FAIL 0 | no |
| scattwin_default_ca | ≥ 1 | pass 1 | pass 1 | not run | no |

### scale_to_reference

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| scale_stable_across_seeds | 1 | pass 0.991 | — | pass 1.01 | yes |
| scale_per_unit_activity_image_independent | 1 | pass 1.07 | — | FAIL 1.1 | yes |
| trimmed_ignores_outlier_bins | ≤ 0.02 | pass 0.015 | — | pass 0.0152 | no |
| sum_changes_with_outlier_bins | ≥ 0.1 | pass 0.355 | — | pass 0.334 | no |

### versions

| case | expected | v8-linux | v9-darwin | v9-linux | smoke |
|---|---|---|---|---|---|
| point_pair_symmetric_about_centre_row | ≤ 0.25 | pass 0.093 | — | pass 0.0541 | no |
| point_pair_symmetric_about_centre_bin | ≤ 0.25 | pass 0.0781 | — | pass 0.0981 | no |
| v9_rejects_unvalidated_orientation | 1 | not run | — | pass 1 | no |
| timeout_enforced | ≤ 10 | pass 1.01 | — | pass 1.02 | no |
<!-- validation:end -->
