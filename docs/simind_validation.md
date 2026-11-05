# SIMIND validation

SIMIND v8 is the supported engine for this release. This page lists what the
connector checks against v8, the biases found in that engine, and the v8 numbers
recorded by the validation suite (see the results at the end).

> **SIMIND v9 is experimental and unvalidated in this release; support is
> planned for 2.1.0.** The v9.0 Linux build is known to fail in SCATTWIN, either
> aborting or returning silently corrupted projections. Do not use v9 for
> release validation or production work. The release tables below contain v8
> evidence only.

- **Input and output checks:** engine output prefixes are capped at 47 characters; up to 99 numbered windows and exact `tot`/`sca`/`pri`/PENETRATE component names are loaded from matching headers; byte offsets and full-pixel payload sizes are checked; measured geometry and source/mu-map grids are checked; DICOM diagnostics, container library isolation, and validation scripts have focused coverage.
- **Window files:** generated `.win` prefixes are capped at 15 characters, and a window file cannot silently follow a changed output prefix. This cap does not apply to runtime `FW`: a v8 raw-engine check read two requested windows from a 16-character `FW` filename. The connector preserves the caller's `/CA` selection and derives the missing order-0 scatter or primary image only when compatible output headers exist. Executor timeouts are exposed through core, native, and PyTomography runs. Window selection is resolved once per `run()`, before file staging, configuration saving, cleanup or execution, and the one selected path is shared by the executor, cleanup protection and `_derive_primary_headers`:
  - **Execution-time precedence:** explicit `FW` → `output_dir/{FW}.win` (an explicit selection overrides connector-written windows); else a current connector-written `{prefix}.win` from `set_energy_windows(...)` (matching this run's output prefix and directory) → that file is used and `FW=prefix` is injected into the run-local switch holder only, so persistent and caller-supplied switch objects are never mutated; else a SCATTWIN run with an existing `scattwin.win` → an actionable `ValueError`; else a clean SCATTWIN default execution with no connector-derived window component. PENETRATE and DUMMY runs never infer a window file from an existing `scattwin.win`; such a file is left untouched and unselected.
  - **Selected-file validation and failure ordering:** a selected window file, explicit or generated, must be a regular file before execution — a missing or non-regular selection raises `FileNotFoundError` rather than falling back to another file. Window resolution runs after output-prefix, scoring-routine and runtime-operator (switch and orbit) validation and before orbit staging, configuration saving, stale-output cleanup or execution, so an invalid setup fails before core-run file staging, configuration saving, cleanup or execution; an adaptor run may already have written its phantom inputs while configuring the phantom before `run()` is called.
  - **Migration (breaking):** a SCATTWIN run that relied on an implicit existing `scattwin.win` now raises. Call `set_energy_windows(...)` to select the connector-written window, or select the default-named file explicitly with `FW="scattwin"` (`connector.add_runtime_switch("FW", "scattwin")` or `connector.run(RuntimeOperator(switches={"FW": "scattwin"}))`). Default-mode engine runs can leave `scattwin.win` behind, so a later run with neither selection needs the same migration.
  - **Adaptor preflight:** `_validate_runtime_operator`, the preflight the native and PyTomography adaptors run before writing inputs, validates switches, generated-window prefix/path consistency and the orbit, but does not run the stale-default guard or the selected-file existence check; both run only inside core `run()`, before core-run staging, configuration saving, stale-output cleanup or execution. A preflight-passing SCATTWIN run can therefore still raise the `ValueError` (or `FileNotFoundError`) inside `run()`, although an adaptor run may already have written its phantom inputs while configuring the phantom before `run()` was called.
  - **CA/collimator validation: unchanged and routine-independent.** `CA` is validated as 0, 1 or 2 whatever the scoring routine, and `CA` remains caller-controlled: the connector validates it when supplied and injects no `/CA`. Validation deliberately stays before the pre-write side effects: moving it later would weaken that protection. No routine-aware validation change was made.
  - **Analytic `.inp`: deliberately retained.** The connector-written `{prefix}.inp` (only the analytic `MultipleInserts` phantom writes it) stays on disk after phantom reset clears `IF`. `.inp` is not an output suffix, nothing downstream reads it, and retention supports inspecting or re-running a configured analytic phantom, so it is not part of cleanup.
  - **Smoke evidence (selection only, not quantitative validation):** the `runtime_switches` table records which raw files each SCATTWIN `/CA` value writes on v8 and the window-file selection cases. These checks exercise selection and readability, not accuracy.
- **Interfile parsing:** rejects non-finite or fractional byte offsets and payloads ending in a partial pixel, and complete extra or short pixels.
- **Experimental v9 output handling:** the experimental v9 code path converts PENETRATE `.hNN` headers alongside their matching `.bNN` payloads. SIMIND v8's shared `.h00` template remains supported. Unknown component 20 is exposed as `b20` without assigning it an inferred physical label. These are unit and header behaviours, not v9 validation evidence.
- **Version selection and centring (experimental v9):** `simind_version=8|9` is available on core/native/PyTomography constructors; `SIMIND_VERSION` supplies the default when omitted, and v8 is the supported fallback. The v9 voxel-map origin compensation is implementation behaviour, not evidence of validated physics: the connector applies offsets only to a copied v9 voxel-run `.smc` (source Indexes 17/18 use effective `PX`; density Indexes 36/37 use Index 31) and preserves the input arrays. It rejects non-default `DI`, density Index 32, or `OR` orientations because those mappings are not validated. The v9 path is not exercised by release validation.
- **Measured-geometry smoke:** the four 16³/eight-view geometry cases (clockwise, counter-clockwise, non-circular, anisotropic) are recorded in the `measured_geometry` table below. Profile: smoke (16³ voxels, 8 views, 10 photon histories per voxel, `quantization_scale` 0.02×default). Each case must reach a correlation above 0.5 with its reference and a centre gap of at most 1 pixel; the full-profile gate (64³, 60 views, 500 histories per voxel, correlation above 0.9) has not been run.
- **MPI normalization:** serial, two-rank replicate and two-rank split totals are compared per build in the `mpi` table below. These are per-build normalization checks, not a cross-version accuracy comparison.
- **Smoke evidence:** which raw files each SCATTWIN `/CA` value writes on v8 is recorded in the `runtime_switches` table below (`ca_raw_files_*`). The v8 manual lists `air`, `tot` and `pri` for `/CA:2`, but the engine also writes `sca`.
- **Reviewed math:** the attenuation tables use MeV and cm²/g. The loader divides input keV by 1000 for interpolation, then multiplies the mass attenuation coefficient by density to obtain cm⁻¹. At 140 keV the checked values are water 0.15454 cm⁻¹ and bone 0.287342 cm⁻¹ at densities 1.0 and 1.85 g/cm³. The bilinear `hu_to_attenuation` maps HU values −1000/0/+1000 to air/water/bone; `mu_map_type="hu"` in `configure_voxel_phantom` uses the Schneider2000 density conversion instead (about 1.02 g/cm³ at 0 HU and 1.61 g/cm³ at 1000 HU). Index 25 carries MBq×seconds per projection; source-map values control histories and remain separate from absolute activity. Activity and acquisition-time doubling are checked in the `activity` table below. An exploratory v8 water/air PENETRATE ratio was 0.6081 versus center-path Beer–Lambert 0.6188 for a 16³, 4 mm point source. Monte Carlo uncertainty was not quantified for that check; it does not establish quantitative calibration.
- **Reviewed geometry/reconstruction:** source and attenuation images must share shape, finite positive voxel sizes, origin, and a native direction matrix where the backend exposes one. Measured-geometry SIRF/reconstruction regressions pass. In the SIRF 3.10.1/STIR 6.4 cache regression, disabling all-views caching reduced first-update residual discrepancy from 0.0222821 to 1.55e-6.
- **Complete workflow smokes:** the same 96³ NEMA phantom at 4 mm, 568.1 MBq, four views, `NN=1`, and `quantization_scale=0.01×DEFAULT` completed with both STIR and SIRF on v8 Linux. Both backends produced finite 96³ reconstructions; the projection total was approximately 66,190. A reduced v8 SIRF scatter-correction run completed one refresh with scale 2019 and scatter sum 2.321×10⁵; its contrast error was 7.8% both with and without correction. These smoke runs do not establish quantitative accuracy or an improvement in reconstruction quality.
- **Other backend evidence:** SIRF 3.10.1 uses STIR 6.4.0. PyTomography 3.4.0 with torch 2.14.1+cpu completed reduced Example 07C on v8 Linux: four views, 32³ images, `NN=1`, and `quantization_scale=0.01×DEFAULT`, producing a finite nonzero reconstruction. The selected backend tests passed (SIRF geometry/reconstruction 66; STIR 3; PyTomography 31). The supported real-engine regression run passed 15 tests on v8 Linux.
- **Core and packaging evidence:** 1,148 local core tests passed; the rebuilt core container passed 1,146 tests plus three import-isolation checks. Ruff, strict offline Sphinx, wheel/sdist resource checks, and `uv lock --check --offline` passed. Build tools are declared in the dev extra, and Compose forwards explicit version selection to all services.
- **Remaining quantitative gates:** calibrated absolute detector sensitivity, sufficient-history attenuation and reconstruction comparisons with uncertainty estimates, and measured-data calibration remain outside the evidence above. Full release validation must establish those references and tolerances before making quantitative claims.

## Findings

- **`simind_mpi` rank seeding:** `simind_mpi` seeds rank r with `RR=r+1`, and a replicate run with a fixed `RR` is refused because every process would then simulate the same histories. The v8 MPI normalization checks are in the table below.
- **STIR SPECTUB mirrors x relative to the mu map** (UCL/STIR#452). With the flip that the examples apply, SIRF and PyTomography agree: the recovered hot-to-background ratios differ by 0.1 % on v8 (2.194 against 2.197). Without the flip the ratio between the two was 1.08.
- **STIR SPECTUB `set_up`:** calling it twice on the same acquisition model gives zero back-projections. This is known STIR behaviour; set up the model once.
- **Per-unit-activity scale:** the scale that matches SPECTUB to SIMIND differs between source distributions by about 7 % on v8 (`scale_per_unit_activity_image_independent`). This is a known mismatch between the SIMIND and SPECTUB models; scale to a reference that has the same source distribution as the object being reconstructed.
- **Analytic against voxel inserts:** the analytic Jaszczak-like phantom gives 1-2 % fewer counts than its voxelised equivalent in the background and cold modes on v8 (total ratio 0.981-0.990; hot mode agrees within 1 %). The cause is not established.
- **`trimmed` scaling is biased low at low histories:** the scale is 0.548 of the true value on the recon OSEM iterate at a tenth of the default quantization on v8, and 0.959 at 64 times those histories (`normalisers_agree_at_high_histories`). The examples use `sum` with smoothing. `trimmed_scale_bias_low_histories` records this documented bias; it is not treated as an unexpected failure.

## Validation results

> Two v8-linux rows in the `recon` table below —
> `normalisers_agree_at_high_histories` and `residual_identity` — are retained
> historical v8 evidence from the 2026-10-18 run, not results of the current
> 2026-10-26 rerun. See "Provenance of the retained v8 rows" after the tables.

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

## Provenance of the retained v8 rows

The `recon` table above is not from a single run. Six of its eight v8-linux
rows are fresh results of the current rerun; the two rows
`normalisers_agree_at_high_histories` (pass 0.959) and `residual_identity`
(pass 1.87e-08) are **historical v8 evidence retained from the previous run,
not current rerun results**:

- **Retained records:** they come from `output/validation/v8-linux.jsonl`
  records 45–46 (an ignored run artifact, not committed), timestamped
  2026-10-18T13:44:54Z and 2026-10-18T13:45:13Z. They were written by
  `bash scripts/run_validation.sh --engine v8` (Docker images, default
  settings) on the validation tree of 2026-10-18 — before the 2.0.1 merge
  (`ef07477`) and before v8 became the supported release engine (`d9003af`).
- **Current rerun:** the rerun under the current tree (records timestamped
  2026-10-26T11:05–11:11Z) reached six of the eight `recon` cases and then
  timed out. All six reached cases passed, including
  `trimmed_scale_bias_low_histories` at 0.548 within its 0.3–0.8 band. The two
  cases above were never reached, so their 2026-10-18 records were retained
  rather than replaced.
- **What remains unverified:** `normalisers_agree_at_high_histories` and
  `residual_identity` have not been reproduced by the current rerun; the table
  publishes them only as retained 2026-10-18 v8 evidence and they should be
  re-run before being read as current results.

