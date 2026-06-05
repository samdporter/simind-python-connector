#!/usr/bin/env python3
"""
Iterative scatter estimation with OSEM + SIMIND (PENETRATE).

Workflow:
1) Initial OSEM reconstruction (optionally with an initial additive estimate)
2) For each outer iteration:
   - Run SIMIND PENETRATE on current image -> get b01 and b02
   - Forward project current image with SIRF
   - Compute global scale = sum(SIRF forward) / sum(b02)
   - Estimate scatter = scale * (b01 - b02)
   - Run OSEM with this scatter as additive term
"""

import argparse
import gc
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from sirf.STIR import (
    AcquisitionData,
    AcquisitionModelUsingMatrix,
    ImageData,
    MessageRedirector,
    RelativeDifferencePrior,
    SeparableGaussianImageFilter,
    SPECTUBMatrix,
)

from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf_simind_connection.core import ScoringRoutine
from sirf_simind_connection.utils import get_array


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _cfg_get(cfg, keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _str_to_optional(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None if value is False else str(value)
    text = str(value)
    return None if text.lower() == "false" else text


def load_spect_inputs(
    data_dir: Path,
    mu_map_filename: Optional[str],
    measured_name: str,
    initial_additive_path: Optional[Path] = None,
):
    measured = AcquisitionData(str(data_dir / measured_name))
    additive = None
    if initial_additive_path is not None:
        additive = AcquisitionData(str(initial_additive_path))

    try:
        initial = ImageData(str(data_dir / "initial_image.hv")).maximum(0)
    except Exception:
        initial = ImageData(str(data_dir / "template_image.hv"))
        initial.fill(1.0)

    if mu_map_filename is not None:
        attenuation = ImageData(str(data_dir / mu_map_filename))
        attn_arr = get_array(attenuation)
        attn_arr = np.flip(attn_arr, axis=-1)
        attenuation.fill(attn_arr)
    else:
        attenuation = initial.get_uniform_copy(0.0)

    return measured, initial, attenuation, additive


def make_spect_model(
    attenuation,
    additive=None,
    resolution_model=(1.31, 0.027, False),
    gauss_fwhm=(6.9, 6.9, 6.9),
    keep_cache=True,
):
    mat = SPECTUBMatrix()
    mat.set_attenuation_image(attenuation)
    mat.set_keep_all_views_in_cache(keep_cache)
    mat.set_resolution_model(
        float(resolution_model[0]),
        float(resolution_model[1]),
        bool(resolution_model[2]),
    )

    model = AcquisitionModelUsingMatrix(mat)
    gauss = SeparableGaussianImageFilter()
    gauss.set_fwhms(tuple(float(x) for x in gauss_fwhm))
    model.set_image_data_processor(gauss)

    if additive is not None:
        model.set_additive_term(additive)
    return model


def make_body_mask_from_attenuation(
    attenuation,
    threshold=0.01,
    opening_radius=0,
    keep_largest_component=False,
):
    """Create image-domain body mask from attenuation map.

    Optional morphology can suppress thin bed/table structures.
    """
    mask = attenuation.get_uniform_copy(0.0)
    mask_arr = get_array(attenuation) > float(threshold)

    if opening_radius > 0 or keep_largest_component:
        try:
            import scipy.ndimage as ndi
        except ImportError as exc:
            raise ImportError(
                "scipy is required for bed-removal morphology in attenuation masks."
            ) from exc

        if opening_radius > 0:
            structure = ndi.iterate_structure(
                ndi.generate_binary_structure(mask_arr.ndim, 1),
                int(opening_radius),
            )
            mask_arr = ndi.binary_opening(mask_arr, structure=structure)
            mask_arr = ndi.binary_closing(mask_arr, structure=structure)

        if keep_largest_component:
            labels, nlabels = ndi.label(mask_arr)
            if nlabels > 0:
                sizes = ndi.sum(mask_arr, labels, index=np.arange(1, nlabels + 1))
                keep_label = int(np.argmax(sizes)) + 1
                mask_arr = labels == keep_label

    mask.fill(mask_arr.astype(np.float32))
    return mask


def median_smooth_scatter(scatter, kernel_size, preserve_sum=True):
    """Apply optional median smoothing to scatter in detector-pixel dimensions."""
    if int(kernel_size) <= 1:
        return scatter.clone()

    if int(kernel_size) % 2 == 0:
        raise ValueError("scatter median kernel size must be odd (e.g. 3, 5)")

    try:
        from scipy.ndimage import median_filter
    except ImportError as exc:
        raise ImportError(
            "scipy is required for median scatter smoothing. "
            "Install scipy or disable smoothing."
        ) from exc

    arr = get_array(scatter)

    # Do not smooth across segment/view indices. Only smooth detector-pixel axes.
    if arr.ndim == 4:
        # Expected [segment, axial, view, tangential]
        filt_size = (1, int(kernel_size), 1, int(kernel_size))
    elif arr.ndim == 3:
        # Expected [axial, view, tangential]
        filt_size = (int(kernel_size), 1, int(kernel_size))
    elif arr.ndim == 2:
        filt_size = (int(kernel_size), int(kernel_size))
    else:
        filt_size = tuple(int(kernel_size) for _ in range(arr.ndim))

    smoothed_arr = median_filter(arr, size=filt_size, mode="nearest")
    smoothed = scatter.get_uniform_copy(0.0)
    smoothed.fill(smoothed_arr.astype(np.float32))

    if preserve_sum:
        old_sum = float(scatter.sum())
        new_sum = float(smoothed.sum())
        if new_sum > 0:
            smoothed *= old_sum / new_sum

    return smoothed


def gaussian_smooth_scatter(scatter, sigma, preserve_sum=True):
    """Apply optional Gaussian smoothing to scatter in detector-pixel dimensions."""
    if float(sigma) <= 0:
        return scatter.clone()

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise ImportError(
            "scipy is required for Gaussian scatter smoothing. "
            "Install scipy or disable smoothing."
        ) from exc

    arr = get_array(scatter)

    # Do not smooth across segment/view indices. Only smooth detector-pixel axes.
    if arr.ndim == 4:
        # Expected [segment, axial, view, tangential]
        filt_sigma = (0.0, float(sigma), 0.0, float(sigma))
    elif arr.ndim == 3:
        # Expected [axial, view, tangential]
        filt_sigma = (float(sigma), 0.0, float(sigma))
    elif arr.ndim == 2:
        filt_sigma = (float(sigma), float(sigma))
    else:
        filt_sigma = tuple(float(sigma) for _ in range(arr.ndim))

    smoothed_arr = gaussian_filter(arr, sigma=filt_sigma, mode="nearest")
    smoothed = scatter.get_uniform_copy(0.0)
    smoothed.fill(smoothed_arr.astype(np.float32))

    if preserve_sum:
        old_sum = float(scatter.sum())
        new_sum = float(smoothed.sum())
        if new_sum > 0:
            smoothed *= old_sum / new_sum

    return smoothed


def compute_scale(
    linear_forward,
    b02,
    method="sum",
    trim_frac=0.0,
    min_linear=0.0,
    min_b02=0.0,
    fallback_if_empty="sum",
):
    method = str(method or "sum").lower()
    if method in ("sum", "global_sum", "mean_sum"):
        b02_counts = float(b02.sum())
        if b02_counts <= 0:
            raise ValueError("b02 counts are non-positive")
        return float(linear_forward.sum()) / b02_counts, b02_counts, None

    if method not in ("trimmed_mean", "trimmed-mean", "trim"):
        raise ValueError(f"Unknown scale method: {method}")

    if trim_frac < 0 or trim_frac >= 0.5:
        raise ValueError("trim_frac must be in [0, 0.5)")

    lin_arr = get_array(linear_forward)
    b02_arr = get_array(b02)
    mask = (lin_arr > float(min_linear)) & (b02_arr > float(min_b02))
    ratios = lin_arr[mask] / b02_arr[mask]
    lin_masked = lin_arr[mask]
    b02_masked = b02_arr[mask]
    num_samples = int(ratios.size)
    if num_samples == 0:
        fallback_method = str(fallback_if_empty or "").lower()
        if fallback_method in ("sum", "global_sum", "mean_sum"):
            b02_counts = float(b02.sum())
            if b02_counts <= 0:
                raise ValueError("b02 counts are non-positive")
            logging.warning(
                "Trimmed scale mask empty (min_linear=%g, min_b02=%g); "
                "falling back to sum scale using CT-masked projections.",
                min_linear,
                min_b02,
            )
            return float(linear_forward.sum()) / b02_counts, b02_counts, 0
        raise ValueError("No samples available for trimmed scale (mask empty)")

    if trim_frac > 0:
        k = int(np.floor(trim_frac * num_samples))
        if 2 * k >= num_samples:
            raise ValueError(
                "trim_frac too large for available samples in trimmed scale"
            )
        if k > 0:
            keep_idx = np.argsort(ratios)[k:-k]
            lin_masked = lin_masked[keep_idx]
            b02_masked = b02_masked[keep_idx]

    b02_retained = float(np.sum(b02_masked))
    if b02_retained <= 0:
        raise ValueError("Retained b02 counts are non-positive after trimming")

    # Robust scalar calibration:
    # 1. threshold bins by minimum counts,
    # 2. trim ratio outliers,
    # 3. compute the calibration from retained sums rather than the mean ratio.
    # This keeps the robustness of ratio trimming while restoring count-weighting.
    scale = float(np.sum(lin_masked)) / b02_retained
    return scale, float(b02.sum()), num_samples


def build_subset_indices(num_views, num_subsets, mode="staggered"):
    if int(num_subsets) <= 0:
        raise ValueError("num_subsets must be >= 1")
    num_views = int(num_views)
    mode = str(mode or "staggered").lower()

    if mode == "staggered":
        return [list(range(i, num_views, int(num_subsets))) for i in range(num_subsets)]

    if mode == "sequential":
        base = num_views // int(num_subsets)
        remainder = num_views % int(num_subsets)
        indices = []
        start = 0
        for i in range(int(num_subsets)):
            size = base + (1 if i < remainder else 0)
            indices.append(list(range(start, start + size)))
            start += size
        return indices

    raise ValueError(f"Unknown subset mode: {mode}")


def _view_axis_for_acq_array(arr):
    if arr.ndim < 2:
        raise ValueError("Acquisition array has too few dimensions")
    return -2


def precompute_subset_sensitivities(
    model,
    measured,
    initial_image,
    subset_indices,
    num_subsets,
    measured_arr=None,
):
    model.num_subsets = int(num_subsets)

    if measured_arr is None:
        measured_arr = get_array(measured)
    view_axis = _view_axis_for_acq_array(measured_arr)
    full_views = int(measured_arr.shape[view_axis])

    model.subset_num = 0
    test_proj = model.forward(initial_image)
    test_views = int(get_array(test_proj).shape[view_axis])
    del test_proj

    use_subset_data = test_views != full_views
    ones_full = None
    if not use_subset_data:
        ones_full = measured.get_uniform_copy(1.0)

    sensitivities = []
    for subset_num, subset_views in enumerate(subset_indices):
        model.subset_num = subset_num
        if use_subset_data:
            ones = measured.get_subset(subset_views)
            ones.fill(1.0)
        else:
            ones = ones_full
        sens = model.backward(ones)
        sensitivities.append(sens)
        if use_subset_data:
            del ones
    return sensitivities


def apply_relative_difference_prior_update(
    current_image,
    prior,
    subset_sensitivity,
    num_subsets,
    eps=1e-8,
):
    grad = prior.gradient(current_image)
    cur_arr = get_array(current_image).astype(np.float32, copy=False)
    grad_arr = get_array(grad).astype(np.float32, copy=False)
    sens_arr = get_array(subset_sensitivity).astype(np.float32, copy=False)
    denom = np.maximum(sens_arr, float(eps))
    cur_arr -= (cur_arr / denom) * (grad_arr / float(num_subsets))
    current_image.fill(np.maximum(cur_arr, 0.0))
    del grad
    return current_image


def run_osem_manual(
    measured,
    model,
    initial_image,
    num_subsets,
    num_epochs,
    subset_indices,
    subset_sensitivities,
    prior=None,
    prior_update_point="post",
    measured_arr=None,
    eps=1e-8,
):
    current_image = initial_image.clone()

    if measured_arr is None:
        measured_arr = get_array(measured)
    measured_arr = measured_arr.astype(np.float32, copy=False)

    model.num_subsets = int(num_subsets)

    view_axis = _view_axis_for_acq_array(measured_arr)
    ratio = measured.get_uniform_copy(1.0)
    ratio_arr = np.ones_like(measured_arr, dtype=np.float32)

    if prior is not None:
        if prior_update_point not in ("pre", "post"):
            raise ValueError("prior_update_point must be 'pre' or 'post'")
        prior.set_up(current_image)

    for _ in range(int(num_epochs)):
        for subset_num in range(int(num_subsets)):
            model.subset_num = subset_num
            subset_views = subset_indices[subset_num]
            if len(subset_views) == 0:
                continue

            if prior is not None and prior_update_point == "pre":
                apply_relative_difference_prior_update(
                    current_image,
                    prior,
                    subset_sensitivities[subset_num],
                    num_subsets,
                    eps=eps,
                )

            estimated_subset = model.forward(current_image)
            est_arr = get_array(estimated_subset).astype(np.float32, copy=False)

            ratio_arr.fill(1.0)
            index = [slice(None)] * measured_arr.ndim
            index[view_axis] = subset_views
            index = tuple(index)
            est_view_dim = est_arr.shape[view_axis]
            if est_view_dim == len(subset_views):
                est_subset_arr = est_arr
            elif est_view_dim == measured_arr.shape[view_axis]:
                est_subset_arr = est_arr[index]
            else:
                raise ValueError(
                    "Estimated subset view dimension mismatch: "
                    f"subset={len(subset_views)}, est_view_dim={est_view_dim}"
                )
            if est_view_dim == len(subset_views):
                ratio_subset = estimated_subset.get_uniform_copy(1.0)
                ratio_subset.fill(measured_arr[index] / (est_subset_arr + float(eps)))
                correction = model.backward(ratio_subset)
                del ratio_subset
            else:
                ratio_arr[index] = measured_arr[index] / (est_subset_arr + float(eps))
                ratio.fill(ratio_arr)
                correction = model.backward(ratio)

            cur_arr = get_array(current_image).astype(np.float32, copy=False)
            corr_arr = get_array(correction).astype(np.float32, copy=False)
            sens_arr = get_array(subset_sensitivities[subset_num]).astype(
                np.float32, copy=False
            )
            denom = np.maximum(sens_arr, float(eps))
            cur_arr *= corr_arr / denom
            current_image.fill(np.maximum(cur_arr, 0.0))

            if prior is not None and prior_update_point == "post":
                apply_relative_difference_prior_update(
                    current_image,
                    prior,
                    subset_sensitivities[subset_num],
                    num_subsets,
                    eps=eps,
                )
            del correction
            del estimated_subset

    return current_image


def damp_additive_update(previous, proposed, alpha):
    alpha = float(alpha)
    if alpha <= 0:
        return previous.clone()
    if alpha >= 1:
        return proposed.clone()
    out = previous.clone()
    prev_arr = get_array(out).astype(np.float32, copy=False)
    new_arr = get_array(proposed).astype(np.float32, copy=False)
    prev_arr *= 1.0 - alpha
    prev_arr += alpha * new_arr
    out.fill(prev_arr)
    return out


def clamp_nonnegative(acq_data):
    arr = get_array(acq_data).astype(np.float32, copy=False)
    np.maximum(arr, 0.0, out=arr)
    acq_data.fill(arr)
    return acq_data


def maybe_smooth(image, enabled):
    if not enabled:
        return image
    out = image.clone()
    gauss = SeparableGaussianImageFilter()
    gauss.set_fwhms((5.0, 5.0, 5.0))
    gauss.apply(out)
    return out


def setup_penetrate_simulator(
    scanner_config,
    output_dir,
    output_prefix,
    measured,
    source_image,
    mu_map,
    sim_cfg,
    total_activity,
    mpi_procs,
):
    simulator = SimindSimulator(
        config_source=SimulationConfig(str(scanner_config)),
        output_dir=str(output_dir),
        output_prefix=output_prefix,
        photon_multiplier=int(sim_cfg["photon_multiplier"]),
        scoring_routine=ScoringRoutine.PENETRATE,
    )

    simulator.set_source(source_image)
    simulator.set_mu_map(mu_map)
    simulator.set_template_sinogram(measured)

    simulator.add_config_value("photon_energy", sim_cfg["photon_energy"])
    simulator.add_config_value("collimator_routine", sim_cfg["collimator_routine"])
    simulator.add_config_value("photon_direction", sim_cfg["photon_direction"])
    simulator.add_config_value("lower_window_threshold", sim_cfg["window_lower"])
    simulator.add_config_value("upper_window_threshold", sim_cfg["window_upper"])
    simulator.add_config_value(
        "source_activity", float(total_activity) * float(sim_cfg["time_per_projection"])
    )
    simulator.add_config_value(
        "step_size_photon_path_simulation", min(source_image.voxel_sizes()) / 5.0
    )
    simulator.add_config_value(
        "cutoff_energy_terminate_photon_history", float(sim_cfg["window_lower"]) * 0.75
    )

    simulator.add_runtime_switch("CC", sim_cfg["collimator"])
    simulator.add_runtime_switch("FI", sim_cfg["source_type"])
    if int(mpi_procs) > 1:
        simulator.add_runtime_switch("MP", int(mpi_procs))

    return simulator


def preflight_simind_runtime(mpi_procs):
    """Validate SIMIND runtime dependencies before starting iterative loops."""
    use_mpi = int(mpi_procs) > 1
    executable = "simind_mpi" if use_mpi else "simind"
    executable_path = shutil.which(executable)
    if executable_path is None:
        raise FileNotFoundError(
            f"Required executable '{executable}' not found in PATH."
        )

    if use_mpi:
        mpirun_path = shutil.which("mpirun")
        if mpirun_path is None:
            raise FileNotFoundError(
                "MPI run requested (--mpi-procs > 1) but 'mpirun' was not found in PATH."
            )
        logging.info("SIMIND preflight: mpirun=%s", mpirun_path)

    logging.info("SIMIND preflight: %s=%s", executable, executable_path)

    ldd_path = shutil.which("ldd")
    if ldd_path is None:
        logging.warning(
            "SIMIND preflight: 'ldd' not found; skipping shared-library dependency check."
        )
        return

    proc = subprocess.run(
        [ldd_path, executable_path],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines()
    missing = [line.strip() for line in combined if "=> not found" in line]
    if missing:
        unresolved = ", ".join(
            line.split("=>", maxsplit=1)[0].strip() for line in missing
        )
        raise RuntimeError(
            f"SIMIND preflight failed: unresolved shared libraries for {executable}: "
            f"{unresolved}. Fix LD_LIBRARY_PATH or runtime environment before running."
        )


def recon_filename(output_dir, num_epochs, num_subsets, index, smoothing):
    suffix = f"recon_osem_i{num_epochs}_s{num_subsets}"
    if smoothing:
        suffix += "_smoothed"
    return output_dir / f"{suffix}_{index}.hv"


def main():
    parser = argparse.ArgumentParser(
        description="Iterative OSEM + SIMIND PENETRATE scatter estimation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("project/config/local.yaml"),
        help="YAML config (project/config/local.yaml style).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Override data dir."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override output dir."
    )
    parser.add_argument(
        "--scanner-config",
        type=Path,
        default=Path("sirf_simind_connection/configs/AnyScan.yaml"),
        help="SIMIND scanner config (.yaml/.smc).",
    )
    parser.add_argument(
        "--simind-parent-dir",
        type=Path,
        default=None,
        help="Working directory used before SIMIND execution.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Outer iterations; default from config.pipeline.num_iterations.",
    )
    parser.add_argument("--num-subsets", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument(
        "--mpi-procs",
        type=int,
        default=1,
        help="If >1, run SIMIND with MP runtime switch.",
    )
    parser.add_argument(
        "--mu-map-filename",
        type=str,
        default=None,
        help="Override mu-map filename; use 'false' for no attenuation.",
    )
    parser.add_argument("--save-components", action="store_true")
    parser.add_argument(
        "--scatter-median-size",
        type=int,
        default=None,
        help="Optional odd kernel size for median smoothing of scatter estimates "
        "(0/1 disables, e.g. 3 enables).",
    )
    parser.add_argument(
        "--scatter-gaussian-sigma",
        type=float,
        default=None,
        help="Optional Gaussian sigma for scatter smoothing (<=0 disables, e.g. 1.0).",
    )
    parser.add_argument(
        "--additive-alpha",
        type=float,
        default=None,
        help="Damping factor for additive updates (0 -> keep old, 1 -> replace).",
    )
    parser.add_argument(
        "--residual-alpha",
        type=float,
        default=None,
        help=(
            "Damping factor for residual updates (0 -> keep old residual, "
            "1 -> replace with current residual)."
        ),
    )
    parser.add_argument(
        "--additive-mode",
        type=str,
        default=None,
        choices=[
            "initial_scatter",
            "update_scatter",
            "initial_scatter_plus_residual",
            "update_scatter_plus_residual",
        ],
        help=(
            "Additive update mode. initial_scatter: fixed initial additive only. "
            "update_scatter: update scatter each iteration. "
            "initial_scatter_plus_residual: fixed initial scatter + residual. "
            "update_scatter_plus_residual: update scatter + residual."
        ),
    )
    parser.add_argument(
        "--additive-clamp",
        action="store_true",
        help="Clamp additive term to non-negative values after updates.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=None,
        help="Attenuation threshold for body mask (default from config, typically 0.01).",
    )
    parser.add_argument(
        "--mask-opening-radius",
        type=int,
        default=None,
        help="Morphological opening radius for bed suppression (0 disables).",
    )
    parser.add_argument(
        "--mask-keep-largest-component",
        action="store_true",
        help="Keep only largest connected component in attenuation mask.",
    )
    parser.add_argument(
        "--scale-method",
        type=str,
        default=None,
        choices=["sum", "trimmed_mean"],
        help=(
            "Scale estimation method: sum (global sum ratio) or trimmed_mean "
            "(trim ratio outliers, then compute a retained ratio-of-sums)."
        ),
    )
    parser.add_argument(
        "--scale-trim-frac",
        type=float,
        default=None,
        help=(
            "Trim fraction for trimmed_mean scale (0 disables trimming; <0.5). "
            "Trimming is applied in ratio space before the retained ratio-of-sums "
            "calibration is computed."
        ),
    )
    parser.add_argument(
        "--scale-min-linear",
        type=float,
        default=None,
        help="Minimum linear-forward value for scale estimation (0 disables).",
    )
    parser.add_argument(
        "--scale-min-b02",
        type=float,
        default=None,
        help="Minimum b02 value for scale estimation (0 disables).",
    )
    parser.add_argument(
        "--save-scale-ratios",
        action="store_true",
        help="Save per-bin scale ratio map and mask (uses scale thresholds).",
    )
    parser.add_argument(
        "--initial-additive-path",
        type=Path,
        default=None,
        help="Optional initial additive estimate (e.g., deep-learned scatter .hs).",
    )
    parser.add_argument(
        "--gc-collect",
        action="store_true",
        help="Force Python garbage collection each iteration to help release memory.",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config.expanduser().resolve())

    data_dir = (
        args.data_dir or Path(_cfg_get(cfg, ["project", "data_dir"]))
    ).expanduser()
    output_dir = (
        args.output_dir or Path(_cfg_get(cfg, ["project", "output_dir"]))
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    simind_parent_dir = args.simind_parent_dir or Path(
        _cfg_get(cfg, ["project", "simind_parent_dir"], ".")
    )
    simind_parent_dir = simind_parent_dir.expanduser().resolve()
    scanner_config = args.scanner_config.expanduser().resolve()

    measured_name = _cfg_get(cfg, ["data", "measured_data_filename"], "peak.hs")
    initial_additive_cfg = _cfg_get(cfg, ["data", "initial_additive_filename"], None)
    initial_additive_path = args.initial_additive_path
    if initial_additive_path is None and initial_additive_cfg:
        initial_additive_path = data_dir / str(initial_additive_cfg)
    if initial_additive_path is not None:
        initial_additive_path = initial_additive_path.expanduser().resolve()
    mu_map_filename = (
        _str_to_optional(args.mu_map_filename)
        if args.mu_map_filename is not None
        else _str_to_optional(
            _cfg_get(cfg, ["data", "mu_map_filename"], "umap_zoomed.hv")
        )
    )

    num_iters = args.num_iterations or int(
        _cfg_get(cfg, ["pipeline", "num_iterations"], 1)
    )
    num_subsets = args.num_subsets or int(
        _cfg_get(cfg, ["osem", "initial_subsets"], 12)
    )
    num_epochs = args.num_epochs or int(_cfg_get(cfg, ["osem", "initial_epochs"], 5))
    smoothing = bool(_cfg_get(cfg, ["osem", "smoothing"], True))
    spect_resolution_model = _cfg_get(
        cfg,
        ["osem", "resolution_model"],
        [1.31, 0.027, False],
    )
    spect_gauss_fwhm = _cfg_get(
        cfg,
        ["osem", "gaussian_image_processor_fwhm"],
        [6.9, 6.9, 6.9],
    )
    scatter_median_size = (
        args.scatter_median_size
        if args.scatter_median_size is not None
        else int(_cfg_get(cfg, ["scatter_estimation", "median_filter_size"], 0))
    )
    scatter_gaussian_sigma = (
        args.scatter_gaussian_sigma
        if args.scatter_gaussian_sigma is not None
        else float(_cfg_get(cfg, ["scatter_estimation", "gaussian_sigma"], 0.0))
    )
    scale_method = (
        args.scale_method
        if args.scale_method is not None
        else _cfg_get(cfg, ["scatter_estimation", "scale_method"], "sum")
    )
    scale_trim_frac = (
        args.scale_trim_frac
        if args.scale_trim_frac is not None
        else float(_cfg_get(cfg, ["scatter_estimation", "scale_trim_frac"], 0.0))
    )
    scale_min_linear = (
        args.scale_min_linear
        if args.scale_min_linear is not None
        else float(_cfg_get(cfg, ["scatter_estimation", "scale_min_linear"], 0.0))
    )
    scale_min_b02 = (
        args.scale_min_b02
        if args.scale_min_b02 is not None
        else float(_cfg_get(cfg, ["scatter_estimation", "scale_min_b02"], 0.0))
    )
    save_scale_ratios = bool(
        _cfg_get(cfg, ["scatter_estimation", "save_scale_ratios"], False)
    )
    if args.save_scale_ratios:
        save_scale_ratios = True
    if scatter_median_size < 0:
        raise ValueError("scatter median kernel size must be >= 0")
    if scatter_median_size > 1 and scatter_median_size % 2 == 0:
        raise ValueError("scatter median kernel size must be odd (e.g. 3, 5)")
    if scatter_gaussian_sigma < 0:
        raise ValueError("scatter gaussian sigma must be >= 0")
    if scale_trim_frac < 0 or scale_trim_frac >= 0.5:
        raise ValueError("scale_trim_frac must be in [0, 0.5)")
    if scale_min_linear < 0 or scale_min_b02 < 0:
        raise ValueError("scale_min_linear and scale_min_b02 must be >= 0")
    mask_threshold = (
        args.mask_threshold
        if args.mask_threshold is not None
        else float(_cfg_get(cfg, ["scatter_estimation", "mask_threshold"], 0.01))
    )
    mask_opening_radius = (
        args.mask_opening_radius
        if args.mask_opening_radius is not None
        else int(_cfg_get(cfg, ["scatter_estimation", "mask_opening_radius"], 2))
    )
    mask_keep_largest = bool(
        args.mask_keep_largest_component
        or _cfg_get(cfg, ["scatter_estimation", "mask_keep_largest_component"], True)
    )
    total_activity = float(_cfg_get(cfg, ["project", "total_activity"], 1.0))
    additive_alpha = (
        args.additive_alpha
        if args.additive_alpha is not None
        else float(_cfg_get(cfg, ["scatter_estimation", "additive_alpha"], 0.2))
    )
    if additive_alpha < 0 or additive_alpha > 1:
        raise ValueError("additive_alpha must be in [0, 1]")
    residual_alpha = (
        args.residual_alpha
        if args.residual_alpha is not None
        else float(_cfg_get(cfg, ["scatter_estimation", "residual_alpha"], 0.2))
    )
    if residual_alpha < 0 or residual_alpha > 1:
        raise ValueError("residual_alpha must be in [0, 1]")
    additive_mode = (
        args.additive_mode
        if args.additive_mode is not None
        else _cfg_get(
            cfg, ["scatter_estimation", "additive_mode"], "update_scatter_plus_residual"
        )
    )
    additive_mode = str(additive_mode).lower().strip()
    if additive_mode not in (
        "initial_scatter",
        "update_scatter",
        "initial_scatter_plus_residual",
        "update_scatter_plus_residual",
    ):
        raise ValueError(
            "additive_mode must be one of: initial_scatter, update_scatter, "
            "initial_scatter_plus_residual, update_scatter_plus_residual"
        )

    update_scatter = additive_mode in (
        "update_scatter",
        "update_scatter_plus_residual",
    )
    use_residual = additive_mode in (
        "initial_scatter_plus_residual",
        "update_scatter_plus_residual",
    )
    use_simind = update_scatter or use_residual
    additive_clamp = (
        args.additive_clamp
        if args.additive_clamp
        else bool(_cfg_get(cfg, ["scatter_estimation", "additive_clamp"], True))
    )

    rdp_weight = float(
        _cfg_get(cfg, ["osem", "relative_difference_prior", "weight"], 0.0)
    )
    rdp_update_point = _cfg_get(
        cfg, ["osem", "relative_difference_prior", "update_point"], "post"
    )
    rdp_gamma = _cfg_get(cfg, ["osem", "relative_difference_prior", "gamma"], None)
    rdp_epsilon = _cfg_get(cfg, ["osem", "relative_difference_prior", "epsilon"], None)
    if rdp_update_point not in ("pre", "post"):
        raise ValueError(
            "relative_difference_prior.update_point must be 'pre' or 'post'"
        )

    sim_cfg = {
        "photon_multiplier": int(_cfg_get(cfg, ["simulation", "photon_multiplier"], 1)),
        "photon_energy": float(_cfg_get(cfg, ["simulation", "photon_energy"], 150.0)),
        "window_lower": float(_cfg_get(cfg, ["simulation", "window_lower"], 75.0)),
        "window_upper": float(_cfg_get(cfg, ["simulation", "window_upper"], 225.0)),
        "time_per_projection": float(
            _cfg_get(cfg, ["simulation", "time_per_projection"], 40.0)
        ),
        "source_type": _cfg_get(cfg, ["simulation", "source_type"], "y90_frey"),
        "collimator": _cfg_get(cfg, ["simulation", "collimator"], "ma-megp"),
        "collimator_routine": int(
            _cfg_get(cfg, ["simulation", "collimator_routine"], 1)
        ),
        "photon_direction": int(_cfg_get(cfg, ["simulation", "photon_direction"], 3)),
    }

    configured_routine = int(_cfg_get(cfg, ["simulation", "scoring_routine"], 4))
    if configured_routine != int(ScoringRoutine.PENETRATE.value):
        logging.warning(
            "Config scoring_routine=%s, but this script enforces PENETRATE (4).",
            configured_routine,
        )

    measured, initial_image, attenuation, additive = load_spect_inputs(
        data_dir,
        mu_map_filename,
        measured_name,
        initial_additive_path=initial_additive_path,
    )
    current_image = initial_image.clone()

    MessageRedirector()

    # Persist the initial additive estimate (iteration 0) for reproducibility.
    # If none provided, write a zero additive term to keep downstream naming consistent.
    iter0_additive = (
        additive.clone() if additive is not None else measured.get_uniform_copy(0.0)
    )
    iter0_scatter_path = output_dir / "mean_iter0_scatter.hs"
    iter0_additive.write(str(iter0_scatter_path))
    (output_dir / "mean_iter0_scatter_scaling.txt").write_text("1\n", encoding="utf-8")

    logging.info(
        "Starting iterative workflow: iterations=%d, subsets=%d, epochs=%d, mpi=%d",
        num_iters,
        num_subsets,
        num_epochs,
        args.mpi_procs,
    )
    logging.info("Using SIMIND parent dir: %s", simind_parent_dir)
    if scatter_median_size > 1:
        logging.info(
            "Scatter median smoothing enabled with kernel size %d",
            scatter_median_size,
        )
    if scatter_gaussian_sigma > 0:
        logging.info(
            "Scatter Gaussian smoothing enabled with sigma %.3f",
            scatter_gaussian_sigma,
        )
    logging.info(
        "Mask settings: threshold=%g, opening_radius=%d, keep_largest_component=%s",
        mask_threshold,
        mask_opening_radius,
        mask_keep_largest,
    )
    logging.info(
        "Scale settings: method=%s, trim_frac=%g, min_linear=%g, min_b02=%g",
        str(scale_method).lower(),
        scale_trim_frac,
        scale_min_linear,
        scale_min_b02,
    )
    logging.info("Additive damping alpha=%g", additive_alpha)
    logging.info("Residual damping alpha=%g", residual_alpha)
    logging.info(
        "Additive mode=%s (update_scatter=%s, residual=%s, clamp_nonnegative=%s)",
        additive_mode,
        update_scatter,
        use_residual,
        additive_clamp,
    )
    if use_simind:
        preflight_simind_runtime(args.mpi_procs)
    if args.gc_collect:
        logging.info("GC collect enabled: forcing gc.collect() each iteration.")
    if save_scale_ratios:
        logging.info("Saving per-bin scale ratios to output directory.")

    attenuation_body_mask = make_body_mask_from_attenuation(
        attenuation,
        threshold=mask_threshold,
        opening_radius=mask_opening_radius,
        keep_largest_component=mask_keep_largest,
    )

    original_cwd = Path.cwd()
    # Chdir to the job-specific output_dir so SIRF writes its tmp sinogram files
    # there rather than to simind_parent_dir, which is shared across all concurrent
    # cluster tasks and causes cross-job file corruption. SimindSimulator handles
    # its own chdir internally.
    os.chdir(output_dir)
    try:
        current_additive = iter0_additive.clone()
        current_residual = measured.get_uniform_copy(0.0)

        full_model = make_spect_model(
            attenuation,
            additive=current_additive,
            resolution_model=spect_resolution_model,
            gauss_fwhm=spect_gauss_fwhm,
            keep_cache=True,
        )
        full_model.set_up(measured, current_image)
        linear_model = full_model.get_linear_acquisition_model()

        measured_arr = get_array(measured)
        view_axis = _view_axis_for_acq_array(measured_arr)
        num_views = int(measured_arr.shape[view_axis])
        subset_indices = build_subset_indices(num_views, num_subsets, mode="staggered")
        subset_sensitivities = precompute_subset_sensitivities(
            full_model,
            measured,
            current_image,
            subset_indices,
            num_subsets,
            measured_arr=measured_arr,
        )

        prior = None
        if rdp_weight > 0:
            prior = RelativeDifferencePrior()
            prior.set_penalisation_factor(float(rdp_weight))
            if rdp_gamma is not None:
                prior.set_gamma(float(rdp_gamma))
            if rdp_epsilon is not None:
                prior.set_epsilon(float(rdp_epsilon))
            logging.info(
                "Using RelativeDifferencePrior: weight=%g, update_point=%s",
                rdp_weight,
                rdp_update_point,
            )

        # Initial OSEM (optionally with an initial additive estimate)
        full_model.set_additive_term(current_additive)
        recon0 = run_osem_manual(
            measured=measured,
            model=full_model,
            initial_image=current_image,
            num_subsets=num_subsets,
            num_epochs=num_epochs,
            subset_indices=subset_indices,
            subset_sensitivities=subset_sensitivities,
            prior=prior,
            prior_update_point=rdp_update_point,
            measured_arr=measured_arr,
        )
        recon0 = maybe_smooth(recon0, smoothing)
        recon0_path = recon_filename(output_dir, num_epochs, num_subsets, 0, smoothing)
        recon0.write(str(recon0_path))
        current_image = recon0.clone()
        logging.info("Initial OSEM saved: %s", recon0_path)
        mask_saved = False

        # Iterative loop
        for iteration in range(1, num_iters + 1):
            logging.info("Iteration %d/%d", iteration, num_iters)

            current_image = current_image.maximum(0)

            additive_for_model = current_additive.clone()
            if use_residual:
                additive_for_model = additive_for_model + current_residual
            if additive_clamp:
                # Signed residuals are folded into the additive term. STIR expects
                # a non-negative effective additive, so project back to positivity
                # before each forward model evaluation.
                additive_for_model = clamp_nonnegative(additive_for_model)
            full_model.set_additive_term(additive_for_model)

            b01 = None
            b02 = None
            b01_scaled = None
            b02_scaled = None
            residual = None
            scatter = None
            linear_forward = None
            full_forward = None
            scale = None
            scale_samples = None

            if use_simind:
                # SIMIND PENETRATE run on current reconstruction (masked in image domain)
                masked_image = current_image.clone()
                masked_image *= attenuation_body_mask
                simulator = setup_penetrate_simulator(
                    scanner_config=scanner_config,
                    output_dir=output_dir,
                    output_prefix=f"output_iter{iteration}",
                    measured=measured,
                    source_image=masked_image,
                    mu_map=attenuation,
                    sim_cfg=sim_cfg,
                    total_activity=total_activity,
                    mpi_procs=args.mpi_procs,
                )
                simulator.run_simulation()

                b01 = simulator.get_all_interactions()
                b02 = simulator.get_geometrically_collimated_primary()
                del simulator

                linear_forward = linear_model.forward(
                    masked_image, subset_num=0, num_subsets=1
                )
                del masked_image

                if args.save_components and not mask_saved:
                    attenuation_body_mask.write(
                        str(output_dir / "attenuation_body_mask.hv")
                    )
                    mask_saved = True

                try:
                    scale, b02_counts, scale_samples = compute_scale(
                        linear_forward,
                        b02,
                        method=scale_method,
                        trim_frac=scale_trim_frac,
                        min_linear=scale_min_linear,
                        min_b02=scale_min_b02,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Scale estimation failed at iteration {iteration}: {exc}"
                    ) from exc

                if save_scale_ratios:
                    lin_arr = get_array(linear_forward)
                    b02_arr = get_array(b02)
                    ratio_mask = (lin_arr > float(scale_min_linear)) & (
                        b02_arr > float(scale_min_b02)
                    )
                    ratio_arr = np.zeros_like(lin_arr, dtype=np.float32)
                    if np.any(ratio_mask):
                        ratio_arr[ratio_mask] = (
                            lin_arr[ratio_mask] / b02_arr[ratio_mask]
                        )
                    ratio_img = b02.get_uniform_copy(0.0)
                    ratio_img.fill(ratio_arr.astype(np.float32))
                    ratio_img.write(
                        str(output_dir / f"mean_iter{iteration}_scale_ratio.hs")
                    )
                    ratio_mask_img = b02.get_uniform_copy(0.0)
                    ratio_mask_img.fill(ratio_mask.astype(np.float32))
                    ratio_mask_img.write(
                        str(output_dir / f"mean_iter{iteration}_scale_ratio_mask.hs")
                    )

                scale_path = output_dir / f"mean_iter{iteration}_scatter_scaling.txt"
                scale_path.write_text(f"{scale:.12g}\n", encoding="utf-8")

                b02_scaled = b02 * scale
                if use_residual:
                    # SIMIND always outputs extent=360.0/start=0.0; measured data
                    # may have slightly different values (e.g. 360.001/360).
                    # Re-wrap b02_scaled into linear_forward's (measured) geometry so
                    # xapyb doesn't reject the mismatch, and so residual stays in
                    # measured geometry — safe for damp_additive_update with alpha=1.
                    b02_scaled_matched = linear_forward.get_uniform_copy(0.0)
                    b02_scaled_matched.fill(get_array(b02_scaled).astype(np.float32))
                    residual = b02_scaled_matched - linear_forward
                    residual_path = output_dir / f"mean_iter{iteration}_residual.hs"
                    residual.write(str(residual_path))
                    current_residual = damp_additive_update(
                        current_residual, residual, residual_alpha
                    )

                if update_scatter:
                    b01_scaled = b01 * scale
                    scatter = b01_scaled - b02_scaled
                    # Re-wrap scatter into measured geometry so current_additive stays
                    # compatible with SIRF operations regardless of additive_alpha.
                    scatter_matched = measured.get_uniform_copy(0.0)
                    scatter_matched.fill(get_array(scatter).astype(np.float32))
                    scatter = scatter_matched
                    if scatter_median_size > 1 or scatter_gaussian_sigma > 0:
                        scatter_raw = scatter.clone()
                    if scatter_median_size > 1:
                        scatter = median_smooth_scatter(scatter, scatter_median_size)
                    if scatter_gaussian_sigma > 0:
                        scatter = gaussian_smooth_scatter(
                            scatter, scatter_gaussian_sigma
                        )
                    if args.save_components and (
                        scatter_median_size > 1 or scatter_gaussian_sigma > 0
                    ):
                        scatter_raw.write(
                            str(output_dir / f"mean_iter{iteration}_scatter_raw.hs")
                        )
                    scatter_path = output_dir / f"mean_iter{iteration}_scatter.hs"
                    scatter.write(str(scatter_path))

                full_forward = full_model.forward(
                    current_image, subset_num=0, num_subsets=1
                )

                if update_scatter:
                    current_additive = damp_additive_update(
                        current_additive, scatter, additive_alpha
                    )

                additive_for_model = current_additive.clone()
                if use_residual:
                    additive_for_model = additive_for_model + current_residual
                if additive_clamp:
                    # See note above: residuals are applied through the effective
                    # additive term, then projected back to non-negativity.
                    additive_for_model = clamp_nonnegative(additive_for_model)
                full_model.set_additive_term(additive_for_model)

                if update_scatter or use_residual:
                    scatter_sum = float(scatter.sum()) if scatter is not None else None
                    residual_sum = (
                        float(residual.sum()) if residual is not None else None
                    )
                    residual_applied_sum = (
                        float(current_residual.sum()) if use_residual else None
                    )
                    if scale_samples is None:
                        logging.info(
                            "Iteration %d scale=%g, linear=%g, full=%g, b01=%g, b02=%g, scatter=%s, residual_raw=%s, residual_applied=%s",
                            iteration,
                            scale,
                            float(linear_forward.sum()),
                            float(full_forward.sum()),
                            float(b01.sum()),
                            float(b02.sum()),
                            f"{scatter_sum:g}" if scatter_sum is not None else "n/a",
                            f"{residual_sum:g}" if residual_sum is not None else "n/a",
                            (
                                f"{residual_applied_sum:g}"
                                if residual_applied_sum is not None
                                else "n/a"
                            ),
                        )
                    else:
                        logging.info(
                            "Iteration %d scale=%g (samples=%d), linear=%g, full=%g, b01=%g, b02=%g, scatter=%s, residual_raw=%s, residual_applied=%s",
                            iteration,
                            scale,
                            scale_samples,
                            float(linear_forward.sum()),
                            float(full_forward.sum()),
                            float(b01.sum()),
                            float(b02.sum()),
                            f"{scatter_sum:g}" if scatter_sum is not None else "n/a",
                            f"{residual_sum:g}" if residual_sum is not None else "n/a",
                            (
                                f"{residual_applied_sum:g}"
                                if residual_applied_sum is not None
                                else "n/a"
                            ),
                        )

                if args.save_components:
                    b01.write(str(output_dir / f"mean_iter{iteration}_b01.hs"))
                    b02.write(str(output_dir / f"mean_iter{iteration}_b02.hs"))
                    if b01_scaled is not None:
                        b01_scaled.write(
                            str(output_dir / f"mean_iter{iteration}_b01_scaled.hs")
                        )
                    b02_scaled.write(
                        str(output_dir / f"mean_iter{iteration}_b02_scaled.hs")
                    )
                    linear_forward.write(
                        str(output_dir / f"mean_iter{iteration}_sirf_linear_forward.hs")
                    )
                    full_forward.write(
                        str(output_dir / f"mean_iter{iteration}_sirf_full_forward.hs")
                    )
            else:
                # No SIMIND needed; keep initial additive only.
                additive_for_model = current_additive.clone()
                if additive_clamp:
                    additive_for_model = clamp_nonnegative(additive_for_model)
                full_model.set_additive_term(additive_for_model)
                logging.info(
                    "Iteration %d: SIMIND skipped (mode=%s).",
                    iteration,
                    additive_mode,
                )

            # OSEM update with new additive scatter
            updated = run_osem_manual(
                measured=measured,
                model=full_model,
                initial_image=current_image,
                num_subsets=num_subsets,
                num_epochs=num_epochs,
                subset_indices=subset_indices,
                subset_sensitivities=subset_sensitivities,
                prior=prior,
                prior_update_point=rdp_update_point,
                measured_arr=measured_arr,
            )

            updated = maybe_smooth(updated, smoothing)
            recon_path = recon_filename(
                output_dir, num_epochs, num_subsets, iteration, smoothing
            )
            updated.write(str(recon_path))
            current_image = updated.clone()
            logging.info("Updated OSEM saved: %s", recon_path)
            del updated
            if full_forward is not None:
                del full_forward
            if linear_forward is not None:
                del linear_forward
            if b01 is not None:
                del b01
            if b02 is not None:
                del b02
            if b01_scaled is not None:
                del b01_scaled
            if b02_scaled is not None:
                del b02_scaled
            if residual is not None:
                del residual
            if scatter is not None:
                del scatter
            if args.gc_collect:
                gc.collect()
    finally:
        os.chdir(original_cwd)

    final_outputs = []
    if update_scatter:
        final_outputs.append(
            f"scatter={output_dir / f'mean_iter{num_iters}_scatter.hs'}"
        )
    if use_residual:
        final_outputs.append(
            f"residual={output_dir / f'mean_iter{num_iters}_residual.hs'}"
        )
    logging.info("Done. Final outputs: %s", ", ".join(final_outputs) or "none")


if __name__ == "__main__":
    main()
