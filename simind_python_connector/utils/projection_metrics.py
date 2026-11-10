"""Small projection-metric helpers used by the worked examples.

They are deliberately plain NumPy: the examples need a shared count scale, a
repeatability measure, a triple-energy-window estimate and a quadrature check,
not a reconstruction framework.
"""

from __future__ import annotations

import numpy as np

from simind_python_connector.core.types import MAX_SOURCE


def projection_view0(projection: np.ndarray) -> np.ndarray:
    """Return the first view of a 3-D or 4-D projection array."""
    if projection.ndim == 4:
        return projection[0, :, 0, :]
    if projection.ndim == 3:
        return projection[0, :, :]
    raise ValueError(
        f"Expected projection with 3 or 4 dimensions, got shape {projection.shape}"
    )


def require_output_keys(outputs: dict[str, object], *keys: str) -> dict[str, object]:
    """Return only the named outputs, or say which ones the run did not write."""
    missing = [key for key in keys if key not in outputs]
    if missing:
        raise KeyError(
            f"SIMIND did not write {', '.join(missing)}; it wrote "
            f"{', '.join(sorted(outputs))}"
        )
    return {key: outputs[key] for key in keys}


def source_weights(source: np.ndarray, quantization_scale: float) -> np.ndarray:
    """Return the integer photon-history weights the connector writes."""
    source = np.asarray(source, dtype=float)
    if source.size == 0:
        raise ValueError("source must not be empty")
    if not 0.0 < quantization_scale <= 1.0:
        raise ValueError("quantization_scale must be in (0, 1]")
    maximum = float(source.max())
    if maximum == 0.0:
        return np.zeros(source.shape, dtype=np.int64)
    return np.clip(
        np.round(source / maximum * (MAX_SOURCE * quantization_scale)),
        0,
        MAX_SOURCE,
    ).astype(np.int64)


def source_weight_sum(source: np.ndarray, quantization_scale: float) -> int:
    """Number of photon histories per projection before the ``NN`` multiplier."""
    return int(source_weights(source, quantization_scale).sum())


def zero_history_fraction(source: np.ndarray, quantization_scale: float) -> float:
    """Fraction of the source weight that integer rounding sends to zero."""
    source = np.asarray(source, dtype=float)
    total = float(source.sum())
    if total == 0.0:
        return 0.0
    weights = source_weights(source, quantization_scale)
    return float(source[weights == 0].sum() / total)


def symmetric_limits(*images: np.ndarray) -> tuple[float, float]:
    """A zero-centred colour scale that fits every image."""
    if not images:
        raise ValueError("symmetric_limits needs at least one image")
    limit = max(float(np.max(np.abs(image))) for image in images)
    if limit == 0.0:
        return 0.0, 0.0
    return -limit, limit


def normalised_rms_difference(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference between two images, relative to their mean magnitude."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} and {b.shape}")
    scale = float(np.mean(np.abs(a) + np.abs(b)) / 2.0)
    if scale == 0.0:
        return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale)


def tew_scatter_estimate(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    lower_width_kev: float,
    photopeak_width_kev: float,
    upper_width_kev: float,
) -> np.ndarray:
    """Triple-energy-window scatter estimate for one photopeak.

    Each side window is scaled to the photopeak width and half of the total is
    assigned to the peak, which assumes the scatter spectrum is linear across
    the three windows. That is an approximation, not a measurement.
    """
    widths = (lower_width_kev, photopeak_width_kev, upper_width_kev)
    if any(width <= 0 for width in widths):
        raise ValueError(f"window widths must be positive, got {widths}")
    return np.asarray(lower) * (
        photopeak_width_kev / (2.0 * lower_width_kev)
    ) + np.asarray(upper) * (photopeak_width_kev / (2.0 * upper_width_kev))


def tew_correction(
    photopeak: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    lower_width_kev: float,
    photopeak_width_kev: float,
    upper_width_kev: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (scatter estimate, corrected, clipped count fraction, clipped pixels).

    The corrected image is the photopeak minus the estimate, clipped at zero.
    The count fraction is the part of the photopeak that clipping removed;
    the pixel fraction is the part of the image that the estimate exceeded.
    """
    photopeak = np.asarray(photopeak)
    estimate = tew_scatter_estimate(
        lower,
        upper,
        lower_width_kev=lower_width_kev,
        photopeak_width_kev=photopeak_width_kev,
        upper_width_kev=upper_width_kev,
    )
    unclipped = photopeak - estimate
    corrected = np.clip(unclipped, a_min=0.0, a_max=None)
    peak_total = float(np.clip(photopeak, 0.0, None).sum())
    clipped_counts = float((corrected - unclipped).sum())
    clipped_count_fraction = clipped_counts / peak_total if peak_total else 0.0
    clipped_pixel_fraction = float(np.mean(unclipped < 0.0))
    return estimate, corrected, clipped_count_fraction, clipped_pixel_fraction


def count_preservation_error(
    outputs: dict[str, object], total_key: str, *component_keys: str
) -> float:
    """Relative difference between a total and the sum of its components."""
    total = float(outputs[total_key].projection.sum())
    components = sum(float(outputs[key].projection.sum()) for key in component_keys)
    if total == 0.0:
        return 0.0 if components == 0.0 else float("inf")
    return (components - total) / total
