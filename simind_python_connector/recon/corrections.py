"""Correction models: full-data additive terms estimated from the current image."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

import numpy as np

from simind_python_connector.core.types import PenetrateOutputType, ScoringRoutine


class CorrectionModel(Protocol):
    def estimate(self, image: Any) -> Any:
        """Return the full-data additive term for image, in the template geometry."""


class ScatterCorrection:
    """SIMIND scatter: what SIMIND records but the fast linear model does not.

    With PENETRATE this is all events minus the geometrically collimated,
    attenuated primary (b01 - b02): phantom scatter, septal penetration,
    collimator scatter and X-rays. With SCATTWIN it is the window's scatter
    output, which holds phantom scatter only.

    With measured, this is Dewaraja et al. 2017's pixel-wise model: the
    measured data times the scatter-to-total ratio of SIMIND's run. It does not
    depend on the projector's scalar scale. With smoothing, SIMIND's scatter
    and total are smoothed with the same Gaussian before the ratio, and the
    scaled result is smoothed again. smoothing_fwhm_bins=3.0 and
    smooth_across_views=True match the paper. The final smoothing is retained
    because the measured noise enters the estimate. The ratio is zero only
    where the smoothed total is not positive; signed scatter is not clipped.
    smooth_across_views wraps around the view axis, so it assumes the views
    cover 360 degrees.
    """

    def __init__(
        self,
        projector: Any,
        background: Any = None,
        window: int = 1,
        smoothing_fwhm_bins: Optional[float] = None,
        measured: Any = None,
        smooth_across_views: bool = False,
    ) -> None:
        self.projector = projector
        self.background = background
        self.window = window
        self.smoothing_fwhm_bins = smoothing_fwhm_bins
        self.measured = measured
        self.smooth_across_views = smooth_across_views

    @property
    def last_scale(self) -> Optional[float]:
        return self.projector.last_scale

    def estimate(self, image: Any) -> Any:
        outputs = self.projector.project(image)
        if self.projector.adaptor.get_scoring_routine() == ScoringRoutine.PENETRATE:
            total = outputs[PenetrateOutputType.ALL_INTERACTIONS.slug]
            scatter = total - outputs[PenetrateOutputType.GEOM_COLL_PRIMARY_ATT.slug]
        else:
            scatter = outputs[f"sca_w{self.window}"]
            total = (
                outputs[f"tot_w{self.window}"] if self.measured is not None else None
            )
        if self.measured is not None:
            scatter = self._scale_to_measured(scatter, total)
        if self.smoothing_fwhm_bins is not None:
            scatter = _smooth_views(
                scatter, self.smoothing_fwhm_bins, self.smooth_across_views
            )
        if self.background is None:
            return scatter
        return self.background + scatter

    def _scale_to_measured(self, scatter: Any, total: Any) -> Any:
        scatter_array = scatter.as_array()
        measured_array = self.measured.as_array()
        if measured_array.shape != scatter_array.shape:
            raise ValueError(
                f"measured shape {measured_array.shape} does not match "
                f"scatter shape {scatter_array.shape}"
            )
        total_array = total.as_array()
        if total_array.shape != scatter_array.shape:
            raise ValueError(
                f"total shape {total_array.shape} does not match "
                f"scatter shape {scatter_array.shape}"
            )
        # Smooth the MC scatter and its total before forming the fraction. In
        # the fixed-primary, noisy-scatter control used to check this order,
        # pre-smoothing reduced the ratio noise in the summed estimate. The
        # direction and size of a noisy ratio's error depend on the covariance
        # between numerator and denominator, so this is an observation about
        # that control, not a general theorem about Monte Carlo ratios, and it
        # does not account for the unresolved pixel-wise drift. The final
        # smoothing in estimate() is retained.
        if self.smoothing_fwhm_bins is not None:
            scatter_array = _smooth_array(
                scatter_array, self.smoothing_fwhm_bins, self.smooth_across_views
            )
            total_array = _smooth_array(
                total_array, self.smoothing_fwhm_bins, self.smooth_across_views
            )
        fraction = np.divide(
            scatter_array,
            total_array,
            out=np.zeros(scatter_array.shape, dtype=np.float64),
            where=total_array > 0,
        )
        out = scatter.clone()
        out.fill((measured_array * fraction).astype(np.float32))
        return out


def _smooth_array(
    array: Any, fwhm_bins: float, across_views: bool = False
) -> np.ndarray:
    """Gaussian-smooth a projection array, keeping its total.

    The total is restored by rescaling only when the input sum stands clear
    of the rounding floor of that sum, so a zero-sum signed field keeps its
    smoothed values instead of being multiplied by an arbitrary factor.
    """
    from scipy.ndimage import gaussian_filter

    values = np.asarray(array, dtype=np.float64)
    sigma = [0.0] * values.ndim
    mode = ["reflect"] * values.ndim
    # Projection arrays end in (axial, view, bin); SIRF may add a TOF axis first.
    sigma[-3] = sigma[-1] = fwhm_bins / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if across_views:
        sigma[-2] = sigma[-1]
        mode[-2] = "wrap"
    smoothed = gaussian_filter(values, sigma, mode=mode)
    # Keep the original total, but only when it stands clear of the rounding
    # floor of the sum. A cancelling signed field sums to about zero, and so
    # does its smoothed version: their ratio is then an arbitrary factor (and
    # zero erases the field), so such a field is left exactly as smoothed.
    original_sum = values.sum()
    if abs(original_sum) > 1e-9 * np.abs(values).sum():
        total = smoothed.sum()
        if total != 0.0:
            smoothed *= original_sum / total
    return smoothed.astype(np.float32)


def _smooth_views(data: Any, fwhm_bins: float, across_views: bool = False) -> Any:
    """Gaussian-smooth each view along its axial and bin axes, keeping the total.

    With across_views, also smooth along the view axis, wrapping round.
    """
    out = data.clone()
    out.fill(_smooth_array(data.as_array(), fwhm_bins, across_views))
    return out


class ResidualCorrection:
    """Fu & Qi residual correction: base_additive + A_acc(x) - A_fast(x).

    As the fast model's additive term, it makes the fast forward projection of
    x match the accurate one. Only forward projections use the accurate model
    (Fu & Qi, Med Phys 37:704, 2010).

    fast_model must be the full-data linear model without an additive term;
    base_additive is whatever the accurate model does not include.
    """

    def __init__(
        self,
        accurate: Callable[[Any], Any],
        fast_model: Any,
        base_additive: Any,
        mask: Any = None,
    ) -> None:
        self.accurate = accurate
        self.fast_model = fast_model
        self.base_additive = base_additive
        self.mask = mask
        self.last_accurate: Any = None

    @property
    def last_scale(self) -> Optional[float]:
        return getattr(self.accurate, "last_scale", None)

    def estimate(self, image: Any) -> Any:
        x = image.maximum(0)
        if self.mask is not None:
            x = x * self.mask
        self.last_accurate = self.accurate(x)
        return self.base_additive + (self.last_accurate - self.fast_model.direct(x))
