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
    scatter fraction scatter / total of SIMIND's run, times the measured data.
    It does not depend on the projector's scalar scale. With smoothing,
    smoothing_fwhm_bins=3.0 and smooth_across_views=True match the paper.
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
            total = outputs.get(f"tot_w{self.window}")
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
        fraction = np.divide(
            scatter_array,
            total_array,
            out=np.zeros(scatter_array.shape, dtype=np.float64),
            where=total_array > 0,
        )
        out = scatter.clone()
        out.fill((measured_array * fraction).astype(np.float32))
        return out


def _smooth_views(data: Any, fwhm_bins: float, across_views: bool = False) -> Any:
    """Gaussian-smooth each view along its axial and bin axes, keeping the total.

    With across_views, also smooth along the view axis, wrapping round.
    """
    from scipy.ndimage import gaussian_filter

    array = np.asarray(data.as_array(), dtype=np.float64)
    sigma = [0.0] * array.ndim
    mode = ["reflect"] * array.ndim
    # Projection arrays end in (axial, view, bin); SIRF may add a TOF axis first.
    sigma[-3] = sigma[-1] = fwhm_bins / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if across_views:
        sigma[-2] = sigma[-1]
        mode[-2] = "wrap"
    smoothed = gaussian_filter(array, sigma, mode=mode)
    total = smoothed.sum()
    if total != 0.0:
        smoothed *= array.sum() / total
    out = data.clone()
    out.fill(smoothed.astype(np.float32))
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
