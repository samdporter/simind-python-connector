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
    """

    def __init__(
        self,
        projector: Any,
        background: Any = None,
        window: int = 1,
        smoothing_fwhm_bins: Optional[float] = None,
    ) -> None:
        self.projector = projector
        self.background = background
        self.window = window
        self.smoothing_fwhm_bins = smoothing_fwhm_bins

    @property
    def last_scale(self) -> Optional[float]:
        return self.projector.last_scale

    def estimate(self, image: Any) -> Any:
        outputs = self.projector.project(image)
        if self.projector.adaptor.get_scoring_routine() == ScoringRoutine.PENETRATE:
            scatter = (
                outputs[PenetrateOutputType.ALL_INTERACTIONS.slug]
                - outputs[PenetrateOutputType.GEOM_COLL_PRIMARY_ATT.slug]
            )
        else:
            scatter = outputs[f"sca_w{self.window}"]
        if self.smoothing_fwhm_bins is not None:
            scatter = _smooth_views(scatter, self.smoothing_fwhm_bins)
        if self.background is None:
            return scatter
        return self.background + scatter


def _smooth_views(data: Any, fwhm_bins: float) -> Any:
    """Gaussian-smooth each view along its axial and bin axes, keeping the total."""
    from scipy.ndimage import gaussian_filter

    array = np.asarray(data.as_array(), dtype=np.float64)
    sigma = [0.0] * array.ndim
    # Projection arrays end in (axial, view, bin); SIRF may add a TOF axis first.
    sigma[-3] = sigma[-1] = fwhm_bins / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    smoothed = gaussian_filter(array, sigma)
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
