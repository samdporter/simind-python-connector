"""Convergence diagnostics for residual correction."""

from __future__ import annotations

from typing import Any

import numpy as np


def _array(data: Any) -> np.ndarray:
    values = data.as_array() if hasattr(data, "as_array") else data
    return np.asarray(values, dtype=np.float64)


def poisson_nll(measured: Any, mean: Any) -> float:
    """Poisson negative log-likelihood without its constant terms.

    sum(mean - measured * log(mean)) over bins with mean > 0; +inf if a bin
    with counts has a mean <= 0.
    """
    measured, mean = _array(measured), _array(mean)
    positive = mean > 0
    if np.any((measured > 0) & ~positive):
        return float("inf")
    return float(np.sum(mean[positive] - measured[positive] * np.log(mean[positive])))


def effective_objective(measured: Any, correction: Any, floor: float = 1e-5) -> float:
    """The accurate model's data term at the image of the last update.

    This diagnostic evaluates ``last_accurate + base_additive``. ``floor``
    floors that reference mean only to keep zero-mean bins with counts finite;
    it is independent of the floor and damping used by an AdditiveUpdater.
    It does not measure the objective optimized by the fast model between
    updates. A rising trend beyond Monte Carlo noise suggests changing the
    update interval or damping.
    """
    if correction.last_accurate is None:
        raise ValueError("effective_objective is available after the first estimate")
    mean = (correction.last_accurate + correction.base_additive).maximum(floor)
    return poisson_nll(measured, mean)
