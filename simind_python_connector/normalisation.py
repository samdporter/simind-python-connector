"""Put SIMIND output on the scale of a reference projection or measured data.

SIMIND output always needs an explicit scale. If the activity is known, tell
SIMIND with SimindPythonConnector.set_activity() (Index 25). Otherwise use
scale_to_reference() against a projection that is already on the wanted
scale. The source map's magnitude and NN only change the number of photon
histories (statistics), never the counts.

Two details of the contract: with method="sum" an all-zero reference gives
0.0, as long as the kept component sum is positive; trim_fraction is only
read, and only validated, when method="trimmed".
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def scale_to_reference(
    component: np.ndarray,
    reference: np.ndarray,
    *,
    method: str = "sum",
    trim_fraction: float = 0.1,
    min_reference: float = 0.0,
    min_component: float = 0.0,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Return s such that s * component is on the scale of reference.

    Apply s to every output of the same SIMIND run. Typical uses:

    - component = SIMIND geometric primary (PENETRATE 'geom_coll_primary',
      or SCATTWIN 'pri_wN' for low-energy isotopes), reference = analytic
      linear forward projection of the same (optionally masked) image;
    - component = SIMIND total ('tot_wN' or 'all_interactions'),
      reference = measured data.

    Pass mask to fit only over part of the data, for example a
    forward-projected body mask thresholded at a fraction of its maximum.

    method="sum": s = sum(reference) / sum(component) over the kept bins.
    method="trimmed": bins must also exceed min_reference/min_component; the
    bins are ranked by reference/component and floor(trim_fraction * n) are
    dropped from each end before taking the ratio of sums.

    Bins where either array is not finite, or outside mask, are never used.
    """
    component = np.asarray(component, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if component.shape != reference.shape:
        raise ValueError(
            f"component and reference shapes differ: {component.shape} vs "
            f"{reference.shape}"
        )

    keep = np.isfinite(component) & np.isfinite(reference)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != component.shape:
            raise ValueError(f"mask shape {mask.shape} differs from {component.shape}")
        keep &= mask

    if method == "trimmed":
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError(f"trim_fraction must be in [0, 0.5), got {trim_fraction}")
        keep &= (reference > min_reference) & (component > min_component)
    elif method != "sum":
        raise ValueError(f"unknown method {method!r}; use 'sum' or 'trimmed'")

    kept_component = component[keep]
    kept_reference = reference[keep]
    if method == "trimmed" and kept_component.size:
        drop = int(np.floor(trim_fraction * kept_component.size))
        if drop:
            order = np.argsort(kept_reference / kept_component)[drop:-drop]
            kept_component = kept_component[order]
            kept_reference = kept_reference[order]

    if kept_component.size == 0:
        raise ValueError("no bins left after masking and trimming")
    component_sum = float(kept_component.sum())
    if component_sum <= 0:
        raise ValueError("the component sum over the kept bins must be > 0")
    return float(kept_reference.sum()) / component_sum


__all__ = ["scale_to_reference"]
