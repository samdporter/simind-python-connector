"""SIMIND as a projector of SIRF images, with its output put on the data scale."""

from __future__ import annotations

from typing import Any, Callable, Optional

from simind_python_connector.normalisation import scale_to_reference


class SimindProjector:
    """Simulate an image with SIMIND in a fixed template geometry.

    SIMIND's output does not follow the magnitude of the image, because the
    source map is always scaled to MAX_SOURCE photon histories. Every call is
    therefore put on the data scale in one of two ways:

    - normalise(outputs, image) returns a factor applied to every output
      (A2 route 2, e.g. reference_normaliser);
    - activity_mbq(image) returns the total activity, which SIMIND is given
      through Index 25 (A2 route 1); no factor is applied.
    """

    def __init__(
        self,
        adaptor: Any,
        template: Any,
        mu_map: Any,
        *,
        normalise: Optional[Callable[[dict, Any], float]] = None,
        activity_mbq: Optional[Callable[[Any], float]] = None,
        time_per_projection_s: float = 1.0,
        mask: Any = None,
        seed: Optional[int] = None,
    ) -> None:
        if (normalise is None) == (activity_mbq is None):
            raise ValueError(
                "Give exactly one of normalise (scale to a reference) or "
                "activity_mbq (known activity)"
            )
        self.adaptor = adaptor
        self.template = template
        self.mu_map = mu_map
        self.normalise = normalise
        self.activity_mbq = activity_mbq
        self.time_per_projection_s = time_per_projection_s
        self.mask = mask
        self.seed = seed
        self.last_scale: Optional[float] = None
        self.last_outputs: Optional[dict[str, Any]] = None
        self._calls = 0

    def project(self, image: Any) -> dict[str, Any]:
        x = image.maximum(0)
        if self.mask is not None:
            x = x * self.mask
        if self._calls == 0:
            self.adaptor.set_mu_map(self.mu_map)
            self.adaptor.set_template(self.template)
        self.adaptor.set_source(x)
        if self.seed is not None:
            # A new seed per call keeps the MC noise of successive updates
            # independent.
            self.adaptor.add_runtime_switch("RR", self.seed + self._calls)
        if self.activity_mbq is not None:
            self.adaptor.set_activity(self.activity_mbq(x), self.time_per_projection_s)
        outputs = self.adaptor.run()
        scale = 1.0 if self.normalise is None else float(self.normalise(outputs, x))
        self._calls += 1
        self.last_scale = scale
        self.last_outputs = {key: value * scale for key, value in outputs.items()}
        return self.last_outputs


def reference_normaliser(
    fast_model: Any,
    component: str = "geom_coll_primary",
    method: str = "sum",
    mask: Any = None,
    **scale_kwargs: Any,
) -> Callable[[dict, Any], float]:
    """Return normalise(outputs, image) for SimindProjector (A2 route 2).

    It scales SIMIND's component to fast_model.direct(image), so fast_model
    must be the full-data linear model, without an additive term. mask and
    scale_kwargs are passed on to scale_to_reference.
    """

    def normalise(outputs: dict, image: Any) -> float:
        return scale_to_reference(
            outputs[component].as_array(),
            fast_model.direct(image).as_array(),
            method=method,
            mask=mask,
            **scale_kwargs,
        )

    return normalise
