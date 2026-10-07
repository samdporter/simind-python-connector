"""When and how the additive term of the forward model is refreshed."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class UpdateSchedule:
    """Iterations at which the additive term is refreshed.

    Iterations are the algorithm's own count; for CIL algorithms over subsets
    that is subiterations.
    """

    every: int
    first: Optional[int] = None
    stop_at: Optional[int] = None

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f"every must be >= 1, got {self.every}")

    def is_due(self, iteration: int) -> bool:
        first = self.every if self.first is None else self.first
        if iteration < first or (iteration - first) % self.every != 0:
            return False
        return self.stop_at is None or iteration < self.stop_at


@dataclass(frozen=True)
class UpdateRecord:
    iteration: int
    scale: Optional[float]  # correction.last_scale, when the correction has one
    additive_sum: float  # sum of the floored term in use
    additive_min: float  # minimum of the unfloored D_k
    seconds: float


class AdditiveUpdater:
    """Additive term eta_k = max(D_k, floor), D_k = (1 - a) D_{k-1} + a estimate(x_k).

    D_k is never floored, so that later damped updates combine correctly;
    only the term handed to the model is.
    """

    def __init__(
        self,
        correction: Any,
        schedule: UpdateSchedule,
        initial_additive: Any,
        damping: float = 1.0,
        floor: float = 1e-5,
    ) -> None:
        if not 0.0 < damping <= 1.0:
            raise ValueError(f"damping must be in (0, 1], got {damping}")
        if not floor > 0.0:
            raise ValueError(f"floor must be > 0, got {floor}")
        self.correction = correction
        self.schedule = schedule
        self.damping = damping
        self.floor = floor
        self.raw = initial_additive.clone()
        self.current = self.raw.maximum(floor)
        self.history: list[UpdateRecord] = []

    def update_now(self, iteration: int, image: Any) -> None:
        start = time.perf_counter()
        estimate = self.correction.estimate(image)
        self.raw = self.raw * (1.0 - self.damping) + estimate * self.damping
        self.current = self.raw.maximum(self.floor)
        self.history.append(
            UpdateRecord(
                iteration=iteration,
                scale=getattr(self.correction, "last_scale", None),
                additive_sum=float(self.current.sum()),
                additive_min=float(self.raw.as_array().min()),
                seconds=time.perf_counter() - start,
            )
        )

    def maybe_update(self, iteration: int, image: Any) -> bool:
        if not self.schedule.is_due(iteration):
            return False
        self.update_now(iteration, image)
        return True
