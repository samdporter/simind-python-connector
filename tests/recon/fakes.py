"""NumPy stand-ins for the SIRF objects and adaptors the recon modules use."""

from __future__ import annotations

import numpy as np

from simind_python_connector.core.types import ScoringRoutine


def _value(other):
    return other.array if isinstance(other, FakeData) else other


class FakeData:
    """SIRF data stand-in; projection arrays end in (axial, view, bin)."""

    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float64)

    def as_array(self):
        return self.array

    def clone(self):
        return FakeData(self.array.copy())

    def fill(self, array):
        self.array = np.asarray(array, dtype=np.float64).reshape(self.array.shape)

    def maximum(self, value):
        return FakeData(np.maximum(self.array, _value(value)))

    def sum(self):
        return float(self.array.sum())

    def get_subset(self, views):
        return FakeData(np.take(self.array, list(views), axis=-2))

    def __add__(self, other):
        return FakeData(self.array + _value(other))

    def __sub__(self, other):
        return FakeData(self.array - _value(other))

    def __mul__(self, other):
        return FakeData(self.array * _value(other))

    __rmul__ = __mul__


class FakeAdaptor:
    """Records calls in order and returns copies of fixed outputs from run()."""

    def __init__(self, outputs, scoring_routine=ScoringRoutine.PENETRATE):
        self.outputs = outputs
        self.scoring_routine = scoring_routine
        self.calls = []

    def set_mu_map(self, mu_map):
        self.calls.append(("set_mu_map", mu_map))

    def set_template(self, template):
        self.calls.append(("set_template", template))

    def set_source(self, source):
        self.calls.append(("set_source", source))

    def set_activity(self, activity_mbq, time_per_projection_s=1.0):
        self.calls.append(("set_activity", activity_mbq, time_per_projection_s))

    def add_runtime_switch(self, switch, value):
        self.calls.append(("add_runtime_switch", switch, value))

    def get_scoring_routine(self):
        return self.scoring_routine

    def run(self):
        self.calls.append(("run",))
        return {key: value.clone() for key, value in self.outputs.items()}

    def names(self):
        return [call[0] for call in self.calls]


class FixedCorrection:
    """CorrectionModel stub that always estimates the same additive term."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def estimate(self, image):
        self.calls += 1
        return self.value.clone()
