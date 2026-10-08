"""Reconstruction with SIMIND corrections (optional).

The core package never imports this subpackage. The CIL driver (recon.cil)
and the SIRF OSEM loop (recon.osem) need those packages and are imported
explicitly.
"""

from simind_python_connector.recon.corrections import (
    CorrectionModel,
    ResidualCorrection,
    ScatterCorrection,
)
from simind_python_connector.recon.diagnostics import (
    effective_objective,
    poisson_nll,
)
from simind_python_connector.recon.projector import (
    SimindComponent,
    SimindProjector,
    reference_normaliser,
)
from simind_python_connector.recon.updates import (
    AdditiveUpdater,
    UpdateRecord,
    UpdateSchedule,
)


__all__ = [
    "AdditiveUpdater",
    "CorrectionModel",
    "ResidualCorrection",
    "ScatterCorrection",
    "SimindComponent",
    "SimindProjector",
    "UpdateRecord",
    "UpdateSchedule",
    "effective_objective",
    "poisson_nll",
    "reference_normaliser",
]
