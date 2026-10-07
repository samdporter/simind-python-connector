"""CIL objectives with an additive term that is refreshed during the run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import cil
from cil.optimisation.functions import (
    KullbackLeibler,
    OperatorCompositionFunction,
    SAGFunction,
    SumFunction,
    SVRGFunction,
)
from cil.optimisation.utilities.callbacks import Callback


class _LinearModelOperator:
    """CIL view of a linear SIRF acquisition model.

    SIRF's direct/adjoint return None when given out=..., but CIL's
    SumFunction and SVRGFunction use the returned object.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def direct(self, x: Any, out: Any = None) -> Any:
        result = self.model.direct(x, out=out)
        return out if out is not None else result

    def adjoint(self, x: Any, out: Any = None) -> Any:
        result = self.model.adjoint(x, out=out)
        return out if out is not None else result

    def range_geometry(self) -> Any:
        return self.model.range_geometry()

    def domain_geometry(self) -> Any:
        return self.model.domain_geometry()

    def norm(self) -> float:
        return self.model.norm()


@dataclass
class SubsetObjectives:
    functions: list  # OperatorCompositionFunction(KullbackLeibler(b_i, eta_i), A_i)
    data: list  # b_i, the measured data of each subset
    view_indices: list  # the views of each subset, in the order of functions
    linear_models: list  # A_i


def build_subset_objectives(
    acquisition_data: Any,
    additive: Any,
    acq_model_factory: Callable[[], Any],
    num_subsets: int,
    initial_image: Any,
    mode: str = "staggered",
    eta_floor: float = 1e-5,
) -> SubsetObjectives:
    """Split the data into subsets, each with its own KL objective.

    Each subset's eta is the matching part of max(additive, eta_floor).
    Selects process-wide SIRF memory storage and leaves it selected. Both
    acquisition inputs are cloned under that scheme before native subsetting;
    existing caller-owned objects are not converted or modified.
    """
    if mode not in ("staggered", "sequential"):
        # "random" shuffles views without a seed that partition_indices could
        # reproduce, so the view indices would not match the subsets.
        raise ValueError(f"mode must be 'staggered' or 'sequential', got {mode!r}")
    import sirf.STIR as sirf
    from sirf.contrib.partitioner import partitioner

    sirf.AcquisitionData.set_storage_scheme("memory")
    acquisition_data = acquisition_data.clone()
    additive = additive.clone()

    data, acq_models, _ = partitioner.data_partition(
        acquisition_data,
        additive,
        acquisition_data.get_uniform_copy(1),
        num_subsets,
        mode=mode,
        initial_image=initial_image,
        create_acq_model=acq_model_factory,
    )
    # partition_indices accepts a Python int, not a NumPy integer.
    num_views = int(acquisition_data.as_array().shape[-2])
    view_indices = partitioner.partition_indices(
        num_subsets, num_views, stagger=(mode == "staggered")
    )
    functions, linear_models = [], []
    for b, acq_model in zip(data, acq_models):
        operator = _LinearModelOperator(acq_model.get_linear_acquisition_model())
        eta = acq_model.get_additive_term().maximum(eta_floor)
        functions.append(
            OperatorCompositionFunction(KullbackLeibler(b=b, eta=eta), operator)
        )
        linear_models.append(operator)
    return SubsetObjectives(functions, data, view_indices, linear_models)


def set_subset_additive(objectives: SubsetObjectives, full_additive: Any) -> None:
    """Give every subset objective the matching views of full_additive.

    KullbackLeibler copies eta when it is created, so each subset gets a new
    KullbackLeibler instead of an edited one. Selects process-wide SIRF memory
    storage and clones full_additive under that scheme before subsetting;
    leaves memory storage selected and does not modify the caller's object.
    """
    import sirf.STIR as sirf

    sirf.AcquisitionData.set_storage_scheme("memory")
    full_additive = full_additive.clone()
    for function, b, views in zip(
        objectives.functions, objectives.data, objectives.view_indices
    ):
        function.function = KullbackLeibler(b=b, eta=full_additive.get_subset(views))


def refresh_stochastic_state(f: Any, x: Any) -> None:
    """Recompute stored gradients after the data term changed.

    SVRG keeps a full gradient at a snapshot and SAG/SAGA keep one gradient
    per subset, all computed with the old additive term. SVRG and SAG are
    SumFunction subclasses, so they are checked first.
    """
    if isinstance(f, SVRGFunction):
        # Private CIL method: there is no public way to retake the snapshot.
        refresh = getattr(f, "_update_full_gradient_and_return", None)
        if refresh is None:
            raise NotImplementedError(
                f"CIL {cil.__version__} has no "
                "SVRGFunction._update_full_gradient_and_return(), so the SVRG "
                "snapshot cannot be refreshed"
            )
        refresh(x)
    elif isinstance(f, SAGFunction):
        f.warm_start_approximate_gradients(x)
        # NumPy's sum loses the SIRF image container; sum the images instead.
        full = f._list_stored_gradients[0].copy()
        for gradient in f._list_stored_gradients[1:]:
            full += gradient
        f._full_gradient_at_iterate = full
    elif isinstance(f, SumFunction):
        for function in f.functions:
            refresh_stochastic_state(function, x)


class CorrectionCallback(Callback):
    """Refresh the additive term on the updater's schedule during Algorithm.run().

    Installed CIL calls back at iteration 0 after initial objective evaluation,
    before the first update; later callbacks follow updates and iteration
    increments. A refresh at iteration k is used from iteration k + 1 on.
    A refresh in the final callback cannot affect the returned reconstruction.
    """

    def __init__(
        self,
        updater: Any,
        objectives: SubsetObjectives,
        on_update: Optional[Callable[[Any, Any], None]] = None,
    ) -> None:
        super().__init__(verbose=0)
        self.updater = updater
        self.objectives = objectives
        self.on_update = on_update

    def prime(self, image: Any) -> None:
        """Start from an estimate at image, before Algorithm.run()."""
        self.updater.update_now(0, image)
        set_subset_additive(self.objectives, self.updater.current)

    def __call__(self, algorithm: Any) -> None:
        if not self.updater.maybe_update(algorithm.iteration, algorithm.solution):
            return
        set_subset_additive(self.objectives, self.updater.current)
        refresh_stochastic_state(algorithm.f, algorithm.solution)
        if self.on_update is not None:
            self.on_update(algorithm, self.updater)
