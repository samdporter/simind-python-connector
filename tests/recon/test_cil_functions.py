import re

import pytest


pytestmark = pytest.mark.requires_cil


def test_build_subset_objectives_rejects_random_partitions():
    from simind_python_connector.recon.cil import build_subset_objectives

    with pytest.raises(ValueError, match="'staggered' or 'sequential'"):
        build_subset_objectives(None, None, None, 2, None, mode="random")


def test_refresh_recurses_into_sums_and_refreshes_svrg():
    from cil.optimisation.functions import SumFunction, SVRGFunction, ZeroFunction

    from simind_python_connector.recon.cil import refresh_stochastic_state

    class _RecordingSVRG(SVRGFunction):
        def __init__(self):  # skip CIL's set-up
            self.points = []

        def _update_full_gradient_and_return(self, x, out=None):
            self.points.append(x)

    svrg = _RecordingSVRG()
    refresh_stochastic_state(SumFunction(svrg, ZeroFunction()), "x")
    assert svrg.points == ["x"]


def test_refresh_without_the_private_svrg_method_names_the_cil_version():
    import cil
    from cil.optimisation.functions import SVRGFunction

    from simind_python_connector.recon.cil import refresh_stochastic_state

    class _NoRefresh(SVRGFunction):
        _update_full_gradient_and_return = None

        def __init__(self):
            pass

    with pytest.raises(NotImplementedError, match=re.escape(cil.__version__)):
        refresh_stochastic_state(_NoRefresh(), "x")
