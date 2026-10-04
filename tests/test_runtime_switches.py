import numpy as np
import pytest

from simind_python_connector.core.config import RuntimeSwitches


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [1, 10, np.int64(5)])
def test_nn_accepts_integers_of_at_least_one(value):
    switches = RuntimeSwitches()
    switches.set_switch("NN", value)
    assert switches.switches["NN"] == value


@pytest.mark.parametrize("value", [0, -1, 0.5, 1.0, True, "2"])
def test_nn_rejects_everything_else(value):
    with pytest.raises(ValueError, match="integer >= 1"):
        RuntimeSwitches().set_switch("NN", value)


@pytest.mark.parametrize(
    "key, value",
    [
        ("CA", 1),
        ("FW", "windows"),
        ("OU", 11),
        ("DP", 0.5),
        ("BG", 1.0),
        ("HO", True),
        ("CO", True),
        ("X1", "h2o"),
        ("X6", "pb_sb2"),
    ],
)
def test_switches_from_the_manual_are_recognised(key, value):
    switches = RuntimeSwitches()
    switches.set_switch(key, value)
    assert switches.switches[key] == value


def test_mp_switch_points_to_set_mpi():
    with pytest.raises(ValueError, match="set_mpi"):
        RuntimeSwitches().set_switch("MP", 4)


def test_none_removes_a_switch():
    switches = RuntimeSwitches()
    switches.set_switch("RR", 5)
    switches.set_switch("RR", None)
    assert "RR" not in switches.switches


@pytest.mark.parametrize("key, value", [("CA", 3), ("CA", 1.0), ("DI", -1)])
def test_ca_and_di_take_zero_one_or_two(key, value):
    with pytest.raises(ValueError, match="0, 1 or 2"):
        RuntimeSwitches().set_switch(key, value)


@pytest.mark.parametrize("key", ["RR", "SC"])
def test_rr_and_sc_require_integers(key):
    with pytest.raises(ValueError, match="integer"):
        RuntimeSwitches().set_switch(key, 1.5)


def test_negative_scatter_order_is_allowed():
    switches = RuntimeSwitches()
    switches.set_switch("SC", -3)
    assert switches.switches["SC"] == -3


def test_switch_can_be_set_by_description():
    switches = RuntimeSwitches()
    switches.set_switch("Random number generator seed", 7)
    assert switches.switches["RR"] == 7


def test_unknown_switch_is_rejected():
    with pytest.raises(ValueError, match="not recognised"):
        RuntimeSwitches().set_switch("ZZ", 1)
