import numpy as np
import pytest

from simind_python_connector.core.config import RuntimeSwitches


pytestmark = pytest.mark.unit


# SIMIND 8 manual: switches that require a value. True renders /KEY with no
# value, so it must be rejected for every one of them. NN, RR, SC, CA and DI
# are also value-taking but are covered by the type tests below.
_VALUE_TAKING_SWITCHES = [
    "1S",
    "A1",
    "A2",
    "A3",
    "BG",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "CC",
    "DF",
    "DP",
    "ES",
    "FD",
    "FI",
    "FS",
    "FW",
    "FZ",
    "IF",
    "IN",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "LO",
    "M1",
    "M2",
    "M3",
    "M4",
    "OR",
    "OU",
    "PR",
    "PX",
    "SF",
    "TH",
    "TS",
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6",
]

# The manual gives these without a value (/HO) or with an optional one (/PU).
_VALUE_LESS_SWITCHES = ["CO", "FE", "HO", "I2", "LF", "PU", "QF", "SB", "UA", "WB"]


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


@pytest.mark.parametrize(
    "key, value", [("CA", 3), ("CA", 1.0), ("CA", True), ("DI", -1), ("DI", True)]
)
def test_ca_and_di_take_zero_one_or_two(key, value):
    with pytest.raises(ValueError, match="0, 1 or 2"):
        RuntimeSwitches().set_switch(key, value)


@pytest.mark.parametrize(
    "key, value", [("RR", 1.5), ("RR", True), ("SC", 1.5), ("SC", True)]
)
def test_rr_and_sc_require_integers(key, value):
    with pytest.raises(ValueError, match="integer"):
        RuntimeSwitches().set_switch(key, value)


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


@pytest.mark.parametrize("switch", _VALUE_TAKING_SWITCHES)
def test_true_is_rejected_for_switches_that_require_a_value(switch):
    with pytest.raises(ValueError, match="requires a value"):
        RuntimeSwitches().set_switch(switch, True)


@pytest.mark.parametrize("switch", _VALUE_LESS_SWITCHES)
def test_true_is_accepted_for_switches_without_a_value(switch):
    switches = RuntimeSwitches()
    switches.set_switch(switch, True)
    assert switches.switches[switch] is True


@pytest.mark.parametrize(
    "key, value",
    [
        ("PX", 0.4),
        ("1S", 3),
        ("LO", 1000),
        ("PR", 1),
        ("TS", 1.5),
        ("A1", 122.0),
        ("M1", 1.0),
        ("IF", "tumours"),
        ("IN", "x4,8x"),
        ("X2", "bone"),
    ],
)
def test_valid_values_are_accepted(key, value):
    switches = RuntimeSwitches()
    switches.set_switch(key, value)
    assert switches.switches[key] == value
