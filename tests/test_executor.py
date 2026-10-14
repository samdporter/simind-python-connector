"""Unit tests for the SIMIND process executor."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simind_python_connector.core.executor import SimindExecutor
from simind_python_connector.core.types import SimulationError


pytestmark = pytest.mark.unit


def _capture_run(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls: list[tuple] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "simind_python_connector.core.executor.subprocess.run", fake_run
    )
    return calls


def _capture_run_with_output(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> list[tuple]:
    calls: list[tuple] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if stdout:
            kwargs["stdout"].write(stdout)
        if stderr:
            kwargs["stderr"].write(stderr)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "simind_python_connector.core.executor.subprocess.run", fake_run
    )
    return calls


def test_executor_uses_explicit_executable_cwd_and_non_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _capture_run(monkeypatch)
    executable = tmp_path / "custom-simind"

    SimindExecutor(executable=executable).run_simulation("case01", cwd=tmp_path)

    command, kwargs = calls[0]
    assert command[0] == str(executable)
    assert kwargs["cwd"] == tmp_path
    assert kwargs["shell"] is False
    assert kwargs["check"] is True


def test_executor_prefers_simind_bin_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMIND_BIN", "/opt/custom/simind")
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01")

    assert calls[0][0][0] == "/opt/custom/simind"


def test_executor_resolves_relative_executable_against_caller_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    simind_dir = tmp_path / "simind"
    simind_dir.mkdir()
    (simind_dir / "simind").write_text("#!/bin/sh\n")
    calls = _capture_run(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    SimindExecutor(executable="./simind/simind").run_simulation(
        "case01", cwd=output_dir
    )

    command, kwargs = calls[0]
    assert command[0] == str(simind_dir / "simind")
    assert kwargs["cwd"] == output_dir


def test_executor_keeps_bare_executable_for_path_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", cwd=tmp_path)

    assert calls[0][0][0] == "simind"


def test_executor_plain_command_includes_orbit_and_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation(
        "case01",
        orbit_file=Path("/tmp/case01.cor"),
        runtime_switches={"NN": 2, "RR": 12345},
    )

    command = calls[0][0]
    assert command[:3] == ["simind", "case01", "case01"]
    assert command[3] == "case01.cor"
    assert command[4] == "/NN:2/RR:12345"


def test_executor_mpi_command_uses_flag_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation(
        "case01",
        runtime_switches={"MP": 4, "NN": 2},
    )

    assert calls[0][0] == [
        "mpirun",
        "-np",
        "4",
        "simind_mpi",
        "case01",
        "case01",
        "/MP/NN:2",
    ]


def test_executor_mpi_uses_simind_mpi_next_to_custom_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIMIND_BIN", str(tmp_path / "bin" / "simind"))
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"MP": 2})

    assert calls[0][0][3] == str(tmp_path / "bin" / "simind_mpi")


def test_executor_mpi_prefers_simind_mpi_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIMIND_MPI_BIN", str(tmp_path / "mpi" / "simind_mpi"))
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"MP": 2})

    assert calls[0][0][3] == str(tmp_path / "mpi" / "simind_mpi")


@pytest.mark.parametrize(
    "bad_prefix",
    [
        "../escape",
        "/absolute",
        "a/b",
        "a\\b",
        "..",
        "case.01",
        "case[1]",
        "Case01",
        "-case",
        "cx-run",
        "case-",
        "case--01",
        "bad prefix",
        "case\x00",
        "",
        123,
        None,
    ],
)
def test_executor_rejects_unsafe_output_prefixes(
    monkeypatch: pytest.MonkeyPatch, bad_prefix: str
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="output_prefix"):
        SimindExecutor().run_simulation(bad_prefix)

    assert calls == []


@pytest.mark.parametrize("good_prefix", ["case01", "case_01", "a", "0_case"])
def test_executor_accepts_safe_output_prefixes(
    monkeypatch: pytest.MonkeyPatch, good_prefix: str
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation(good_prefix)

    assert calls[0][0][1] == good_prefix


def test_executor_translates_os_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_oserror(command, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(
        "simind_python_connector.core.executor.subprocess.run", raise_oserror
    )

    with pytest.raises(SimulationError, match="Unable to execute"):
        SimindExecutor().run_simulation("case01")


def test_executor_translates_called_process_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_called_process_error(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(
        "simind_python_connector.core.executor.subprocess.run",
        raise_called_process_error,
    )

    with pytest.raises(SimulationError, match="failed"):
        SimindExecutor().run_simulation("case01")


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_executor_detects_exit_zero_error_number(
    monkeypatch: pytest.MonkeyPatch, stream: str
) -> None:
    _capture_run_with_output(
        monkeypatch, **{stream: "simulation started\nSIMIND error number: 42\n"}
    )

    with pytest.raises(SimulationError, match=r"error number 42"):
        SimindExecutor().run_simulation("case01")


def test_executor_detects_divider_only_error_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(monkeypatch, stdout="========================= Error\n")

    with pytest.raises(SimulationError, match="SIMIND reported an error:"):
        SimindExecutor().run_simulation("case01")


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "relative error estimate: 0.01\n",
        "no errors were found\n",
        "Error: something benign happened\n",
    ],
)
def test_executor_ignores_benign_error_words(
    monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    _capture_run_with_output(monkeypatch, stdout=stdout)

    SimindExecutor().run_simulation("case01")


@pytest.mark.parametrize(
    "comment",
    [
        "SIMIND error number: 7",
        "No input file found for the spectra:test.isd",
    ],
)
def test_executor_ignores_fatal_looking_comment_echoed_under_banner(
    monkeypatch: pytest.MonkeyPatch, comment: str
) -> None:
    # SIMIND echoes the configured comment verbatim on the single line after
    # OTHER INFORMATION; a comment that spells a diagnostic must not fail a
    # successful run.
    _capture_run_with_output(
        monkeypatch,
        stdout=(
            " Simulation completed\n"
            " OTHER INFORMATION\n"
            f" {comment}\n"
            " Compiled 2025:10:06 with intel Linux \n"
        ),
    )

    SimindExecutor().run_simulation("case01")


@pytest.mark.parametrize(
    "stdout",
    [
        "SIMIND error number: 7\n OTHER INFORMATION\n Compiled\n",
        " OTHER INFORMATION\n safe comment\n Compiled\n========== Error\n",
    ],
)
def test_executor_detects_diagnostics_outside_the_echoed_comment(
    monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    _capture_run_with_output(monkeypatch, stdout=stdout)

    with pytest.raises(SimulationError, match="SIMIND reported an error"):
        SimindExecutor().run_simulation("case01")


def test_executor_replays_captured_diagnostics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _capture_run_with_output(
        monkeypatch, stdout="simind stdout\n", stderr="simind stderr\n"
    )

    SimindExecutor().run_simulation("case01")

    captured = capsys.readouterr()
    assert "simind stdout" in captured.out
    assert "simind stderr" in captured.err


def test_executor_error_excerpt_is_bounded_but_keeps_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = ("x" * 3000) + "\nSIMIND error number: 123\n" + ("y" * 3000)
    _capture_run_with_output(monkeypatch, stdout=output)

    with pytest.raises(SimulationError) as excinfo:
        SimindExecutor().run_simulation("case01")

    message = str(excinfo.value)
    assert "error number 123" in message
    assert "y" * 10 not in message
    excerpt = message.split("): ", 1)[1]
    assert len(excerpt) <= 2000


def test_executor_forwards_timeout_to_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", timeout=5)

    assert calls[0][1]["timeout"] == 5


@pytest.mark.parametrize(
    "timeout",
    [True, "1", 0, -1, float("nan"), float("inf")],
)
def test_executor_rejects_invalid_timeouts(
    monkeypatch: pytest.MonkeyPatch, timeout
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(
        SimulationError, match=r"timeout must be a finite number > 0 seconds"
    ):
        SimindExecutor().run_simulation("case01", timeout=timeout)

    assert calls == []


def test_executor_translates_timeout_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    monkeypatch.setattr(
        "simind_python_connector.core.executor.subprocess.run", raise_timeout
    )

    with pytest.raises(SimulationError, match="timed out after 3 seconds") as excinfo:
        SimindExecutor().run_simulation("case01", timeout=3)

    assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)


def test_executor_real_subprocess_timeout_raises_promptly(tmp_path: Path) -> None:
    # The prefix names the script the interpreter runs, so it must be valid.
    (tmp_path / "case01").write_text("import time\ntime.sleep(30)\n")

    with pytest.raises(SimulationError, match="timed out after 0.2 seconds"):
        SimindExecutor(executable=sys.executable).run_simulation(
            "case01", cwd=tmp_path, timeout=0.2
        )


@pytest.mark.parametrize("mp_value", [1, 4, "4"])
def test_executor_accepts_valid_mp(monkeypatch: pytest.MonkeyPatch, mp_value) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"MP": mp_value})

    command = calls[0][0]
    assert command[0:3] == ["mpirun", "-np", str(int(mp_value))]
    assert "/MP" in command


def test_executor_keeps_mp_switch_flag_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"MP": 4, "NN": 2})

    command = calls[0][0]
    assert command[-1] == "/MP/NN:2"
    assert "/MP:4" not in command


@pytest.mark.parametrize(
    "mp_value",
    [0, -1, 2.5, None, True, "abc"],
)
def test_executor_rejects_invalid_mp(monkeypatch: pytest.MonkeyPatch, mp_value) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="MP must be an integer >= 1"):
        SimindExecutor().run_simulation("case01", runtime_switches={"MP": mp_value})

    assert calls == []


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_executor_detects_missing_spectra_input(
    monkeypatch: pytest.MonkeyPatch, stream: str
) -> None:
    _capture_run_with_output(
        monkeypatch, **{stream: "No input file found for the spectra:case01.isd\n"}
    )

    with pytest.raises(SimulationError, match="SIMIND reported an error") as excinfo:
        SimindExecutor().run_simulation("case01")

    assert "case01.isd" in str(excinfo.value)


def test_executor_nonzero_exit_includes_labelled_stream_excerpts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(
        monkeypatch,
        stdout="stdout diagnostic\n",
        stderr="stderr diagnostic\n",
        returncode=1,
    )

    with pytest.raises(SimulationError) as excinfo:
        SimindExecutor().run_simulation("case01")

    message = str(excinfo.value)
    assert "SIMIND execution failed" in message
    assert "returned non-zero exit status 1" in message
    assert "stdout:" in message
    assert "stdout diagnostic" in message
    assert "stderr:" in message
    assert "stderr diagnostic" in message
    assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)


def test_executor_accepts_prefix_at_length_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)
    prefix = "a" * 47

    SimindExecutor().run_simulation(prefix)

    assert calls[0][0][1] == prefix


def test_executor_rejects_prefix_over_length_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match=r"at most 47 characters, got 48"):
        SimindExecutor().run_simulation("a" * 48)

    assert calls == []


@pytest.mark.parametrize("nn_value", [1, 2, np.int64(3)])
def test_executor_accepts_valid_integer_nn(
    monkeypatch: pytest.MonkeyPatch, nn_value
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"NN": nn_value})

    assert calls[0][0][-1] == f"/NN:{nn_value}"


@pytest.mark.parametrize(
    "nn_value",
    [2.5, 0, -1, "3", None, True],
)
def test_executor_rejects_invalid_nn(monkeypatch: pytest.MonkeyPatch, nn_value) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="NN"):
        SimindExecutor().run_simulation("case01", runtime_switches={"NN": nn_value})

    assert calls == []


@pytest.mark.parametrize(
    "runtime_switches",
    [{"RR": None}, {"RR": True}, {"PX": False}],
)
def test_executor_rejects_none_or_bool_switch_value(
    monkeypatch: pytest.MonkeyPatch, runtime_switches
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="requires a value"):
        SimindExecutor().run_simulation("case01", runtime_switches=runtime_switches)

    assert calls == []


@pytest.mark.parametrize("value", ["bad name", "line\nbreak", "nul\x00value"])
def test_executor_rejects_illegal_string_switch_before_execution(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError):
        SimindExecutor().run_simulation("case01", runtime_switches={"FI": value})

    assert calls == []


def test_executor_serialize_runtime_switches_preserves_order_and_mp_count() -> None:
    blob, mp_count = SimindExecutor._serialize_runtime_switches(
        {"NN": 2, "MP": 4, "RR": 12345}
    )

    assert blob == "/NN:2/MP/RR:12345"
    assert mp_count == "4"
