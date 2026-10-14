"""Unit tests for the SIMIND process executor."""

import subprocess
import sys
from fractions import Fraction
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


def test_executor_resolves_relative_simind_bin_against_caller_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    simind_dir = tmp_path / "simind"
    simind_dir.mkdir()
    (simind_dir / "simind").write_text("#!/bin/sh\n")
    monkeypatch.setenv("SIMIND_BIN", "./simind/simind")
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", cwd=tmp_path / "elsewhere")

    assert calls[0][0][0] == str(simind_dir / "simind")


def test_executor_accepts_executable_path_containing_spaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    simind_dir = tmp_path / "simind dir"
    simind_dir.mkdir()
    (simind_dir / "simind").write_text("#!/bin/sh\n")
    calls = _capture_run(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    SimindExecutor(executable="./simind dir/simind").run_simulation(
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


def test_executor_defaults_to_path_lookup_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01")

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


def test_executor_rejects_whitespace_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="start with a lowercase letter"):
        SimindExecutor().run_simulation("bad prefix")

    assert calls == []


def test_executor_rejects_nul_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="start with a lowercase letter"):
        SimindExecutor().run_simulation("case\x00")

    assert calls == []


def test_executor_rejects_empty_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="start with a lowercase letter"):
        SimindExecutor().run_simulation("")

    assert calls == []


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


def test_executor_detects_exit_zero_error_banner_on_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(
        monkeypatch,
        stdout="simulation started\nSIMIND error number: 42\n",
    )

    with pytest.raises(SimulationError, match=r"error number 42"):
        SimindExecutor().run_simulation("case01")


def test_executor_detects_exit_zero_error_banner_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(
        monkeypatch,
        stderr="fatal condition\nSIMIND error number: 7\n",
    )

    with pytest.raises(SimulationError, match=r"error number 7"):
        SimindExecutor().run_simulation("case01")


def test_executor_does_not_match_diagnostic_split_across_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Neither stream contains the full numbered pattern on its own.
    _capture_run_with_output(
        monkeypatch,
        stdout="SIMIND",
        stderr=" error number: 1\n",
    )

    SimindExecutor().run_simulation("case01")


def test_executor_detects_error_banner_case_and_whitespace_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(
        monkeypatch,
        stdout="SiMiNd   ErRoR   NuMbEr :   123\n",
    )

    with pytest.raises(SimulationError, match=r"error number 123"):
        SimindExecutor().run_simulation("case01")


def test_executor_detects_divider_only_error_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(monkeypatch, stdout="========================= Error\n")

    with pytest.raises(SimulationError, match="SIMIND reported an error:"):
        SimindExecutor().run_simulation("case01")


def test_executor_ignores_lowercase_divider_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(monkeypatch, stdout="========================= error\n")

    SimindExecutor().run_simulation("case01")


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "relative error estimate: 0.01\n",
        "no errors were found\n",
        "Error: something benign happened\n",
        "=== Error handling summary ===\n",
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
        "SIMIND error number: 0",
        "SIMIND error number: 7",
        "========================= Error",
        "OTHER INFORMATION",
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


def test_executor_blank_comment_echo_under_banner_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A blank comment is echoed as a single space; it must not be read as a
    # divider or error line.
    _capture_run_with_output(
        monkeypatch,
        stdout=(
            " Simulation completed\n"
            " OTHER INFORMATION\n"
            " \n"
            " Compiled 2025:10:06 with intel Linux \n"
        ),
    )

    SimindExecutor().run_simulation("case01")


def test_executor_ignores_fatal_looking_comment_with_crlf_line_endings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(
        monkeypatch,
        stdout=(
            " Simulation completed\r\n"
            " OTHER INFORMATION\r\n"
            " SIMIND error number: 7\r\n"
            " Compiled 2025:10:06 with intel Linux\r\n"
        ),
    )

    SimindExecutor().run_simulation("case01")


@pytest.mark.parametrize(
    "stdout",
    [
        # Before the banner.
        "SIMIND error number: 7\n OTHER INFORMATION\n Compiled\n",
        "========================= Error\n OTHER INFORMATION\n Compiled\n",
        # After the echoed comment (which is itself a fatal-looking comment).
        " OTHER INFORMATION\n SIMIND error number: 0\n"
        " Compiled\nSIMIND error number: 7\n",
        " OTHER INFORMATION\n safe comment\n Compiled\n========== Error\n",
    ],
)
def test_executor_detects_diagnostics_outside_the_echoed_comment(
    monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    _capture_run_with_output(monkeypatch, stdout=stdout)

    with pytest.raises(SimulationError, match="SIMIND reported an error"):
        SimindExecutor().run_simulation("case01")


def test_executor_banner_lookalike_comment_does_not_hide_later_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(
        monkeypatch,
        stdout=(
            " Simulation completed\n"
            " OTHER INFORMATION\n"
            " OTHER INFORMATION\n"
            " Compiled 2025:10:06 with intel Linux \n"
            "SIMIND error number: 7\n"
        ),
    )

    with pytest.raises(SimulationError, match=r"error number 7"):
        SimindExecutor().run_simulation("case01")


def test_executor_error_number_without_banner_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(monkeypatch, stdout="fatal\nSIMIND error number: 7\n")

    with pytest.raises(SimulationError, match=r"error number 7"):
        SimindExecutor().run_simulation("case01")


def test_executor_error_number_zero_outside_echo_region_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Error number 0 is not ignored globally; only the echoed comment is exempt.
    _capture_run_with_output(monkeypatch, stdout="fatal\nSIMIND error number: 0\n")

    with pytest.raises(SimulationError, match=r"error number 0"):
        SimindExecutor().run_simulation("case01")


def test_executor_replays_raw_echoed_comment_without_redaction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _capture_run_with_output(
        monkeypatch,
        stdout=(
            " OTHER INFORMATION\n"
            " SIMIND error number: 0\n"
            " Compiled\n"
            "SIMIND error number: 7\n"
        ),
    )

    with pytest.raises(SimulationError, match=r"error number 7"):
        SimindExecutor().run_simulation("case01")

    assert " SIMIND error number: 0" in capsys.readouterr().out


def test_executor_echo_region_on_stdout_does_not_mask_stderr_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only stdout carries the banner and echo; stderr's genuinely numbered
    # diagnostic must still be reported.
    _capture_run_with_output(
        monkeypatch,
        stdout=" OTHER INFORMATION\n SIMIND error number: 0\n Compiled\n",
        stderr="SIMIND error number: 9\n",
    )

    with pytest.raises(SimulationError, match=r"error number 9"):
        SimindExecutor().run_simulation("case01")


def test_executor_does_not_match_echoed_diagnostic_split_across_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(
        monkeypatch,
        stdout=" OTHER INFORMATION\n SIMIND\n",
        stderr=" error number: 1\n",
    )

    SimindExecutor().run_simulation("case01")


def test_executor_error_after_echo_region_keeps_bounded_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (
        " OTHER INFORMATION\n"
        " safe comment\n" + ("x" * 3000) + "\nSIMIND error number: 42\n" + ("y" * 3000)
    )
    _capture_run_with_output(monkeypatch, stdout=output)

    with pytest.raises(SimulationError) as excinfo:
        SimindExecutor().run_simulation("case01")

    message = str(excinfo.value)
    assert "error number 42" in message
    assert "y" * 10 not in message
    excerpt = message.split("): ", 1)[1]
    assert len(excerpt) <= 2000


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


def test_executor_invalid_utf8_does_not_hide_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        kwargs["stdout"].buffer.write(b"SIMIND error number: 9\n\xff\xfe")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "simind_python_connector.core.executor.subprocess.run", fake_run
    )

    with pytest.raises(SimulationError, match=r"error number 9"):
        SimindExecutor().run_simulation("case01")


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


def test_executor_nonzero_exit_replays_diagnostics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _capture_run_with_output(monkeypatch, stderr="crash output\n", returncode=1)

    with pytest.raises(SimulationError, match="SIMIND execution failed"):
        SimindExecutor().run_simulation("case01")

    assert "crash output" in capsys.readouterr().err


def test_executor_forwards_timeout_to_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", timeout=5)

    assert calls[0][1]["timeout"] == 5


def test_executor_defaults_to_unlimited_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01")

    assert calls[0][1]["timeout"] is None


def test_executor_accepts_large_representable_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", timeout=10**6)

    assert calls[0][1]["timeout"] == 1e6


def test_executor_rejects_unrepresentable_timeout_without_launch(
    tmp_path: Path,
) -> None:
    # No subprocess.run mock: the real validation must reject the huge int
    # before subprocess.run computes its float deadline (which would overflow).
    with pytest.raises(
        SimulationError, match=r"timeout must be a finite number > 0 seconds"
    ):
        SimindExecutor(executable=sys.executable).run_simulation(
            "case01", cwd=tmp_path, timeout=10**400
        )


@pytest.mark.parametrize(
    "timeout",
    [True, False, "1", 0, 0.0, -1, -0.5, float("nan"), float("inf"), float("-inf")],
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


def test_executor_validate_timeout_returns_float_or_none() -> None:
    assert SimindExecutor.validate_timeout(None) is None

    validated = SimindExecutor.validate_timeout(5)
    assert validated == 5.0
    assert isinstance(validated, float)


@pytest.mark.parametrize(
    "timeout",
    [True, False, "1", 0, 0.0, -1, -0.5, float("nan"), float("inf"), float("-inf")],
)
def test_executor_validate_timeout_rejects_invalid_values(timeout) -> None:
    with pytest.raises(
        SimulationError, match=r"timeout must be a finite number > 0 seconds"
    ):
        SimindExecutor.validate_timeout(timeout)


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


@pytest.mark.parametrize("mp_value", [1, 2, 4, 2.0, "4", np.int32(4), np.int64(4)])
def test_executor_accepts_valid_mp(monkeypatch: pytest.MonkeyPatch, mp_value) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"MP": mp_value})

    command = calls[0][0]
    assert command[0:3] == ["mpirun", "-np", str(int(mp_value))]
    assert "/MP" in command


def test_executor_accepts_large_integer_mp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)
    large_mp = 10**400

    SimindExecutor().run_simulation("case01", runtime_switches={"MP": large_mp})

    assert calls[0][0][1:3] == ["-np", str(large_mp)]


@pytest.mark.parametrize(
    "mp_value",
    ["\u0664"],
)
def test_executor_rejects_unusable_digit_string_mp(
    monkeypatch: pytest.MonkeyPatch, mp_value
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="MP must be an integer >= 1"):
        SimindExecutor().run_simulation("case01", runtime_switches={"MP": mp_value})

    assert calls == []


def test_executor_normalises_integral_float_mp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"MP": 2.0})

    assert calls[0][0][1:3] == ["-np", "2"]


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
    [
        0,
        -1,
        2.5,
        None,
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        float("nan"),
        float("inf"),
        "",
        "4.0",
        "abc",
        "-2",
    ],
)
def test_executor_rejects_invalid_mp(monkeypatch: pytest.MonkeyPatch, mp_value) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="MP must be an integer >= 1"):
        SimindExecutor().run_simulation("case01", runtime_switches={"MP": mp_value})

    assert calls == []


def test_executor_mp_digit_string_matches_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"MP": 4})
    SimindExecutor().run_simulation("case01", runtime_switches={"MP": "4"})

    assert calls[0][0] == calls[1][0]


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


@pytest.mark.parametrize(
    "line",
    [
        "No   input file   found   for   the   spectra :   case01.isd",
        "nO InPuT FILE fOuNd FOR THE SPECTRA:CASE01.ISD",
        "No input file found for the spectra:\tcase01.isd\r",
    ],
)
def test_executor_matches_missing_spectra_case_and_whitespace(
    monkeypatch: pytest.MonkeyPatch, line: str
) -> None:
    _capture_run_with_output(monkeypatch, stdout=line + "\n")

    with pytest.raises(SimulationError, match="SIMIND reported an error"):
        SimindExecutor().run_simulation("case01")


def test_executor_missing_spectra_filename_survives_long_preamble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "x" * 3000 + "\nNo input file found for the spectra:spectra_123.isd\n"
    _capture_run_with_output(monkeypatch, stdout=output)

    with pytest.raises(SimulationError) as excinfo:
        SimindExecutor().run_simulation("case01")

    assert "spectra_123.isd" in str(excinfo.value)


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "No input file found for the spectra:case01.dat\n",
        "No input file found for the image:case01.isd\n",
        "No input file found for the spectra:.isd\n",
        "No input file found for the spectra:case01.isd extra\n",
        "no input files were found\n",
    ],
)
def test_executor_ignores_near_miss_missing_spectra(
    monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    _capture_run_with_output(monkeypatch, stdout=stdout)

    SimindExecutor().run_simulation("case01")


def test_executor_does_not_match_missing_spectra_split_across_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(
        monkeypatch,
        stdout="No input file found for the spectra",
        stderr=":case01.isd\n",
    )

    SimindExecutor().run_simulation("case01")


def test_executor_ignores_missing_spectra_echoed_in_config_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A valid run echoes the configured comment verbatim under OTHER
    # INFORMATION; a comment that spells the missing-spectra line must not be
    # mistaken for the fatal diagnostic.
    _capture_run_with_output(
        monkeypatch,
        stdout=(
            " Simulation completed\n"
            " OTHER INFORMATION\n"
            " No input file found for the spectra:test.isd\n"
            " Compiled 2025:10:06 with intel Linux \n"
        ),
    )

    SimindExecutor().run_simulation("case01")


def test_executor_missing_spectra_before_the_comment_echo_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(
        monkeypatch,
        stdout=" No  input file found for the spectra:case01.isd\n" + " " * 20 + "\n",
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


def test_executor_nonzero_exit_excerpts_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(
        monkeypatch, stdout="o" * 3000, stderr="e" * 3000, returncode=1
    )

    with pytest.raises(SimulationError) as excinfo:
        SimindExecutor().run_simulation("case01")

    message = str(excinfo.value)
    assert "o" * 2500 not in message
    assert "e" * 2500 not in message


@pytest.mark.parametrize(
    "bad_prefix",
    [123, None, Path("case01"), b"case01", ["case01"]],
)
def test_executor_rejects_non_string_output_prefix(
    monkeypatch: pytest.MonkeyPatch, bad_prefix
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="output_prefix must be a string"):
        SimindExecutor().run_simulation(bad_prefix)

    assert calls == []


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


@pytest.mark.parametrize(
    "nn_value", [1, 2, 2147483647, 2147483648, 10**400, np.int64(3)]
)
def test_executor_accepts_valid_integer_nn(
    monkeypatch: pytest.MonkeyPatch, nn_value
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"NN": nn_value})

    assert calls[0][0][-1] == f"/NN:{nn_value}"


@pytest.mark.parametrize("nn_value", [2.0, 2147483647.0, np.float64(2.0)])
def test_executor_normalises_integral_float_nn(
    monkeypatch: pytest.MonkeyPatch, nn_value
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"NN": nn_value})

    assert calls[0][0][-1] == f"/NN:{int(nn_value)}"


def test_executor_nn_integer_and_integral_float_serialise_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"NN": 2})
    SimindExecutor().run_simulation("case01", runtime_switches={"NN": 2.0})

    assert calls[0][0] == calls[1][0]
    assert calls[0][0][-1] == "/NN:2"


def test_executor_normalises_integral_rational_nn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"NN": Fraction(2, 1)})

    assert calls[0][0][-1] == "/NN:2"


@pytest.mark.parametrize("switch", ["MP", "NN"])
def test_executor_rejects_non_integral_rational_before_execution(
    monkeypatch: pytest.MonkeyPatch, switch
) -> None:
    calls = _capture_run(monkeypatch)
    # Exactly 9007199254740992.5, which a float round-trip would lose.
    value = Fraction(18014398509481985, 2)

    with pytest.raises(SimulationError, match=switch):
        SimindExecutor().run_simulation("case01", runtime_switches={switch: value})

    assert calls == []


@pytest.mark.parametrize(
    "nn_value",
    [
        2.5,
        0,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        "3",
        None,
        True,
        False,
        np.bool_(True),
    ],
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


@pytest.mark.parametrize("switch", ["FI", "CC", "RR"])
@pytest.mark.parametrize(
    "value",
    ["1 2", "bad name", "tab\tvalue", "line\nbreak", "nul\x00value"],
)
def test_executor_rejects_illegal_string_switch_before_execution(
    monkeypatch: pytest.MonkeyPatch, switch: str, value: str
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError):
        SimindExecutor().run_simulation("case01", runtime_switches={switch: value})

    assert calls == []


def test_executor_serialize_runtime_switches_preserves_order_and_mp_count() -> None:
    blob, mp_count = SimindExecutor._serialize_runtime_switches(
        {"NN": 2, "MP": 4, "RR": 12345}
    )

    assert blob == "/NN:2/MP/RR:12345"
    assert mp_count == "4"


def test_executor_serialize_runtime_switches_rejects_illegal_value() -> None:
    with pytest.raises(SimulationError, match="whitespace"):
        SimindExecutor._serialize_runtime_switches({"RR": "1 2"})
