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


def _write_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_simind.py"
    script.write_text(f"#!{sys.executable}\n{body}")
    script.chmod(0o755)
    return script


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


def test_executor_mpi_replicate_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation(
        "case01", runtime_switches={"NN": 2}, mpi_processes=4
    )

    assert calls[0][0] == [
        "mpirun",
        "-np",
        "4",
        "simind_mpi",
        "case01",
        "case01",
        "/NN:2",
    ]


def test_executor_mpi_split_command_appends_mp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation(
        "case01",
        orbit_file=Path("/tmp/case01_orbit.cor"),
        runtime_switches={"NN": 2},
        mpi_processes=4,
        split_projections=True,
    )

    assert calls[0][0] == [
        "mpirun",
        "-np",
        "4",
        "simind_mpi",
        "case01",
        "case01",
        "case01_orbit.cor",
        "/NN:2/MP",
    ]


def test_executor_mpi_binary_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMIND_MPI_BIN", "/opt/simind/simind_mpi")
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", mpi_processes=2)

    assert calls[0][0][3] == "/opt/simind/simind_mpi"


def test_executor_mpi_explicit_binary_beats_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIMIND_MPI_BIN", "/opt/simind/simind_mpi")
    calls = _capture_run(monkeypatch)

    SimindExecutor(mpi_executable="/custom/simind_mpi").run_simulation(
        "case01", mpi_processes=2
    )

    assert calls[0][0][3] == "/custom/simind_mpi"


def test_executor_mpi_uses_simind_mpi_next_to_custom_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIMIND_BIN", str(tmp_path / "bin" / "simind"))
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", mpi_processes=2)

    assert calls[0][0][3] == str(tmp_path / "bin" / "simind_mpi")


def test_executor_mpi_uses_sibling_of_relative_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIMIND_BIN", "./simind/simind")
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", mpi_processes=2, cwd=tmp_path / "out")

    assert calls[0][0][3] == str(tmp_path / "simind" / "simind_mpi")


def test_executor_mpi_uses_sibling_of_home_relative_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMIND_BIN", "~/simind/bin/simind")
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", mpi_processes=2)

    expected = (
        Path("~/simind/bin/simind").expanduser().absolute().with_name("simind_mpi")
    )
    assert calls[0][0][3] == str(expected)


def test_executor_mpi_accepts_space_containing_sibling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    simind_dir = tmp_path / "simind dir"
    monkeypatch.setenv("SIMIND_BIN", str(simind_dir / "simind"))
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", mpi_processes=2)

    assert calls[0][0][3] == str(simind_dir / "simind_mpi")


def test_executor_split_requires_mpi_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run(monkeypatch)
    with pytest.raises(SimulationError, match="mpi_processes"):
        SimindExecutor().run_simulation("case01", split_projections=True)


@pytest.mark.parametrize(
    "count", [1, 2, 4, 2.0, "4", np.int32(4), np.int64(4), np.float64(4.0)]
)
def test_executor_accepts_valid_mpi_process_counts(
    monkeypatch: pytest.MonkeyPatch, count: object
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", mpi_processes=count)

    assert calls[0][0][:3] == ["mpirun", "-np", str(int(count))]


def test_executor_accepts_large_integer_mpi_process_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)
    count = 10**400

    SimindExecutor().run_simulation("case01", mpi_processes=count)

    assert calls[0][0][:3] == ["mpirun", "-np", str(count)]


def test_executor_wraps_digit_string_count_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (
        hasattr(sys, "set_int_max_str_digits")
        and hasattr(sys, "get_int_max_str_digits")
    ):
        pytest.skip("sys.set_int_max_str_digits is unavailable")

    calls = _capture_run(monkeypatch)
    previous_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        with pytest.raises(SimulationError, match="mpi_processes"):
            SimindExecutor().run_simulation("case01", mpi_processes="9" * 700)
    finally:
        sys.set_int_max_str_digits(previous_limit)

    assert calls == []


def test_executor_huge_integral_fraction_count_does_not_leak_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An integral Fraction beyond the float range must not leak the
    # OverflowError that float coercion would raise; conversion failures become
    # SimulationError. The low digit limit makes the serialisation failure
    # deterministic instead of depending on the interpreter default.
    if not (
        hasattr(sys, "set_int_max_str_digits")
        and hasattr(sys, "get_int_max_str_digits")
    ):
        pytest.skip("sys.set_int_max_str_digits is unavailable")

    calls = _capture_run(monkeypatch)
    previous_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        with pytest.raises(SimulationError, match="mpi_processes"):
            SimindExecutor().run_simulation(
                "case01", mpi_processes=Fraction(10**700, 1)
            )
    finally:
        sys.set_int_max_str_digits(previous_limit)

    assert calls == []


@pytest.mark.parametrize(
    "count",
    [
        0,
        -1,
        1.5,
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        float("nan"),
        float("inf"),
        "2.5",
        "abc",
        "",
    ],
)
def test_executor_rejects_invalid_mpi_process_counts_before_launch(
    monkeypatch: pytest.MonkeyPatch, count: object
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="mpi_processes"):
        SimindExecutor().run_simulation("case01", mpi_processes=count)

    assert calls == []


def test_executor_renders_valueless_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation(
        "case01", runtime_switches={"IF": "case01.inp", "HO": True, "NN": 3}
    )

    assert calls[0][0][-1] == "/IF:case01.inp/HO/NN:3"


def test_executor_rejects_whitespace_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="whitespace"):
        SimindExecutor().run_simulation("bad prefix")

    assert calls == []


def test_executor_rejects_nul_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="NUL"):
        SimindExecutor().run_simulation("case\x00")

    assert calls == []


def test_executor_rejects_empty_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="empty"):
        SimindExecutor().run_simulation("")

    assert calls == []


@pytest.mark.parametrize("bad_prefix", ["hyphen-prefix", "case-", "case--01"])
def test_executor_rejects_hyphenated_prefixes_before_launch(
    monkeypatch: pytest.MonkeyPatch, bad_prefix: str
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="argument delimiter"):
        SimindExecutor().run_simulation(bad_prefix)

    assert calls == []


@pytest.mark.parametrize("good_prefix", ["case01", "case_01", "a", "0_case"])
def test_executor_accepts_safe_prefixes(
    monkeypatch: pytest.MonkeyPatch, good_prefix: str
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation(good_prefix)

    assert calls[0][0][1:3] == [good_prefix, good_prefix]


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


def test_executor_detects_divider_only_error_banner_on_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(monkeypatch, stdout="========================= Error\n")

    with pytest.raises(SimulationError, match="SIMIND reported an error:"):
        SimindExecutor().run_simulation("case01")


def test_executor_detects_divider_only_error_banner_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run_with_output(monkeypatch, stderr="=== Error\n")

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


def test_executor_replay_drops_nul_bytes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _capture_run_with_output(monkeypatch, stdout="\0\nbanner\n")

    SimindExecutor().run_simulation("case01")

    assert capsys.readouterr().out == "\nbanner\n"


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


def test_executor_real_subprocess_replays_both_streams_on_success(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    script = _write_script(
        tmp_path,
        "import sys\nprint('real stdout')\nprint('real stderr', file=sys.stderr)\n",
    )

    SimindExecutor(executable=script).run_simulation("case01", cwd=tmp_path)

    captured = capsys.readouterr()
    assert "real stdout" in captured.out
    assert "real stderr" in captured.err


def test_executor_real_subprocess_replays_both_streams_on_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    script = _write_script(
        tmp_path,
        "import sys\nprint('out before crash')\n"
        "print('err before crash', file=sys.stderr)\nsys.exit(3)\n",
    )

    with pytest.raises(SimulationError, match="SIMIND execution failed"):
        SimindExecutor(executable=script).run_simulation("case01", cwd=tmp_path)

    captured = capsys.readouterr()
    assert "out before crash" in captured.out
    assert "err before crash" in captured.err


def test_executor_real_subprocess_large_dual_stream_output_does_not_deadlock(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # Well past the OS pipe buffer on both streams, so capturing into pipes
    # without draining them would block the child.
    size = 200_000
    script = _write_script(
        tmp_path,
        "import sys\n"
        f"sys.stdout.write('o' * {size} + '\\n')\n"
        f"sys.stderr.write('e' * {size} + '\\n')\n",
    )

    SimindExecutor(executable=script).run_simulation("case01", cwd=tmp_path)

    captured = capsys.readouterr()
    assert "o" * size in captured.out
    assert "e" * size in captured.err


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
        # After the echoed comment (itself a fatal-looking comment).
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
    monkeypatch: pytest.MonkeyPatch, bad_prefix: object
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


@pytest.mark.unit
def test_executor_forwards_timeout(monkeypatch: pytest.MonkeyPatch):
    calls = _capture_run(monkeypatch)
    SimindExecutor().run_simulation("case01", timeout=5)
    assert calls[0][1]["timeout"] == 5


@pytest.mark.unit
def test_executor_timeout_stops_a_sleeping_process(tmp_path: Path):
    script = _write_script(tmp_path, "import time\ntime.sleep(30)\n")
    with pytest.raises(SimulationError, match="timed out after 0.2 seconds"):
        SimindExecutor(executable=script).run_simulation(
            "case01", cwd=tmp_path, timeout=0.2
        )


def test_executor_rejects_prefix_over_length_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match=r"at most 47 characters, got 48"):
        SimindExecutor().run_simulation("a" * 48)

    assert calls == []


@pytest.mark.parametrize("switch", ["NN", "RR", "CA"])
def test_executor_validates_direct_runtime_switches(
    monkeypatch: pytest.MonkeyPatch, switch: str
) -> None:
    # A direct executor call cannot bypass the public switch contract.
    invalid = {"NN": 2.0, "RR": "abc", "CA": 3}[switch]
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match=switch):
        SimindExecutor().run_simulation("case01", runtime_switches={switch: invalid})

    assert calls == []


def test_executor_direct_switches_preserve_bare_and_none_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation(
        "case01", runtime_switches={"HO": True, "RR": None, "CA": 1}
    )

    assert calls[0][0][-1] == "/HO/CA:1"


def test_executor_direct_ca_switch_retains_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", runtime_switches={"CA": 1})

    assert calls[0][0][-1] == "/CA:1"


def test_executor_serialize_switches_preserves_bare_none_and_mp() -> None:
    token = SimindExecutor._serialize_switches(
        {"IF": "case01.inp", "HO": True, "RR": None, "NN": 3},
        split_projections=True,
    )

    assert token == "/IF:case01.inp/HO/NN:3/MP"


def test_executor_serialize_switches_empty_returns_empty_token() -> None:
    assert SimindExecutor._serialize_switches(None) == ""
    assert SimindExecutor._serialize_switches({}) == ""


@pytest.mark.parametrize("switch", ["FI", "CC"])
@pytest.mark.parametrize(
    "value",
    ["bad name", "bad\tname", "bad\nname", "bad\rname", "bad\x00name"],
)
def test_executor_serialize_switches_rejects_whitespace_and_nul(
    switch: str, value: str
) -> None:
    with pytest.raises(SimulationError):
        SimindExecutor._serialize_switches({switch: value})


def test_executor_serialize_switches_rejects_invalid_value() -> None:
    # The helper folds the public switch contract into SimulationError, the
    # same way run_simulation did inline.
    with pytest.raises(SimulationError, match="NN"):
        SimindExecutor._serialize_switches({"NN": 0})
