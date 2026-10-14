"""Unit tests for the SIMIND process executor."""

import subprocess
import sys
from pathlib import Path

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


def test_executor_split_requires_mpi_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run(monkeypatch)
    with pytest.raises(SimulationError, match="mpi_processes"):
        SimindExecutor().run_simulation("case01", split_projections=True)


@pytest.mark.parametrize("count", [1, 4, "4"])
def test_executor_accepts_valid_mpi_process_counts(
    monkeypatch: pytest.MonkeyPatch, count: object
) -> None:
    monkeypatch.delenv("SIMIND_BIN", raising=False)
    monkeypatch.delenv("SIMIND_MPI_BIN", raising=False)
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation("case01", mpi_processes=count)

    assert calls[0][0][:3] == ["mpirun", "-np", str(int(count))]


@pytest.mark.parametrize(
    "count",
    [0, -1, 1.5, True, "abc"],
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


@pytest.mark.parametrize(
    "bad_prefix",
    ["hyphen-prefix", "case--01", "bad prefix", "case\x00", "", 123, None],
)
def test_executor_rejects_unsafe_output_prefixes(
    monkeypatch: pytest.MonkeyPatch, bad_prefix: object
) -> None:
    calls = _capture_run(monkeypatch)

    with pytest.raises(SimulationError):
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


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_executor_detects_exit_zero_error_number(
    monkeypatch: pytest.MonkeyPatch, stream: str
) -> None:
    _capture_run_with_output(
        monkeypatch, **{stream: "simulation started\nSIMIND error number: 42\n"}
    )

    with pytest.raises(SimulationError, match=r"error number 42"):
        SimindExecutor().run_simulation("case01")


def test_executor_detects_divider_only_error_banner_on_stdout(
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


def test_executor_serialize_switches_preserves_bare_none_and_mp() -> None:
    token = SimindExecutor._serialize_switches(
        {"IF": "case01.inp", "HO": True, "RR": None, "NN": 3},
        split_projections=True,
    )

    assert token == "/IF:case01.inp/HO/NN:3/MP"
