"""Unit tests for the SIMIND process executor."""

import subprocess
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
    calls = _capture_run(monkeypatch)

    SimindExecutor().run_simulation(
        "case01",
        runtime_switches={"MP": 4, "NN": 2},
    )

    command = calls[0][0]
    assert command[:6] == ["mpirun", "-np", "4", "simind", "case01", "case01"]
    assert command[-2:] == ["-p", "/MP/NN:2"]
    assert "/MP:4" not in command[-1]


def test_executor_rejects_whitespace_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="whitespace"):
        SimindExecutor().run_simulation("bad prefix")


def test_executor_rejects_nul_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="NUL"):
        SimindExecutor().run_simulation("case\x00")


def test_executor_rejects_empty_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_run(monkeypatch)

    with pytest.raises(SimulationError, match="empty"):
        SimindExecutor().run_simulation("")


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
