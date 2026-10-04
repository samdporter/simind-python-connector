"""Tests for example-runner scripts and container helper scripts."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.unit

VALIDATION_SCRIPT = ROOT / "scripts" / "run_container_validation.sh"


def test_run_all_examples_returns_nonzero_when_an_example_fails(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_all_examples", ROOT / "scripts" / "run_all_examples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "EXAMPLES",
        [
            {
                "name": "01_basic_simulation.py",
                "description": "d",
                "module": "m",
                "estimated_time": "1 minute",
            },
        ],
    )
    monkeypatch.setattr(module, "check_dependencies", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    monkeypatch.setattr(
        module,
        "run_example_subprocess",
        lambda example_file: (False, "Exit code 1: boom"),
    )

    assert module.main() == 1


def test_run_all_examples_returns_zero_when_all_succeed(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_all_examples", ROOT / "scripts" / "run_all_examples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "EXAMPLES",
        [
            {
                "name": "01_basic_simulation.py",
                "description": "d",
                "module": "m",
                "estimated_time": "1 minute",
            },
        ],
    )
    monkeypatch.setattr(module, "check_dependencies", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    monkeypatch.setattr(
        module,
        "run_example_subprocess",
        lambda example_file: (True, "Completed successfully"),
    )

    assert module.main() == 0


@pytest.fixture
def fake_bin(tmp_path: Path):
    """Directory holding fake python3/simind executables recording argv."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    argv_log = tmp_path / "argv.log"

    python3 = bin_dir / "python3"
    python3.write_text(f'#!/bin/sh\necho "$@" >> {argv_log}\nexit 0\n')
    for executable in (python3,):
        executable.chmod(executable.stat().st_mode | stat.S_IEXEC)

    simind = bin_dir / "simind"
    simind.write_text("#!/bin/sh\nexit 0\n")
    simind.chmod(simind.stat().st_mode | stat.S_IEXEC)

    smc_dir = tmp_path / "smc_dir"
    smc_dir.mkdir()

    return {"bin": bin_dir, "log": argv_log, "smc_dir": smc_dir}


def test_container_validation_script_rejects_no_build_without_image(
    tmp_path: Path, fake_bin
):
    """--no-build must fail fast when the selected image is missing."""
    docker_bin = fake_bin["bin"] / "docker"
    docker_bin.write_text(
        "#!/bin/sh\ncase \"$1 $2\" in *'image inspect'*) exit 1;; esac\nexit 0\n"
    )
    docker_bin.chmod(docker_bin.stat().st_mode | stat.S_IEXEC)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin['bin']}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_container_validation.sh"),
            "--only-core",
            "--no-build",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )

    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize(
    "script_name",
    ["run_container_validation.sh", "run_container_examples.sh"],
)
def test_container_scripts_forward_smc_env_without_compose_no_build(
    tmp_path: Path, script_name: str
):
    """--no-build is a script-level flag only, and custom SIMIND data
    locations must reach SIMIND via SMC_DIR/SIMIND_DATA_DIR too."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "docker").mkdir()
    shutil.copy(ROOT / "scripts" / script_name, repo / "scripts" / script_name)
    shutil.copy(ROOT / "docker" / "compose.yaml", repo / "docker" / "compose.yaml")

    repo_simind = repo / "simind"
    repo_simind.mkdir()
    executable = repo_simind / "simind"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_log = tmp_path / "argv.log"
    docker = bin_dir / "docker"
    docker.write_text(f'#!/bin/sh\necho "$@" >> {argv_log}\nexit 0\n')
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(repo / "scripts" / script_name), "--only-core", "--no-build"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    recorded = argv_log.read_text().splitlines()
    assert any("image inspect" in line for line in recorded)
    assert "--no-build" not in argv_log.read_text().split()

    compose_runs = [
        line for line in recorded if line.startswith("compose ") and " run " in line
    ]
    assert compose_runs, recorded
    expected_env = {
        "SIMIND_BIN": "/workspace/simind/simind",
        "SIMIND_SMC_DIR": "/workspace/simind/smc_dir/",
        "SIMIND_DATA_DIR": "/workspace/simind/smc_dir/",
        "SMC_DIR": "/workspace/simind/smc_dir/",
    }
    if script_name == "run_container_validation.sh":
        expected_env["SIMIND_MPI_BIN"] = "/workspace/simind/simind_mpi"
    for line in compose_runs:
        tokens = line.split()
        env_pairs = {
            tokens[i + 1].split("=", 1)[0]: tokens[i + 1].split("=", 1)[1]
            for i, token in enumerate(tokens)
            if token == "-e"
        }
        assert env_pairs == expected_env, line


def test_dockerignore_excludes_local_heavy_paths():
    ignore = (ROOT / ".dockerignore").read_text().splitlines()
    for pattern in (
        ".git",
        ".venv",
        "simind",
        "dist",
        "build",
        "*.tar.gz",
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        "coverage.xml",
        "docs/_build",
        "output",
        "results",
    ):
        assert pattern in ignore, pattern
    # Packaged preset must never be excluded
    assert all(not line.endswith("input.smc") for line in ignore)


@pytest.mark.unit
def test_validation_script_require_simind_fails_without_binary(
    tmp_path: Path, fake_bin
):
    # Bare PATH without a fake simind so the runner must treat it as absent.
    bare_bin = tmp_path / "bare-bin"
    bare_bin.mkdir()

    env = dict(os.environ)
    env["PATH"] = f"{bare_bin}:{env['PATH']}"
    env.pop("SMC_DIR", None)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_container_examples.sh"),
            "--only-core",
            "--require-simind",
            "--simind-path",
            str(tmp_path / "does-not-exist" / "simind"),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert result.returncode == 1, result.stdout + result.stderr


def test_validation_script_forwards_simind_mpi_bin_in_run_service():
    script = VALIDATION_SCRIPT.read_text()
    assert '-e "SIMIND_MPI_BIN=$(dirname "$container_bin")/simind_mpi"' in script


def test_validation_script_requires_sibling_simind_mpi_for_simind_tests():
    script = VALIDATION_SCRIPT.read_text()
    assert (
        '[[ "$RUN_SIMIND_TESTS" == "1" && "$SIMIND_AVAILABLE" -eq 1 ]]; then' in script
    )
    assert 'SIMIND_MPI_PATH="$(dirname "$SIMIND_PATH")/simind_mpi"' in script
    assert '[[ ! -x "$SIMIND_MPI_PATH" ]]' in script


def test_validation_script_runs_simind_mpi_test_in_python_service():
    script = VALIDATION_SCRIPT.read_text()
    assert (
        'run_service python "python -m pytest tests/test_simind_mpi.py -q -s"' in script
    )


def test_validation_script_sirf_simind_selection_ignores_simind_mpi_test():
    script = VALIDATION_SCRIPT.read_text()
    sirf_line = next(
        line
        for line in script.splitlines()
        if "run_service sirf" in line and "SIMIND_MARKERS" in line
    )
    assert "--ignore=tests/test_simind_mpi.py" in sirf_line


def test_validation_script_fails_without_sibling_simind_mpi(tmp_path):
    """Missing sibling simind_mpi is a hard failure, not a skip."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "docker").mkdir()
    shutil.copy(VALIDATION_SCRIPT, repo / "scripts" / "run_container_validation.sh")
    shutil.copy(ROOT / "docker" / "compose.yaml", repo / "docker" / "compose.yaml")

    repo_simind = repo / "simind"
    repo_simind.mkdir()
    executable = repo_simind / "simind"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    # No repo/simind/simind_mpi sibling.

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Fail loudly if docker is ever reached; the script must stop before that.
    docker = bin_dir / "docker"
    docker.write_text("#!/bin/sh\necho 'docker must not be called' >&2\nexit 99\n")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "run_container_validation.sh"),
            "--only-core",
            "--with-simind",
            "--require-simind",
            "--no-build",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "simind_mpi" in result.stderr
