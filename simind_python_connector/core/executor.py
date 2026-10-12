"""Minimal SIMIND process runner used by connector-first APIs."""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

from .config import _validate_switch_value
from .types import OUTPUT_PREFIX_PATTERN, SimulationError


_SIMIND_ERROR_NUMBER_PATTERN = re.compile(
    r"\bSIMIND\s+error\s+number\s*:\s*(\d+)\b", re.IGNORECASE
)
_SIMIND_ERROR_DIVIDER_PATTERN = re.compile(
    r"^[ \t]*={3,}[ \t]*Error[ \t]*$", re.MULTILINE
)
_ERROR_EXCERPT_LIMIT = 2000


class SimindExecutor:
    """Run SIMIND as a subprocess with optional MPI switch support."""

    def __init__(self, executable: Union[str, Path, None] = None) -> None:
        self.logger = logging.getLogger(__name__)
        if executable is not None:
            self.executable = str(executable)
        else:
            self.executable = os.environ.get("SIMIND_BIN") or "simind"

    def run_simulation(
        self,
        output_prefix: str,
        orbit_file: Optional[Path] = None,
        runtime_switches: Optional[Dict] = None,
        cwd: Union[str, Path, None] = None,
        *,
        timeout: Union[int, float, None] = None,
    ) -> None:
        validated_timeout = self.validate_timeout(timeout)

        mp_present = False
        mp_value = None
        if runtime_switches:
            for key, value in runtime_switches.items():
                if str(key).lower() == "mp":
                    mp_present = True
                    mp_value = value
                    break

        # shell=False passes the executable to execve as argv[0], so spaces
        # in its path are safe; other tokens must stay whitespace-free.
        validated_simind = self._validate_cli_token(
            self._resolve_executable(), allow_whitespace=True
        )
        validated_output_prefix = self._validate_cli_token(output_prefix)
        if not re.fullmatch(OUTPUT_PREFIX_PATTERN, validated_output_prefix):
            raise SimulationError(
                "output_prefix must start with a lowercase letter or digit and "
                "contain only lowercase letters, digits and '_'; a hyphen is an "
                "argument delimiter in SIMIND V8 and would split the positional "
                f"filename arguments, got {output_prefix!r}"
            )
        validated_orbit_name = (
            self._validate_cli_token(orbit_file.name) if orbit_file else None
        )
        validated_mp_value = self._validate_mp_count(mp_value) if mp_present else None

        validated_switch_blob = None
        if runtime_switches:
            switch_parts = []
            for key, value in runtime_switches.items():
                if str(key).upper() == "MP":
                    switch_parts.append(f"/{key}")
                else:
                    switch_parts.append(f"/{key}:{value}")
            if switch_parts:
                validated_switch_blob = self._validate_cli_token("".join(switch_parts))

        if validated_mp_value is not None:
            # /MP splits the projections over the MPI processes, which only
            # the separate simind_mpi binary supports.
            command = [
                "mpirun",
                "-np",
                validated_mp_value,
                self._mpi_executable(validated_simind),
                validated_output_prefix,
                validated_output_prefix,
            ]
            if validated_orbit_name is not None:
                command.append(validated_orbit_name)
            if validated_switch_blob is not None:
                command.append(validated_switch_blob)
        else:
            command = [
                validated_simind,
                validated_output_prefix,
                validated_output_prefix,
            ]
            if validated_orbit_name is not None:
                command.append(validated_orbit_name)
            if validated_switch_blob is not None:
                command.append(validated_switch_blob)

        self.logger.info("Running SIMIND: %s", " ".join(command))
        with (
            tempfile.TemporaryFile(
                mode="w+", encoding="utf-8", errors="replace"
            ) as stdout_file,
            tempfile.TemporaryFile(
                mode="w+", encoding="utf-8", errors="replace"
            ) as stderr_file,
        ):
            try:
                subprocess.run(
                    command,
                    check=True,
                    shell=False,
                    cwd=cwd,
                    timeout=validated_timeout,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
            except OSError as exc:
                self._read_and_replay(stdout_file, stderr_file)
                raise SimulationError(
                    f"Unable to execute SIMIND command: {exc}"
                ) from exc
            except subprocess.CalledProcessError as exc:
                self._read_and_replay(stdout_file, stderr_file)
                raise SimulationError(f"SIMIND execution failed: {exc}") from exc
            except subprocess.TimeoutExpired as exc:
                self._read_and_replay(stdout_file, stderr_file)
                raise SimulationError(
                    f"SIMIND execution timed out after {timeout} seconds."
                ) from exc
            stdout_text, stderr_text = self._read_and_replay(stdout_file, stderr_file)

        # Each stream is checked on its own: concatenating them could match a
        # diagnostic split across the boundary that neither stream contains.
        self._raise_for_reported_error(stdout_text)
        self._raise_for_reported_error(stderr_text)

    @staticmethod
    def validate_timeout(
        timeout: Union[int, float, None],
    ) -> Optional[float]:
        """Return ``timeout`` as a float, rejecting unsafe values.

        ``timeout`` must be a finite number > 0 seconds; ``None`` passes
        through as ``None``.
        """
        if timeout is None:
            return None
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise SimulationError(
                f"timeout must be a finite number > 0 seconds, got {timeout!r}"
            )
        try:
            # subprocess computes a float deadline, so an int it cannot
            # represent must be rejected before the launch.
            validated_timeout = float(timeout)
        except OverflowError as exc:
            raise SimulationError(
                f"timeout must be a finite number > 0 seconds, got {timeout!r}"
            ) from exc
        if not math.isfinite(validated_timeout) or validated_timeout <= 0:
            raise SimulationError(
                f"timeout must be a finite number > 0 seconds, got {timeout!r}"
            )
        return validated_timeout

    @staticmethod
    def _read_and_replay(stdout_file, stderr_file) -> tuple[str, str]:
        stdout_file.seek(0)
        stdout_text = stdout_file.read()
        stderr_file.seek(0)
        stderr_text = stderr_file.read()
        sys.stdout.write(stdout_text)
        sys.stderr.write(stderr_text)
        return stdout_text, stderr_text

    @staticmethod
    def _raise_for_reported_error(output: str) -> None:
        number_match = _SIMIND_ERROR_NUMBER_PATTERN.search(output)
        if number_match:
            excerpt = output[:_ERROR_EXCERPT_LIMIT]
            raise SimulationError(
                f"SIMIND reported an error (error number "
                f"{number_match.group(1)}): {excerpt}"
            )
        if _SIMIND_ERROR_DIVIDER_PATTERN.search(output):
            excerpt = output[:_ERROR_EXCERPT_LIMIT]
            raise SimulationError(f"SIMIND reported an error: {excerpt}")

    @staticmethod
    def _validate_mp_count(value: object) -> str:
        candidate = value
        if isinstance(value, str):
            if not (value.isascii() and value.isdigit()):
                raise SimulationError(f"MP must be an integer >= 1, got {value!r}")
            try:
                candidate = int(value)
            except ValueError as exc:
                raise SimulationError(
                    f"MP must be an integer >= 1, got {value!r}"
                ) from exc
        try:
            _validate_switch_value("MP", candidate)
            return str(int(candidate))
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

    def _resolve_executable(self) -> str:
        """Make path-bearing executables absolute before ``cwd`` applies.

        subprocess resolves relative paths against the child ``cwd`` (the
        output directory), so they must be anchored to the caller's working
        directory here. Bare names keep their PATH-lookup semantics.
        """
        executable = self.executable
        if os.sep in executable or (os.altsep and os.altsep in executable):
            return str(Path(executable).expanduser().absolute())
        return executable

    @staticmethod
    def _mpi_executable(executable: str) -> str:
        """$SIMIND_MPI_BIN, else simind_mpi next to a path-given serial binary."""
        mpi_executable = os.environ.get("SIMIND_MPI_BIN")
        if mpi_executable:
            return str(Path(mpi_executable).expanduser().absolute())
        if os.sep in executable or (os.altsep and os.altsep in executable):
            return str(Path(executable).with_name("simind_mpi"))
        return "simind_mpi"

    @staticmethod
    def _validate_cli_token(value: object, *, allow_whitespace: bool = False) -> str:
        """Validate command tokens before subprocess invocation.

        This executor always runs with ``shell=False``, and each token is
        validated to reject empty values, NUL bytes, and whitespace.
        Executables are exempt from the whitespace check because they are
        passed as ``argv[0]``.
        """
        token = str(value)
        if not token:
            raise SimulationError("Encountered empty command token for SIMIND call.")
        if "\x00" in token:
            raise SimulationError("SIMIND command token contains NUL byte.")
        if not allow_whitespace and any(char.isspace() for char in token):
            raise SimulationError(
                f"SIMIND command token contains whitespace: {token!r}"
            )
        return token
