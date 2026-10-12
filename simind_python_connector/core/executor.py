"""Minimal SIMIND process runner used by connector-first APIs."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

from .types import OUTPUT_PREFIX_PATTERN, SimulationError


_SIMIND_ERROR_NUMBER_PATTERN = re.compile(
    r"\bSIMIND\s+error\s+number\s*:\s*(\d+)\b", re.IGNORECASE
)
_SIMIND_ERROR_DIVIDER_PATTERN = re.compile(
    r"^[ \t]*={3,}[ \t]*Error[ \t]*$", re.MULTILINE
)
_ERROR_EXCERPT_LIMIT = 2000


class SimindExecutor:
    """Run SIMIND as a subprocess, optionally under MPI with simind_mpi."""

    def __init__(
        self,
        executable: Union[str, Path, None] = None,
        mpi_executable: Union[str, Path, None] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        if executable is not None:
            self.executable = str(executable)
        else:
            self.executable = os.environ.get("SIMIND_BIN") or "simind"
        if mpi_executable is not None:
            self.mpi_executable = str(mpi_executable)
        else:
            self.mpi_executable = os.environ.get("SIMIND_MPI_BIN") or None

    def run_simulation(
        self,
        output_prefix: str,
        orbit_file: Optional[Path] = None,
        runtime_switches: Optional[Dict] = None,
        cwd: Union[str, Path, None] = None,
        mpi_processes: Optional[Union[int, float, str]] = None,
        split_projections: bool = False,
    ) -> None:
        prefix = self._validate_cli_token(output_prefix)
        if not re.fullmatch(OUTPUT_PREFIX_PATTERN, prefix):
            raise SimulationError(
                f"output_prefix must contain only lowercase letters, digits and "
                f"'_', never '-' (SIMIND treats '-' as an argument delimiter, "
                f"which would split the filename arguments), got {prefix!r}"
            )

        mpi_count = (
            self._validate_mp_count(mpi_processes)
            if mpi_processes is not None
            else None
        )

        if split_projections and mpi_count is None:
            raise SimulationError("split_projections needs mpi_processes")

        switch_parts = []
        for key, value in (runtime_switches or {}).items():
            # True stands for a switch without a value, e.g. /HO.
            switch_parts.append(f"/{key}" if value is True else f"/{key}:{value}")
        if split_projections:
            switch_parts.append("/MP")

        # SIMIND manual (MPI section): MPI runs use the separate simind_mpi
        # build. Without /MP every rank simulates all projections and the
        # results are summed; /MP shares the projections out instead.
        if mpi_count is not None:
            command = [
                "mpirun",
                "-np",
                mpi_count,
                self._validate_cli_token(
                    self._mpi_executable(self.executable),
                    allow_whitespace=True,
                ),
            ]
        else:
            command = [
                self._validate_cli_token(
                    self._resolve_executable(self.executable),
                    allow_whitespace=True,
                )
            ]

        command += [prefix, prefix]
        if orbit_file is not None:
            command.append(self._validate_cli_token(orbit_file.name))
        if switch_parts:
            command.append(self._validate_cli_token("".join(switch_parts)))

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
            stdout_text, stderr_text = self._read_and_replay(stdout_file, stderr_file)

        # Each stream is checked on its own: concatenating them could match a
        # diagnostic split across the boundary that neither stream contains.
        self._raise_for_reported_error(stdout_text)
        self._raise_for_reported_error(stderr_text)

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
    def _resolve_executable(executable: str) -> str:
        """Make path-bearing executables absolute before ``cwd`` applies.

        subprocess resolves relative paths against the child ``cwd`` (the
        output directory), so they must be anchored to the caller's working
        directory here. Bare names keep their PATH-lookup semantics.
        """
        if os.sep in executable or (os.altsep and os.altsep in executable):
            return str(Path(executable).expanduser().absolute())
        return executable

    def _mpi_executable(self, serial_executable: str) -> str:
        """Explicit or $SIMIND_MPI_BIN, else simind_mpi next to a path binary."""
        if self.mpi_executable is not None:
            return self._resolve_executable(self.mpi_executable)
        serial = self._resolve_executable(serial_executable)
        if os.sep in serial or (os.altsep and os.altsep in serial):
            return str(Path(serial).with_name("simind_mpi"))
        return "simind_mpi"

    @staticmethod
    def _validate_mp_count(value: object) -> str:
        """Normalise an MPI process count to a plain integer string.

        Accepts Python ints, integral floats and digit-only strings so callers
        can pass a count from configuration without surprises. ``bool`` is
        rejected explicitly, and round-tripping through ``float`` is avoided so
        very large ints keep their exact value.
        """
        message = f"mpi_processes must be an integer >= 1, got {value!r}"
        if isinstance(value, bool):
            raise SimulationError(message)
        if isinstance(value, int):
            count = value
        elif isinstance(value, float) and value.is_integer():
            count = int(value)
        elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
            count = int(value)
        else:
            raise SimulationError(message)
        if count < 1:
            raise SimulationError(message)
        return str(count)

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
