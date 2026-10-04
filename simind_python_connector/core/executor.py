"""Minimal SIMIND process runner used by connector-first APIs."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

from .types import SimulationError


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
            self.mpi_executable = os.environ.get("SIMIND_MPI_BIN") or "simind_mpi"

    def run_simulation(
        self,
        output_prefix: str,
        orbit_file: Optional[Path] = None,
        runtime_switches: Optional[Dict] = None,
        cwd: Union[str, Path, None] = None,
        mpi_processes: Optional[int] = None,
        split_projections: bool = False,
    ) -> None:
        if split_projections and mpi_processes is None:
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
        if mpi_processes is not None:
            command = [
                "mpirun",
                "-np",
                self._validate_cli_token(mpi_processes),
                self._validate_cli_token(
                    self._resolve_executable(self.mpi_executable),
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

        prefix = self._validate_cli_token(output_prefix)
        command += [prefix, prefix]
        if orbit_file is not None:
            command.append(self._validate_cli_token(orbit_file.name))
        if switch_parts:
            command.append(self._validate_cli_token("".join(switch_parts)))

        self.logger.info("Running SIMIND: %s", " ".join(command))
        try:
            subprocess.run(command, check=True, shell=False, cwd=cwd)
        except OSError as exc:
            raise SimulationError(f"Unable to execute SIMIND command: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            raise SimulationError(f"SIMIND execution failed: {exc}") from exc

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
