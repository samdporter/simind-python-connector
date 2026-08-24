"""Minimal SIMIND process runner used by connector-first APIs."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

from .types import SimulationError


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
    ) -> None:
        mp_value = None
        if runtime_switches:
            for key, value in runtime_switches.items():
                if str(key).lower() == "mp":
                    mp_value = value
                    break

        # shell=False passes the executable to execve as argv[0], so spaces
        # in its path are safe; other tokens must stay whitespace-free.
        validated_simind = self._validate_cli_token(
            self._resolve_executable(), allow_whitespace=True
        )
        validated_output_prefix = self._validate_cli_token(output_prefix)
        validated_orbit_name = (
            self._validate_cli_token(orbit_file.name) if orbit_file else None
        )
        validated_mp_value = (
            self._validate_cli_token(mp_value) if mp_value is not None else None
        )

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
            command = [
                "mpirun",
                "-np",
                validated_mp_value,
                validated_simind,
                validated_output_prefix,
                validated_output_prefix,
            ]
            if validated_orbit_name is not None:
                command.append(validated_orbit_name)
            command.append("-p")
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
        try:
            subprocess.run(command, check=True, shell=False, cwd=cwd)
        except OSError as exc:
            raise SimulationError(f"Unable to execute SIMIND command: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            raise SimulationError(f"SIMIND execution failed: {exc}") from exc

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
