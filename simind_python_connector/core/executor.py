"""Minimal SIMIND process runner used by connector-first APIs."""

from __future__ import annotations

import logging
import numbers
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
# Confirmed fatal exit-0 diagnostic: SIMIND prints this on a complete line,
# with variable spacing and casing, when a spectra .isd input cannot be read.
_SIMIND_MISSING_INPUT_PATTERN = re.compile(
    r"^[ \t]*No[ \t]+input[ \t]+file[ \t]+found[ \t]+for[ \t]+the[ \t]+spectra"
    r"[ \t]*:[ \t]*(\S+\.isd)[ \t]*\r?$",
    re.IGNORECASE | re.MULTILINE,
)
# SIMIND echoes the configured comment, verbatim, under this banner on every
# successful run. That echo can spell an error-number, divider or
# missing-spectra diagnostic, so those scans exclude only the echoed line; the
# genuine fatal aborts during input setup and never reaches the banner.
_SIMIND_OTHER_INFORMATION_PATTERN = re.compile(
    r"^[ \t]*OTHER[ \t]+INFORMATION[ \t]*\r?$", re.MULTILINE
)
_ERROR_EXCERPT_LIMIT = 2000
# SIMIND V8 copies output names into a fixed 60-character Fortran buffer and
# silently truncates anything longer, dropping the final extension character so
# no readable SPECT header remains. The longest engine-written name adds a
# "_tot_w{N}.a00" suffix (10 characters plus the digits of the window number),
# so 47 keeps the name within 59 characters even with a two-digit window.
_OUTPUT_PREFIX_MAX_LENGTH = 47


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
        prefix = self._validate_output_prefix(output_prefix)

        mpi_count = (
            self._validate_mp_count(mpi_processes)
            if mpi_processes is not None
            else None
        )

        if split_projections and mpi_count is None:
            raise SimulationError("split_projections needs mpi_processes")

        switch_parts = []
        for key, value in (runtime_switches or {}).items():
            # None removes a switch, matching RuntimeSwitches.set_switch.
            if value is None:
                continue
            switch = str(key).upper()
            try:
                _validate_switch_value(switch, value)
            except ValueError as exc:
                raise SimulationError(str(exc)) from exc
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
                stdout_text, stderr_text = self._read_and_replay(
                    stdout_file, stderr_file
                )
                raise SimulationError(
                    f"SIMIND execution failed: {exc}\n"
                    f"stdout:\n{stdout_text[:_ERROR_EXCERPT_LIMIT]}\n"
                    f"stderr:\n{stderr_text[:_ERROR_EXCERPT_LIMIT]}"
                ) from exc
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
    def _blank_echoed_comment(output: str) -> str:
        """Blank SIMIND's echoed config comment for diagnostic scanning.

        SIMIND echoes the configured comment verbatim on the single physical
        line after the OTHER INFORMATION banner. That comment is user-controlled,
        so a comment that spells an error-number or divider diagnostic would
        otherwise fail a successful run. Only that line is blanked; every line
        before and after the banner is still scanned.
        """
        banner = _SIMIND_OTHER_INFORMATION_PATTERN.search(output)
        if banner is None:
            return output
        line_start = output.find("\n", banner.end())
        if line_start == -1:
            return output
        line_start += 1
        line_end = output.find("\n", line_start)
        if line_end == -1:
            line_end = len(output)
        # Same-length blank keeps line boundaries and offsets intact.
        return output[:line_start] + " " * (line_end - line_start) + output[line_end:]

    @staticmethod
    def _raise_for_reported_error(output: str) -> None:
        scan_text = SimindExecutor._blank_echoed_comment(output)
        number_match = _SIMIND_ERROR_NUMBER_PATTERN.search(scan_text)
        if number_match:
            excerpt = output[:_ERROR_EXCERPT_LIMIT]
            raise SimulationError(
                f"SIMIND reported an error (error number "
                f"{number_match.group(1)}): {excerpt}"
            )
        if _SIMIND_ERROR_DIVIDER_PATTERN.search(scan_text):
            excerpt = output[:_ERROR_EXCERPT_LIMIT]
            raise SimulationError(f"SIMIND reported an error: {excerpt}")
        banner = _SIMIND_OTHER_INFORMATION_PATTERN.search(output)
        scan_end = banner.start() if banner else len(output)
        missing_input_match = _SIMIND_MISSING_INPUT_PATTERN.search(output, 0, scan_end)
        if missing_input_match:
            # The matched line is carried in the message itself so it survives
            # even when it falls outside the bounded excerpt window.
            diagnostic = missing_input_match.group(0).strip()
            excerpt = output[:_ERROR_EXCERPT_LIMIT]
            raise SimulationError(f"SIMIND reported an error ({diagnostic}): {excerpt}")

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

        Accepts Python and NumPy integers, integral reals (including NumPy
        floats) and ASCII digit-only strings so callers can pass a count from
        configuration without surprises. ``bool`` is rejected explicitly, and
        round-tripping through ``float`` is avoided so very large ints keep
        their exact value. Conversion failures (for example a digit string that
        trips the interpreter's int-string limit) become ``SimulationError``.
        """
        message = "mpi_processes must be an integer >= 1"
        if isinstance(value, bool):
            raise SimulationError(message)
        if isinstance(value, str):
            if not (value.isascii() and value.isdigit()):
                raise SimulationError(message)
            try:
                count = int(value)
            except (ValueError, OverflowError) as exc:
                raise SimulationError(message) from exc
        elif isinstance(value, numbers.Integral):
            count = int(value)
        elif isinstance(value, numbers.Real):
            # Compare against int(value) instead of coercing through float: a
            # huge integral Fraction overflows the float coercion, and exact
            # integrality must be preserved.
            try:
                count = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SimulationError(message) from exc
            if value != count:
                raise SimulationError(message)
        else:
            raise SimulationError(message)
        if count < 1:
            raise SimulationError(message)
        try:
            return str(count)
        except (ValueError, OverflowError) as exc:
            raise SimulationError(message) from exc

    @staticmethod
    def _validate_output_prefix(prefix: object) -> str:
        """Return ``prefix`` after validating it as a SIMIND output prefix.

        Non-strings are rejected rather than coerced, and the length is capped
        so SIMIND's own output names stay inside the engine's fixed 60-character
        buffer instead of being silently truncated.
        """
        if not isinstance(prefix, str):
            raise SimulationError(
                f"output_prefix must be a string, got {type(prefix).__name__}"
            )
        token = SimindExecutor._validate_cli_token(prefix)
        if len(token) > _OUTPUT_PREFIX_MAX_LENGTH:
            raise SimulationError(
                f"output_prefix must be at most {_OUTPUT_PREFIX_MAX_LENGTH} "
                f"characters, got {len(token)}"
            )
        if not re.fullmatch(OUTPUT_PREFIX_PATTERN, token):
            raise SimulationError(
                f"output_prefix must contain only lowercase letters, digits and "
                f"'_', never '-' (SIMIND treats '-' as an argument delimiter, "
                f"which would split the filename arguments), got {token!r}"
            )
        return token

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
