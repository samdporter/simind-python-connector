import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from simind_python_connector.core.types import PenetrateOutputType
from simind_python_connector.utils.interfile import InterfileHeader


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class ConversionConfig:
    """Configuration for SIMIND to STIR conversion."""

    radius_scale_factor: float = 1.0  # pass-through; SIMIND writes Radius in mm
    angle_offset: float = 180.0  # degrees
    default_number_format: str = "float"
    ignored_patterns: List[str] = None

    def __post_init__(self):
        if self.ignored_patterns is None:
            self.ignored_patterns = [
                "program",
                "patient",
                "institution",
                "contact",
                "ID",
                "exam type",
                "detector head",
                "number of images/energy window",
                "time per projection",
                "data description",
                "total number of images",
                "acquisition mode",
            ]


class ConversionRule:
    """Base class for conversion rules."""

    def matches(self, line: str) -> bool:
        """Check if this rule applies to the given line."""
        raise NotImplementedError

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Convert the line and return new line plus updated context."""
        raise NotImplementedError


class RadiusConversionRule(ConversionRule):
    """Convert radius values with scaling."""

    def __init__(self, scale_factor: float = 1.0):  # Default to no scaling
        self.scale_factor = scale_factor

    def matches(self, line: str) -> bool:
        return "Radius" in line and ":=" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            # Apply the configured scale factor (default 1.0 keeps SIMIND's
            # millimetre Radius values unchanged).
            radius_value = float(line.split()[-1]) * self.scale_factor
            return f"Radius := {radius_value}", context
        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to convert radius line '{line}': {e}")
            return line, context


class OrbitFileRule(ConversionRule):
    """Process non-circular orbit file reference and insert Radii array."""

    def __init__(self, input_file_dir: Optional[Path] = None):
        self.input_file_dir = input_file_dir
        self.orbit_file_processed = False

    def matches(self, line: str) -> bool:
        return ";# Non-Uniform Orbit File" in line and not self.orbit_file_processed

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            # Extract orbit filename
            orbit_filename = line.split(":=")[-1].strip()

            # Build full path to orbit file
            if self.input_file_dir:
                orbit_path = self.input_file_dir / orbit_filename
            else:
                orbit_path = Path(orbit_filename)

            if not orbit_path.exists():
                logging.warning(f"Orbit file not found: {orbit_path}")
                self.orbit_file_processed = True
                return line, context

            # Read radii from orbit file (first column, in cm)
            radii_cm = []
            with open(orbit_path, "r") as f:
                for file_line in f:
                    parts = file_line.strip().split()
                    if parts:
                        radii_cm.append(float(parts[0]))

            # Convert cm to mm
            radii_mm = [int(round(r * 10)) for r in radii_cm]

            # Format as STIR Radii array
            radii_str = ", ".join(str(r) for r in radii_mm)
            radii_line = f"Radii := {{{radii_str}}}"

            logging.info(
                f"Converted {len(radii_mm)} radii from orbit file {orbit_filename}"
            )
            self.orbit_file_processed = True

            # Return both the commented orbit file line and the new Radii line
            return (
                f";# Non-Uniform Orbit File := {orbit_filename}\n{radii_line}",
                context,
            )

        except Exception as e:
            logging.warning(f"Failed to process orbit file from line '{line}': {e}")
            self.orbit_file_processed = True
            return line, context


class StartAngleConversionRule(ConversionRule):
    """Convert start angle with offset."""

    def __init__(self, angle_offset: float = 180.0):
        self.angle_offset = angle_offset

    def matches(self, line: str) -> bool:
        return "start angle" in line and ":=" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            angle = float(line.split()[3]) + self.angle_offset
            return f"start angle := {angle % 360}", context
        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to convert start angle line '{line}': {e}")
            return line, context


class RotationDirectionRule(ConversionRule):
    """Track rotation direction for context."""

    def matches(self, line: str) -> bool:
        return "CCW" in line or "CW" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if "CCW" in line:
            context["rotation_direction"] = "CCW"
        elif "CW" in line:
            context["rotation_direction"] = "CW"
        return line, context


class NumberFormatRule(ConversionRule):
    """Convert number format specifications."""

    def matches(self, line: str) -> bool:
        return "!number format := short float" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return "!number format := float", context


class OrbitConversionRule(ConversionRule):
    """Convert orbit specifications."""

    def matches(self, line: str) -> bool:
        return "orbit" in line and "noncircular" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return "orbit := non-circular", context


class ImageDurationRule(ConversionRule):
    """Convert image duration to STIR format."""

    def matches(self, line: str) -> bool:
        return "image duration" in line and ":=" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            parts = line.split()
            duration = parts[4]
            return (
                f"number of time frames := 1\nimage duration (sec) [1] := {duration}",
                context,
            )
        except (IndexError, ValueError) as e:
            logging.warning(f"Failed to convert image duration line '{line}': {e}")
            return line, context


class EnergyWindowRule(ConversionRule):
    """Convert energy window specifications."""

    def __init__(self, window_type: str):
        self.window_type = window_type  # "lower" or "upper"

    def matches(self, line: str) -> bool:
        return f";energy window {self.window_type} level" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            value = line.split()[-1]
            return f"energy window {self.window_type} level[1] := {value}", context
        except IndexError as e:
            logging.warning(f"Failed to convert energy window line '{line}': {e}")
            return line, context


class DataFileNameRule(ConversionRule):
    """Convert data file name references."""

    def __init__(self, override_filename: Optional[str] = None):
        self.override_filename = override_filename

    def matches(self, line: str) -> bool:
        return "!name of data file" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            if self.override_filename:
                # Use the override filename
                return f"!name of data file := {self.override_filename}", context
            else:
                # Use existing filename from the header line
                file = Path(line.split()[5])
                return f"!name of data file := {file.stem + file.suffix}", context
        except IndexError as e:
            logging.warning(f"Failed to convert data file name line '{line}': {e}")
            return line, context


class IgnorePatternRule(ConversionRule):
    """Add semicolon to ignored pattern lines."""

    def __init__(self, patterns: List[str]):
        self.patterns = patterns

    def matches(self, line: str) -> bool:
        return any(pattern in line for pattern in self.patterns)

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return ";" + line, context


class SimindToStirConverter:
    """Converts SIMIND Interfile headers (.h00) into STIR headers (.hs)."""

    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.input_file_dir = None  # Will be set during convert_file
        self.rules = self._create_rules()
        self.logger = logging.getLogger(__name__)

    def _create_rules(
        self, data_file_override: Optional[str] = None
    ) -> List[ConversionRule]:
        """Create conversion rules in order of priority."""
        return [
            # Data-file names must be rewritten before ignore rules run:
            # ignored substrings such as "patient" otherwise comment out
            # data files whose paths happen to contain them.
            DataFileNameRule(data_file_override),
            IgnorePatternRule(self.config.ignored_patterns),
            OrbitFileRule(self.input_file_dir),  # Process orbit file before other rules
            RadiusConversionRule(self.config.radius_scale_factor),
            StartAngleConversionRule(self.config.angle_offset),
            RotationDirectionRule(),
            NumberFormatRule(),
            OrbitConversionRule(),
            ImageDurationRule(),
            EnergyWindowRule("lower"),
            EnergyWindowRule("upper"),
        ]

    def convert_line(
        self, line: str, context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Convert a single line using the first matching rule."""
        line = line.strip()

        for rule in self.rules:
            if rule.matches(line):
                return rule.convert(line, context)

        return line, context

    @contextmanager
    def _safe_file_operation(self, input_file: str, output_file: str):
        """Context manager for safe file operations with cleanup."""
        temp_file = output_file + ".tmp"
        try:
            with open(input_file, "r") as f_in, open(temp_file, "w") as f_out:
                yield f_in, f_out

            # Only replace original if conversion succeeded
            if os.path.exists(temp_file):
                if os.path.exists(output_file):
                    os.remove(output_file)
                os.rename(temp_file, output_file)
        except Exception:
            # Clean up temp file if something went wrong
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    def convert_file(
        self,
        input_filename: str,
        output_filename: Optional[str] = None,
        data_file: Optional[str] = None,
    ) -> None:
        """Convert a SIMIND header file to STIR format."""

        if not input_filename.endswith(".h00"):
            raise ValueError("Input file must have .h00 extension")

        if output_filename is None:
            output_filename = input_filename.replace(".h00", ".hs")

        # Set input directory for orbit file resolution
        self.input_file_dir = Path(input_filename).parent

        # Create rules with optional data file override
        if data_file is not None:
            self.rules = self._create_rules(data_file)
        else:
            # Recreate rules to pick up the new input_file_dir
            self.rules = self._create_rules()

        context = {"rotation_direction": None}

        try:
            with self._safe_file_operation(input_filename, output_filename) as (
                f_in,
                f_out,
            ):
                for line_num, line in enumerate(f_in, 1):
                    try:
                        converted_line, context = self.convert_line(line, context)
                        f_out.write(converted_line + "\n")
                    except Exception as e:
                        self.logger.error(
                            f"Error converting line {line_num}: {line.strip()}"
                        )
                        self.logger.error(f"Error details: {e}")
                        # Write original line as fallback
                        f_out.write(line)

            self.logger.info(
                f"Successfully converted {input_filename} to {output_filename}"
            )
            if data_file:
                self.logger.info(f"Used data file override: {data_file}")

        except Exception as e:
            self.logger.error(f"Failed to convert {input_filename}: {e}")
            raise
        finally:
            # Reset rules to default after conversion
            if data_file is not None:
                self.rules = self._create_rules()

    def create_penetrate_headers_from_template(
        self, h00_file: str, output_prefix: str, output_dir: str
    ) -> Dict[str, Path]:
        """
        Create one STIR header per PENETRATE component from the single .h00 file.

        The penetrate routine writes one .h00 file pointing to a non-existent
        .a00, plus one .bXX binary file per component. This writes a .hs
        header for each .bXX file found.

        Returns:
            Dictionary mapping component slugs to the written header paths.
        """
        output_dir = Path(output_dir)
        outputs: Dict[str, Path] = {}

        # First convert the template .h00 to .hs format
        template_hs = h00_file.replace(".h00", "_template.hs")
        self.convert_file(h00_file, template_hs)

        template_header = InterfileHeader.from_file(template_hs)

        # Look for .bXX files and create headers for each
        for component in PenetrateOutputType:
            binary_file = output_dir / f"{output_prefix}.b{component.value:02d}"

            if binary_file.exists():
                try:
                    component_hs = output_dir / (
                        f"{output_prefix}_component_{component.value:02d}.hs"
                    )

                    component_header = template_header.copy()
                    study_base = Path(binary_file.name).stem
                    component_header.set("!name of data file", binary_file.name)
                    component_header.set(
                        "patient name", f"{component.slug}_{binary_file.name}"
                    )
                    component_header.set("!study ID", study_base)
                    component_header.set("data description", component.description)
                    component_header.write(component_hs)

                    outputs[component.slug] = component_hs

                    self.logger.info(
                        f"Created STIR header for {component.slug}: {component_hs.name}"
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to create header for {binary_file}: {e}"
                    )

        # Clean up template file
        if os.path.exists(template_hs):
            os.remove(template_hs)

        return outputs

    def find_penetrate_h00_file(
        self, output_prefix: str, output_dir: str
    ) -> Optional[str]:
        """
        Find the single .h00 file created by penetrate routine.

        Args:
            output_prefix: Prefix used for output files
            output_dir: Directory containing output files

        Returns:
            Path to the .h00 file, or None if not found
        """
        output_dir = Path(output_dir)

        # Look for .h00 file with the output prefix
        h00_files = list(output_dir.glob(f"{output_prefix}*.h00"))

        if len(h00_files) == 1:
            return str(h00_files[0])
        elif len(h00_files) == 0:
            self.logger.warning(f"No .h00 file found with prefix {output_prefix}")
            return None
        else:
            raise ValueError(
                f"Multiple .h00 files match prefix {output_prefix!r} in "
                f"{output_dir}: {[f.name for f in sorted(h00_files)]}. "
                "Remove stale outputs or use a distinct output prefix."
            )
