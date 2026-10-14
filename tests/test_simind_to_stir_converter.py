"""Tests for SIMIND to STIR converter."""

import logging
from pathlib import Path

import pytest

from simind_python_connector.configs import get
from simind_python_connector.connectors import SimindPythonConnector
from simind_python_connector.converters import simind_to_stir
from simind_python_connector.converters.simind_to_stir import (
    ConversionConfig,
    DataFileNameRule,
    EnergyWindowRule,
    ImageDurationRule,
    NumberFormatRule,
    OrbitConversionRule,
    OrbitFileRule,
    RadiusConversionRule,
    SimindToStirConverter,
    StartAngleConversionRule,
)
from simind_python_connector.core.types import (
    OutputError,
    PenetrateOutputType,
    ScoringRoutine,
)
from simind_python_connector.utils.interfile_parser import (
    InterfileHeader,
    parse_interfile_line,
)


@pytest.mark.unit
class TestConversionRules:
    """Test individual conversion rules."""

    def test_radius_conversion_rule(self):
        """Test radius conversion honours its configured scale factor."""
        scaling_rule = RadiusConversionRule(scale_factor=2.0)

        # Test matching
        assert scaling_rule.matches("Radius := 134")
        assert not scaling_rule.matches("Radii := {134}")
        assert not scaling_rule.matches("start angle := 0")

        # Test conversion applies the factor
        line, context = scaling_rule.convert("Radius := 134.5", {})
        assert line == "Radius := 269.0"

        # Identity scaling passes the value through unchanged
        identity_rule = RadiusConversionRule(scale_factor=1.0)
        line, _ = identity_rule.convert("Radius := 134.5", {})
        assert line == "Radius := 134.5"

    def test_data_file_line_survives_ignored_substrings(self, tmp_path):
        """A data-file path containing ignored substrings must stay active."""
        h00_file = tmp_path / "output.h00"
        h00_file.write_text(
            "\n".join(
                [
                    "!INTERFILE :=",
                    "!name of data file := patient.a00",
                    "patient name := John Doe",
                    "!study ID := study123",
                    "!END OF INTERFILE :=",
                ]
            )
        )

        hs_file = tmp_path / "output.hs"
        SimindToStirConverter().convert_file(str(h00_file), str(hs_file))

        content_lines = [line.strip() for line in hs_file.read_text().splitlines()]
        active_data_file = [
            line for line in content_lines if line.startswith("!name of data file")
        ]
        assert active_data_file == ["!name of data file := patient.a00"]
        assert ";patient name := John Doe" in content_lines
        assert ";!study ID := study123" in content_lines

    def test_orbit_file_name_with_ignored_word_still_adds_radii(self, tmp_path):
        (tmp_path / "patient.cor").write_text("20.0 0\n25.0 0\n")
        h00_file = tmp_path / "output.h00"
        h00_file.write_text(
            "\n".join(
                [
                    "!INTERFILE :=",
                    "!name of data file := output.a00",
                    ";# Non-Uniform Orbit File := patient.cor",
                    "!END OF INTERFILE :=",
                ]
            )
        )

        hs_file = tmp_path / "output.hs"
        SimindToStirConverter().convert_file(str(h00_file), str(hs_file))

        assert "Radii := {200, 250}" in hs_file.read_text()

    def test_header_written_elsewhere_points_at_the_input_data(self, tmp_path):
        h00_file = tmp_path / "output.h00"
        h00_file.write_text(
            "!INTERFILE :=\n!name of data file := output.a00\n!END OF INTERFILE :=\n"
        )
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        hs_file = other_dir / "output.hs"
        SimindToStirConverter().convert_file(str(h00_file), str(hs_file))

        key, value = parse_interfile_line(
            next(
                line
                for line in hs_file.read_text().splitlines()
                if line.startswith("!name of data file")
            )
        )
        assert Path(value) == (tmp_path / "output.a00").resolve()

    def test_start_angle_conversion_rule(self):
        """Test start angle conversion with offset."""
        rule = StartAngleConversionRule(angle_offset=180.0)

        # Test matching
        assert rule.matches("start angle := 0.000000")
        assert not rule.matches("Radius := 134")

        # Test conversion - 0 + 180 = 180
        line, context = rule.convert("start angle := 0.000000", {})
        assert "start angle := 180" in line

        # Test conversion - 90 + 180 = 270
        line, context = rule.convert("start angle := 90.000000", {})
        assert "start angle := 270" in line

        # Test wrap-around - 270 + 180 = 450 % 360 = 90
        line, context = rule.convert("start angle := 270.000000", {})
        assert "start angle := 90" in line

    def test_number_format_rule(self):
        """Test number format conversion."""
        rule = NumberFormatRule()

        # Test matching
        assert rule.matches("!number format := short float")
        assert not rule.matches("!number format := float")

        # Test conversion
        line, context = rule.convert("!number format := short float", {})
        assert line == "!number format := float"

    def test_orbit_conversion_rule(self):
        """Test orbit specification conversion."""
        rule = OrbitConversionRule()

        # Test matching
        assert rule.matches("orbit := noncircular")
        assert not rule.matches("orbit := circular")

        # Test conversion
        line, context = rule.convert("orbit := noncircular", {})
        assert line == "orbit := non-circular"

    def test_image_duration_rule(self):
        """Test image duration conversion."""
        rule = ImageDurationRule()

        # Test matching
        assert rule.matches("image duration (sec) := 120.000000")
        assert not rule.matches("time per projection (sec) := 1.0")

        # Test conversion
        line, context = rule.convert("image duration (sec) := 120.000000", {})
        assert "number of time frames := 1" in line
        assert "image duration (sec) [1] := 120.000000" in line

    def test_energy_window_rule(self):
        """Test energy window conversion."""
        rule_lower = EnergyWindowRule("lower")
        rule_upper = EnergyWindowRule("upper")

        # Test matching
        assert rule_lower.matches(";energy window lower level := 126.16")
        assert rule_upper.matches(";energy window upper level := 36.84")
        assert not rule_lower.matches(";energy window upper level := 36.84")

        # Test conversion
        line, context = rule_lower.convert(";energy window lower level := 126.16", {})
        assert line == "energy window lower level[1] := 126.16"

        line, context = rule_upper.convert(";energy window upper level := 36.84", {})
        assert line == "energy window upper level[1] := 36.84"

    def test_data_file_name_rule(self):
        """Test data file name conversion."""
        rule = DataFileNameRule()

        # Test matching
        assert rule.matches("!name of data file := /path/to/output.a00")

        # Test conversion
        line, context = rule.convert("!name of data file := /path/to/output.a00", {})
        assert line == "!name of data file := output.a00"

        # Test with override
        rule_override = DataFileNameRule(override_filename="custom.b01")
        line, context = rule_override.convert("!name of data file := output.a00", {})
        assert line == "!name of data file := custom.b01"


@pytest.mark.unit
class TestOrbitFileRule:
    """Test orbit file processing rule."""

    def test_orbit_file_rule_with_valid_file(self, tmp_path):
        """Test orbit file conversion with valid .cor file."""
        # Create a test orbit file
        orbit_file = tmp_path / "test.cor"
        radii_cm = [14.0, 15.5, 16.0, 14.0]  # cm
        with open(orbit_file, "w") as f:
            for r in radii_cm:
                f.write(f"      {r:6.3f}    64\n")

        # Create rule with input directory
        rule = OrbitFileRule(input_file_dir=tmp_path)

        # Test matching
        assert rule.matches(";# Non-Uniform Orbit File := test.cor")
        assert not rule.matches("orbit := noncircular")

        # Test conversion
        line, context = rule.convert(";# Non-Uniform Orbit File := test.cor", {})

        # Check that Radii line was created
        assert "Radii := {140, 155, 160, 140}" in line
        assert ";# Non-Uniform Orbit File := test.cor" in line

    def test_orbit_file_rule_missing_file(self, tmp_path):
        """Test orbit file rule when .cor file doesn't exist."""
        rule = OrbitFileRule(input_file_dir=tmp_path)

        # Test with non-existent file
        line, context = rule.convert(";# Non-Uniform Orbit File := missing.cor", {})

        # Should return original line when file not found
        assert line == ";# Non-Uniform Orbit File := missing.cor"

    def test_orbit_file_rule_only_processes_once(self, tmp_path):
        """Test that orbit file is only processed once."""
        orbit_file = tmp_path / "test.cor"
        with open(orbit_file, "w") as f:
            f.write("14.0    64\n15.0    64\n")

        rule = OrbitFileRule(input_file_dir=tmp_path)

        # First call should process
        assert rule.matches(";# Non-Uniform Orbit File := test.cor")
        line1, _ = rule.convert(";# Non-Uniform Orbit File := test.cor", {})
        assert "Radii" in line1

        # Second call should not match (already processed)
        assert not rule.matches(";# Non-Uniform Orbit File := test.cor")


@pytest.mark.unit
class TestSimindToStirConverter:
    """Test the main converter class."""

    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = SimindToStirConverter()
        assert converter is not None
        assert converter.config is not None
        assert len(converter.rules) > 0

    def test_converter_with_custom_config(self):
        """Test converter with custom configuration."""
        config = ConversionConfig(
            radius_scale_factor=1.0,
            angle_offset=90.0,
        )
        converter = SimindToStirConverter(config)
        assert converter.config.radius_scale_factor == 1.0

    def test_penetrate_headers_apply_enum_metadata(self, tmp_path):
        """Ensure penetrate metadata from enum is propagated to headers."""
        converter = SimindToStirConverter()
        h00_file = tmp_path / "output.h00"
        h00_file.write_text(
            "\n".join(
                [
                    "!name of data file := output.a00",
                    "patient name := phantom",
                    "!study ID := study123",
                    "data description := placeholder",
                    "!END OF INTERFILE :=",
                ]
            )
        )
        component = PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT_BACK
        binary_file = tmp_path / f"output.b{component.value:02d}"
        binary_file.write_bytes(b"\x00\x00\x00\x00")

        outputs = converter.create_penetrate_headers_from_template(
            str(h00_file), "output", str(tmp_path)
        )
        header_path = tmp_path / f"output_component_{component.value:02d}.hs"
        header_text = header_path.read_text()

        assert component.slug in outputs
        assert f"patient name := {component.slug}_{binary_file.name}" in header_text
        assert component.description in header_text

    def test_convert_line(self):
        """Test single line conversion."""
        converter = SimindToStirConverter()

        # Test number format conversion
        line, context = converter.convert_line("!number format := short float", {})
        assert line == "!number format := float"

        # Test start angle conversion
        line, context = converter.convert_line("start angle := 0.000000", {})
        assert "180" in line

    def test_parse_interfile_line(self):
        """Test interfile line parsing using shared parser."""
        # Test valid parameter line
        key, value = parse_interfile_line("!matrix size [1] := 128")
        assert key == "!matrix size [1]"
        assert value == "128"

        # Test with spaces
        key, value = parse_interfile_line("scaling factor (mm/pixel) [1] := 4.419600")
        assert key == "scaling factor (mm/pixel) [1]"
        assert value == "4.419600"

        # Test comment line
        key, value = parse_interfile_line(";# This is a comment")
        assert key is None
        assert value is None

        # Test section header
        key, value = parse_interfile_line("!GENERAL DATA :=")
        assert key is None
        assert value is None


@pytest.mark.unit
class TestConverterWithRealData:
    """Test converter with realistic SIMIND data."""

    def create_sample_h00_file(self, tmp_path, include_orbit=False):
        """Create a sample SIMIND .h00 file."""
        h00_file = tmp_path / "output.h00"

        content = """!INTERFILE :=
!imaging modality := nucmed
!originating system := simind
!version of keys := 3.3
program author := M Ljungberg, Lund University

!GENERAL DATA :=
!data offset in bytes := 0
!name of data file := output.a00
patient name := test_patient

!GENERAL IMAGE DATA :=
!type of data := tomographic
imagedata byte order := LITTLEENDIAN
;energy window lower level := 126.16
;energy window upper level := 36.84
!number of energy windows := 1
!matrix size [1] := 128
!matrix size [2] := 128
!number format := short float
!number of bytes per pixel := 4
scaling factor (mm/pixel) [1] := 4.419600
scaling factor (mm/pixel) [2] := 4.419600

!SPECT STUDY (General) :=
!extent of rotation := 360
!process status := acquired
!number of projections := 60
image duration (sec) := 60.000000

!SPECT STUDY (acquired data) :=
"""

        if include_orbit:
            content += "orbit := noncircular\n"
            content += ";# Non-Uniform Orbit File := output.cor\n"

            # Create corresponding .cor file
            cor_file = tmp_path / "output.cor"
            with open(cor_file, "w") as f:
                for i in range(60):
                    radius = 15.0 + (i % 10) * 0.5  # Varying radius
                    f.write(f"      {radius:6.3f}    64\n")
        else:
            content += "orbit := circular\n"

        content += """!direction of rotation := CW
start angle := 0.000000

!END OF INTERFILE :=
"""

        with open(h00_file, "w") as f:
            f.write(content)

        return h00_file

    def test_convert_file_basic(self, tmp_path):
        """Test basic file conversion."""
        h00_file = self.create_sample_h00_file(tmp_path, include_orbit=False)
        hs_file = tmp_path / "output.hs"

        converter = SimindToStirConverter()
        converter.convert_file(str(h00_file), str(hs_file))

        # Check output file exists
        assert hs_file.exists()

        # Read and verify content
        with open(hs_file, "r") as f:
            content = f.read()

            # Check key conversions
            assert "!number format := float" in content
            assert "start angle := 180" in content
            assert "orbit := circular" in content
            assert "number of time frames := 1" in content
            assert "image duration (sec) [1] := 60.000000" in content
            assert "energy window lower level[1] := 126.16" in content

    def test_convert_file_with_orbit(self, tmp_path):
        """Test file conversion with non-circular orbit."""
        h00_file = self.create_sample_h00_file(tmp_path, include_orbit=True)
        hs_file = tmp_path / "output.hs"

        converter = SimindToStirConverter()
        converter.convert_file(str(h00_file), str(hs_file))

        # Check output file exists
        assert hs_file.exists()

        # Read and verify content
        with open(hs_file, "r") as f:
            content = f.read()

            # Check orbit-specific conversions
            assert "orbit := non-circular" in content
            assert "Radii := {" in content

            # Verify Radii array has correct number of values
            radii_line = [line for line in content.split("\n") if "Radii :=" in line][0]
            radii_count = radii_line.count(",") + 1
            assert radii_count == 60  # Should match number of projections

            # Verify radii are in mm (should be 150-195 based on our test data)
            radii_str = radii_line.split("{")[1].split("}")[0]
            radii = [int(x.strip()) for x in radii_str.split(",")]
            assert all(140 <= r <= 200 for r in radii)

    def test_convert_file_default_output_name(self, tmp_path):
        """Test that output filename defaults to .hs extension."""
        h00_file = self.create_sample_h00_file(tmp_path)

        converter = SimindToStirConverter()
        converter.convert_file(str(h00_file))

        # Default output should be .hs
        hs_file = tmp_path / "output.hs"
        assert hs_file.exists()

    def test_read_parameter(self, tmp_path):
        """Test reading parameter from header file."""
        h00_file = self.create_sample_h00_file(tmp_path)
        hs_file = tmp_path / "output.hs"

        converter = SimindToStirConverter()
        converter.convert_file(str(h00_file), str(hs_file))

        # Read parameters
        matrix_size = converter.read_parameter(str(hs_file), "!matrix size [1]")
        assert matrix_size == "128"

        scaling = converter.read_parameter(
            str(hs_file), "scaling factor (mm/pixel) [1]"
        )
        assert scaling == "4.419600"

    def test_edit_parameter(self, tmp_path):
        """Test editing parameter in header file."""
        h00_file = self.create_sample_h00_file(tmp_path)
        hs_file = tmp_path / "output.hs"

        converter = SimindToStirConverter()
        converter.convert_file(str(h00_file), str(hs_file))

        # Edit a parameter
        converter.edit_parameter(str(hs_file), "!matrix size [1]", "256")

        # Verify edit
        new_value = converter.read_parameter(str(hs_file), "!matrix size [1]")
        assert new_value == "256"


@pytest.mark.unit
def test_conversion_config_defaults():
    """Test ConversionConfig default values."""
    config = ConversionConfig()

    # Default passes SIMIND's millimetre Radius values through unchanged
    assert config.radius_scale_factor == 1.0
    assert config.angle_offset == 180.0
    assert config.default_number_format == "float"
    assert len(config.ignored_patterns) > 0
    assert "program" in config.ignored_patterns
    assert "patient" in config.ignored_patterns


@pytest.mark.unit
class TestSirfResolutionIsLazy:
    """SIRF types resolve only where a native object is actually loaded."""

    @staticmethod
    def write_h00(tmp_path) -> Path:
        h00_file = tmp_path / "output.h00"
        h00_file.write_text(
            "\n".join(
                [
                    "!name of data file := output.a00",
                    "!matrix size [1] := 128",
                    "!END OF INTERFILE :=",
                ]
            )
        )
        return h00_file

    @staticmethod
    def write_hs(tmp_path) -> Path:
        hs_file = tmp_path / "output.hs"
        hs_file.write_text("!matrix size [1] := 128\n")
        return hs_file

    @staticmethod
    def forbidden_sirf_resolution():
        def _resolve():
            raise AssertionError("SIRF resolution must not run")

        return _resolve

    @staticmethod
    def fake_sirf_types():
        class FakeAcquisitionData:
            def __init__(self, path):
                self.path = path

        def _resolve():
            return type(None), FakeAcquisitionData, True

        return FakeAcquisitionData, _resolve

    def test_file_only_paths_never_resolve_sirf(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            simind_to_stir, "get_sirf_types", self.forbidden_sirf_resolution()
        )

        h00_file = self.write_h00(tmp_path)
        hs_file = tmp_path / "output.hs"
        converter = SimindToStirConverter()

        converter.convert_file(str(h00_file), str(hs_file))
        assert converter.edit_parameter(str(hs_file), "!matrix size [1]", "256") is None
        assert converter.add_parameter(str(hs_file), "!extra", "1") is None

    def test_convert_file_return_object_resolves_sirf_locally(
        self, tmp_path, monkeypatch
    ):
        fake_type, resolve = self.fake_sirf_types()
        monkeypatch.setattr(simind_to_stir, "BACKEND_AVAILABLE", False)
        monkeypatch.setattr(simind_to_stir, "get_sirf_types", resolve)

        hs_file = tmp_path / "output.hs"
        result = SimindToStirConverter().convert_file(
            str(self.write_h00(tmp_path)), str(hs_file), return_object=True
        )

        assert isinstance(result, fake_type)
        assert result.path == str(hs_file)

    def test_convert_file_return_object_prefers_factory(self, tmp_path, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(simind_to_stir, "BACKEND_AVAILABLE", True)
        monkeypatch.setattr(
            simind_to_stir, "create_acquisition_data", lambda _path: sentinel
        )
        monkeypatch.setattr(
            simind_to_stir, "get_sirf_types", self.forbidden_sirf_resolution()
        )

        result = SimindToStirConverter().convert_file(
            str(self.write_h00(tmp_path)),
            str(tmp_path / "output.hs"),
            return_object=True,
        )

        assert result is sentinel

    def test_edit_parameter_return_object_resolves_sirf_locally(
        self, tmp_path, monkeypatch
    ):
        fake_type, resolve = self.fake_sirf_types()
        monkeypatch.setattr(simind_to_stir, "BACKEND_AVAILABLE", False)
        monkeypatch.setattr(simind_to_stir, "get_sirf_types", resolve)

        hs_file = self.write_hs(tmp_path)
        result = SimindToStirConverter().edit_parameter(
            str(hs_file), "!matrix size [1]", "256", return_object=True
        )

        assert isinstance(result, fake_type)
        assert result.path == str(hs_file)

    def test_add_parameter_return_object_resolves_sirf_locally(
        self, tmp_path, monkeypatch
    ):
        fake_type, resolve = self.fake_sirf_types()
        monkeypatch.setattr(simind_to_stir, "BACKEND_AVAILABLE", False)
        monkeypatch.setattr(simind_to_stir, "get_sirf_types", resolve)

        hs_file = self.write_hs(tmp_path)
        result = SimindToStirConverter().add_parameter(
            str(hs_file), "!extra", "1", return_object=True
        )

        assert isinstance(result, fake_type)
        assert result.path == str(hs_file)

    def test_load_penetrate_output_resolves_sirf_when_no_factory(
        self, tmp_path, monkeypatch
    ):
        fake_type, resolve = self.fake_sirf_types()
        header = tmp_path / "component.hs"
        header.write_text("")
        monkeypatch.setattr(simind_to_stir, "BACKEND_AVAILABLE", False)
        monkeypatch.setattr(simind_to_stir, "create_acquisition_data", None)
        monkeypatch.setattr(simind_to_stir, "get_sirf_types", resolve)

        result = SimindToStirConverter()._load_penetrate_output(header)

        assert isinstance(result, fake_type)
        assert result.path == str(header)

    def test_load_penetrate_output_broken_sirf_warns_and_falls_back(
        self, tmp_path, monkeypatch, caplog
    ):
        header = tmp_path / "component.hs"
        header.write_text("")

        def broken_get_sirf_types():
            raise RuntimeError("broken SIRF init")

        monkeypatch.setattr(simind_to_stir, "BACKEND_AVAILABLE", False)
        monkeypatch.setattr(simind_to_stir, "create_acquisition_data", None)
        monkeypatch.setattr(simind_to_stir, "get_sirf_types", broken_get_sirf_types)

        with caplog.at_level(logging.DEBUG):
            result = SimindToStirConverter()._load_penetrate_output(header)

        assert result == str(header)
        fallbacks = [
            record
            for record in caplog.records
            if record.getMessage().startswith("Falling back to file path")
        ]
        assert len(fallbacks) == 1
        assert fallbacks[0].levelno == logging.WARNING
        assert "broken SIRF init" in fallbacks[0].getMessage()


MISSING_BACKEND_ERROR = "Neither SIRF nor STIR Python found"


def make_penetrate_outputs(tmp_path) -> Path:
    """Create the PENETRATE .h00 template plus one binary per component."""
    h00_file = tmp_path / "output.h00"
    h00_file.write_text(
        "\n".join(
            [
                "!name of data file := output.a00",
                "patient name := phantom",
                "!study ID := study123",
                "data description := placeholder",
                "!END OF INTERFILE :=",
            ]
        )
    )
    for component in PenetrateOutputType:
        (tmp_path / f"output.b{component.value:02d}").write_bytes(b"\x00\x00\x00\x00")
    return h00_file


def component_header(tmp_path, component) -> Path:
    return tmp_path / f"output_component_{component.value:02d}.hs"


def penetrate_template_path(h00_file) -> Path:
    """Scratch .hs path: rewrite the terminal .h00 of the filename only."""
    h00_file = Path(h00_file)
    return h00_file.with_name(h00_file.name[: -len(".h00")] + "_template.hs")


@pytest.mark.unit
class TestPenetrateFallbackLogging:
    """Missing backends are an expected fallback, so they must not warn."""

    @staticmethod
    def patch_missing_backend(monkeypatch):
        def _factory(_header_path):
            raise ImportError(MISSING_BACKEND_ERROR)

        monkeypatch.setattr(simind_to_stir, "BACKEND_AVAILABLE", True)
        monkeypatch.setattr(simind_to_stir, "create_acquisition_data", _factory)

    def test_missing_backend_falls_back_at_debug_without_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        h00_file = make_penetrate_outputs(tmp_path)
        self.patch_missing_backend(monkeypatch)

        with caplog.at_level(logging.DEBUG):
            outputs = SimindToStirConverter().create_penetrate_headers_from_template(
                str(h00_file), "output", str(tmp_path)
            )

        assert set(outputs) == {component.slug for component in PenetrateOutputType}
        for component in PenetrateOutputType:
            header = component_header(tmp_path, component)
            assert header.exists()
            assert outputs[component.slug] == str(header)

        noisy = [
            record
            for record in caplog.records
            if record.levelno >= logging.WARNING
            and "Falling back to file path" in record.getMessage()
        ]
        assert noisy == []

        fallbacks = [
            record
            for record in caplog.records
            if record.getMessage().startswith("Falling back to file path")
        ]
        assert len(fallbacks) == len(PenetrateOutputType)
        assert all(record.levelno == logging.DEBUG for record in fallbacks)
        for component in PenetrateOutputType:
            header = str(component_header(tmp_path, component))
            matching = [record for record in fallbacks if header in record.getMessage()]
            assert len(matching) == 1
            assert MISSING_BACKEND_ERROR in matching[0].getMessage()

    def test_genuine_load_failure_warns_and_falls_back(
        self, tmp_path, monkeypatch, caplog
    ):
        h00_file = make_penetrate_outputs(tmp_path)

        def _factory(_header_path):
            raise OSError("corrupt interfile data")

        monkeypatch.setattr(simind_to_stir, "BACKEND_AVAILABLE", True)
        monkeypatch.setattr(simind_to_stir, "create_acquisition_data", _factory)

        with caplog.at_level(logging.DEBUG):
            outputs = SimindToStirConverter().create_penetrate_headers_from_template(
                str(h00_file), "output", str(tmp_path)
            )

        assert set(outputs) == {component.slug for component in PenetrateOutputType}
        for component in PenetrateOutputType:
            assert outputs[component.slug] == str(component_header(tmp_path, component))

        fallbacks = [
            record
            for record in caplog.records
            if record.getMessage().startswith("Falling back to file path")
        ]
        assert len(fallbacks) == len(PenetrateOutputType)
        assert all(record.levelno == logging.WARNING for record in fallbacks)
        assert all(
            "corrupt interfile data" in record.getMessage() for record in fallbacks
        )

    @staticmethod
    def patch_failing_write(monkeypatch, failing_name, error):
        original_write = InterfileHeader.write

        def failing_write(self, path, *args, **kwargs):
            if Path(path).name == failing_name:
                raise error
            return original_write(self, path, *args, **kwargs)

        monkeypatch.setattr(InterfileHeader, "write", failing_write)

    def test_component_header_permission_error_propagates(self, tmp_path, monkeypatch):
        h00_file = make_penetrate_outputs(tmp_path)
        component = PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT_BACK
        self.patch_failing_write(
            monkeypatch,
            component_header(tmp_path, component).name,
            PermissionError("permission denied writing header"),
        )

        with pytest.raises(PermissionError, match="permission denied"):
            SimindToStirConverter().create_penetrate_headers_from_template(
                str(h00_file), "output", str(tmp_path)
            )

    def test_success_path_removes_template(self, tmp_path):
        h00_file = make_penetrate_outputs(tmp_path)
        template_hs = penetrate_template_path(h00_file)

        SimindToStirConverter().create_penetrate_headers_from_template(
            str(h00_file), "output", str(tmp_path)
        )

        assert not template_hs.exists()

    def test_component_write_failure_removes_template_and_preserves_inputs(
        self, tmp_path, monkeypatch
    ):
        h00_file = make_penetrate_outputs(tmp_path)
        template_hs = penetrate_template_path(h00_file)
        h00_before = h00_file.read_bytes()
        binaries_before = {
            path.name: path.read_bytes() for path in tmp_path.glob("output.b*")
        }

        component = PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT_BACK
        self.patch_failing_write(
            monkeypatch,
            component_header(tmp_path, component).name,
            OSError("disk full while writing header"),
        )

        with pytest.raises(OSError, match="disk full while writing header"):
            SimindToStirConverter().create_penetrate_headers_from_template(
                str(h00_file), "output", str(tmp_path)
            )

        assert not template_hs.exists()
        assert h00_file.read_bytes() == h00_before
        after = {path.name: path.read_bytes() for path in tmp_path.glob("output.b*")}
        assert after == binaries_before

    def test_template_parse_failure_propagates_and_removes_template(
        self, tmp_path, monkeypatch
    ):
        h00_file = make_penetrate_outputs(tmp_path)
        template_hs = penetrate_template_path(h00_file)

        def failing_from_file(filename):
            assert Path(filename) == template_hs
            raise ValueError("corrupt template header")

        monkeypatch.setattr(
            InterfileHeader, "from_file", staticmethod(failing_from_file)
        )

        with pytest.raises(ValueError, match="corrupt template header"):
            SimindToStirConverter().create_penetrate_headers_from_template(
                str(h00_file), "output", str(tmp_path)
            )

        assert not template_hs.exists()

    def test_conversion_failure_removes_template(self, tmp_path, monkeypatch):
        h00_file = make_penetrate_outputs(tmp_path)
        template_hs = penetrate_template_path(h00_file)

        def failing_convert_file(
            self, input_filename, output_filename=None, *args, **kwargs
        ):
            Path(output_filename).write_text("partial template")
            raise OSError("conversion failed")

        monkeypatch.setattr(SimindToStirConverter, "convert_file", failing_convert_file)

        with pytest.raises(OSError, match="conversion failed"):
            SimindToStirConverter().create_penetrate_headers_from_template(
                str(h00_file), "output", str(tmp_path)
            )

        assert not template_hs.exists()

    def test_cleanup_failure_preserves_primary_exception(self, tmp_path, monkeypatch):
        h00_file = make_penetrate_outputs(tmp_path)
        template_hs = penetrate_template_path(h00_file)

        component = PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT_BACK
        self.patch_failing_write(
            monkeypatch,
            component_header(tmp_path, component).name,
            PermissionError("permission denied writing header"),
        )

        cleanup_attempts = []
        original_remove = simind_to_stir.os.remove

        def failing_remove(path, *args, **kwargs):
            if Path(path) == template_hs:
                cleanup_attempts.append(Path(path))
                raise RuntimeError("cleanup denied")
            return original_remove(path, *args, **kwargs)

        monkeypatch.setattr(simind_to_stir.os, "remove", failing_remove)

        with pytest.raises(PermissionError, match="permission denied"):
            SimindToStirConverter().create_penetrate_headers_from_template(
                str(h00_file), "output", str(tmp_path)
            )

        assert cleanup_attempts == [template_hs]

    def test_header_write_failure_surfaces(self, tmp_path, monkeypatch):
        h00_file = make_penetrate_outputs(tmp_path)
        component = PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT_BACK
        self.patch_failing_write(
            monkeypatch,
            component_header(tmp_path, component).name,
            OSError("disk full while writing header"),
        )

        with pytest.raises(OSError, match="disk full while writing header"):
            SimindToStirConverter().create_penetrate_headers_from_template(
                str(h00_file), "output", str(tmp_path)
            )

    def test_component_header_write_failure_becomes_output_error(
        self, tmp_path, monkeypatch
    ):
        make_penetrate_outputs(tmp_path)
        component = PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT_BACK
        self.patch_failing_write(
            monkeypatch,
            component_header(tmp_path, component).name,
            PermissionError("permission denied writing header"),
        )
        connector = SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path,
            output_prefix="output",
        )
        connector.add_config_value(84, ScoringRoutine.PENETRATE.value)

        with pytest.raises(OutputError) as excinfo:
            connector._ensure_interfile_headers()

        assert isinstance(excinfo.value.__cause__, PermissionError)

    def test_successful_factory_result_is_returned(self, tmp_path, monkeypatch):
        h00_file = make_penetrate_outputs(tmp_path)
        backend_object = object()
        loaded_paths = []

        def _factory(header_path):
            loaded_paths.append(header_path)
            return backend_object

        monkeypatch.setattr(simind_to_stir, "BACKEND_AVAILABLE", True)
        monkeypatch.setattr(simind_to_stir, "create_acquisition_data", _factory)

        outputs = SimindToStirConverter().create_penetrate_headers_from_template(
            str(h00_file), "output", str(tmp_path)
        )

        assert len(outputs) == len(PenetrateOutputType)
        assert all(value is backend_object for value in outputs.values())
        assert loaded_paths == [
            str(component_header(tmp_path, component))
            for component in PenetrateOutputType
        ]


@pytest.mark.unit
class TestPenetrateTemplateScratch:
    """Rejected inputs must fail before any scratch path is computed."""

    @staticmethod
    def snapshot(directory) -> dict:
        return {path.name: path.read_bytes() for path in Path(directory).iterdir()}

    @pytest.mark.parametrize("name", ["case.H00", "case.txt", "case"])
    def test_rejected_input_is_kept_byte_for_byte(self, tmp_path, name):
        make_penetrate_outputs(tmp_path)
        rejected = tmp_path / name
        rejected.write_bytes(b"caller-owned bytes")
        before = self.snapshot(tmp_path)

        with pytest.raises(ValueError, match="must have .h00 extension"):
            SimindToStirConverter().create_penetrate_headers_from_template(
                str(rejected), "output", str(tmp_path)
            )

        assert self.snapshot(tmp_path) == before
        assert rejected.read_bytes() == b"caller-owned bytes"

    def test_rejected_input_in_h00_named_directory_touches_nothing(self, tmp_path):
        nested = tmp_path / "run.h00.dir"
        nested.mkdir()
        make_penetrate_outputs(nested)
        rejected = nested / "case.txt"
        rejected.write_bytes(b"caller-owned bytes")
        # Rewriting ".h00" across the whole path would resolve the scratch file
        # here; rejecting the input must never reach it.
        decoy_dir = tmp_path / "run_template.hs.dir"
        decoy_dir.mkdir()
        (decoy_dir / "case.txt").write_bytes(b"unrelated bytes")
        nested_before = self.snapshot(nested)
        decoy_before = self.snapshot(decoy_dir)

        with pytest.raises(ValueError, match="must have .h00 extension"):
            SimindToStirConverter().create_penetrate_headers_from_template(
                str(rejected), "output", str(nested)
            )

        assert self.snapshot(nested) == nested_before
        assert self.snapshot(decoy_dir) == decoy_before

    def test_valid_input_in_h00_named_directory_uses_sibling_scratch(
        self, tmp_path, monkeypatch
    ):
        nested = tmp_path / "run.h00.dir"
        nested.mkdir()
        h00_file = make_penetrate_outputs(nested)
        scratch = penetrate_template_path(h00_file)

        seen = []
        original_from_file = InterfileHeader.from_file

        def recording_from_file(filename):
            seen.append(Path(filename))
            return original_from_file(filename)

        monkeypatch.setattr(
            InterfileHeader, "from_file", staticmethod(recording_from_file)
        )

        outputs = SimindToStirConverter().create_penetrate_headers_from_template(
            str(h00_file), "output", str(nested)
        )

        assert set(outputs) == {component.slug for component in PenetrateOutputType}
        assert scratch == nested / "output_template.hs"
        assert seen == [scratch]
        assert not scratch.exists()
        for component in PenetrateOutputType:
            assert component_header(nested, component).exists()


@pytest.mark.unit
class TestFindPenetrateH00File:
    """The output prefix must only claim its own PENETRATE outputs."""

    def test_overlapping_prefixes_do_not_collide(self, tmp_path):
        (tmp_path / "s.h00").write_text("x")
        (tmp_path / "s2.h00").write_text("x")
        (tmp_path / "sx.h00").write_text("x")

        found = SimindToStirConverter().find_penetrate_h00_file("s", str(tmp_path))

        assert found == str(tmp_path / "s.h00")
        assert SimindToStirConverter().find_penetrate_h00_file(
            "s2", str(tmp_path)
        ) == str(tmp_path / "s2.h00")

    def test_underscore_suffixed_output_matches_prefix(self, tmp_path):
        (tmp_path / "s_extra.h00").write_text("x")

        found = SimindToStirConverter().find_penetrate_h00_file("s", str(tmp_path))

        assert found == str(tmp_path / "s_extra.h00")

    def test_genuine_ambiguity_still_raises(self, tmp_path):
        (tmp_path / "s.h00").write_text("x")
        (tmp_path / "s_extra.h00").write_text("x")

        with pytest.raises(ValueError, match="Multiple .h00 files match prefix 's'"):
            SimindToStirConverter().find_penetrate_h00_file("s", str(tmp_path))

    def test_no_match_returns_none_with_warning(self, tmp_path, caplog):
        (tmp_path / "s2.h00").write_text("x")

        with caplog.at_level(logging.WARNING):
            found = SimindToStirConverter().find_penetrate_h00_file("s", str(tmp_path))

        assert found is None
        assert any(
            record.levelno == logging.WARNING
            and "No .h00 file found with prefix s" in record.getMessage()
            for record in caplog.records
        )

    def test_directory_named_like_output_is_ignored(self, tmp_path, caplog):
        (tmp_path / "s.h00").mkdir()

        with caplog.at_level(logging.WARNING):
            found = SimindToStirConverter().find_penetrate_h00_file("s", str(tmp_path))

        assert found is None
