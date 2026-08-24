import os
import tempfile
from pathlib import Path

import pytest

from simind_python_connector import SimulationConfig
from simind_python_connector.configs import get


# Most tests here are unit tests that don't require SIRF
pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_text() -> str:
    return (FIXTURES / "minimal_variable_sections.smc").read_text()


def test_smc_parser_uses_section_cursor_and_preserves_counts():
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")

    assert len(config.data) == 120
    assert config.get_value(11) == pytest.approx(11.0)
    assert config.get_value(42) == pytest.approx(42.0)
    assert config.get_value(78) == pytest.approx(78.0)
    assert config.get_value(79) == pytest.approx(79.0)
    assert config.get_value(81) == pytest.approx(81.0)
    assert config.get_value(82) == pytest.approx(82.0)
    assert config.flags == "TFTFFFFFFFFFFFFFFFFFFFFFFFFFFF"
    assert config.text_variables == {
        1: "first text variable",
        2: "second text variable",
    }
    assert config.data_files == {
        1: "phantom.dat",
        2: "source.dat",
        3: "attenuation.dat",
    }


def test_yaml_smc_yaml_round_trip_preserves_parameter_flags_and_files(tmp_path):
    original = SimulationConfig(get("AnyScan.yaml"))
    original.set_value(11, 5.0)
    original.set_data_file("phantom_soft_tissue", "custom_phantom")
    original.set_data_file(12, "custom_last_file")
    smc_path = original.save_file(tmp_path / "round_trip.smc")
    restored = SimulationConfig(smc_path)

    assert restored.get_value(11) == pytest.approx(5.0)
    assert restored.get_data_file("phantom_soft_tissue") == "custom_phantom"
    assert restored.get_data_file(12) == "custom_last_file"
    assert len(restored.flags) == len(original.flags)


def test_smc_round_trip_preserves_empty_comment(tmp_path):
    original = SimulationConfig(get("AnyScan.yaml"))
    original.set_comment("")
    smc_path = original.save_file(tmp_path / "empty_comment.smc")

    restored = SimulationConfig(smc_path)

    assert restored.get_comment() == ""


def test_data_file_description_accessors_create_and_return_entries():
    config = SimulationConfig(get("AnyScan.yaml"))
    config.set_data_file(12, "new_file")

    assert config.get_data_file(12) == "new_file"
    config.set_data_file("unknown_file_4", "replacement")
    assert config.get_data_file("unknown_file_4") == "replacement"


def test_uppercase_yaml_suffix_is_loaded_as_yaml(tmp_path):
    source = tmp_path / "config.YAML"
    source.write_text(Path(get("AnyScan.yaml")).read_text())

    config = SimulationConfig(source)

    assert config.get_value("photon_energy") == pytest.approx(150.0)


def test_smc_missing_basic_value_raises_value_error(tmp_path):
    text = _fixture_text().replace("7.0E+00 ", "", 1)
    path = tmp_path / "missing_value.smc"
    path.write_text(text)

    with pytest.raises(ValueError, match="basic data"):
        SimulationConfig(path)


def test_smc_flag_count_mismatch_raises_value_error(tmp_path):
    text = _fixture_text().replace(
        "     30  # Simulation flags", "     29  # Simulation flags"
    )
    path = tmp_path / "flag_mismatch.smc"
    path.write_text(text)

    with pytest.raises(ValueError, match="[Ss]imulation flag"):
        SimulationConfig(path)


def test_smc_text_count_larger_than_lines_raises_value_error(tmp_path):
    text = _fixture_text().replace(
        "      2  # Text Variables", "      5  # Text Variables"
    )
    path = tmp_path / "text_overflow.smc"
    path.write_text(text)

    with pytest.raises(ValueError, match="[Tt]ext"):
        SimulationConfig(path)


def test_smc_data_file_count_larger_than_lines_raises_value_error(tmp_path):
    text = _fixture_text().replace("     3  # Data files", "     9  # Data files")
    path = tmp_path / "data_overflow.smc"
    path.write_text(text)

    with pytest.raises(ValueError, match="[Dd]ata file"):
        SimulationConfig(path)


def test_smc_non_numeric_basic_value_raises_value_error(tmp_path):
    text = _fixture_text().replace("3.0E+00", "abc", 1)
    path = tmp_path / "non_numeric.smc"
    path.write_text(text)

    with pytest.raises(ValueError, match="basic data"):
        SimulationConfig(path)


def test_simulation_config_loading():
    """Test loading SimulationConfig from file."""
    config = SimulationConfig(get("AnyScan.yaml"))
    assert config is not None


def test_config_yaml_loading():
    """Test loading configuration from YAML file."""
    config = SimulationConfig(get("AnyScan.yaml"))

    # Test basic parameter access
    photon_energy = config.get_value("photon_energy")
    assert photon_energy > 0

    # Test flag access
    spect_study = config.get_flag("simulate_spect_study")
    assert isinstance(spect_study, bool)


@pytest.mark.skipif(
    "CI" in os.environ or "GITHUB_ACTIONS" in os.environ,
    reason="SMC config files not available in CI environment",
)
def test_config_smc_loading():
    """Test loading configuration from SMC file."""
    config = SimulationConfig(get("input.smc"))
    assert config is not None

    # Test parameter access
    photon_energy = config.get_value(1)  # Access by index
    assert photon_energy >= 0


def test_config_modification():
    """Test modifying configuration values."""
    config = SimulationConfig(get("AnyScan.yaml"))

    # Test value modification
    config.get_value("photon_energy")
    config.set_value("photon_energy", 150.0)
    assert config.get_value("photon_energy") == 150.0

    # Test flag modification
    config.set_flag("write_results_to_screen", True)
    assert config.get_flag("write_results_to_screen")


def test_config_yaml_export():
    """Test exporting configuration to YAML."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = SimulationConfig(get("AnyScan.yaml"))

        # Modify a parameter
        config.set_value("photon_energy", 140.0)

        # Export to YAML
        yaml_path = Path(temp_dir) / "test_config.yaml"
        config.export_yaml(yaml_path)

        # Verify file was created
        assert yaml_path.exists()

        # Load the exported config and verify
        new_config = SimulationConfig(yaml_path)
        assert new_config.get_value("photon_energy") == 140.0


def test_config_validation():
    """Test configuration parameter validation."""
    config = SimulationConfig(get("AnyScan.yaml"))

    # Test that validation runs without error
    # Note: validation returns True if no warnings, False if there are warnings
    result = config.validate_parameters()
    assert isinstance(result, bool)
