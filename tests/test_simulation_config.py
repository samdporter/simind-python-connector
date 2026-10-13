import math
import os
import tempfile
from pathlib import Path

import pytest
import yaml

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


def test_smc_yaml_round_trip_keeps_unnamed_values_and_flags(tmp_path):
    original = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    original.data[60] = 61.5
    original.flags = original.flags[:15] + "T" + original.flags[16:]
    yaml_path = tmp_path / "round_trip.yaml"
    original.export_yaml(yaml_path)

    restored = SimulationConfig(yaml_path)

    assert restored.data == original.data
    assert restored.flags == original.flags


def test_smc_yaml_smc_round_trip_keeps_unnamed_extended_slots(tmp_path):
    original = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    original.data[60] = 61.5
    original.data[115] = -9.25
    original.flags = original.flags[:24] + "T" + original.flags[25:]
    yaml_path = tmp_path / "extended_round_trip.yaml"
    original.export_yaml(yaml_path)

    smc_path = SimulationConfig(yaml_path).save_file(tmp_path / "extended_round_trip")
    restored = SimulationConfig(smc_path)

    assert restored.data == original.data
    assert restored.flags == original.flags


def test_yaml_without_other_fields_still_loads(tmp_path):
    original = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    original.data[60] = 61.5
    yaml_path = tmp_path / "with_other_fields.yaml"
    original.export_yaml(yaml_path)

    document = yaml.safe_load(yaml_path.read_text())
    document.pop("other_parameters")
    document.pop("other_flags")
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text(yaml.dump(document, sort_keys=False))

    restored = SimulationConfig(legacy_path)

    assert restored.get_value(11) == pytest.approx(11.0)
    assert len(restored.data) == 101
    assert restored.flags == original.flags[:15] + "F" * 15


def test_smc_yaml_round_trip_keeps_named_values_files_and_text_variables(tmp_path):
    original = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    original.set_value("photon_energy", 140.0)
    original.set_data_file("phantom_soft_tissue", "custom_phantom.dat")
    original.set_data_file(2, "custom_source.dat")
    original.text_variables[1] = "custom text variable"
    yaml_path = tmp_path / "named_round_trip.yaml"
    original.export_yaml(yaml_path)

    restored = SimulationConfig(yaml_path)

    assert restored.get_value("photon_energy") == pytest.approx(140.0)
    assert restored.get_data_file("phantom_soft_tissue") == "custom_phantom.dat"
    assert restored.get_data_file(2) == "custom_source.dat"
    assert restored.text_variables == original.text_variables


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


SHORT_SMC = (
    "SMCV2\n"
    "short-config\n"
    "      1  # Basic Change data\n"
    "1.0E+00\n"
    "      1  # Simulation flags\n"
    "T\n"
    "      0  # Text Variables\n"
    "      0  # Data files\n"
)


def _write_yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# A) Fixed-width save
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value, match",
    [
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (float("-inf"), "finite"),
        (1e100, "12-character SMC field"),
        (1e-300, "12-character SMC field"),
    ],
)
def test_smc_save_rejects_unrepresentable_value_without_truncating(
    tmp_path, bad_value, match
):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.data[0] = bad_value
    destination = tmp_path / "guarded.smc"
    destination.write_text("sentinel")

    with pytest.raises(ValueError, match=match):
        config.save_file(destination)

    assert destination.read_text() == "sentinel"


@pytest.mark.parametrize("bad_value", ["not-a-number", object(), 10**400])
def test_smc_save_rejects_non_numeric_value_without_truncating(tmp_path, bad_value):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.data[0] = bad_value
    destination = tmp_path / "numeric_guard.smc"
    destination.write_text("sentinel")

    with pytest.raises(ValueError, match="index 1"):
        config.save_file(destination)

    assert destination.read_text() == "sentinel"


def test_smc_save_does_not_create_destination_for_invalid_value(tmp_path):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.data[0] = 1e100
    destination = tmp_path / "never_written.smc"

    with pytest.raises(ValueError, match="index 1"):
        config.save_file(destination)

    assert not destination.exists()


@pytest.mark.parametrize("bad_comment", [None, 7])
def test_smc_save_rejects_non_string_comment_without_touching_destination(
    tmp_path, bad_comment
):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.set_comment(bad_comment)
    destination = tmp_path / "bad_comment.smc"
    destination.write_bytes(b"sentinel")

    with pytest.raises(ValueError, match="comment"):
        config.save_file(destination)

    assert destination.read_bytes() == b"sentinel"


def test_smc_save_rejects_null_flags_without_touching_destination(tmp_path):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.flags = None
    destination = tmp_path / "bad_flags.smc"
    destination.write_bytes(b"sentinel")

    with pytest.raises(ValueError, match="flags"):
        config.save_file(destination)

    assert destination.read_bytes() == b"sentinel"


@pytest.mark.parametrize("bad_value", [None, 5, object()])
def test_smc_save_rejects_non_string_text_variable_without_touching_destination(
    tmp_path, bad_value
):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.text_variables[1] = bad_value
    destination = tmp_path / "bad_text_variable.smc"
    destination.write_bytes(b"sentinel")

    with pytest.raises(ValueError, match=r"text variables.*slot 1"):
        config.save_file(destination)

    assert destination.read_bytes() == b"sentinel"


def test_smc_save_rejects_null_data_file_without_touching_destination(tmp_path):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.data_files[2] = None
    destination = tmp_path / "bad_filepath.smc"
    destination.write_bytes(b"sentinel")

    with pytest.raises(ValueError, match=r"data files.*slot 2"):
        config.save_file(destination)

    assert destination.read_bytes() == b"sentinel"


def test_smc_save_fills_missing_slots_with_none(tmp_path):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.text_variables = {2: "only"}
    config.data_files = {3: "crystal.dat"}

    lines = config.save_file(tmp_path / "gaps.smc").read_text().splitlines()

    text_header = lines.index("2  # Text Variables")
    assert lines[text_header + 1] == "none"
    assert lines[text_header + 2] == "only"

    file_header = lines.index("3 # Data files")
    assert lines[file_header + 1] == "none".ljust(60)
    assert lines[file_header + 2] == "none".ljust(60)
    assert lines[file_header + 3] == "crystal.dat".ljust(60)


def test_smc_save_writes_expected_bytes(tmp_path):
    source = tmp_path / "short.smc"
    source.write_text(SHORT_SMC)
    config = SimulationConfig(source)

    smc_path = config.save_file(tmp_path / "short_out.smc")

    zero = " 0.00000E+00"
    fields = [" 0.10000E+01"] + [zero] * 119
    basic_lines = ["".join(fields[i : i + 5]) for i in range(0, 120, 5)]
    comment = "short-config" + " " * (70 - len("short-config"))
    expected = "".join(
        [
            "SMCV2\n",
            f"{comment}\n",
            "   120  # Basic Change data\n",
            *[f"{line}\n" for line in basic_lines],
            "1  # Simulation flags\n",
            "T\n",
            "0  # Text Variables\n",
            "0 # Data files\n",
        ]
    )

    assert smc_path.read_text() == expected


def test_smc_save_writes_twelve_character_basic_fields(tmp_path):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.data[0] = -0.0
    config.data[1] = -12.5

    smc_path = config.save_file(tmp_path / "widths.smc")
    basic_lines = smc_path.read_text().splitlines()[3:27]

    assert len(basic_lines) == 24
    for line in basic_lines:
        assert len(line) == 60
        for start in range(0, 60, 12):
            assert len(line[start : start + 12]) == 12


def test_smc_round_trip_canonicalises_negative_zero_and_negatives(tmp_path):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.data[0] = -0.0
    config.data[1] = -12.5
    config.data[2] = -0.0

    restored = SimulationConfig(config.save_file(tmp_path / "negatives.smc"))

    assert restored.data[0] == 0.0
    assert math.copysign(1.0, restored.data[0]) == 1.0
    assert restored.data[1] == pytest.approx(-12.5)
    assert restored.data[2] == 0.0
    assert math.copysign(1.0, restored.data[2]) == 1.0


@pytest.mark.parametrize("value", [9.9999e98, 1e-100, 9.999999e-101])
def test_smc_save_accepts_values_inside_exponent_boundary(tmp_path, value):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.data[0] = value

    restored = SimulationConfig(config.save_file(tmp_path / "boundary_ok.smc"))

    assert restored.data[0] == pytest.approx(value)


@pytest.mark.parametrize("value", [9.999999e98, 1e99, 1e-101])
def test_smc_save_rejects_values_past_exponent_boundary(tmp_path, value):
    config = SimulationConfig(FIXTURES / "minimal_variable_sections.smc")
    config.data[0] = value

    with pytest.raises(ValueError, match="index 1"):
        config.save_file(tmp_path / "boundary_bad.smc")


# ---------------------------------------------------------------------------
# B) Short-config diagnostics
# ---------------------------------------------------------------------------


def test_short_smc_print_config_only_prints_present_named_entries(tmp_path, capsys):
    path = tmp_path / "short.smc"
    path.write_text(SHORT_SMC)
    config = SimulationConfig(path)

    config.print_config()

    output = capsys.readouterr().out
    assert "photon_energy" in output
    assert "write_results_to_screen" in output
    # flag 4 (include_collimator) and crystal thickness (index 9) are absent.
    assert "include_collimator" not in output
    assert "crystal_thickness" not in output


def test_short_smc_validate_parameters_reports_missing_without_mutating(
    tmp_path, capsys
):
    path = tmp_path / "short.smc"
    path.write_text(SHORT_SMC)
    config = SimulationConfig(path)
    data_before = list(config.data)
    flags_before = config.flags

    result = config.validate_parameters()

    output = capsys.readouterr().out
    assert result is False
    assert "not present" in output
    assert config.data == data_before
    assert config.flags == flags_before
    assert config.data == [1.0]


# ---------------------------------------------------------------------------
# C) YAML robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "- one\n- two\n", "just a scalar\n", "42\n"])
def test_yaml_non_mapping_root_raises(tmp_path, text):
    with pytest.raises(ValueError, match="root"):
        SimulationConfig(_write_yaml(tmp_path, text))


def test_yaml_empty_mapping_root_is_a_valid_defaults_config(tmp_path):
    config = SimulationConfig(_write_yaml(tmp_path, "{}\n"))

    assert config.get_value("photon_energy") == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "parameters: [1, 2, 3]\n",
        "parameters:\n  source: [1]\n",
        "parameters:\n  source:\n    photon_energy: 5\n",
        "data_files: [1]\n",
        "text_variables: [1]\n",
        "other_parameters: [1]\n",
    ],
)
def test_yaml_non_mapping_sections_groups_and_entries_raise(tmp_path, text):
    with pytest.raises(ValueError, match="must be a mapping"):
        SimulationConfig(_write_yaml(tmp_path, text))


def test_yaml_scalar_data_file_entry_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="data_files"):
        SimulationConfig(_write_yaml(tmp_path, "data_files:\n  '1': x\n"))


def test_yaml_quoted_indices_are_normalised(tmp_path):
    text = (
        "parameters:\n"
        "  source:\n"
        "    photon_energy:\n"
        "      index: '1'\n"
        "      value: 140.0\n"
        "simulation_flags:\n"
        "  write_results_to_screen:\n"
        "    index: '1'\n"
        "    enabled: true\n"
        "data_files:\n"
        "  phantom_soft_tissue:\n"
        "    index: '1'\n"
        "    filepath: my_phantom\n"
        "text_variables:\n"
        "  '1': first\n"
        "  '3': third\n"
        "other_parameters:\n"
        "  '61': 5.0\n"
        "other_flags:\n"
        "  '16': true\n"
    )
    config = SimulationConfig(_write_yaml(tmp_path, text))

    assert config.get_value("photon_energy") == pytest.approx(140.0)
    assert config.get_flag("write_results_to_screen") is True
    assert config.get_data_file("phantom_soft_tissue") == "my_phantom"
    assert config.text_variables == {1: "first", 3: "third"}
    assert config.data[60] == pytest.approx(5.0)
    assert config.flags[15] == "T"


@pytest.mark.parametrize(
    "bad_key",
    ["'0'", "-1", "true", "1.5", "'abc'", "lint"],
)
def test_yaml_invalid_indices_are_rejected(tmp_path, bad_key):
    text = f"other_parameters:\n  {bad_key}: 1.0\n"

    with pytest.raises(ValueError, match="positive integer"):
        SimulationConfig(_write_yaml(tmp_path, text))


def test_yaml_invalid_parameter_index_field_is_rejected(tmp_path):
    text = (
        "parameters:\n"
        "  source:\n"
        "    photon_energy:\n"
        "      index: 0\n"
        "      value: 140.0\n"
    )

    with pytest.raises(ValueError, match="positive integer"):
        SimulationConfig(_write_yaml(tmp_path, text))


@pytest.mark.parametrize(
    "section, first, second",
    [
        ("text_variables", "'1': a", "1: b"),
        ("other_parameters", "'61': 1.0", "61: 2.0"),
        ("other_flags", "'16': true", "16: false"),
    ],
)
def test_yaml_colliding_normalised_indices_are_rejected(
    tmp_path, section, first, second
):
    text = f"{section}:\n  {first}\n  {second}\n"

    with pytest.raises(ValueError, match="duplicate"):
        SimulationConfig(_write_yaml(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        (
            "parameters:\n"
            "  source:\n"
            "    photon_energy:\n"
            "      index: 1\n"
            "      value: 140.0\n"
            "  phantom:\n"
            "    phantom_type:\n"
            "      index: 1\n"
            "      value: 2.0\n"
        ),
        (
            "data_files:\n"
            "  phantom_soft_tissue:\n"
            "    index: 1\n"
            "    filepath: a\n"
            "  phantom_bone:\n"
            "    index: 1\n"
            "    filepath: b\n"
        ),
        (
            "simulation_flags:\n"
            "  write_results_to_screen:\n"
            "    index: 1\n"
            "    enabled: true\n"
            "  write_images_to_files:\n"
            "    index: 1\n"
            "    enabled: false\n"
        ),
    ],
)
def test_yaml_colliding_canonical_index_fields_are_rejected(tmp_path, text):
    with pytest.raises(ValueError, match="duplicate index"):
        SimulationConfig(_write_yaml(tmp_path, text))


def test_yaml_sparse_text_variable_slots_are_retained(tmp_path):
    config = SimulationConfig(_write_yaml(tmp_path, "text_variables:\n  '2': only\n"))

    assert config.text_variables == {2: "only"}


def test_yaml_canonical_data_file_entries_are_preserved(tmp_path):
    text = (
        "data_files:\n"
        "  phantom_soft_tissue:\n"
        "    index: 1\n"
        "    filepath: phantom.dat\n"
        "  crystal_material:\n"
        "    index: 4\n"
        "    filepath: crystal.dat\n"
    )
    config = SimulationConfig(_write_yaml(tmp_path, text))

    assert config.get_data_file("phantom_soft_tissue") == "phantom.dat"
    assert config.get_data_file("crystal_material") == "crystal.dat"
    assert config.data_files[2] == "none"
