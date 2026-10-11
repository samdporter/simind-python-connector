import pytest

from simind_python_connector.configs import get
from simind_python_connector.core.config import SimulationConfig


pytestmark = pytest.mark.unit


def _value(cfg: SimulationConfig, key: str) -> float:
    return float(cfg.get_value(key))


def test_example_yaml_uses_smaller_projection_space_than_anyscan():
    anyscan = SimulationConfig(get("AnyScan.yaml"))
    example = SimulationConfig(get("Example.yaml"))

    assert _value(example, "spect_no_projections") < _value(
        anyscan, "spect_no_projections"
    )
    assert _value(example, "matrix_size_image_i") < _value(
        anyscan, "matrix_size_image_i"
    )
    assert _value(example, "matrix_size_image_j") < _value(
        anyscan, "matrix_size_image_j"
    )
    assert _value(example, "number_density_images") < _value(
        anyscan, "number_density_images"
    )
    assert _value(example, "matrix_size_density_map_i") < _value(
        anyscan, "matrix_size_density_map_i"
    )
    assert _value(example, "matrix_size_source_map_i") < _value(
        anyscan, "matrix_size_source_map_i"
    )


def test_example_yaml_enables_spect_flag():
    example = SimulationConfig(get("Example.yaml"))
    assert example.get_flag("simulate_spect_study")


def test_example_simind_check_accepts_simind_bin(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    helpers_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "_python_connector_helpers.py"
    )
    spec = importlib.util.spec_from_file_location("_helpers", helpers_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    executable = tmp_path / "custom_simind"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("SIMIND_BIN", str(executable))

    helpers.require_simind()

    monkeypatch.delenv("SIMIND_BIN")
    with pytest.raises(RuntimeError, match="SIMIND executable not found"):
        helpers.require_simind()
