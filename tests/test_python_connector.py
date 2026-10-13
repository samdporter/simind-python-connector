import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from simind_python_connector.configs import get
from simind_python_connector.connectors import RuntimeOperator, SimindPythonConnector
from simind_python_connector.core import SimulationConfig
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    MAX_SOURCE,
    OutputError,
    ScoringRoutine,
    SimindError,
    SimulationError,
)


def _case_insensitive_filesystem() -> bool:
    with tempfile.TemporaryDirectory() as directory:
        probe = Path(directory) / "case_probe"
        probe.write_bytes(b"probe")
        return (Path(directory) / "CASE_PROBE").exists()


CASE_INSENSITIVE_FS = _case_insensitive_filesystem()


@pytest.mark.unit
def test_python_connector_requires_run_before_get_outputs(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    with pytest.raises(RuntimeError, match="Run the connector first"):
        connector.get_outputs()


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_prefix",
    [
        "../escape",
        "/absolute",
        "",
        "a/b",
        "a\\b",
        "..",
        "case.01",
        "case[1]",
        "Case01",
        "-case",
        "cx-run",
        "case-",
        "case--01",
    ],
)
def test_python_connector_rejects_unsafe_output_prefixes(
    tmp_path: Path, bad_prefix: str
):
    output_dir = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="output_prefix"):
        SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=output_dir,
            output_prefix=bad_prefix,
        )
    assert not output_dir.exists()


@pytest.mark.unit
@pytest.mark.parametrize("good_prefix", ["case_01", "case01", "0_case", "a"])
def test_python_connector_accepts_safe_output_prefixes(
    tmp_path: Path, good_prefix: str
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path / good_prefix,
        output_prefix=good_prefix,
    )
    assert connector.output_prefix == good_prefix


@pytest.mark.unit
def test_python_connector_rejects_non_finite_quantization_scale(tmp_path: Path):
    import math

    for bad_scale in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="quantization_scale"):
            SimindPythonConnector(
                config_source=get("AnyScan.yaml"),
                output_dir=tmp_path,
                output_prefix="case01",
                quantization_scale=bad_scale,
            )


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_sets_all_map_dimensions(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    source = np.zeros((2, 3, 4), dtype=np.float32)
    source[0, 0, 0] = 1.0

    connector.configure_voxel_phantom(
        source=source,
        mu_map=np.zeros_like(source),
        voxel_size_mm=4.0,
    )

    config = connector.get_config()
    assert config.get_value(78) == pytest.approx(4)  # density map i = dim_x
    assert config.get_value(79) == pytest.approx(4)  # source map i = dim_x
    assert config.get_value(81) == pytest.approx(3)  # density map j = dim_y
    assert config.get_value(82) == pytest.approx(3)  # source map j = dim_y
    assert config.get_value(34) == pytest.approx(2)  # number of density images


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_keeps_the_config_projection_size(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    config = connector.get_config()
    preset = [config.get_value(index) for index in (28, 76, 77)]
    source = np.ones((2, 3, 4), dtype=np.float32)

    connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)
    assert [config.get_value(index) for index in (28, 76, 77)] == preset

    connector.add_config_value(28, 0.25)
    connector.add_config_value(76, 96)
    connector.add_config_value(77, 40)
    connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)
    assert config.get_value(28) == pytest.approx(0.25)
    assert (config.get_value(76), config.get_value(77)) == (96, 40)


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_keeps_the_config_scoring_routine(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(84, ScoringRoutine.PENETRATE.value)
    source = np.ones((2, 3, 4), dtype=np.float32)

    connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)

    assert int(connector.get_config().get_value(84)) == ScoringRoutine.PENETRATE.value


def _run_without_simind(connector: SimindPythonConnector) -> None:
    connector.executor.run_simulation = lambda *a, **k: None  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run()


def _record_executor_run(connector: SimindPythonConnector) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, timeout=None
    ):
        captured["output_prefix"] = output_prefix
        captured["orbit_file"] = orbit_file
        captured["runtime_switches"] = dict(runtime_switches or {})
        captured["cwd"] = cwd
        captured["timeout"] = timeout

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    return captured


def _write_smc_with_basic_value_count(
    source: Path, destination: Path, value_count: int
) -> Path:
    """Write a genuine SMC whose basic-data section holds ``value_count`` values.

    SIMIND writes basic data as fixed 12-character fields, so slicing the raw
    section at that width reproduces a short config the parser reads as-is.
    """
    lines = source.read_text().splitlines()
    basic_header = next(
        index for index, line in enumerate(lines) if "# Basic Change data" in line
    )
    flags_header = next(
        index for index, line in enumerate(lines) if "# Simulation flags" in line
    )
    basic_text = "".join(lines[basic_header + 1 : flags_header])
    kept = basic_text[: value_count * 12]
    output = lines[:basic_header]
    output.append(f"   {value_count}  # Basic Change data")
    output.extend(kept[start : start + 60] for start in range(0, len(kept), 60))
    output.extend(lines[flags_header:])
    destination.write_text("\n".join(output) + "\n")
    return destination


@pytest.mark.unit
def test_run_warns_when_the_phantom_is_larger_than_the_field_of_view(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.add_config_value(28, 0.4)
    connector.add_config_value(76, 8)  # 32 mm across
    connector.add_config_value(77, 8)  # 32 mm axially
    source = np.ones((4, 16, 16), dtype=np.float32)  # 64 mm across, 16 mm axially
    connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)

    with caplog.at_level("WARNING"):
        _run_without_simind(connector)
    assert "wider than the projection field of view" in caplog.text
    assert "longer than" not in caplog.text

    caplog.clear()
    connector.add_config_value(76, 32)
    connector.add_config_value(77, 2)  # 8 mm axially
    with caplog.at_level("WARNING"):
        _run_without_simind(connector)
    assert "longer than the projection field of view" in caplog.text
    assert "wider than" not in caplog.text


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_rejects_non_finite_values(
    tmp_path: Path,
):
    import math

    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    good = np.zeros((2, 3, 4), dtype=np.float32)

    for bad_value in (math.nan, math.inf, -math.inf):
        bad = good.copy()
        bad[0, 0, 0] = bad_value
        with pytest.raises(ValueError, match="finite"):
            connector.configure_voxel_phantom(source=bad, mu_map=good)
        with pytest.raises(ValueError, match="finite"):
            connector.configure_voxel_phantom(source=good, mu_map=bad)


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_rejects_negative_values(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    good = np.zeros((2, 3, 4), dtype=np.float32)
    negative = good.copy()
    negative[0, 0, 0] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        connector.configure_voxel_phantom(source=negative, mu_map=good)
    with pytest.raises(ValueError, match="non-negative"):
        connector.configure_voxel_phantom(source=good, mu_map=negative)


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_rejects_empty_arrays(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    empty = np.zeros((0, 0, 0), dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        connector.configure_voxel_phantom(source=empty, mu_map=empty)


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_rejects_invalid_scoring_routine(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    source = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="scoring_routine"):
        connector.configure_voxel_phantom(
            source=source,
            mu_map=source.copy(),
            scoring_routine=1.5,
        )
    with pytest.raises(ValueError, match="scoring_routine"):
        connector.configure_voxel_phantom(
            source=source,
            mu_map=source.copy(),
            scoring_routine=999,
        )


@pytest.mark.unit
def test_python_connector_runtime_operator_switches_are_one_shot(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=tmp_path / "run1",
        output_prefix="case01",
    )

    captured_switches: list[dict] = []

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, timeout=None
    ):
        captured_switches.append(dict(runtime_switches or {}))

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run(RuntimeOperator(switches={"RR": 111}))
    connector.run()

    assert "RR" not in connector.runtime_switches.switches
    assert captured_switches[0].get("RR") == 111
    assert "RR" not in captured_switches[1]


@pytest.mark.unit
def test_python_connector_passes_output_dir_as_cwd(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=tmp_path / "isolated",
        output_prefix="case01",
    )

    seen_cwd: list[object] = []

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, timeout=None
    ):
        seen_cwd.append(cwd)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run()

    assert seen_cwd == [connector.output_dir]
    assert Path.cwd() != connector.output_dir


@pytest.mark.unit
def test_python_connector_accepts_quantization_scale_parameter(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        quantization_scale=0.05 * DEFAULT_QUANTIZATION_SCALE,
    )
    assert connector.quantization_scale == pytest.approx(
        0.05 * DEFAULT_QUANTIZATION_SCALE
    )


@pytest.mark.unit
def test_source_ceiling_is_signed_int16():
    assert MAX_SOURCE == np.iinfo(np.int16).max
    assert MAX_SOURCE == 32767


def _written_source(tmp_path: Path, scale: float, values: list[float]) -> np.ndarray:
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        quantization_scale=scale,
    )
    source = np.zeros((1, 1, len(values)), dtype=np.float32)
    source[0, 0, :] = values
    source_path, _ = connector.configure_voxel_phantom(
        source, np.zeros_like(source), voxel_size_mm=4.0
    )
    return np.fromfile(source_path, dtype=np.uint16).reshape(source.shape)


@pytest.mark.unit
def test_python_connector_explicit_full_scale_uses_signed_range(tmp_path: Path):
    written = _written_source(tmp_path, 1.0, [1.0, 0.6, 0.3])
    assert tuple(written[0, 0]) == (32767, 19660, 9830)


@pytest.mark.unit
def test_python_connector_explicit_scale_is_not_auto_migrated(tmp_path: Path):
    written = _written_source(tmp_path, 0.05, [1.0, 0.6, 0.3])
    assert tuple(written[0, 0]) == (1638, 983, 492)


@pytest.mark.unit
@pytest.mark.parametrize("old_scale", [1.0, 0.05, 0.02, 0.01])
def test_migrated_scale_matches_legacy_source_bytes(tmp_path: Path, old_scale: float):
    written = _written_source(
        tmp_path, old_scale * DEFAULT_QUANTIZATION_SCALE, [1.0, 0.6, 0.3]
    )
    source = np.array([[[1.0, 0.6, 0.3]]], dtype=np.float32)
    expected = np.clip(
        np.round(source / source.max() * (500 * old_scale)), 0, 500
    ).astype(np.uint16)
    assert np.array_equal(written, expected)


@pytest.mark.unit
def test_migrated_non_integral_budget_rounds_once(tmp_path: Path):
    written = _written_source(tmp_path, 0.0032 * DEFAULT_QUANTIZATION_SCALE, [1.0, 0.3])
    assert tuple(written[0, 0]) == (2, 0)


@pytest.mark.unit
def test_python_connector_density_capped_at_signed_int16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.get_config().set_flag(11, True)
    monkeypatch.setattr(
        "simind_python_connector.connectors.python_connector.attenuation_to_density",
        lambda mu_map, energy: np.array([[[0.15, 10.0, 40.0]]], dtype=np.float32),
    )
    source = np.ones((1, 1, 3), dtype=np.float32)
    mu_map = np.zeros_like(source)

    _, density_path = connector.configure_voxel_phantom(
        source, mu_map, voxel_size_mm=4.0
    )

    written = np.fromfile(density_path, dtype=np.uint16)
    assert tuple(written) == (150, 10000, 32767)


@pytest.mark.unit
@pytest.mark.parametrize("scale", [0.0, -1.0, 2.0])
def test_python_connector_rejects_quantization_scale_outside_unit_interval(
    tmp_path: Path, scale: float
):
    # Above 1 the source is clipped at MAX_SOURCE, which changes its shape.
    with pytest.raises(ValueError, match=r"quantization_scale must be in \(0, 1\]"):
        SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path,
            output_prefix="case01",
            quantization_scale=scale,
        )


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, -1, 1.5, None, True])
def test_python_connector_rejects_invalid_nn(tmp_path: Path, value):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path
    )
    with pytest.raises(ValueError, match="NN"):
        connector.add_runtime_switch("NN", value)


@pytest.mark.unit
def test_python_connector_accepts_integral_nn(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path
    )
    connector.add_runtime_switch("NN", 3)
    connector.add_runtime_switch("Multiplier for scaling the number of counts", 2.0)
    assert connector.runtime_switches.switches["NN"] == 2.0


@pytest.mark.unit
@pytest.mark.parametrize("value", [np.int32(4), np.int64(4)])
def test_python_connector_accepts_numpy_integer_mp(tmp_path: Path, value):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path
    )
    connector.add_runtime_switch("MP", value)
    assert connector.runtime_switches.switches["MP"] == value


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_python_connector_rejects_boolean_mp(tmp_path: Path, value):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path
    )
    with pytest.raises(ValueError, match="MP must be an integer >= 1"):
        connector.add_runtime_switch("MP", value)


@pytest.mark.unit
def test_python_connector_validates_numpy_integer_mp_in_runtime_operator(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path
    )
    connector._validate_runtime_operator(RuntimeOperator(switches={"MP": np.int64(4)}))


@pytest.mark.unit
def test_python_connector_runtime_operator_numpy_integer_mp_is_one_shot(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    captured = _record_executor_run(connector)

    connector.run(RuntimeOperator(switches={"MP": np.int64(4)}))

    assert captured["runtime_switches"] == {"MP": np.int64(4)}
    assert "MP" not in connector.runtime_switches.switches


@pytest.mark.unit
def test_python_connector_keeps_integral_float_nn_stored_value(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    captured = _record_executor_run(connector)

    connector.run(RuntimeOperator(switches={"NN": 2.0}))

    assert captured["runtime_switches"] == {"NN": 2.0}
    assert "NN" not in connector.runtime_switches.switches


@pytest.mark.unit
def test_python_connector_run_returns_numpy_outputs(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    captured: dict[str, object] = {}

    def fake_run_simulation(
        output_prefix: str,
        orbit_file=None,
        runtime_switches=None,
        cwd=None,
        timeout=None,
    ) -> None:
        captured["output_prefix"] = output_prefix
        captured["orbit_file"] = orbit_file
        captured["runtime_switches"] = dict(runtime_switches or {})

        projection = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        data_path = tmp_path / f"{output_prefix}_tot_w1.a00"
        header_path = tmp_path / f"{output_prefix}_tot_w1.hs"

        projection.tofile(data_path)
        header_path.write_text(
            "\n".join(
                [
                    "!INTERFILE :=",
                    "!number format := float",
                    "!number of bytes per pixel := 4",
                    "imagedata byte order := LITTLEENDIAN",
                    "!matrix size [1] := 4",
                    "!matrix size [2] := 3",
                    "!matrix size [3] := 2",
                    f"!name of data file := {data_path.name}",
                    "!END OF INTERFILE :=",
                ]
            )
        )

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector.config.set_flag(5, False)
    connector.config.set_flag(14, False)
    outputs = connector.run(RuntimeOperator(switches={"NN": 2, "RR": 12345}))

    # SPECT Interfile headers are needed to read the outputs.
    assert connector.config.get_flag(5)
    assert connector.config.get_flag(14)
    assert captured["output_prefix"] == "case01"
    assert captured["runtime_switches"] == {"NN": 2, "RR": 12345}

    assert "tot_w1" in outputs
    result = outputs["tot_w1"]
    assert result.header_path == (tmp_path / "case01_tot_w1.hs").resolve()
    assert result.data_path == (tmp_path / "case01_tot_w1.a00").resolve()
    assert result.projection.shape == (2, 3, 4)
    expected_sum = float(np.arange(24, dtype=np.float32).sum())
    assert float(result.projection.sum()) == expected_sum


@pytest.mark.unit
def test_python_connector_raises_for_malformed_interfile_outputs(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    def fake_run_simulation(
        output_prefix: str,
        orbit_file=None,
        runtime_switches=None,
        cwd=None,
        timeout=None,
    ) -> None:
        good_projection = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        good_data_path = tmp_path / f"{output_prefix}_tot_w1.a00"
        good_header_path = tmp_path / f"{output_prefix}_tot_w1.hs"
        good_projection.tofile(good_data_path)
        good_header_path.write_text(
            "\n".join(
                [
                    "!INTERFILE :=",
                    "!number format := float",
                    "!number of bytes per pixel := 4",
                    "imagedata byte order := LITTLEENDIAN",
                    "!matrix size [1] := 4",
                    "!matrix size [2] := 3",
                    "!matrix size [3] := 2",
                    f"!name of data file := {good_data_path.name}",
                    "!END OF INTERFILE :=",
                ]
            )
        )

        bad_projection = np.arange(16, dtype=np.float32).reshape(4, 4)
        bad_data_path = tmp_path / f"{output_prefix}_air_w1.a00"
        bad_header_path = tmp_path / f"{output_prefix}_air_w1.hs"
        bad_projection.tofile(bad_data_path)
        bad_header_path.write_text(
            "\n".join(
                [
                    "!INTERFILE :=",
                    "!number format := float",
                    "!number of bytes per pixel := 4",
                    "imagedata byte order := LITTLEENDIAN",
                    "!matrix size [1] := 4",
                    "!matrix size [2] := 4",
                    "!matrix size [3] := 4",
                    f"!name of data file := {bad_data_path.name}",
                    "!END OF INTERFILE :=",
                ]
            )
        )

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="case01_air_w1.hs could not be read"):
        connector.run(RuntimeOperator(switches={"NN": 1}))


@pytest.mark.unit
def test_python_connector_penetrate_uses_bxx_component_headers(
    tmp_path: Path, monkeypatch
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(84, 4)

    def fake_run_simulation(
        output_prefix: str,
        orbit_file=None,
        runtime_switches=None,
        cwd=None,
        timeout=None,
    ) -> None:
        (tmp_path / f"{output_prefix}.h00").write_text(
            "\n".join(["!INTERFILE :=", "!END OF INTERFILE :="])
        )

        projection = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        data_path = tmp_path / f"{output_prefix}.b01"
        header_path = tmp_path / f"{output_prefix}_component_01.hs"
        projection.tofile(data_path)
        header_path.write_text(
            "\n".join(
                [
                    "!INTERFILE :=",
                    "!number format := float",
                    "!number of bytes per pixel := 4",
                    "imagedata byte order := LITTLEENDIAN",
                    "!matrix size [1] := 4",
                    "!matrix size [2] := 3",
                    "!matrix size [3] := 2",
                    f"!name of data file := {data_path.name}",
                    "!END OF INTERFILE :=",
                ]
            )
        )

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    monkeypatch.setattr(
        connector.converter,
        "find_penetrate_h00_file",
        lambda output_prefix, output_dir: str(tmp_path / f"{output_prefix}.h00"),
    )
    monkeypatch.setattr(
        connector.converter,
        "create_penetrate_headers_from_template",
        lambda h00_file, output_prefix, output_dir: {},
    )

    outputs = connector.run(RuntimeOperator(switches={"NN": 1}))

    assert "all_interactions" in outputs
    result = outputs["all_interactions"]
    assert result.data_path == (tmp_path / "case01.b01").resolve()
    assert result.projection.shape == (2, 3, 4)


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_writes_input_files(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    source = np.zeros((4, 5, 6), dtype=np.float32)
    source[1:3, 2:4, 2:5] = 1.0
    mu_map = np.full_like(source, 0.15, dtype=np.float32)

    source_path, density_path = connector.configure_voxel_phantom(
        source=source,
        mu_map=mu_map,
        voxel_size_mm=4.0,
        scoring_routine=1,
    )

    assert source_path.exists()
    assert density_path.exists()
    assert source_path.name == "case01_src.smi"
    assert density_path.name == "case01_dns.dmi"
    assert connector.runtime_switches.switches["PX"] == pytest.approx(0.4)

    source_u16 = np.fromfile(source_path, dtype=np.uint16)
    density_u16 = np.fromfile(density_path, dtype=np.uint16)
    assert source_u16.size == source.size
    assert density_u16.size == mu_map.size
    assert source_u16.max() > 0


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_rejects_shape_mismatch(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    source = np.zeros((4, 5, 6), dtype=np.float32)
    mu_map = np.zeros((4, 5, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="identical shapes"):
        connector.configure_voxel_phantom(source=source, mu_map=mu_map)


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_rejects_non_3d_inputs(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    source_2d = np.zeros((4, 5), dtype=np.float32)
    mu_2d = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="must both be 3D arrays"):
        connector.configure_voxel_phantom(source=source_2d, mu_map=mu_2d)

    source_4d = np.zeros((2, 3, 4, 5), dtype=np.float32)
    mu_4d = np.zeros((2, 3, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="must both be 3D arrays"):
        connector.configure_voxel_phantom(source=source_4d, mu_map=mu_4d)


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_rejects_non_positive_voxel_size(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    source = np.zeros((4, 5, 6), dtype=np.float32)
    mu_map = np.zeros_like(source)

    with pytest.raises(ValueError, match="voxel_size_mm must be > 0"):
        connector.configure_voxel_phantom(
            source=source,
            mu_map=mu_map,
            voxel_size_mm=0.0,
        )

    with pytest.raises(ValueError, match="voxel_size_mm must be > 0"):
        connector.configure_voxel_phantom(
            source=source,
            mu_map=mu_map,
            voxel_size_mm=-4.0,
        )


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_zeroes_density_when_attenuation_off(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.get_config().set_flag(11, False)

    source = np.zeros((4, 5, 6), dtype=np.float32)
    source[1:3, 1:4, 1:5] = 1.0
    mu_map = np.full_like(source, 0.25, dtype=np.float32)

    _, density_path = connector.configure_voxel_phantom(source=source, mu_map=mu_map)
    density_u16 = np.fromfile(density_path, dtype=np.uint16)
    assert density_u16.size == mu_map.size
    assert np.all(density_u16 == 0)


@pytest.mark.unit
def test_python_connector_negative_index_1_converts_the_mu_map_at_its_magnitude(
    tmp_path: Path,
):
    source = np.ones((2, 3, 4), dtype=np.float32)
    mu_map = np.full_like(source, 0.15)
    densities = []
    for energy in (140.0, -140.0):
        connector = SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path / str(energy),
            output_prefix="case01",
        )
        connector.get_config().set_flag(11, True)
        connector.add_config_value(1, energy)
        _, density_path = connector.configure_voxel_phantom(source, mu_map)
        densities.append(np.fromfile(density_path, dtype=np.uint16))

    assert densities[0].min() > 900  # about water
    assert np.array_equal(densities[0], densities[1])


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_accepts_scoring_routine_enum(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    source = np.zeros((4, 5, 6), dtype=np.float32)
    mu_map = np.zeros_like(source)

    connector.configure_voxel_phantom(
        source=source,
        mu_map=mu_map,
        scoring_routine=ScoringRoutine.PENETRATE,
    )

    assert int(connector.get_config().get_value(84)) == ScoringRoutine.PENETRATE.value


@pytest.mark.unit
def test_python_connector_set_energy_windows_writes_window_file(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    connector.set_energy_windows([126.0], [154.0], [0])
    window_file = tmp_path / "case01.win"
    assert window_file.exists()

    lines = [line.strip() for line in window_file.read_text().splitlines() if line]
    assert lines[0] == "126.0,154.0,0"


@pytest.mark.unit
def test_python_connector_cleanup_preserves_window_file_it_wrote(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.set_energy_windows([126.0], [154.0], [0])
    stale = tmp_path / "case01_tot_w1.hs"
    stale.write_text("stale")

    connector._clear_previous_outputs()

    assert (tmp_path / "case01.win").exists()
    assert not stale.exists()


@pytest.mark.unit
def test_python_connector_cleanup_removes_stale_window_file(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    stale_window = tmp_path / "case01.win"
    stale_window.write_text("126.0,154.0,0\n")

    connector._clear_previous_outputs()

    assert not stale_window.exists()


@pytest.mark.unit
def test_python_connector_cleanup_only_protects_window_file_it_wrote(tmp_path: Path):
    written_dir = tmp_path / "written"
    rerun_dir = tmp_path / "rerun"
    written_dir.mkdir()
    rerun_dir.mkdir()
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=written_dir,
        output_prefix="case01",
    )
    connector.set_energy_windows([126.0], [154.0], [0])

    stale_window = rerun_dir / "case01.win"
    stale_window.write_text("stale")
    connector.output_dir = rerun_dir

    connector._clear_previous_outputs()

    assert not stale_window.exists()
    assert (written_dir / "case01.win").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "orbit_name",
    [
        "case01_src.smi",
        "case01_dns.dmi",
        "case01.win",
        "case01.smc",
        "orbit-with-spaces-and-hyphens.cor",
    ],
)
def test_python_connector_stages_orbit_away_from_protected_inputs(
    tmp_path: Path, orbit_name: str
):
    output_dir = tmp_path / "out"
    source_dir = tmp_path / "in"
    output_dir.mkdir()
    source_dir.mkdir()
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=output_dir,
        output_prefix="case01",
    )
    source = np.zeros((2, 3, 4), dtype=np.float32)
    source_path, density_path = connector.configure_voxel_phantom(source, source.copy())
    connector.set_energy_windows([126.0], [154.0], [0])

    orbit_source = source_dir / orbit_name
    orbit_source.write_bytes(b"orbit-original")
    inputs_before = {
        source_path: source_path.read_bytes(),
        density_path: density_path.read_bytes(),
        output_dir / "case01.win": (output_dir / "case01.win").read_bytes(),
    }

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=orbit_source))

    staged = output_dir / "case01_orbit.cor"
    assert captured["orbit_file"] == staged
    assert staged.read_bytes() == b"orbit-original"
    assert orbit_source.read_bytes() == b"orbit-original"
    for path, contents in inputs_before.items():
        assert path.read_bytes() == contents


@pytest.mark.unit
def test_python_connector_stages_orbit_already_inside_output_dir(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    orbit_source = tmp_path / "custom-orbit.cor"
    orbit_source.write_bytes(b"inside-output-dir")

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=orbit_source))

    staged = tmp_path / "case01_orbit.cor"
    assert captured["orbit_file"] == staged
    assert staged.read_bytes() == b"inside-output-dir"
    assert orbit_source.read_bytes() == b"inside-output-dir"


@pytest.mark.unit
def test_python_connector_orbit_already_canonical_is_untouched(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    canonical = tmp_path / "case01_orbit.cor"
    canonical.write_bytes(b"canonical-orbit")

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=canonical))

    assert captured["orbit_file"] == canonical
    assert canonical.read_bytes() == b"canonical-orbit"
    assert canonical.is_file()
    assert not canonical.is_symlink()


@pytest.mark.unit
def test_python_connector_orbit_rerun_replaces_staged_copy(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    first = tmp_path / "first-orbit.cor"
    second = tmp_path / "second-orbit.cor"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=first))
    assert (tmp_path / "case01_orbit.cor").read_bytes() == b"first"

    connector.run(RuntimeOperator(orbit_file=second))
    assert (tmp_path / "case01_orbit.cor").read_bytes() == b"second"

    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"


@pytest.mark.unit
def test_python_connector_orbit_replaces_preexisting_symlink_alias(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    orbit_source = tmp_path / "real-orbit.cor"
    orbit_source.write_bytes(b"orbit-data")
    canonical = tmp_path / "case01_orbit.cor"
    canonical.symlink_to(orbit_source)

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=orbit_source))

    assert captured["orbit_file"] == canonical
    assert not canonical.is_symlink()
    assert canonical.read_bytes() == b"orbit-data"
    assert orbit_source.read_bytes() == b"orbit-data"


@pytest.mark.unit
def test_python_connector_orbit_canonical_symlink_is_left_alone(tmp_path: Path):
    # The caller may supply the canonical path itself as a symlink. It must be
    # returned untouched, never unlinked and never resolved into a copy.
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    referent = tmp_path / "referent-orbit.cor"
    referent.write_bytes(b"referent-data")
    canonical = tmp_path / "case01_orbit.cor"
    canonical.symlink_to(referent)

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=canonical))

    assert captured["orbit_file"] == canonical
    assert canonical.is_symlink()
    assert canonical.read_bytes() == b"referent-data"
    assert referent.read_bytes() == b"referent-data"


@pytest.mark.unit
def test_python_connector_cleanup_preserves_orbit_source_in_output_dir(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    orbit_source = tmp_path / "case01_radii.win"
    orbit_source.write_bytes(b"radii-orbit")

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=orbit_source))

    assert captured["orbit_file"] == tmp_path / "case01_orbit.cor"
    assert orbit_source.read_bytes() == b"radii-orbit"
    assert (tmp_path / "case01_orbit.cor").read_bytes() == b"radii-orbit"


@pytest.mark.unit
def test_python_connector_cleanup_preserves_canonical_symlink_referent(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    referent = tmp_path / "case01_radii.win"
    referent.write_bytes(b"radii-referent")
    canonical = tmp_path / "case01_orbit.cor"
    canonical.symlink_to(referent)

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=canonical))

    assert captured["orbit_file"] == canonical
    assert canonical.is_symlink()
    assert referent.read_bytes() == b"radii-referent"
    assert canonical.read_bytes() == b"radii-referent"


@pytest.mark.unit
@pytest.mark.skipif(
    not CASE_INSENSITIVE_FS,
    reason="needs a case-insensitive filesystem to alias differently-cased names",
)
def test_python_connector_orbit_case_insensitive_alias_is_preserved(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    canonical = connector.output_dir / "case01_orbit.cor"
    canonical.write_bytes(b"canonical-data")
    alias = connector.output_dir / "CASE01_ORBIT.COR"

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=alias))

    assert captured["orbit_file"] == canonical
    assert canonical.read_bytes() == b"canonical-data"
    assert alias.read_bytes() == b"canonical-data"


@pytest.mark.unit
def test_python_connector_orbit_hardlink_alias_is_preserved(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    source = connector.output_dir / "real-orbit.cor"
    source.write_bytes(b"hardlink-data")
    canonical = connector.output_dir / "case01_orbit.cor"
    os.link(source, canonical)

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=source))

    assert captured["orbit_file"] == canonical
    assert source.read_bytes() == b"hardlink-data"
    assert canonical.read_bytes() == b"hardlink-data"


@pytest.mark.unit
@pytest.mark.skipif(
    not CASE_INSENSITIVE_FS,
    reason="needs a case-insensitive filesystem to alias differently-cased names",
)
def test_python_connector_cleanup_preserves_differently_cased_orbit_source(
    tmp_path: Path,
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=output_dir,
        output_prefix="case01",
    )
    source = output_dir / "case01_radii.win"
    source.write_bytes(b"source-data")
    supplied = output_dir / "Case01_Radii.Win"

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=supplied))

    assert captured["orbit_file"] == output_dir / "case01_orbit.cor"
    assert source.read_bytes() == b"source-data"


@pytest.mark.unit
def test_python_connector_orbit_external_symlink_to_canonical_target_survives(
    tmp_path: Path,
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=output_dir,
        output_prefix="case01",
    )
    canonical = output_dir / "case01_orbit.cor"
    canonical.write_bytes(b"canonical-data")
    external = tmp_path / "external-orbit.cor"
    external.symlink_to(canonical)

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=external))

    assert captured["orbit_file"] == canonical
    assert external.is_symlink()
    assert canonical.read_bytes() == b"canonical-data"


@pytest.mark.unit
@pytest.mark.skipif(
    not CASE_INSENSITIVE_FS,
    reason="needs a case-insensitive filesystem to alias differently-cased names",
)
def test_python_connector_orbit_external_symlink_differently_cased_referent_survives(
    tmp_path: Path,
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=output_dir,
        output_prefix="case01",
    )
    canonical = output_dir / "case01_orbit.cor"
    canonical.write_bytes(b"canonical-data")
    external = tmp_path / "external-orbit.cor"
    external.symlink_to(output_dir / "CASE01_ORBIT.COR")

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=external))

    assert captured["orbit_file"] == canonical
    assert external.is_symlink()
    assert canonical.read_bytes() == b"canonical-data"


@pytest.mark.unit
def test_python_connector_cleanup_preserves_hardlink_alias_of_orbit_source(
    tmp_path: Path,
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=output_dir,
        output_prefix="case01",
    )
    source = tmp_path / "real-orbit.cor"
    source.write_bytes(b"alias-data")
    alias = output_dir / "case01_extra.win"
    os.link(source, alias)

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=source))

    assert captured["orbit_file"] == output_dir / "case01_orbit.cor"
    assert alias.read_bytes() == b"alias-data"
    assert source.read_bytes() == b"alias-data"


@pytest.mark.unit
def test_python_connector_cleanup_preserves_symlink_alias_of_orbit_source(
    tmp_path: Path,
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=output_dir,
        output_prefix="case01",
    )
    source = tmp_path / "real-orbit.cor"
    source.write_bytes(b"alias-data")
    alias = output_dir / "case01_extra.win"
    alias.symlink_to(source)

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=source))

    assert captured["orbit_file"] == output_dir / "case01_orbit.cor"
    assert alias.is_symlink()
    assert source.read_bytes() == b"alias-data"


@pytest.mark.unit
def test_python_connector_cleanup_preserves_differently_cased_prefix_cor_orbit(
    tmp_path: Path,
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=output_dir,
        output_prefix="case01",
    )
    source = output_dir / "Case01.Cor"
    source.write_bytes(b"source-data")

    captured = _record_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=source))

    assert captured["orbit_file"] == output_dir / "case01_orbit.cor"
    assert source.read_bytes() == b"source-data"


@pytest.mark.unit
def test_python_connector_forwards_timeout_for_one_run_only(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    timeouts: list[object] = []

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, timeout=None
    ):
        timeouts.append(timeout)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]

    connector.run(RuntimeOperator(timeout=30.0))
    connector.run()

    assert timeouts == [30.0, None]


@pytest.mark.unit
@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), "5"])
def test_python_connector_invalid_timeout_preserves_existing_outputs(
    tmp_path: Path, timeout
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    stale = tmp_path / "case01_tot_w1.hs"
    stale.write_text("stale")
    orbit_source = tmp_path / "orbit.cor"
    orbit_source.write_text("orbit")
    staged_orbit = tmp_path / "case01_orbit.cor"

    with pytest.raises(
        SimulationError, match=r"timeout must be a finite number > 0 seconds"
    ):
        connector.run(RuntimeOperator(timeout=timeout, orbit_file=orbit_source))

    assert stale.read_text() == "stale"
    assert not staged_orbit.exists()


@pytest.mark.unit
def test_python_connector_run_leaves_index_26_and_nn_untouched(tmp_path: Path):
    # Non-regression guard: the prefix/orbit/timeout plumbing must not touch
    # Index 26 (photon histories), the /NN count multiplier or quantization.
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        quantization_scale=0.5,
    )
    connector.add_config_value(26, 123456)
    connector.add_runtime_switch("NN", 3)

    source = np.full((2, 3, 4), 4.0, dtype=np.float32)
    source_path, _ = connector.configure_voxel_phantom(source, source.copy())
    captured = _record_executor_run(connector)
    connector.run()

    assert connector.get_config().get_value(26) == pytest.approx(123456)
    assert captured["runtime_switches"]["NN"] == 3
    assert connector.quantization_scale == pytest.approx(0.5)
    source_u16 = np.fromfile(source_path, dtype=np.uint16)
    assert int(source_u16.max()) == round(MAX_SOURCE * 0.5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "routine",
    [ScoringRoutine.LIST_MODE, ScoringRoutine.FORCED_COLLIMATION],
)
def test_python_connector_configure_rejects_unsupported_scoring_routine(
    tmp_path: Path, routine: ScoringRoutine
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(84, ScoringRoutine.SCATTWIN.value)
    source = np.ones((2, 3, 4), dtype=np.float32)
    config_before = [connector.get_config().get_value(i) for i in (5, 84)]

    match = rf"{routine.name} \({routine.value}\) is not supported"
    with pytest.raises(ValueError, match=match):
        connector.configure_voxel_phantom(
            source, source.copy(), scoring_routine=routine
        )
    with pytest.raises(ValueError, match=match):
        connector.configure_voxel_phantom(
            source, source.copy(), scoring_routine=int(routine.value)
        )

    assert [connector.get_config().get_value(i) for i in (5, 84)] == config_before
    assert not (tmp_path / "case01_src.smi").exists()
    assert not (tmp_path / "case01_dns.dmi").exists()


@pytest.mark.unit
def test_python_connector_configure_rejects_unsupported_config_routine(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(84, ScoringRoutine.FORCED_COLLIMATION.value)
    source = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match=r"FORCED_COLLIMATION \(3\) is not supported"):
        connector.configure_voxel_phantom(source, source.copy())

    assert connector.get_config().get_value(84) == (
        ScoringRoutine.FORCED_COLLIMATION.value
    )
    assert not (tmp_path / "case01_src.smi").exists()
    assert not (tmp_path / "case01_dns.dmi").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "routine",
    [ScoringRoutine.LIST_MODE, ScoringRoutine.FORCED_COLLIMATION],
)
def test_python_connector_config_only_run_rejects_unsupported_routine(
    tmp_path: Path, routine: ScoringRoutine
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(84, routine.value)
    stale = tmp_path / "case01_tot_w1.hs"
    stale.write_text("stale")
    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

    with pytest.raises(
        ValueError, match=rf"{routine.name} \({routine.value}\) is not supported"
    ):
        connector.run()

    assert launched == []
    assert stale.read_text() == "stale"
    assert not (tmp_path / "case01.smc").exists()
    assert not (tmp_path / "case01_orbit.cor").exists()


@pytest.mark.unit
def test_python_connector_run_rejects_routine_changed_after_configure(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    source = np.ones((2, 3, 4), dtype=np.float32)
    source_path, density_path = connector.configure_voxel_phantom(
        source, source.copy(), scoring_routine=ScoringRoutine.SCATTWIN
    )
    connector.add_config_value(84, ScoringRoutine.LIST_MODE.value)
    stale = tmp_path / "case01_tot_w1.hs"
    stale.write_text("stale")
    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

    with pytest.raises(ValueError, match=r"LIST_MODE \(2\) is not supported"):
        connector.run()

    assert launched == []
    assert stale.read_text() == "stale"
    assert source_path.exists()
    assert density_path.exists()
    assert not (tmp_path / "case01.smc").exists()


@pytest.mark.unit
@pytest.mark.parametrize("routine", [0, 1, 4])
def test_python_connector_accepts_supported_scoring_routines(
    tmp_path: Path, routine: int
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    source = np.ones((2, 3, 4), dtype=np.float32)

    connector.configure_voxel_phantom(source, source.copy(), scoring_routine=routine)

    assert int(connector.get_config().get_value(84)) == routine


@pytest.mark.unit
def test_python_connector_explicit_scoring_override_ignores_config_value(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    # An explicit override is authoritative even when the config value is one
    # the connector does not support.
    connector.add_config_value(84, ScoringRoutine.LIST_MODE.value)
    source = np.ones((2, 3, 4), dtype=np.float32)

    connector.configure_voxel_phantom(
        source, source.copy(), scoring_routine=ScoringRoutine.SCATTWIN
    )

    assert int(connector.get_config().get_value(84)) == ScoringRoutine.SCATTWIN.value


@pytest.mark.unit
def test_python_connector_does_not_round_fractional_config_routine(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    # 0.6 rounds to a supported routine (1) but is not an integral enum value.
    connector.add_config_value(84, 0.6)
    source = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="not an integral"):
        connector.configure_voxel_phantom(source, source.copy())
    with pytest.raises(ValueError, match="not an integral"):
        connector.run()


@pytest.mark.unit
def test_python_connector_short_config_defaults_scoring_to_dummy(tmp_path: Path):
    short_smc = _write_smc_with_basic_value_count(
        get("input.smc"), tmp_path / "short.smc", 83
    )
    connector = SimindPythonConnector(
        config_source=short_smc, output_dir=tmp_path, output_prefix="case01"
    )
    with pytest.raises(IndexError):
        connector.get_config().get_value("scoring_routine")  # genuinely absent

    source = np.ones((2, 3, 4), dtype=np.float32)
    source_path, density_path = connector.configure_voxel_phantom(source, source.copy())
    assert source_path.exists()
    assert density_path.exists()

    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

    # A DUMMY routine takes the non-PENETRATE path, so the run fails on the
    # missing SPECT headers rather than on missing PENETRATE headers.
    with pytest.raises(OutputError, match="No SPECT projection headers"):
        connector.run()

    assert launched == [True]
    assert connector._is_penetrate_routine() is False
    assert SimulationConfig(tmp_path / "case01.smc").get_value(84) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(0.6, id="fractional"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-inf"),
        pytest.param(-float("inf"), id="negative-inf"),
        pytest.param(10**1000, id="huge-integer"),
        pytest.param(ScoringRoutine.LIST_MODE.value, id="unsupported"),
    ],
)
def test_python_connector_invalid_present_scoring_has_no_side_effects(
    tmp_path: Path, bad_value: object
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(84, bad_value)
    source = np.ones((2, 3, 4), dtype=np.float32)

    stale = tmp_path / "case01_tot_w1.hs"
    stale.write_bytes(b"prior header")
    source_map = tmp_path / "case01_src.smi"
    source_map.write_bytes(b"prior source")
    density_map = tmp_path / "case01_dns.dmi"
    density_map.write_bytes(b"prior density")
    prior_config = tmp_path / "case01.smc"
    prior_config.write_bytes(b"prior config")
    orbit = tmp_path / "orbit.cor"
    orbit.write_bytes(b"prior orbit")

    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="scoring_routine"):
        connector.configure_voxel_phantom(source, source.copy())
    with pytest.raises(ValueError, match="scoring_routine"):
        connector.run(RuntimeOperator(orbit_file=orbit))

    assert launched == []
    assert stale.read_bytes() == b"prior header"
    assert source_map.read_bytes() == b"prior source"
    assert density_map.read_bytes() == b"prior density"
    assert prior_config.read_bytes() == b"prior config"
    assert orbit.read_bytes() == b"prior orbit"
    assert not (tmp_path / "case01_orbit.cor").exists()


@pytest.mark.unit
def test_python_connector_rejects_single_voxel_phantom(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    source = np.ones((1, 1, 1), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "SIMIND does not support a 1x1x1 voxel phantom; pad it to at least "
            "two voxels in one dimension."
        ),
    ):
        connector.configure_voxel_phantom(source, source.copy())

    assert not (tmp_path / "case01_src.smi").exists()
    assert not (tmp_path / "case01_dns.dmi").exists()


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
def test_python_connector_accepts_single_slice_phantoms(
    tmp_path: Path, shape: tuple[int, int, int]
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    source = np.ones(shape, dtype=np.float32)

    source_path, density_path = connector.configure_voxel_phantom(source, source.copy())

    assert source_path.exists()
    assert density_path.exists()


@pytest.mark.unit
def test_python_connector_rejects_source_quantized_to_all_zeros_at_tiny_scale(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        quantization_scale=1e-6,
    )
    source = np.ones((2, 3, 4), dtype=np.float32)
    index_5_before = connector.get_config().get_value(5)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "quantization_scale rounds the non-zero source to all zeros; "
            "increase quantization_scale."
        ),
    ):
        connector.configure_voxel_phantom(source, source.copy())

    assert connector.get_config().get_value(5) == index_5_before
    assert not (tmp_path / "case01_src.smi").exists()
    assert not (tmp_path / "case01_dns.dmi").exists()


@pytest.mark.unit
def test_python_connector_rejects_source_at_the_rounding_threshold(tmp_path: Path):
    # The migrated scale reproduces the legacy 500 * 0.001 == 0.5 budget, which
    # NumPy rounds to even (0): every voxel vanishes.
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        quantization_scale=0.001 * DEFAULT_QUANTIZATION_SCALE,
    )
    source = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="rounds the non-zero source to all zeros"):
        connector.configure_voxel_phantom(source, source.copy())


@pytest.mark.unit
def test_python_connector_accepts_source_above_the_rounding_threshold(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        quantization_scale=0.002 * DEFAULT_QUANTIZATION_SCALE,
    )
    source = np.ones((2, 3, 4), dtype=np.float32)

    source_path, _ = connector.configure_voxel_phantom(source, source.copy())

    assert int(np.fromfile(source_path, dtype=np.uint16).max()) == 1


@pytest.mark.unit
def test_python_connector_all_zero_source_is_still_accepted(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        quantization_scale=1e-6,
    )
    source = np.zeros((2, 3, 4), dtype=np.float32)

    source_path, _ = connector.configure_voxel_phantom(source, source.copy())

    assert source_path.exists()
    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run()
    assert launched == [True]


@pytest.mark.unit
@pytest.mark.parametrize("bad_scale", ["not-a-number", object()])
def test_python_connector_quantization_scale_conversion_failure_is_normalized(
    tmp_path: Path, bad_scale: object
):
    with pytest.raises(ValueError, match="quantization_scale"):
        SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path,
            output_prefix="case01",
            quantization_scale=bad_scale,
        )


@pytest.mark.unit
def test_python_connector_huge_quantization_scale_is_normalized(tmp_path: Path):
    with pytest.raises(ValueError, match="quantization_scale"):
        SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path,
            output_prefix="case01",
            quantization_scale=10**1000,
        )


@pytest.mark.unit
def test_python_connector_rejects_orbit_directory_before_staging(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    orbit_dir = tmp_path / "orbit-dir"
    orbit_dir.mkdir()
    canonical = tmp_path / "case01_orbit.cor"
    canonical.write_bytes(b"existing")
    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="orbit_file must be a regular file"):
        connector.run(RuntimeOperator(orbit_file=orbit_dir))

    assert launched == []
    assert canonical.read_bytes() == b"existing"
    assert orbit_dir.is_dir()
    assert not (tmp_path / "case01.smc").exists()


@pytest.mark.unit
def test_python_connector_source_write_failure_is_output_error(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    (tmp_path / "case01_src.smi").mkdir()
    source = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(OutputError, match="source map") as excinfo:
        connector.configure_voxel_phantom(source, source.copy())

    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.unit
def test_python_connector_density_write_failure_is_output_error(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    (tmp_path / "case01_dns.dmi").mkdir()
    source = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(OutputError, match="density map") as excinfo:
        connector.configure_voxel_phantom(source, source.copy())

    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.unit
def test_python_connector_window_write_failure_is_output_error(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    (tmp_path / "case01.win").mkdir()

    with pytest.raises(OutputError, match="energy-window") as excinfo:
        connector.set_energy_windows([126.0], [154.0], [0])

    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.unit
def test_python_connector_config_save_failure_is_output_error(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    (tmp_path / "case01.smc").mkdir()
    connector.executor.run_simulation = lambda *a, **k: None  # type: ignore[assignment]

    with pytest.raises(OutputError, match="configuration") as excinfo:
        connector.run()

    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.unit
def test_python_connector_orbit_staging_failure_is_output_error(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    orbit_source = tmp_path / "orbit.cor"
    orbit_source.write_bytes(b"orbit")
    (tmp_path / "case01_orbit.cor").mkdir()
    connector.executor.run_simulation = lambda *a, **k: None  # type: ignore[assignment]

    with pytest.raises(OutputError, match="orbit") as excinfo:
        connector.run(RuntimeOperator(orbit_file=orbit_source))

    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.unit
def test_python_connector_cleanup_failure_is_output_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    stale = tmp_path / "case01_tot_w1.hs"
    stale.write_text("stale")

    def fail_unlink(self: Path, *args, **kwargs):
        raise PermissionError("injected unlink failure")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    connector.executor.run_simulation = lambda *a, **k: None  # type: ignore[assignment]

    with pytest.raises(OutputError, match="stale output") as excinfo:
        connector.run()

    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.unit
def test_python_connector_deleted_output_dir_is_output_error(tmp_path: Path):
    output_dir = tmp_path / "gone"
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=output_dir,
        output_prefix="case01",
    )
    output_dir.rmdir()
    source = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(OutputError, match="does not exist or is not a directory"):
        connector.configure_voxel_phantom(source, source.copy())
    with pytest.raises(OutputError, match="does not exist or is not a directory"):
        connector.run()

    assert not output_dir.exists()


@pytest.mark.unit
def test_python_connector_missing_spect_headers_raise_output_error(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.executor.run_simulation = lambda *a, **k: None  # type: ignore[assignment]

    with pytest.raises(OutputError, match="No SPECT projection headers") as excinfo:
        connector.run()

    assert "case01" in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


@pytest.mark.unit
def test_python_connector_missing_penetrate_headers_raise_output_error(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(84, ScoringRoutine.PENETRATE.value)
    connector.executor.run_simulation = lambda *a, **k: None  # type: ignore[assignment]

    with pytest.raises(OutputError, match="No PENETRATE") as excinfo:
        connector.run()

    assert "case01" in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


@pytest.mark.unit
def test_output_error_is_also_an_os_error():
    assert issubclass(OutputError, SimindError)
    assert issubclass(OutputError, OSError)

    error = OutputError("failed to write source map")
    assert str(error) == "failed to write source map"

    with pytest.raises(OSError, match="failed to write source map"):
        raise error


@pytest.mark.unit
def test_python_connector_wrapped_write_failure_catchable_as_os_error(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    (tmp_path / "case01_src.smi").mkdir()
    source = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(OSError, match="source map") as excinfo:
        connector.configure_voxel_phantom(source, source.copy())

    assert isinstance(excinfo.value, OutputError)
    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.unit
def test_python_connector_missing_spect_headers_catchable_as_file_not_found(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.executor.run_simulation = lambda *a, **k: None  # type: ignore[assignment]

    with pytest.raises(
        FileNotFoundError, match="No SPECT projection headers"
    ) as excinfo:
        connector.run()

    assert isinstance(excinfo.value, OutputError)
    assert "case01" in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


@pytest.mark.unit
def test_python_connector_missing_penetrate_headers_catchable_as_file_not_found(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(84, ScoringRoutine.PENETRATE.value)
    connector.executor.run_simulation = lambda *a, **k: None  # type: ignore[assignment]

    with pytest.raises(FileNotFoundError, match="No PENETRATE") as excinfo:
        connector.run()

    assert isinstance(excinfo.value, OutputError)
    assert "case01" in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


@pytest.mark.unit
def test_python_connector_rejects_over_long_prefix_at_construction(tmp_path: Path):
    with pytest.raises(ValueError, match="at most 47 characters"):
        SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path,
            output_prefix="a" * 48,
        )


@pytest.mark.unit
def test_python_connector_over_long_prefix_preserves_outputs_before_launch(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    over_long = "a" * 48
    config_path = tmp_path / f"{over_long}.smc"
    header_path = tmp_path / f"{over_long}_tot_w1.hs"
    config_path.write_bytes(b"pre-existing config")
    header_path.write_bytes(b"pre-existing header")
    connector.output_prefix = over_long
    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="at most 47 characters"):
        connector.run()

    assert launched == []
    assert config_path.read_bytes() == b"pre-existing config"
    assert header_path.read_bytes() == b"pre-existing header"


@pytest.mark.unit
def test_python_connector_over_long_prefix_rejected_before_writing_inputs(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    over_long = "a" * 48
    connector.output_prefix = over_long
    source = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="at most 47 characters"):
        connector.configure_voxel_phantom(source, source.copy())
    with pytest.raises(ValueError, match="at most 47 characters"):
        connector.set_energy_windows([126.0], [154.0], [0])

    assert not (tmp_path / f"{over_long}_src.smi").exists()
    assert not (tmp_path / f"{over_long}_dns.dmi").exists()
    assert not (tmp_path / f"{over_long}.win").exists()


@pytest.mark.unit
def test_python_connector_set_energy_windows_accepts_ten_windows(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    count = 10

    connector.set_energy_windows(
        [126.0 + i for i in range(count)],
        [127.0 + i for i in range(count)],
        [0] * count,
    )

    window_file = tmp_path / "case01.win"
    assert window_file.exists()
    lines = [line for line in window_file.read_text().splitlines() if line]
    assert len(lines) == count + 1  # all-zero orders append a scatter line


@pytest.mark.unit
def test_python_connector_set_energy_windows_rejects_too_many_windows(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    count = 100

    with pytest.raises(ValueError, match="At most 99 energy windows"):
        connector.set_energy_windows(
            [126.0] * count,
            [154.0] * count,
            [0] * count,
        )

    assert not (tmp_path / "case01.win").exists()
