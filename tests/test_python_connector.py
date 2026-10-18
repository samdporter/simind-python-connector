import logging
import os
import re
import sys
import tempfile
import warnings
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import simind_python_connector.connectors.python_connector as connector_mod
from simind_python_connector.configs import get
from simind_python_connector.connectors import RuntimeOperator, SimindPythonConnector
from simind_python_connector.core.config import SimulationConfig
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    MAX_SOURCE,
    ScoringRoutine,
    SimulationError,
)
from simind_python_connector.phantoms import VoxelPhantom
from simind_python_connector.utils.interfile import (
    InterfileHeader,
    ProjectionGeometry,
    check_geometry_match,
    read_projection_geometry,
)


def _case_insensitive_filesystem() -> bool:
    with tempfile.TemporaryDirectory() as directory:
        probe = Path(directory) / "case_probe"
        probe.write_bytes(b"probe")
        return (Path(directory) / "CASE_PROBE").exists()


CASE_INSENSITIVE_FS = _case_insensitive_filesystem()


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
        "case-01",
        "case-",
        "case--01",
        "case 01",
        "café",
    ],
)
def test_python_connector_rejects_unsafe_output_prefixes(
    tmp_path: Path, bad_prefix: str
):
    with pytest.raises(ValueError, match="output_prefix"):
        SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path,
            output_prefix=bad_prefix,
        )


@pytest.mark.unit
@pytest.mark.parametrize("good_prefix", ["a", "case01", "case_01", "9case"])
def test_python_connector_accepts_safe_output_prefixes(
    tmp_path: Path, good_prefix: str
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
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
def test_python_connector_configure_voxel_phantom_keeps_projection_matrix(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(28, 0.25)
    connector.add_config_value(76, 96)
    connector.add_config_value(77, 40)

    source = np.zeros((2, 4, 3), dtype=np.float32)
    source[0, 0, 0] = 1.0

    connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)

    config = connector.get_config()
    assert config.get_value(28) == pytest.approx(0.25)
    assert (config.get_value(76), config.get_value(77)) == (96, 40)
    assert config.get_value(78) == pytest.approx(3)  # density map i = dim_x
    assert config.get_value(79) == pytest.approx(3)  # source map i = dim_x
    assert config.get_value(81) == pytest.approx(4)  # density map j = dim_y
    assert config.get_value(82) == pytest.approx(4)  # source map j = dim_y
    assert config.get_value(34) == pytest.approx(2)  # number of density images


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
    with pytest.raises(ValueError, match="scoring_routine"):
        connector.configure_voxel_phantom(
            source=source,
            mu_map=source.copy(),
            scoring_routine=True,
        )


@pytest.mark.unit
@pytest.mark.parametrize("bad_value", [True, 1.5, 999])
def test_python_connector_configure_phantom_rejects_invalid_voxel_scoring_routine(
    tmp_path: Path, bad_value
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    activity = np.ones((2, 3, 4), dtype=np.float32)
    density = np.ones_like(activity)

    with pytest.raises(ValueError, match="scoring_routine"):
        connector.configure_phantom(
            VoxelPhantom(activity, density, (4.0, 4.0, 4.0)),
            scoring_routine=bad_value,
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
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
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
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
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
def test_python_connector_rejects_non_positive_quantization_scale(tmp_path: Path):
    with pytest.raises(ValueError, match=r"quantization_scale must be in \(0, 1\]"):
        SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path,
            output_prefix="case01",
            quantization_scale=0.0,
        )


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
        **kwargs,
    ) -> None:
        captured["output_prefix"] = output_prefix
        captured["orbit_file"] = orbit_file
        captured["runtime_switches"] = dict(runtime_switches or {})
        captured.update(kwargs)

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
    outputs = connector.run(RuntimeOperator(switches={"NN": 2, "RR": 12345}), timeout=5)

    # SPECT Interfile headers are needed to read the outputs.
    assert connector.config.get_flag(5)
    assert connector.config.get_flag(14)
    assert captured["output_prefix"] == "case01"
    assert captured["runtime_switches"] == {"NN": 2, "RR": 12345}
    assert captured["timeout"] == 5

    assert "tot_w1" in outputs
    result = outputs["tot_w1"]
    assert result.header_path == (tmp_path / "case01_tot_w1.hs").resolve()
    assert result.data_path == (tmp_path / "case01_tot_w1.a00").resolve()
    assert result.projection.shape == (2, 3, 4)
    expected_sum = float(np.arange(24, dtype=np.float32).sum())
    assert float(result.projection.sum()) == expected_sum


@pytest.mark.unit
def test_simind_version_uses_env_default_and_explicit_value_wins(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("SIMIND_VERSION", "9")
    env_connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path / "env"
    )
    explicit_connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path / "explicit",
        simind_version=8,
    )
    assert env_connector.simind_version == 9
    assert explicit_connector.simind_version == 8


@pytest.mark.unit
@pytest.mark.parametrize("version", [7, 8.5, True, "10"])
def test_simind_version_rejects_unsupported_values(tmp_path: Path, version):
    with pytest.raises(ValueError, match="simind_version must be 8 or 9"):
        SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path,
            simind_version=version,
        )


@pytest.mark.unit
def test_v9_voxel_center_offsets_apply_only_to_saved_config_and_do_not_accumulate(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        simind_version=9,
    )
    connector.add_config_value(17, 0.1)
    connector.add_config_value(18, 0.2)
    connector.add_config_value(36, 0.3)
    connector.add_config_value(37, -0.2)
    connector.configure_voxel_phantom(
        np.ones((4, 4, 4), dtype=np.float32),
        np.zeros((4, 4, 4), dtype=np.float32),
        voxel_size_mm=4.42,
    )
    connector.add_runtime_switch("PX", 0.5)
    connector.config.set_flag(5, False)
    connector.config.set_flag(14, False)
    source_bytes = (tmp_path / "case01_src.smi").read_bytes()
    captured = []

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        saved = SimulationConfig(Path(cwd) / f"{output_prefix}.smc")
        saved.import_smc(saved.filepath)
        captured.append(
            (
                *(saved.get_value(i) for i in (17, 18, 36, 37)),
                saved.get_flag(5),
                saved.get_flag(14),
            )
        )

    connector.executor.run_simulation = fake_run_simulation
    connector._ensure_interfile_headers = lambda: []
    connector._load_projection_outputs = lambda _: {}
    connector.run()
    connector.run()

    expected = (0.6, -0.3, 0.742, -0.642, True, True)
    assert [item[:4] for item in captured] == [
        pytest.approx(expected[:4]),
        pytest.approx(expected[:4]),
    ]
    assert [item[4:] for item in captured] == [(True, True), (True, True)]
    assert tuple(connector.get_config().get_value(i) for i in (17, 18, 36, 37)) == (
        0.1,
        0.2,
        0.3,
        -0.2,
    )
    assert (tmp_path / "case01_src.smi").read_bytes() == source_bytes


@pytest.mark.unit
@pytest.mark.parametrize("orientation", ["DI", "density", "OR"])
def test_v9_voxel_centering_rejects_unsupported_map_orientations(
    tmp_path: Path, orientation: str
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        simind_version=9,
    )
    connector.configure_voxel_phantom(
        np.ones((4, 4, 4), dtype=np.float32),
        np.zeros((4, 4, 4), dtype=np.float32),
        voxel_size_mm=4.42,
    )
    if orientation == "DI":
        connector.runtime_switches.set_switch("DI", 1)
    elif orientation == "density":
        connector.add_config_value(32, 1)
    else:
        connector.runtime_switches.set_switch("OR", 2)
    stale_output = tmp_path / "output_tot_w1.a00"
    stale_output.write_text("stale")
    called = False

    def fake_run_simulation(*args, **kwargs):
        nonlocal called
        called = True

    connector.executor.run_simulation = fake_run_simulation
    with pytest.raises(ValueError, match="v9 voxel centring requires DI=0"):
        connector.run()
    assert not called
    assert stale_output.read_text() == "stale"
    assert not (tmp_path / "output.smc").exists()


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
        **kwargs,
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
def test_energy_window_filename_prefix_limit_is_15_characters(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="a" * 15,
    )
    connector.set_energy_windows([126], [154], [0])
    existing = tmp_path / ("b" * 16 + ".win")
    existing.write_bytes(b"existing")
    connector.output_prefix = "b" * 16
    with pytest.raises(ValueError, match="15 characters"):
        connector.set_energy_windows([126], [154], [0])
    assert existing.read_bytes() == b"existing"


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
def test_python_connector_rejects_invalid_nn(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    with pytest.raises(ValueError, match="integer >= 1"):
        connector.add_runtime_switch("NN", 0.5)


@pytest.mark.unit
def test_python_connector_set_mpi_validates_processes(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="integer >= 1"):
            connector.set_mpi(bad)
    with pytest.raises(ValueError, match="needs processes"):
        connector.set_mpi(None, split_projections=True)


@pytest.mark.unit
def test_python_connector_split_mpi_needs_divisible_projections(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.add_config_value(29, 24)
    connector.set_mpi(5, split_projections=True)
    with pytest.raises(ValueError, match="multiple of the MPI processes"):
        connector.run()
    assert not (tmp_path / "case01.smc").exists()


@pytest.mark.unit
def test_python_connector_split_mpi_rejects_unrepresentable_projection_count(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    # A huge integral count cannot be converted to float; it must fail with the
    # clean split-projections configuration error, not a raw OverflowError.
    connector.add_config_value(29, Fraction(10**400, 1))
    connector.set_mpi(4, split_projections=True)
    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="number of projections"):
        connector.run()

    assert launched == []
    assert not (tmp_path / "case01.smc").exists()


@pytest.mark.unit
def test_python_connector_split_mpi_reports_huge_count_without_digit_limit(
    tmp_path: Path,
):
    if not hasattr(sys, "set_int_max_str_digits"):
        pytest.skip("interpreter has no int-string digit limit")
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    original_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        # 701 decimal digits: repr()/str() of this value would themselves raise
        # the digit-limit ValueError if the error message interpolated it.
        connector.add_config_value(29, 10**700)
        connector.set_mpi(4, split_projections=True)
        launched: list[object] = []
        connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

        with pytest.raises(ValueError, match="number of projections") as excinfo:
            connector.run()
    finally:
        sys.set_int_max_str_digits(original_limit)

    assert "digit limit" not in str(excinfo.value).lower()
    assert launched == []
    assert not (tmp_path / "case01.smc").exists()


@pytest.mark.unit
def test_python_connector_split_mpi_rejects_penetrate(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.add_config_value(29, 24)
    connector.add_config_value(84, ScoringRoutine.PENETRATE.value)
    connector.set_mpi(2, split_projections=True)
    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="PENETRATE"):
        connector.run()

    assert launched == []
    assert not (tmp_path / "case01.smc").exists()


@pytest.mark.unit
@pytest.mark.parametrize("via_operator", [False, True])
def test_python_connector_replicate_mpi_rejects_fixed_rr(
    tmp_path: Path, via_operator: bool
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    operator = None
    if via_operator:
        operator = RuntimeOperator(switches={"RR": 12345})
    else:
        connector.add_runtime_switch("RR", 12345)
    connector.set_mpi(2)
    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="RR"):
        connector.run(operator)

    assert launched == []
    assert not (tmp_path / "case01.smc").exists()


@pytest.mark.unit
def test_python_connector_replicate_mpi_runs_when_operator_clears_rr(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.add_runtime_switch("RR", 12345)
    connector.set_mpi(2)
    captured = {}

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        captured.update(runtime_switches=runtime_switches, **kwargs)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run(RuntimeOperator(switches={"RR": None}))

    assert "RR" not in captured["runtime_switches"]
    assert captured["mpi_processes"] == 2


@pytest.mark.unit
@pytest.mark.parametrize("processes, split", [(1, False), (2, True)])
def test_python_connector_mpi_allows_fixed_rr_without_replicas(
    tmp_path: Path, processes: int, split: bool
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.add_config_value(29, 24)
    connector.add_runtime_switch("RR", 12345)
    connector.set_mpi(processes, split_projections=split)
    launched: list[object] = []

    def fake_run_simulation(*args, **kwargs):
        launched.append(True)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run()

    assert launched == [True]


@pytest.mark.unit
def test_python_connector_passes_mpi_settings_to_executor(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.add_config_value(29, 24)
    connector.set_mpi(4, split_projections=True)
    captured = {}

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        captured.update(kwargs)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run()

    assert captured == {"mpi_processes": 4, "split_projections": True, "timeout": None}


@pytest.mark.unit
def test_python_connector_rejects_quantization_scale_above_one(tmp_path: Path):
    with pytest.raises(ValueError, match=r"quantization_scale must be in \(0, 1\]"):
        SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path,
            output_prefix="case01",
            quantization_scale=1.5,
        )


@pytest.mark.unit
def test_python_connector_default_scale_matches_legacy_source_values(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    source = np.zeros((4, 4, 4), dtype=np.float32)
    source[1, 1, 1], source[1, 1, 2], source[1, 1, 3] = 1.0, 0.6, 0.3
    source_path, _ = connector.configure_voxel_phantom(
        source, np.zeros_like(source), voxel_size_mm=4.0
    )
    written = np.fromfile(source_path, dtype=np.uint16).reshape(source.shape)
    assert (written[1, 1, 1], written[1, 1, 2], written[1, 1, 3]) == (500, 300, 150)


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
def test_python_connector_density_capped_at_signed_int16(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.get_config().set_flag(11, True)
    source = np.ones((1, 1, 3), dtype=np.float32)
    mu_map = np.zeros_like(source)
    mu_map[0, 0, :] = [0.15, 10.0, 40.0]
    _, density_path = connector.configure_voxel_phantom(
        source, mu_map, voxel_size_mm=4.0, mu_map_type="density"
    )
    written = np.fromfile(density_path, dtype=np.uint16)
    assert tuple(written) == (150, 10000, 32767)


@pytest.mark.unit
def test_python_connector_warns_when_activity_gets_no_histories(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    source = np.full((5, 5, 4), 0.0005, dtype=np.float32)
    source[0, 0, 0] = 1.0
    with caplog.at_level("WARNING"):
        connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)
    assert "gets no photon histories" in caplog.text


@pytest.mark.unit
def test_python_connector_negative_photon_energy_converts_like_positive(
    tmp_path: Path,
):
    written = []
    for energy in (140.0, -140.0):
        connector = SimindPythonConnector(
            config_source=get("Example.yaml"),
            output_dir=tmp_path / str(energy),
            output_prefix="case01",
        )
        connector.get_config().set_flag(11, True)
        connector.add_config_value(1, energy)
        source = np.ones((4, 4, 4), dtype=np.float32)
        _, density_path = connector.configure_voxel_phantom(
            source, np.full_like(source, 0.15), voxel_size_mm=4.0
        )
        written.append(np.fromfile(density_path, dtype=np.uint16))
    assert written[0].max() > 900  # water-like density, not ~0
    assert np.array_equal(written[0], written[1])


def _write_projection(tmp_path: Path, name: str, shape: tuple, declared: tuple):
    data_path = tmp_path / f"{name}.a00"
    np.zeros(shape, dtype=np.float32).tofile(data_path)
    (tmp_path / f"{name}.hs").write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                f"!matrix size [1] := {declared[2]}",
                f"!matrix size [2] := {declared[1]}",
                f"!matrix size [3] := {declared[0]}",
                f"!name of data file := {data_path.name}",
                "!END OF INTERFILE :=",
            ]
        )
    )


@pytest.mark.unit
def test_python_connector_raises_on_unparseable_output(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        _write_projection(tmp_path, "case01_tot_w1", (2, 3, 4), (2, 3, 4))
        _write_projection(tmp_path, "case01_air_w1", (4, 4), (4, 4, 4))

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    with pytest.raises(SimulationError, match="case01_air_w1"):
        connector.run()


@pytest.mark.unit
def test_python_connector_copies_orbit_file_to_lower_case_name(tmp_path: Path):
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    orbit = source_dir / "MyOrbit.COR"
    orbit.write_text("15.0\n")
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path / "out",
        output_prefix="case01",
    )
    captured = {}

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        captured["orbit_file"] = orbit_file

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run(RuntimeOperator(orbit_file=orbit))
    connector.run(RuntimeOperator(orbit_file=tmp_path / "out" / "case01_orbit.cor"))

    assert captured["orbit_file"] == tmp_path / "out" / "case01_orbit.cor"
    assert (tmp_path / "out" / "case01_orbit.cor").read_text() == "15.0\n"


def _record_orbit_executor_run(connector: SimindPythonConnector) -> dict:
    captured: dict[str, object] = {}

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        captured["orbit_file"] = orbit_file

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    return captured


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

    captured = _record_orbit_executor_run(connector)
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
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    orbit_source = tmp_path / "custom-orbit.cor"
    orbit_source.write_bytes(b"inside-output-dir")

    captured = _record_orbit_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=orbit_source))

    staged = tmp_path / "case01_orbit.cor"
    assert captured["orbit_file"] == staged
    assert staged.read_bytes() == b"inside-output-dir"
    assert orbit_source.read_bytes() == b"inside-output-dir"


@pytest.mark.unit
def test_python_connector_orbit_already_canonical_is_untouched(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    canonical = tmp_path / "case01_orbit.cor"
    canonical.write_bytes(b"canonical-orbit")

    captured = _record_orbit_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=canonical))

    assert captured["orbit_file"] == canonical
    assert canonical.read_bytes() == b"canonical-orbit"
    assert canonical.is_file()
    assert not canonical.is_symlink()


@pytest.mark.unit
def test_python_connector_orbit_rerun_replaces_staged_copy(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    first = tmp_path / "first-orbit.cor"
    second = tmp_path / "second-orbit.cor"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    _record_orbit_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=first))
    assert (tmp_path / "case01_orbit.cor").read_bytes() == b"first"

    connector.run(RuntimeOperator(orbit_file=second))
    assert (tmp_path / "case01_orbit.cor").read_bytes() == b"second"

    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"


@pytest.mark.unit
def test_python_connector_orbit_replaces_preexisting_symlink_alias(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    orbit_source = tmp_path / "real-orbit.cor"
    orbit_source.write_bytes(b"orbit-data")
    canonical = tmp_path / "case01_orbit.cor"
    canonical.symlink_to(orbit_source)

    captured = _record_orbit_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=orbit_source))

    assert captured["orbit_file"] == canonical
    assert not canonical.is_symlink()
    assert canonical.read_bytes() == b"orbit-data"
    assert orbit_source.read_bytes() == b"orbit-data"


@pytest.mark.unit
def test_python_connector_orbit_canonical_symlink_is_left_alone(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    referent = tmp_path / "referent-orbit.cor"
    referent.write_bytes(b"referent-data")
    canonical = tmp_path / "case01_orbit.cor"
    canonical.symlink_to(referent)

    captured = _record_orbit_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=canonical))

    assert captured["orbit_file"] == canonical
    assert canonical.is_symlink()
    assert canonical.read_bytes() == b"referent-data"
    assert referent.read_bytes() == b"referent-data"


@pytest.mark.unit
def test_python_connector_cleanup_preserves_orbit_source_in_output_dir(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    orbit_source = tmp_path / "case01_radii.win"
    orbit_source.write_bytes(b"radii-orbit")

    captured = _record_orbit_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=orbit_source))

    assert captured["orbit_file"] == tmp_path / "case01_orbit.cor"
    assert orbit_source.read_bytes() == b"radii-orbit"
    assert (tmp_path / "case01_orbit.cor").read_bytes() == b"radii-orbit"


@pytest.mark.unit
def test_python_connector_cleanup_preserves_canonical_symlink_referent(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    referent = tmp_path / "case01_radii.win"
    referent.write_bytes(b"radii-referent")
    canonical = tmp_path / "case01_orbit.cor"
    canonical.symlink_to(referent)

    captured = _record_orbit_executor_run(connector)
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
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    canonical = connector.output_dir / "case01_orbit.cor"
    canonical.write_bytes(b"canonical-data")
    alias = connector.output_dir / "CASE01_ORBIT.COR"

    captured = _record_orbit_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=alias))

    assert captured["orbit_file"] == canonical
    assert canonical.read_bytes() == b"canonical-data"
    assert alias.read_bytes() == b"canonical-data"


@pytest.mark.unit
def test_python_connector_orbit_hardlink_alias_is_preserved(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    source = connector.output_dir / "real-orbit.cor"
    source.write_bytes(b"hardlink-data")
    canonical = connector.output_dir / "case01_orbit.cor"
    os.link(source, canonical)

    captured = _record_orbit_executor_run(connector)
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

    captured = _record_orbit_executor_run(connector)
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

    captured = _record_orbit_executor_run(connector)
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

    captured = _record_orbit_executor_run(connector)
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
    alias = output_dir / "case01_radii.win"
    os.link(source, alias)

    captured = _record_orbit_executor_run(connector)
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
    alias = output_dir / "case01_radii.win"
    alias.symlink_to(source)

    captured = _record_orbit_executor_run(connector)
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

    captured = _record_orbit_executor_run(connector)
    connector.run(RuntimeOperator(orbit_file=source))

    assert captured["orbit_file"] == output_dir / "case01_orbit.cor"
    assert source.read_bytes() == b"source-data"


@pytest.mark.unit
def test_python_connector_rejects_orbit_at_the_config_destination(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    config_path = tmp_path / "case01.smc"
    config_path.write_bytes(b"caller-orbit")
    header_path = tmp_path / "case01_tot_w1.hs"
    header_path.write_bytes(b"prior-header")
    captured = _record_orbit_executor_run(connector)

    with pytest.raises(ValueError, match="orbit_file"):
        connector.run(RuntimeOperator(orbit_file=config_path))

    assert "orbit_file" not in captured
    assert config_path.read_bytes() == b"caller-orbit"
    assert header_path.read_bytes() == b"prior-header"
    assert not (tmp_path / "case01_orbit.cor").exists()


@pytest.mark.unit
@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_python_connector_rejects_config_link_alias_of_supplied_orbit(
    tmp_path: Path, link: str
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    external = tmp_path / "radii.cor"
    external.write_bytes(b"external-radii")
    config_path = tmp_path / "case01.smc"
    if link == "symlink":
        config_path.symlink_to(external)
    else:
        os.link(external, config_path)
    captured = _record_orbit_executor_run(connector)

    with pytest.raises(ValueError, match="orbit_file"):
        connector.run(RuntimeOperator(orbit_file=external))

    assert "orbit_file" not in captured
    assert external.read_bytes() == b"external-radii"
    assert not (tmp_path / "case01_orbit.cor").exists()
    if link == "symlink":
        assert config_path.is_symlink()
    else:
        assert os.stat(config_path).st_ino == os.stat(external).st_ino


@pytest.mark.unit
@pytest.mark.skipif(
    not CASE_INSENSITIVE_FS,
    reason="needs a case-insensitive filesystem to alias differently-cased names",
)
def test_python_connector_rejects_differently_cased_config_alias(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    alias = tmp_path / "CASE01.SMC"
    alias.write_bytes(b"caller-orbit")
    captured = _record_orbit_executor_run(connector)

    with pytest.raises(ValueError, match="orbit_file"):
        connector.run(RuntimeOperator(orbit_file=alias))

    assert "orbit_file" not in captured
    assert alias.read_bytes() == b"caller-orbit"


@pytest.mark.unit
def test_python_connector_external_orbit_with_input_like_basename_is_accepted(
    tmp_path: Path,
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
    external = source_dir / "case01_src.smi"
    external.write_bytes(b"external-orbit")
    captured = _record_orbit_executor_run(connector)

    connector.run(RuntimeOperator(orbit_file=external))

    assert captured["orbit_file"] == output_dir / "case01_orbit.cor"
    assert external.read_bytes() == b"external-orbit"
    assert (output_dir / "case01_orbit.cor").read_bytes() == b"external-orbit"


@pytest.mark.unit
def test_python_connector_removes_stale_result_files(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    for name in (
        "case01.res",
        "case01.bis",
        "case01.spe",
        "case01.cor",
        "case01_orbit.cor",
    ):
        (tmp_path / name).write_text("old")
    seen = {}

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        seen["files"] = sorted(path.name for path in tmp_path.iterdir())

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run()

    for name in ("case01.res", "case01.bis", "case01.spe", "case01.cor"):
        assert name not in seen["files"]
    assert "case01_orbit.cor" in seen["files"]


def _write_named_projection(tmp_path: Path, name: str, values: np.ndarray):
    data_path = tmp_path / f"{name}.a00"
    values.astype("<f4").tofile(data_path)
    (tmp_path / f"{name}.hs").write_text(
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


@pytest.mark.unit
def test_python_connector_derives_primary_from_total_and_scatter(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.set_energy_windows([126.0], [154.0], [0])
    total = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 10
    scatter = np.ones((2, 3, 4), dtype=np.float32)

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        _write_named_projection(tmp_path, "case01_tot_w1", total)
        _write_named_projection(tmp_path, "case01_sca_w1", scatter)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    outputs = connector.run()

    assert np.array_equal(outputs["pri_w1"].projection, total - scatter)
    assert outputs["pri_w1"].metadata["derived"] == "tot - sca"
    assert "derived" not in outputs["tot_w1"].metadata


@pytest.mark.unit
def test_python_connector_derives_scatter_from_total_and_primary(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.set_energy_windows([126.0], [154.0], [0])
    total = np.full((2, 3, 4), 10.0, dtype=np.float32)
    primary = np.full((2, 3, 4), 6.0, dtype=np.float32)

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        _write_named_projection(tmp_path, "case01_tot_w1", total)
        _write_named_projection(tmp_path, "case01_pri_w1", primary)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    outputs = connector.run()

    assert np.array_equal(outputs["sca_w1"].projection, total - primary)
    assert outputs["sca_w1"].metadata["derived"] == "tot - pri"


@pytest.mark.unit
def test_python_connector_keeps_user_ca_and_simind_primary(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.add_runtime_switch("CA", 2)
    primary = np.full((2, 3, 4), 5.0, dtype=np.float32)
    captured = {}

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        captured["switches"] = dict(runtime_switches)
        _write_named_projection(tmp_path, "case01_tot_w1", primary + 1)
        _write_named_projection(tmp_path, "case01_pri_w1", primary)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    outputs = connector.run()

    assert captured["switches"]["CA"] == 2
    assert np.array_equal(outputs["pri_w1"].projection, primary)
    assert "derived" not in outputs["pri_w1"].metadata


@pytest.mark.unit
def test_python_connector_derives_primary_only_for_order_zero_windows(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.set_energy_windows([126.0, 126.0], [154.0, 154.0], [0, 1])
    total = np.full((2, 3, 4), 10.0, dtype=np.float32)
    scatter = np.full((2, 3, 4), 1.0, dtype=np.float32)

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        for window in (1, 2):
            _write_named_projection(tmp_path, f"case01_tot_w{window}", total)
            _write_named_projection(tmp_path, f"case01_sca_w{window}", scatter)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    outputs = connector.run()

    assert np.array_equal(outputs["pri_w1"].projection, total - scatter)
    assert outputs["pri_w1"].metadata["derived"] == "tot - sca"
    assert "pri_w2" not in outputs


@pytest.mark.unit
def test_python_connector_derives_from_fw_window_orders(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.set_energy_windows([126.0], [154.0], [0])
    connector.add_runtime_switch("FW", "windowprefix1234")
    (tmp_path / "windowprefix1234.win").write_text("126.0,154.0,0\n92.0,124.0,0\n")
    total = np.full((2, 3, 4), 10.0, dtype=np.float32)
    scatter = np.full((2, 3, 4), 1.0, dtype=np.float32)

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        for window in (1, 2):
            _write_named_projection(tmp_path, f"case01_tot_w{window}", total)
            _write_named_projection(tmp_path, f"case01_sca_w{window}", scatter)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    outputs = connector.run()

    assert np.array_equal(outputs["pri_w1"].projection, total - scatter)
    assert np.array_equal(outputs["pri_w2"].projection, total - scatter)


@pytest.mark.unit
def test_python_connector_skips_pri_for_nonzero_order_window(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.set_energy_windows([126.0], [154.0], [0])
    connector.add_runtime_switch("FW", "case01_custom")
    (tmp_path / "case01_custom.win").write_text("126.0,154.0,0\n126.0,154.0,1\n")
    total = np.full((2, 3, 4), 10.0, dtype=np.float32)
    scatter = np.full((2, 3, 4), 1.0, dtype=np.float32)

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        for window in (1, 2):
            _write_named_projection(tmp_path, f"case01_tot_w{window}", total)
            _write_named_projection(tmp_path, f"case01_sca_w{window}", scatter)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    outputs = connector.run()

    assert "pri_w1" in outputs
    assert "pri_w2" not in outputs


@pytest.mark.unit
def test_python_connector_cleanup_protects_fw_window_file(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.add_runtime_switch("FW", "case01_custom")
    keep = tmp_path / "case01_custom.win"
    stale = tmp_path / "case01_old.win"
    keep.write_text("126.0,154.0,0\n")
    stale.write_text("126.0,154.0,0\n")
    seen = {}

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        seen["files"] = sorted(path.name for path in tmp_path.iterdir())

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run()

    assert "case01_custom.win" in seen["files"]
    assert "case01_old.win" not in seen["files"]


@pytest.mark.unit
def test_set_activity_writes_index_25_as_activity_times_time(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.set_activity(np.float32(250.0), time_per_projection_s=20.0)
    assert connector.get_config().get_value(25) == pytest.approx(5000.0)

    connector.set_activity(3.0)
    assert connector.get_config().get_value(25) == pytest.approx(3.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "activity, seconds",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (1.0, 0.0),
        (1.0, float("inf")),
        (True, 1.0),
        (1.0, True),
        (np.bool_(True), 1.0),
        (1.0, np.bool_(True)),
        ("not-a-number", 1.0),
        (1.0, None),
        (1e308, 1e308),  # product overflows to inf
        (5e-324, 0.5),  # product underflows to 0.0
    ],
)
def test_set_activity_rejects_non_positive_or_non_finite(
    tmp_path: Path, activity, seconds
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    with pytest.raises(ValueError, match="finite and > 0"):
        connector.set_activity(activity, seconds)


def _acquisition(**changes):
    values = dict(
        num_projections=60,
        extent_deg=360.0,
        direction="CW",
        start_angle_deg=180.0,
        radius_mm=250.0,
        radii_mm=None,
        num_bins=64,
        num_axial=32,
        bin_size_mm=4.42,
        axial_size_mm=4.42,
        image_duration_s=None,
    )
    values.update(changes)
    return ProjectionGeometry(**values)


@pytest.mark.unit
@pytest.mark.parametrize("direction, sign", [("CW", 1.0), ("CCW", -1.0)])
def test_configure_acquisition_maps_to_simind_indices(tmp_path: Path, direction, sign):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.configure_acquisition(
        _acquisition(direction=direction, start_angle_deg=90.0)
    )
    config = connector.get_config()
    assert config.get_value(29) == 60
    assert config.get_value(30) == pytest.approx(sign * 360.0)
    assert config.get_value(41) == pytest.approx(270.0)
    assert config.get_value(12) == pytest.approx(25.0)
    assert config.get_value(28) == pytest.approx(0.442)
    assert (config.get_value(76), config.get_value(77)) == (64, 32)
    assert config.get_flag(5)
    assert not (tmp_path / "case01_acquisition.cor").exists()


@pytest.mark.unit
@pytest.mark.parametrize("direction, sign", [("CW", 1.0), ("CCW", -1.0)])
def test_configure_acquisition_clamps_extent_within_tolerance_for_simind(
    tmp_path: Path, direction, sign
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    geometry = _acquisition(direction=direction, extent_deg=360.001)
    connector.configure_acquisition(geometry)

    assert connector.get_config().get_value(30) == pytest.approx(sign * 360.0)
    # the template geometry itself keeps the measured value
    assert geometry.extent_deg == pytest.approx(360.001)
    assert check_geometry_match(_acquisition(direction=direction), geometry) == []


@pytest.mark.unit
def test_configure_acquisition_writes_orbit_file_and_run_uses_it(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    radii = tuple(240.0 + i for i in range(60))
    connector.configure_acquisition(_acquisition(radius_mm=None, radii_mm=radii))
    orbit = tmp_path / "case01_acquisition.cor"
    captured = []

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        captured.append(orbit_file)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run()
    lines = orbit.read_text().splitlines()
    assert lines[0] == "      24.000    32" and lines[-1] == "      29.900    32"
    assert len(lines) == 60

    override = tmp_path / "other.cor"
    override.write_text("15.0\n")
    connector.run(RuntimeOperator(orbit_file=override))
    connector.run()

    assert captured == [orbit, tmp_path / "case01_orbit.cor", orbit]
    assert orbit.exists()

    # The centre bin follows a matrix size changed after the template.
    connector.add_config_value(76, 128)
    connector.run()
    assert orbit.read_text().splitlines()[0] == "      24.000    64"


@pytest.mark.unit
def test_index_12_edit_does_not_clear_a_stored_non_circular_orbit(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    radii = tuple(240.0 + i for i in range(60))
    connector.configure_acquisition(_acquisition(radius_mm=None, radii_mm=radii))
    orbit = tmp_path / "case01_acquisition.cor"

    connector.add_config_value(12, 15.0)
    assert connector.get_config().get_value(12) == pytest.approx(15.0)

    captured = []

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        captured.append((connector.get_config().get_value(12), orbit_file))

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]

    connector.run()
    connector.run()

    assert captured == [(pytest.approx(15.0), orbit)] * 2
    lines = orbit.read_text().splitlines()
    assert lines[0] == "      24.000    32" and lines[-1] == "      29.900    32"
    assert len(lines) == 60


@pytest.mark.unit
def test_python_connector_explicit_orbit_at_acquisition_path_is_preserved(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    _non_circular_acquisition(connector)
    acquisition = tmp_path / "case01_acquisition.cor"
    explicit = b"      24.000    99\n" * 60
    acquisition.write_bytes(explicit)
    captured = _record_orbit_executor_run(connector)

    connector.run(RuntimeOperator(orbit_file=acquisition))

    assert captured["orbit_file"] == tmp_path / "case01_orbit.cor"
    assert acquisition.read_bytes() == explicit
    assert (tmp_path / "case01_orbit.cor").read_bytes() == explicit

    # Without an explicit orbit the stored radii regenerate the acquisition
    # orbit, using the current Index 76 centre bin.
    connector.add_config_value(76, 128)
    connector.run()

    assert captured["orbit_file"] == acquisition
    lines = acquisition.read_text().splitlines()
    assert lines[0] == "      24.000    64"
    assert len(lines) == 60


@pytest.mark.unit
def test_python_connector_explicit_orbit_aliased_to_acquisition_path_is_preserved(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    _non_circular_acquisition(connector)
    external = tmp_path / "elsewhere.cor"
    external.write_bytes(b"aliased-explicit")
    acquisition = tmp_path / "case01_acquisition.cor"
    acquisition.symlink_to(external)
    captured = _record_orbit_executor_run(connector)

    connector.run(RuntimeOperator(orbit_file=external))

    assert captured["orbit_file"] == tmp_path / "case01_orbit.cor"
    assert external.read_bytes() == b"aliased-explicit"
    assert acquisition.is_symlink()
    assert (tmp_path / "case01_orbit.cor").read_bytes() == b"aliased-explicit"


@pytest.mark.unit
def test_replacing_a_non_circular_acquisition_with_a_circular_one_clears_the_orbit(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.configure_acquisition(
        _acquisition(radius_mm=None, radii_mm=tuple(240.0 + i for i in range(60)))
    )
    connector.add_config_value(12, 15.0)

    connector.configure_acquisition(_acquisition())

    assert connector.get_config().get_value(12) == pytest.approx(25.0)
    captured = []
    connector.executor.run_simulation = (  # type: ignore[assignment]
        lambda *args, **kwargs: captured.append(args[1])
    )
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]

    connector.run()

    assert captured == [None]


@pytest.mark.unit
def test_configure_acquisition_treats_equal_radii_as_circular(tmp_path: Path):
    header = InterfileHeader.from_text(
        "!number of projections := 3\n"
        "!extent of rotation := 360\n"
        "!direction of rotation := CW\n"
        "start angle := 180\n"
        "orbit := non-circular\n"
        "Radii := {250, 250, 250}\n"
        "!matrix size [1] := 64\n"
        "scaling factor (mm/pixel) [1] := 4.42\n"
        "!matrix size [2] := 32\n"
        "scaling factor (mm/pixel) [2] := 4.42\n"
    )
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.configure_acquisition(read_projection_geometry(header))
    assert connector.get_config().get_value(12) == pytest.approx(25.0)
    assert not (tmp_path / "case01_acquisition.cor").exists()


@pytest.mark.unit
def test_configure_acquisition_treats_an_equal_radii_tuple_as_circular(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.configure_acquisition(
        _acquisition(
            num_projections=3,
            radius_mm=None,
            radii_mm=(250.0, 250.0, 250.0),
        )
    )
    assert connector.get_config().get_value(12) == pytest.approx(25.0)
    assert not (tmp_path / "case01_acquisition.cor").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "changes, message",
    [
        ({"axial_size_mm": 4.0}, "one projection pixel size"),
        ({"extent_deg": 0.0}, "extent"),
        ({"extent_deg": 360.6}, "extent"),
        ({"extent_deg": 400.0}, "extent"),
        ({"radius_mm": None, "radii_mm": (250.0, 260.0)}, "radii"),
    ],
)
def test_configure_acquisition_validation(tmp_path: Path, changes, message):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    with pytest.raises(ValueError, match=message):
        connector.configure_acquisition(_acquisition(**changes))


@pytest.mark.unit
def test_acquisition_values_win_over_voxel_phantom_in_either_order(tmp_path: Path):
    source = np.ones((8, 8, 8), dtype=np.float32)
    for order in ("phantom_first", "acquisition_first"):
        connector = SimindPythonConnector(
            config_source=get("Example.yaml"),
            output_dir=tmp_path / order,
            output_prefix="case01",
        )
        if order == "phantom_first":
            connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)
            connector.configure_acquisition(_acquisition())
        else:
            connector.configure_acquisition(_acquisition())
            connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)
        config = connector.get_config()
        assert config.get_value(28) == pytest.approx(0.442)
        assert (config.get_value(76), config.get_value(77)) == (64, 32)


@pytest.mark.unit
def test_run_warns_when_phantom_is_wider_than_the_field_of_view(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.configure_acquisition(_acquisition(num_bins=8))  # 35 mm field of view
    source = np.ones((8, 16, 16), dtype=np.float32)  # 64 mm across
    connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)
    connector.executor.run_simulation = lambda *a, **k: None  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    with caplog.at_level("WARNING"):
        connector.run()
    assert "wider than the projection field of view" in caplog.text


@pytest.mark.unit
def test_run_warns_when_phantom_is_longer_than_the_field_of_view(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.add_config_value(28, 0.4)
    connector.add_config_value(76, 64)
    connector.add_config_value(77, 8)  # 32 mm axially
    source = np.ones((16, 8, 8), dtype=np.float32)  # 64 mm axially
    connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)
    connector.executor.run_simulation = lambda *a, **k: None  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    with caplog.at_level("WARNING"):
        connector.run()
    assert "longer than the projection field of view" in caplog.text
    assert "wider than" not in caplog.text


def _write_density(connector, mu_map, **kwargs):
    connector.get_config().set_flag(11, True)
    source = np.ones_like(mu_map, dtype=np.float32)
    _, density_path = connector.configure_voxel_phantom(
        source=source, mu_map=mu_map, **kwargs
    )
    return np.fromfile(density_path, dtype=np.uint16)


@pytest.mark.unit
def test_configure_voxel_phantom_accepts_density_input(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    written = _write_density(
        connector, np.full((3, 4, 5), 1.5, dtype=np.float32), mu_map_type="density"
    )
    assert np.all(written == 1500)


@pytest.mark.unit
def test_configure_voxel_phantom_accepts_hu_input_with_negative_values(
    tmp_path: Path, monkeypatch
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    hu_map = np.full((3, 4, 5), -1000.0, dtype=np.float32)
    captured = {}

    def fake_hu_to_density_schneider(values):
        captured["input"] = values.copy()
        return np.full_like(values, 1.25)

    monkeypatch.setattr(
        connector_mod, "hu_to_density_schneider", fake_hu_to_density_schneider
    )
    written = _write_density(connector, hu_map, mu_map_type="hu")
    assert np.all(written == 1250)
    assert np.array_equal(captured["input"], hu_map)


@pytest.mark.unit
def test_configure_voxel_phantom_uses_explicit_mu_map_energy(tmp_path: Path):
    mu = np.full((3, 4, 5), 0.096, dtype=np.float32)  # water at 511 keV
    at_511 = _write_density(
        SimindPythonConnector(get("Example.yaml"), tmp_path / "a", "case01"),
        mu,
        mu_map_energy_kev=511.0,
    )
    at_config = _write_density(
        SimindPythonConnector(get("Example.yaml"), tmp_path / "b", "case01"), mu
    )
    assert abs(int(at_511[0]) - 1000) < 30
    assert at_config[0] < 700  # read as 140 keV, the same mu means less density


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mu_map_type": "unsupported"}, "mu_map_type must be one of"),
        ({"mu_map_energy_kev": 0.0}, "mu_map_energy_kev must be > 0"),
    ],
)
def test_configure_voxel_phantom_rejects_bad_mu_arguments(
    tmp_path: Path, kwargs, message
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    source = np.ones((3, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        connector.configure_voxel_phantom(source, np.ones_like(source), **kwargs)


@pytest.mark.unit
def test_configure_voxel_phantom_rejects_negative_attenuation(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    source = np.ones((3, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        connector.configure_voxel_phantom(source, -np.ones_like(source))


@pytest.mark.unit
def test_configure_voxel_phantom_anisotropic_voxels(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    source = np.ones((10, 6, 6), dtype=np.float32)
    connector.configure_voxel_phantom(
        source, np.zeros_like(source), voxel_size_mm=(8.0, 4.0, 4.0)
    )
    config = connector.get_config()
    assert config.get_value(2) == pytest.approx(10 * 0.8 / 2)
    assert config.get_value(5) == pytest.approx(10 * 0.8 / 2)
    assert config.get_value(31) == pytest.approx(0.4)
    assert connector.runtime_switches.switches["PX"] == pytest.approx(0.4)
    assert connector.runtime_switches.switches["TH"] == pytest.approx(0.8)


@pytest.mark.unit
def test_configure_voxel_phantom_isotropic_voxels_remove_a_stale_th(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    source = np.ones((4, 4, 4), dtype=np.float32)
    connector.configure_voxel_phantom(
        source, np.zeros_like(source), voxel_size_mm=(8.0, 4.0, 4.0)
    )
    assert connector.runtime_switches.switches["TH"] == pytest.approx(0.8)

    connector.configure_voxel_phantom(
        source, np.zeros_like(source), voxel_size_mm=(4.0, 4.0, 4.0)
    )

    assert "TH" not in connector.runtime_switches.switches


@pytest.mark.unit
def test_configure_voxel_phantom_needs_square_in_plane_voxels(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    source = np.ones((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="square in-plane"):
        connector.configure_voxel_phantom(
            source, np.zeros_like(source), (4.0, 3.0, 4.0)
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "voxel_size_mm",
    [(), (4.0,), (4.0, 4.0), (4.0, 4.0, 4.0, 4.0)],
)
def test_configure_voxel_phantom_rejects_bad_voxel_size_tuples(
    tmp_path: Path, voxel_size_mm
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    source = np.ones((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        connector.configure_voxel_phantom(source, np.zeros_like(source), voxel_size_mm)
    assert str(excinfo.value) == "voxel_size_mm must be a scalar or a (z, y, x) tuple"


@pytest.mark.unit
def test_run_warns_when_no_phantom_is_configured(tmp_path: Path, caplog):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.executor.run_simulation = lambda *args, **kwargs: None
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        connector.run()
    assert "No phantom configured" in caplog.text


@pytest.mark.unit
def test_configure_voxel_phantom_suppresses_the_warning(tmp_path: Path, caplog):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    source = np.zeros((4, 4, 4), dtype=np.float32)
    connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)
    connector.executor.run_simulation = lambda *args, **kwargs: None
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        connector.run()
    assert "No phantom configured" not in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("table_directory", ["tutorial", "smc_dir"])
def test_find_zub_table_expands_home_relative_simind_bin(
    tmp_path, monkeypatch, table_directory
):
    home = tmp_path / "home"
    installation = home / "simind"
    installation.mkdir(parents=True)
    executable = installation / "simind"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    table = installation / table_directory / "phantom.zub"
    table.parent.mkdir()
    table.write_text("== V8.0 Code Section 1\n")
    empty_path = tmp_path / "empty_path"
    empty_path.mkdir()
    for variable in ("SIMIND_SMC_DIR", "SMC_DIR", "SIMIND_DATA_DIR", "SIMIND_ROOT"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(empty_path))
    monkeypatch.setenv("SIMIND_BIN", "~/simind/simind")
    assert connector_mod._find_zub_table() == table.resolve()


# ---------------------------------------------------------------------------
# Preflight and phantom safeguards (semantic port from 1.1.2)
# ---------------------------------------------------------------------------


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
def test_python_connector_voxel_phantom_route_rejects_unsupported_scoring(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    activity = np.ones((2, 3, 4), dtype=np.float32)
    seeded, _ = _seed_existing_inputs(tmp_path)

    with pytest.raises(ValueError, match=r"LIST_MODE \(2\) is not supported"):
        connector.configure_phantom(
            VoxelPhantom(activity, np.ones_like(activity), (4.0, 4.0, 4.0)),
            scoring_routine=ScoringRoutine.LIST_MODE,
        )

    for path, original in seeded.items():
        assert path.read_bytes() == original, path


@pytest.mark.unit
@pytest.mark.parametrize(
    "routine",
    [ScoringRoutine.DUMMY, ScoringRoutine.SCATTWIN, ScoringRoutine.PENETRATE],
)
def test_python_connector_accepts_supported_scoring_routines(
    tmp_path: Path, routine: ScoringRoutine
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    source = np.ones((2, 3, 4), dtype=np.float32)

    connector.configure_voxel_phantom(source, source.copy(), scoring_routine=routine)

    assert int(connector.get_config().get_value(84)) == routine.value


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
@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(Decimal("1.00000000000000001"), id="fractional-decimal"),
        pytest.param(Fraction(10**17 + 1, 10**17), id="fractional-fraction"),
        pytest.param(np.float64(1.5), id="fractional-float"),
    ],
)
def test_python_connector_rejects_fractional_scoring_without_float_round(
    tmp_path: Path, bad_value: object
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    # A float round would collapse the exact fractions to 1.0 and accept them.
    connector.add_config_value(84, bad_value)
    source = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="not an integral"):
        connector.configure_voxel_phantom(source, source.copy())
    with pytest.raises(ValueError, match="not an integral"):
        connector.run()


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_value",
    [True, False, np.bool_(True), np.bool_(False)],
    ids=["bool-true", "bool-false", "np-bool-true", "np-bool-false"],
)
def test_python_connector_rejects_boolean_scoring_value(
    tmp_path: Path, bad_value: object
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(84, bad_value)
    source = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="scoring_routine"):
        connector.configure_voxel_phantom(source, source.copy())
    with pytest.raises(ValueError, match="scoring_routine"):
        connector.run()


@pytest.mark.unit
@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(1, 1, id="int"),
        pytest.param(np.int64(1), 1, id="np-int"),
        pytest.param(Decimal(1), 1, id="integral-decimal"),
        pytest.param(Fraction(1, 1), 1, id="integral-fraction"),
        pytest.param(1.0, 1, id="integral-float"),
    ],
)
def test_python_connector_accepts_integral_scoring_values(
    tmp_path: Path, value: object, expected: int
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(84, value)
    source = np.ones((2, 3, 4), dtype=np.float32)

    connector.configure_voxel_phantom(source, source.copy())

    assert connector._config_scoring_routine().value == expected


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
    with pytest.raises(FileNotFoundError, match="No projection headers"):
        connector.run()

    assert launched == [True]
    assert connector._is_penetrate_routine() is False


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

    seeded, orbit = _seed_existing_inputs(tmp_path)
    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="scoring_routine"):
        connector.configure_voxel_phantom(source, source.copy())
    with pytest.raises(ValueError, match="scoring_routine"):
        connector.run(RuntimeOperator(orbit_file=orbit))

    assert launched == []
    for path, original in seeded.items():
        assert path.read_bytes() == original
    assert (tmp_path / "case01_orbit.cor").read_bytes() == b"prior:case01_orbit.cor"


@pytest.mark.unit
def test_python_connector_config_only_run_rejects_unsupported_routine(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
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
    assert not (tmp_path / "case01.smc").exists()
    assert not (tmp_path / "case01_orbit.cor").exists()


def _seed_existing_inputs(
    tmp_path: Path, prefix: str = "case01"
) -> tuple[dict[Path, bytes], Path]:
    """Seed maps, config, a stale header and the canonical orbit target."""
    orbit = tmp_path / "orbit.cor"
    orbit.write_bytes(b"prior orbit")
    paths = [
        tmp_path / f"{prefix}_src.smi",
        tmp_path / f"{prefix}_dns.dmi",
        tmp_path / f"{prefix}.smc",
        tmp_path / f"{prefix}_tot_w1.hs",
        tmp_path / f"{prefix}_orbit.cor",
    ]
    for path in paths:
        path.write_bytes(f"prior:{path.name}".encode())
    return {path: path.read_bytes() for path in [*paths, orbit]}, orbit


@pytest.mark.unit
def test_python_connector_run_rejects_routine_changed_after_configure(tmp_path: Path):
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
def test_python_connector_single_voxel_keeps_prior_phantom_state(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    good = np.ones((2, 3, 4), dtype=np.float32)
    source_path, density_path = connector.configure_voxel_phantom(good, good.copy())
    index_5_before = connector.get_config().get_value(5)
    source_bytes = source_path.read_bytes()
    density_bytes = density_path.read_bytes()

    with pytest.raises(ValueError, match="1x1x1"):
        connector.configure_voxel_phantom(
            np.ones((1, 1, 1), np.float32), np.ones((1, 1, 1), np.float32)
        )

    assert connector.get_config().get_value(5) == index_5_before
    assert source_path.read_bytes() == source_bytes
    assert density_path.read_bytes() == density_bytes


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
    # The default scale reproduces the legacy 500 budget, so 0.001 of it gives
    # a peak of 0.5, which NumPy rounds to even (0): every voxel vanishes.
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
def test_python_connector_invalid_inputs_leave_seeded_state_untouched(tmp_path: Path):
    """Every rejected input path preserves inputs and the canonical orbit."""
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    seeded, orbit = _seed_existing_inputs(tmp_path)
    launched: list[object] = []
    connector.executor.run_simulation = lambda *a, **k: launched.append(True)  # type: ignore[assignment]

    source = np.ones((2, 3, 4), dtype=np.float32)

    # 1x1x1 voxel input
    with pytest.raises(ValueError, match="1x1x1"):
        connector.configure_voxel_phantom(
            np.ones((1, 1, 1), np.float32), np.ones((1, 1, 1), np.float32)
        )

    # nonzero source that quantizes entirely to zero
    tiny = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        quantization_scale=1e-6,
    )
    with pytest.raises(ValueError, match="rounds the non-zero source to all zeros"):
        tiny.configure_voxel_phantom(source, source.copy())

    # invalid present scoring routine on the array route and the run route
    connector.add_config_value(84, ScoringRoutine.LIST_MODE.value)
    with pytest.raises(ValueError, match="scoring_routine"):
        connector.configure_voxel_phantom(source, source.copy())
    with pytest.raises(ValueError, match="scoring_routine"):
        connector.run(RuntimeOperator(orbit_file=orbit))

    # invalid runtime orbit directory
    connector.add_config_value(84, ScoringRoutine.SCATTWIN.value)
    orbit_dir = tmp_path / "orbit-dir"
    orbit_dir.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        connector.run(RuntimeOperator(orbit_file=orbit_dir))

    # over-long prefix
    connector.output_prefix = "a" * 48
    with pytest.raises(ValueError, match="at most 47 characters"):
        connector.run()
    with pytest.raises(ValueError, match="at most 47 characters"):
        connector.set_energy_windows([126.0], [154.0], [0])

    assert launched == []
    for path, original in seeded.items():
        assert path.read_bytes() == original, path
    assert not (tmp_path / "case01_orbit.cor.tmp").exists()


def _seed_preflight_state(tmp_path: Path, prefix: str = "case01") -> dict[Path, bytes]:
    """Seed every file run() could overwrite or clear before the executor."""
    paths = [
        tmp_path / f"{prefix}_src.smi",
        tmp_path / f"{prefix}_dns.dmi",
        tmp_path / f"{prefix}.win",
        tmp_path / f"{prefix}.smc",
        tmp_path / f"{prefix}_tot_w1.hs",
        tmp_path / f"{prefix}_orbit.cor",
        tmp_path / f"{prefix}_acquisition.cor",
    ]
    for path in paths:
        path.write_bytes(f"prior:{path.name}".encode())
    return {path: path.read_bytes() for path in paths}


def _non_circular_acquisition(connector: SimindPythonConnector) -> None:
    connector.configure_acquisition(
        _acquisition(radius_mm=None, radii_mm=tuple(240.0 + i for i in range(60)))
    )


def _launch_then_fail(launched: list[object]):
    def fake_run_simulation(*args, **kwargs):
        launched.append(True)
        raise SimulationError("executor must not be reached")

    return fake_run_simulation


@pytest.mark.unit
@pytest.mark.parametrize("switch", ["FI", "CC"])
@pytest.mark.parametrize("value", ["bad name", "bad\tname", "bad\nname", "bad\x00name"])
def test_python_connector_rejects_illegal_switch_before_acquisition_orbit(
    tmp_path: Path, switch: str, value: str
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    # Variable radii make run() regenerate the acquisition orbit before the
    # supplied orbit would be staged and the config saved.
    _non_circular_acquisition(connector)
    seeded = _seed_preflight_state(tmp_path)
    launched: list[object] = []
    connector.executor.run_simulation = _launch_then_fail(launched)  # type: ignore[assignment]

    with pytest.raises(SimulationError):
        connector.run(RuntimeOperator(switches={switch: value}))

    assert launched == []
    for path, original in seeded.items():
        assert path.read_bytes() == original, path


@pytest.mark.unit
@pytest.mark.parametrize("operator", [None, RuntimeOperator()])
def test_python_connector_rejects_invalid_persistent_switch_before_writes(
    tmp_path: Path, operator
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    _non_circular_acquisition(connector)
    connector.add_runtime_switch("FI", "bad name")
    seeded = _seed_preflight_state(tmp_path)
    launched: list[object] = []
    connector.executor.run_simulation = _launch_then_fail(launched)  # type: ignore[assignment]

    with pytest.raises(SimulationError):
        connector.run(operator)

    assert launched == []
    for path, original in seeded.items():
        assert path.read_bytes() == original, path


@pytest.mark.unit
def test_python_connector_valid_runtime_value_overrides_invalid_persistent_switch(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_runtime_switch("FI", "bad name")
    captured: list[dict] = []
    connector.executor.run_simulation = (  # type: ignore[assignment]
        lambda *args, **kwargs: captured.append(dict(args[2] or {}))
    )
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]

    connector.run(RuntimeOperator(switches={"FI": "good.inp"}))

    assert captured == [{"FI": "good.inp"}]
    assert connector.runtime_switches.switches == {"FI": "bad name"}


@pytest.mark.unit
def test_python_connector_none_removal_of_invalid_persistent_switch_succeeds(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_runtime_switch("FI", "bad name")
    captured: list[dict] = []
    connector.executor.run_simulation = (  # type: ignore[assignment]
        lambda *args, **kwargs: captured.append(dict(args[2] or {}))
    )
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]

    connector.run(RuntimeOperator(switches={"FI": None}))

    assert captured == [{}]
    assert connector.runtime_switches.switches == {"FI": "bad name"}


@pytest.mark.unit
def test_python_connector_bare_switch_stays_bare_and_persistent_state_unchanged(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_runtime_switch("HO", True)
    captured: list[dict] = []
    connector.executor.run_simulation = (  # type: ignore[assignment]
        lambda *args, **kwargs: captured.append(dict(args[2] or {}))
    )
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]

    connector.run(RuntimeOperator(switches={"NN": 3}))

    assert captured == [{"HO": True, "NN": 3}]
    token = connector.executor._serialize_switches(captured[0])
    assert token == "/HO/NN:3"
    assert ":True" not in token
    assert connector.runtime_switches.switches == {"HO": True}


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
    assert len(lines) == count


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


class _LaunchReached(Exception):
    pass


def _warnings_before_launch(
    monkeypatch,
    tmp_path: Path,
    *,
    system: str,
    version: int,
    routine: ScoringRoutine,
) -> list[warnings.WarningMessage]:
    monkeypatch.setattr(connector_mod.platform, "system", lambda: system)
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        simind_version=version,
    )
    connector.add_config_value(84, routine.value)

    def fake_run_simulation(*args, **kwargs):
        raise _LaunchReached

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_LaunchReached):
            connector.run()
    return caught


@pytest.mark.unit
def test_python_connector_warns_about_v9_linux_scattwin(monkeypatch, tmp_path: Path):
    caught = _warnings_before_launch(
        monkeypatch,
        tmp_path,
        system="Linux",
        version=9,
        routine=ScoringRoutine.SCATTWIN,
    )

    defect = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(defect) == 1
    assert "SCATTWIN" in str(defect[0].message)
    assert "PENETRATE" in str(defect[0].message)
    assert defect[0].filename == __file__


@pytest.mark.unit
@pytest.mark.parametrize(
    ("system", "version", "routine"),
    [
        ("Linux", 8, ScoringRoutine.SCATTWIN),
        ("Linux", 9, ScoringRoutine.PENETRATE),
        ("Darwin", 9, ScoringRoutine.SCATTWIN),
    ],
)
def test_python_connector_does_not_warn_outside_v9_linux_scattwin(
    monkeypatch, tmp_path: Path, system: str, version: int, routine: ScoringRoutine
):
    caught = _warnings_before_launch(
        monkeypatch, tmp_path, system=system, version=version, routine=routine
    )

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
