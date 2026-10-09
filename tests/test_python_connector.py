import logging
from pathlib import Path

import numpy as np
import pytest

import simind_python_connector.connectors.python_connector as connector_mod
from simind_python_connector.configs import get
from simind_python_connector.connectors import RuntimeOperator, SimindPythonConnector
from simind_python_connector.core.types import ScoringRoutine, SimulationError
from simind_python_connector.utils.interfile import (
    InterfileHeader,
    ProjectionGeometry,
    check_geometry_match,
    read_projection_geometry,
)


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
def test_python_connector_rejects_unsafe_output_prefixes(tmp_path: Path):
    for bad_prefix in ("../escape", "/absolute", "", "a/b", "a\\b", ".."):
        with pytest.raises(ValueError, match="output_prefix"):
            SimindPythonConnector(
                config_source=get("AnyScan.yaml"),
                output_dir=tmp_path,
                output_prefix=bad_prefix,
            )


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
    assert config.get_value(76) == pytest.approx(4)  # columns = max(dim_x, dim_y)
    assert config.get_value(77) == pytest.approx(2)  # rows = dim_z (axial)
    assert config.get_value(78) == pytest.approx(4)  # density map i = dim_x
    assert config.get_value(79) == pytest.approx(4)  # source map i = dim_x
    assert config.get_value(81) == pytest.approx(3)  # density map j = dim_y
    assert config.get_value(82) == pytest.approx(3)  # source map j = dim_y
    assert config.get_value(34) == pytest.approx(2)  # number of density images


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_uses_axial_projection_rows(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    # dim_y (4) exceeds dim_x (3), so the columns follow dim_y.
    source = np.zeros((2, 4, 3), dtype=np.float32)
    source[0, 0, 0] = 1.0

    connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)

    config = connector.get_config()
    assert config.get_value(76) == pytest.approx(4)  # columns = max(dim_x, dim_y)
    assert config.get_value(77) == pytest.approx(2)  # rows = dim_z (axial)
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


@pytest.mark.unit
def test_python_connector_runtime_operator_switches_are_one_shot(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir=tmp_path / "run1",
        output_prefix="case01",
    )

    captured_switches: list[dict] = []

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
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
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
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
        quantization_scale=0.05,
    )
    assert connector.quantization_scale == pytest.approx(0.05)


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
    outputs = connector.run(RuntimeOperator(switches={"NN": 2, "RR": 12345}))

    assert captured["output_prefix"] == "case01"
    assert captured["runtime_switches"] == {"NN": 2, "RR": 12345, "CA": 1}

    assert "tot_w1" in outputs
    result = outputs["tot_w1"]
    assert result.header_path == (tmp_path / "case01_tot_w1.hs").resolve()
    assert result.data_path == (tmp_path / "case01_tot_w1.a00").resolve()
    assert result.projection.shape == (2, 3, 4)
    expected_sum = float(np.arange(24, dtype=np.float32).sum())
    assert float(result.projection.sum()) == expected_sum


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

    assert captured == {"mpi_processes": 4, "split_projections": True}


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
def test_python_connector_full_scale_source_values(tmp_path: Path):
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
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
    ):
        _write_projection(tmp_path, "case01_tot_w1", (2, 3, 4), (2, 3, 4))
        _write_projection(tmp_path, "case01_air_w1", (4, 4), (4, 4, 4))

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    with pytest.raises(SimulationError, match="case01_air_w1"):
        connector.run()


@pytest.mark.unit
def test_python_connector_rejects_upper_case_prefix(tmp_path: Path):
    with pytest.raises(ValueError, match="lower case"):
        SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path,
            output_prefix="Case01",
        )


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
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
    ):
        captured["orbit_file"] = orbit_file

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run(RuntimeOperator(orbit_file=orbit))
    connector.run(RuntimeOperator(orbit_file=tmp_path / "out" / "case01_orbit.cor"))

    assert captured["orbit_file"] == tmp_path / "out" / "case01_orbit.cor"
    assert (tmp_path / "out" / "case01_orbit.cor").read_text() == "15.0\n"


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
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
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
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
    ):
        _write_named_projection(tmp_path, "case01_tot_w1", total)
        _write_named_projection(tmp_path, "case01_sca_w1", scatter)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    outputs = connector.run()

    assert np.array_equal(outputs["pri_w1"].projection, total - scatter)
    assert outputs["pri_w1"].metadata["derived"] == "tot - sca"
    assert "derived" not in outputs["tot_w1"].metadata


@pytest.mark.unit
def test_python_connector_keeps_user_ca_and_simind_primary(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path, output_prefix="case01"
    )
    connector.add_runtime_switch("CA", 2)
    primary = np.full((2, 3, 4), 5.0, dtype=np.float32)
    captured = {}

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
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
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
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
    connector.add_runtime_switch("FW", "case01_custom")
    (tmp_path / "case01_custom.win").write_text("126.0,154.0,0\n92.0,124.0,0\n")
    total = np.full((2, 3, 4), 10.0, dtype=np.float32)
    scatter = np.full((2, 3, 4), 1.0, dtype=np.float32)

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
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
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
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
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
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
    lines = orbit.read_text().splitlines()
    assert lines[0] == "      24.000    32" and lines[-1] == "      29.900    32"
    assert len(lines) == 60

    captured = []

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
    ):
        captured.append(orbit_file)

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run()
    override = tmp_path / "other.cor"
    override.write_text("15.0\n")
    connector.run(RuntimeOperator(orbit_file=override))
    connector.run()

    assert captured == [orbit, tmp_path / "case01_orbit.cor", orbit]
    assert orbit.exists()


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
    assert config.get_value(28) == pytest.approx(0.4)
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
