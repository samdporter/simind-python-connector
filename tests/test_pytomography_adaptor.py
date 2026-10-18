import os
import types
from pathlib import Path

import numpy as np
import pytest

from simind_python_connector.configs import get
from simind_python_connector.connectors.python_connector import (
    ProjectionResult,
    RuntimeOperator,
)
from simind_python_connector.connectors.pytomography_adaptor import (
    PyTomographySimindAdaptor,
)
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
    SimulationError,
)


torch = pytest.importorskip("torch")


pytestmark = pytest.mark.requires_pytomography


@pytest.mark.unit
def test_pytomography_adaptor_preserves_projection_shape(tmp_path: Path, monkeypatch):
    connector = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        voxel_size_mm=4.0,
    )

    source = torch.zeros((6, 8, 8), dtype=torch.float32)
    source[2:5, 2:6, 2:6] = 1.0
    mu_map = torch.zeros_like(source)
    mu_map[source > 0] = 0.15

    connector.set_source(source)
    connector.set_mu_map(mu_map)
    connector.set_energy_windows([126], [154], [0])

    projection = np.arange(8 * 5 * 7, dtype=np.float32).reshape(8, 5, 7)
    fake_result = ProjectionResult(
        projection=projection,
        header_path=tmp_path / "case01_tot_w1.hs",
        data_path=tmp_path / "case01_tot_w1.a00",
        metadata={"key": "tot_w1"},
    )

    monkeypatch.setattr(
        connector.python_connector,
        "run",
        lambda runtime_operator=None, timeout=None: {"tot_w1": fake_result},
    )

    outputs = connector.run()

    assert "tot_w1" in outputs
    assert tuple(outputs["tot_w1"].shape) == (8, 7, 5)
    assert (
        connector.get_output_header_path("tot_w1") == fake_result.header_path.resolve()
    )


@pytest.mark.unit
def test_pytomography_adaptor_avoids_wrapping_pytomography_methods(tmp_path: Path):
    connector = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    assert not hasattr(connector, "build_system_matrix")
    assert not hasattr(connector, "get_pytomography_metadata")


@pytest.mark.unit
def test_pytomography_adaptor_forwards_connector_wiring_and_axis_order(
    tmp_path: Path, monkeypatch
):
    connector = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        voxel_size_mm=3.5,
        scoring_routine=ScoringRoutine.PENETRATE,
        mu_map_type="hu",
        mu_map_energy_kev=511.0,
    )

    source_xyz = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    mu_xyz = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 10.0
    connector.set_source(source_xyz)
    connector.set_mu_map(mu_xyz)
    connector.set_energy_windows([75.0, 120.0], [85.0, 140.0], [0, 1])

    captured: dict[str, object] = {}

    def fake_configure_voxel_phantom(
        source,
        mu_map,
        voxel_size_mm,
        scoring_routine,
        mu_map_type,
        mu_map_energy_kev,
    ):
        captured["source"] = np.asarray(source)
        captured["mu_map"] = np.asarray(mu_map)
        captured["voxel_size_mm"] = voxel_size_mm
        captured["scoring_routine"] = scoring_routine
        captured["mu_map_type"] = mu_map_type
        captured["mu_map_energy_kev"] = mu_map_energy_kev
        return (tmp_path / "case01_src.smi", tmp_path / "case01_dns.dmi")

    def fake_set_energy_windows(lower_bounds, upper_bounds, scatter_orders):
        captured["windows"] = (
            list(lower_bounds),
            list(upper_bounds),
            list(scatter_orders),
        )

    projection = np.arange(8 * 5 * 7, dtype=np.float32).reshape(8, 5, 7)
    raw_outputs = {
        "tot_w1": ProjectionResult(
            projection=projection,
            header_path=tmp_path / "case01_tot_w1.hs",
            data_path=tmp_path / "case01_tot_w1.a00",
            metadata={"component": "tot_w1"},
        ),
        "sca_w1": ProjectionResult(
            projection=projection + 1.0,
            header_path=tmp_path / "case01_sca_w1.hs",
            data_path=tmp_path / "case01_sca_w1.a00",
            metadata={"component": "sca_w1"},
        ),
        "pri_w1": ProjectionResult(
            projection=projection + 2.0,
            header_path=tmp_path / "case01_pri_w1.hs",
            data_path=tmp_path / "case01_pri_w1.a00",
            metadata={"component": "pri_w1"},
        ),
        "air_w1": ProjectionResult(
            projection=projection + 3.0,
            header_path=tmp_path / "case01_air_w1.hs",
            data_path=tmp_path / "case01_air_w1.a00",
            metadata={"component": "air_w1"},
        ),
    }

    def fake_run(runtime_operator=None, timeout=None):
        captured["runtime_operator"] = runtime_operator
        return raw_outputs

    monkeypatch.setattr(
        connector.python_connector,
        "configure_voxel_phantom",
        fake_configure_voxel_phantom,
    )
    monkeypatch.setattr(
        connector.python_connector,
        "set_energy_windows",
        fake_set_energy_windows,
    )
    monkeypatch.setattr(
        connector.python_connector,
        "run",
        fake_run,
    )

    runtime_operator = RuntimeOperator(switches={"RR": 12345})
    outputs = connector.run(runtime_operator=runtime_operator)

    expected_source_zyx = source_xyz.flip(2).permute(2, 1, 0).numpy()
    expected_mu_zyx = mu_xyz.flip(2).permute(2, 1, 0).numpy()
    assert np.array_equal(captured["source"], expected_source_zyx)
    assert np.array_equal(captured["mu_map"], expected_mu_zyx)
    assert captured["voxel_size_mm"] == pytest.approx(3.5)
    assert captured["scoring_routine"] == ScoringRoutine.PENETRATE
    assert captured["mu_map_type"] == "hu"
    assert captured["mu_map_energy_kev"] == 511.0
    assert captured["windows"] == ([75.0, 120.0], [85.0, 140.0], [0, 1])
    assert captured["runtime_operator"] is runtime_operator

    assert torch.equal(connector.get_total_output(window=1), outputs["tot_w1"])
    assert torch.equal(connector.get_scatter_output(window=1), outputs["sca_w1"])
    assert torch.equal(connector.get_primary_output(window=1), outputs["pri_w1"])
    assert torch.equal(connector.get_air_output(window=1), outputs["air_w1"])


@pytest.mark.unit
def test_pytomography_axis_helpers_roundtrip_through_simind_order() -> None:
    xyz = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    zyx = PyTomographySimindAdaptor.to_simind_image_axes(xyz)
    restored = PyTomographySimindAdaptor.from_simind_image_axes(zyx)

    assert tuple(zyx.shape) == (4, 3, 2)
    assert tuple(restored.shape) == (2, 3, 4)
    assert torch.equal(restored, xyz)


@pytest.mark.unit
@pytest.mark.parametrize("direction", ["from", "to"])
def test_pytomography_axis_helpers_match_real_simind_reader(
    tmp_path: Path, direction: str
) -> None:
    from pytomography.io.SPECT import simind as pytomo_simind

    # Distinct dimensions and voxel values expose any axis or slice reversal.
    simind_zyx = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    simind_zyx.tofile(tmp_path / "asymmetric.ict")
    header_path = tmp_path / "asymmetric.hct"
    header_path.write_text(
        "!INTERFILE :=\n"
        "name of data file := asymmetric.ict\n"
        "matrix size [1] := 2\n"
        "matrix size [2] := 3\n"
        "matrix size [3] := 4\n"
        "!END OF INTERFILE :=\n"
    )
    pytomo_xyz = pytomo_simind.get_attenuation_map(
        str(header_path), smi_index_22=3
    ).cpu()
    simind_tensor = torch.from_numpy(simind_zyx)

    if direction == "from":
        actual = PyTomographySimindAdaptor.from_simind_image_axes(simind_tensor)
        expected = pytomo_xyz
    else:
        actual = PyTomographySimindAdaptor.to_simind_image_axes(pytomo_xyz)
        expected = simind_tensor

    assert actual.is_contiguous()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.integration
@pytest.mark.requires_simind
def test_pytomography_simind_smoke_preserves_axial_source_position(
    tmp_path: Path,
) -> None:
    """Sparse, two-view smoke test of the source-to-projection slice order."""
    adaptor = PyTomographySimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="axial_smoke",
        photon_multiplier=1,
        voxel_size_mm=4.0,
        quantization_scale=0.01 * DEFAULT_QUANTIZATION_SCALE,
    )
    source_xyz = torch.zeros((16, 16, 16), dtype=torch.float32)
    source_xyz[7:9, 7:9, 3:5] = 1.0
    adaptor.set_source(source_xyz)
    adaptor.set_mu_map(torch.zeros_like(source_xyz))
    adaptor.set_energy_windows([126], [154], [0])
    adaptor.add_config_value(19, 2)
    adaptor.add_config_value(53, 0)
    adaptor.add_config_value(29, 2)
    # Projections on the source grid: 16 x 16 bins of 4 mm (Index 28 in cm).
    adaptor.add_config_value(28, 0.4)
    adaptor.add_config_value(76, 16)
    adaptor.add_config_value(77, 16)
    adaptor.add_runtime_switch("CC", "ma-lehr")
    adaptor.add_runtime_switch("RR", 7)
    adaptor.run()

    projection = adaptor.get_total_output().cpu()
    assert tuple(projection.shape) == (2, 16, 16)
    axial_counts = projection.sum(dim=(0, 1))
    assert axial_counts.sum() > 0
    axial_centre = ((axial_counts * torch.arange(16)).sum() / axial_counts.sum()).item()
    print(f"SIMIND smoke test: axial source centre=3.5, projection={axial_centre:.3f}")
    assert axial_centre == pytest.approx(3.5, abs=1.5)


@pytest.mark.integration
@pytest.mark.requires_simind
def test_pytomography_penetrate_smoke_reconstructs_off_centre_source(
    tmp_path: Path,
) -> None:
    """Smoke test: OSEM from the PENETRATE header puts the source where it was."""
    from pytomography import algorithms, likelihoods
    from pytomography.io.SPECT import simind as pytomo_simind
    from pytomography.projectors.SPECT import SPECTSystemMatrix

    adaptor = PyTomographySimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir=tmp_path,
        output_prefix="penetrate_smoke",
        voxel_size_mm=4.0,
        quantization_scale=0.05 * DEFAULT_QUANTIZATION_SCALE,
        scoring_routine=ScoringRoutine.PENETRATE,
    )
    source_xyz = torch.zeros((16, 16, 16), dtype=torch.float32)
    source_xyz[11:13, 3:5, 7:9] = 1.0
    adaptor.set_source(source_xyz)
    adaptor.set_mu_map(torch.zeros_like(source_xyz))
    adaptor.set_energy_windows([126], [154], [0])
    adaptor.add_config_value(19, 2)
    adaptor.add_config_value(53, 0)
    adaptor.add_config_value(29, 8)
    adaptor.add_config_value(28, 0.4)
    adaptor.add_config_value(76, 16)
    adaptor.add_config_value(77, 16)
    adaptor.add_runtime_switch("CC", "ma-lehr")
    adaptor.add_runtime_switch("RR", 7)
    adaptor.run()

    object_meta, proj_meta = pytomo_simind.get_metadata(
        str(adaptor.get_output_header_path("all_interactions"))
    )
    system_matrix = SPECTSystemMatrix([], [], object_meta, proj_meta)
    likelihood = likelihoods.PoissonLogLikelihood(
        system_matrix, adaptor.get_outputs()["all_interactions"]
    )
    recon = algorithms.OSEM(likelihood)(2, n_subsets=4).cpu()

    weights = recon.sum(dim=2)
    x, y = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    centre = ((weights * x).sum() / weights.sum(), (weights * y).sum() / weights.sum())
    assert centre[0].item() == pytest.approx(11.5, abs=1.0)
    assert centre[1].item() == pytest.approx(3.5, abs=1.0)


def _write_simind_header(path: Path, data_file: str, start_angle: float) -> None:
    path.write_text(
        "!INTERFILE :=\n"
        "program version := V8.0\n"
        f"!name of data file := {data_file}\n"
        "!total number of images := 2\n"
        "!matrix size [1] := 4\n"
        "!matrix size [2] := 3\n"
        "!number format := short float\n"
        "!number of bytes per pixel := 4\n"
        "scaling factor (mm/pixel) [1] := 4.0\n"
        "scaling factor (mm/pixel) [2] := 4.0\n"
        "!extent of rotation := 360\n"
        "!number of projections := 2\n"
        "!direction of rotation := CCW\n"
        f"start angle := {start_angle}\n"
        "!END OF INTERFILE :=\n"
    )


def _wire_raw_outputs(connector, monkeypatch, raw_outputs) -> None:
    monkeypatch.setattr(
        connector.python_connector,
        "run",
        lambda runtime_operator=None, timeout=None: raw_outputs,
    )


@pytest.mark.unit
@pytest.mark.parametrize("simind_header", ["case01.h00", "case01.h01"])
def test_pytomography_penetrate_header_keeps_simind_start_angle(
    tmp_path: Path, monkeypatch, simind_header: str
):
    """v8 writes one shared .h00, v9 a .hNN per component; the .hs is STIR's."""
    from pytomography.io.SPECT import simind as pytomo_simind

    connector = _make_wired_connector(tmp_path, monkeypatch)
    projection = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    projection.tofile(tmp_path / "case01.b01")
    _write_simind_header(tmp_path / simind_header, "case01.a00", 0.0)
    _write_simind_header(tmp_path / "case01_component_01.hs", "case01.b01", 180.0)
    _wire_raw_outputs(
        connector,
        monkeypatch,
        {
            "all_interactions": ProjectionResult(
                projection=projection,
                header_path=tmp_path / "case01_component_01.hs",
                data_path=tmp_path / "case01.b01",
                metadata={},
            )
        },
    )

    outputs = connector.run()

    header = connector.get_output_header_path("all_interactions")
    _, proj_meta = pytomo_simind.get_metadata(str(header))
    assert proj_meta.angles[0] == 0
    expected = torch.from_numpy(projection[:, ::-1, :].transpose(0, 2, 1).copy())
    assert torch.equal(outputs["all_interactions"].cpu(), expected)
    assert torch.equal(pytomo_simind.get_projections(str(header)).cpu(), expected)


@pytest.mark.unit
def test_pytomography_derived_primary_header_keeps_simind_start_angle(
    tmp_path: Path, monkeypatch
):
    """A derived pri_wN has only a .hs copied from tot_wN's converted header."""
    from pytomography.io.SPECT import simind as pytomo_simind

    connector = _make_wired_connector(tmp_path, monkeypatch)
    total = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    primary = total / 2
    total.tofile(tmp_path / "case01_tot_w1.a00")
    primary.tofile(tmp_path / "case01_pri_w1.a00")
    _write_simind_header(tmp_path / "case01_tot_w1.h00", "case01_tot_w1.a00", 0.0)
    _write_simind_header(tmp_path / "case01_pri_w1.hs", "case01_pri_w1.a00", 180.0)
    _wire_raw_outputs(
        connector,
        monkeypatch,
        {
            "pri_w1": ProjectionResult(
                projection=primary,
                header_path=tmp_path / "case01_pri_w1.hs",
                data_path=tmp_path / "case01_pri_w1.a00",
                metadata={"derived": "tot - sca"},
            )
        },
    )

    outputs = connector.run()

    header = connector.get_output_header_path("pri_w1")
    _, proj_meta = pytomo_simind.get_metadata(str(header))
    assert proj_meta.angles[0] == 0
    expected = torch.from_numpy(primary[:, ::-1, :].transpose(0, 2, 1).copy())
    assert torch.equal(outputs["pri_w1"].cpu(), expected)


@pytest.mark.unit
def test_pytomography_h00_branch_uses_pyto_reader(tmp_path: Path, monkeypatch):
    """When a SIMIND .h00 exists the PyTomography reader supplies projections."""
    import simind_python_connector.connectors.pytomography_adaptor as pyta

    connector = _make_wired_connector(tmp_path, monkeypatch)

    h00_path = tmp_path / "case01_tot_w1.h00"
    h00_path.write_text("!INTERFILE :=\n!END OF INTERFILE :=\n")

    reader_tensor = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    monkeypatch.setattr(
        pyta,
        "pytomo_simind",
        types.SimpleNamespace(
            get_projections=lambda path: (
                reader_tensor.clone()
                if str(path).endswith(".h00")
                else (_ for _ in ()).throw(AssertionError(path))
            )
        ),
    )

    outputs = connector.run()

    assert torch.equal(outputs["tot_w1"], reader_tensor)
    assert connector.get_output_header_path("tot_w1") == h00_path.resolve()


@pytest.mark.unit
def test_pytomography_fallback_branch_preserves_projection_values(
    tmp_path: Path, monkeypatch
):
    """Without an .h00 the converter-generated data and .hs header are used."""
    import simind_python_connector.connectors.pytomography_adaptor as pyta

    connector = _make_wired_connector(tmp_path, monkeypatch)
    monkeypatch.setattr(pyta, "pytomo_simind", None)

    outputs = connector.run()
    raw = connector._raw["tot_w1"].projection

    # PyTomography's (view, bin, axial) order with the axial direction reversed.
    expected = torch.from_numpy(raw[:, ::-1, :].transpose(0, 2, 1).copy())
    assert torch.equal(outputs["tot_w1"], expected)
    assert (
        connector.get_output_header_path("tot_w1")
        == (tmp_path / "case01_tot_w1.hs").resolve()
    )


@pytest.mark.unit
def test_pytomography_fallback_branch_matches_real_reader(tmp_path: Path, monkeypatch):
    """Without PyTomography the fallback matches its reader."""
    import simind_python_connector.connectors.pytomography_adaptor as pyta

    real_reader = pytest.importorskip("pytomography.io.SPECT.simind")
    connector = _make_wired_connector(tmp_path, monkeypatch)
    raw = connector._raw["tot_w1"].projection
    raw.tofile(tmp_path / "reference.a00")
    header = tmp_path / "reference.h00"
    header.write_text(
        "!INTERFILE :=\n"
        "!name of data file := reference.a00\n"
        "!number format := short float\n"
        "!number of bytes per pixel := 4\n"
        "!total number of images := 2\n"
        "!matrix size [1] := 4\n"
        "!matrix size [2] := 3\n"
        "!END OF INTERFILE :=\n"
    )
    expected = real_reader.get_projections(str(header)).cpu()
    monkeypatch.setattr(pyta, "pytomo_simind", None)

    outputs = connector.run()

    assert tuple(outputs["tot_w1"].shape) == (2, 4, 3)
    torch.testing.assert_close(outputs["tot_w1"], expected, rtol=0, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("order", [1.5, -0.5, -1])
def test_pytomography_rejects_invalid_scatter_orders(tmp_path: Path, order) -> None:
    connector = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"), output_dir=tmp_path
    )
    with pytest.raises(ValueError, match="Scatter orders"):
        connector.set_energy_windows([126], [154], [order])


def _make_wired_connector(tmp_path: Path, monkeypatch):
    from simind_python_connector.connectors.python_connector import ProjectionResult

    connector = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        voxel_size_mm=4.0,
    )

    source = torch.zeros((4, 3, 2), dtype=torch.float32)
    source[0, 0, 0] = 1.0
    connector.set_source(source)
    connector.set_mu_map(torch.zeros_like(source))
    connector.set_energy_windows([126], [154], [0])

    projection = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    raw_outputs = {
        "tot_w1": ProjectionResult(
            projection=projection,
            header_path=tmp_path / "case01_tot_w1.hs",
            data_path=tmp_path / "case01_tot_w1.a00",
            metadata={"k": "v"},
        )
    }

    connector._raw = raw_outputs
    monkeypatch.setattr(
        connector.python_connector,
        "run",
        lambda runtime_operator=None, timeout=None: raw_outputs,
    )
    return connector


@pytest.mark.unit
def test_pytomography_adaptor_set_activity_delegates(tmp_path: Path):
    adaptor = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    adaptor.set_activity(100.0, 15.0)
    assert adaptor.get_config().get_value(25) == pytest.approx(100.0 * 15.0)


def _seed_adaptor_inputs(tmp_path: Path, prefix: str = "case01") -> dict[Path, bytes]:
    paths = [
        tmp_path / f"{prefix}_src.smi",
        tmp_path / f"{prefix}_dns.dmi",
        tmp_path / f"{prefix}.win",
    ]
    for path in paths:
        path.write_bytes(f"existing:{path.name}".encode())
    return {path: path.read_bytes() for path in paths}


def _invalid_runtime_operator(
    kind: str, tmp_path: Path
) -> tuple[RuntimeOperator, type[Exception]]:
    if kind == "switch":
        return RuntimeOperator(switches={"NN": 0}), ValueError
    if kind == "orbit":
        orbit_dir = tmp_path / "orbit-dir"
        orbit_dir.mkdir(exist_ok=True)
        return RuntimeOperator(orbit_file=orbit_dir), ValueError
    raise AssertionError(f"unknown runtime-operator kind {kind!r}")


def _make_preflight_adaptor(tmp_path: Path) -> PyTomographySimindAdaptor:
    adaptor = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        voxel_size_mm=4.0,
    )
    adaptor.set_source(torch.ones((2, 3, 4), dtype=torch.float32))
    adaptor.set_mu_map(torch.zeros((2, 3, 4), dtype=torch.float32))
    adaptor.set_energy_windows([126.0], [154.0], [0])
    return adaptor


@pytest.mark.parametrize("kind", ["switch", "orbit"])
def test_pytomography_adaptor_rejects_invalid_runtime_operator_before_writes(
    kind: str, tmp_path: Path, monkeypatch
) -> None:
    adaptor = _make_preflight_adaptor(tmp_path)
    seeded = _seed_adaptor_inputs(tmp_path)
    launched: list[object] = []
    monkeypatch.setattr(
        adaptor.python_connector.executor,
        "run_simulation",
        lambda *args, **kwargs: launched.append((args, kwargs)),
    )
    operator, expected = _invalid_runtime_operator(kind, tmp_path)

    with pytest.raises(expected):
        adaptor.run(runtime_operator=operator)

    assert launched == []
    for path, original in seeded.items():
        assert path.read_bytes() == original


def _seed_full_preflight_state(
    tmp_path: Path, prefix: str = "case01"
) -> dict[Path, bytes]:
    """Seed every file the connector could overwrite or clear before launch."""
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


@pytest.mark.parametrize("value", ["bad name", "bad\tname", "bad\nname", "bad\x00name"])
def test_pytomography_adaptor_rejects_illegal_switch_before_any_write(
    value: str, tmp_path: Path, monkeypatch
) -> None:
    adaptor = _make_preflight_adaptor(tmp_path)
    seeded = _seed_full_preflight_state(tmp_path)
    launched: list[object] = []

    def run_then_fail(*args, **kwargs):
        launched.append(True)
        raise SimulationError("executor must not be reached")

    monkeypatch.setattr(
        adaptor.python_connector.executor, "run_simulation", run_then_fail
    )

    with pytest.raises(SimulationError):
        adaptor.run(runtime_operator=RuntimeOperator(switches={"FI": value}))

    assert launched == []
    for path, original in seeded.items():
        assert path.read_bytes() == original, path


def test_pytomography_adaptor_rejects_invalid_persistent_switch_before_any_write(
    tmp_path: Path, monkeypatch
) -> None:
    adaptor = _make_preflight_adaptor(tmp_path)
    adaptor.add_runtime_switch("FI", "bad name")
    seeded = _seed_full_preflight_state(tmp_path)
    launched: list[object] = []

    def run_then_fail(*args, **kwargs):
        launched.append(True)
        raise SimulationError("executor must not be reached")

    monkeypatch.setattr(
        adaptor.python_connector.executor, "run_simulation", run_then_fail
    )

    with pytest.raises(SimulationError):
        adaptor.run()

    assert launched == []
    for path, original in seeded.items():
        assert path.read_bytes() == original, path


def test_pytomography_adaptor_accepts_valid_runtime_operator(
    tmp_path: Path, monkeypatch
) -> None:
    adaptor = _make_preflight_adaptor(tmp_path)
    monkeypatch.setattr(
        adaptor.python_connector, "run", lambda runtime_operator=None, timeout=None: {}
    )

    outputs = adaptor.run(runtime_operator=RuntimeOperator(switches={"RR": 12345}))

    assert outputs == {}


def _alias_destination(destination: Path, external: Path, alias: str) -> Path:
    """Put an orbit at destination, directly or through a link; return its path."""
    if alias == "direct":
        destination.write_bytes(b"orbit-bytes")
        return destination
    if alias == "symlink":
        destination.symlink_to(external)
        return external
    os.link(external, destination)
    return external


@pytest.mark.parametrize("destination", ["src", "density"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_pytomography_adaptor_rejects_orbit_conflicting_with_its_inputs(
    destination: str, alias: str, tmp_path: Path, monkeypatch
) -> None:
    adaptor = _make_preflight_adaptor(tmp_path)
    suffix = "_src.smi" if destination == "src" else "_dns.dmi"
    destination_path = tmp_path / f"case01{suffix}"
    external = tmp_path / "external-orbit.cor"
    external.write_bytes(b"orbit-bytes")
    orbit = _alias_destination(destination_path, external, alias)
    launched: list[object] = []
    monkeypatch.setattr(
        adaptor.python_connector.executor,
        "run_simulation",
        lambda *args, **kwargs: launched.append(True),
    )

    with pytest.raises(ValueError, match="orbit_file"):
        adaptor.run(runtime_operator=RuntimeOperator(orbit_file=orbit))

    assert launched == []
    assert external.read_bytes() == b"orbit-bytes"
    if alias == "direct":
        assert destination_path.read_bytes() == b"orbit-bytes"
    elif alias == "symlink":
        assert destination_path.is_symlink()
    else:
        assert os.stat(destination_path).st_ino == os.stat(external).st_ino


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_pytomography_adaptor_rejects_orbit_at_the_window_destination(
    alias: str, tmp_path: Path, monkeypatch
) -> None:
    adaptor = _make_preflight_adaptor(tmp_path)
    window_path = tmp_path / "case01.win"
    external = tmp_path / "external-orbit.cor"
    external.write_bytes(b"orbit-bytes")
    orbit = _alias_destination(window_path, external, alias)
    launched: list[object] = []
    monkeypatch.setattr(
        adaptor.python_connector.executor,
        "run_simulation",
        lambda *args, **kwargs: launched.append(True),
    )

    with pytest.raises(ValueError, match="orbit_file"):
        adaptor.run(runtime_operator=RuntimeOperator(orbit_file=orbit))

    assert launched == []
    assert external.read_bytes() == b"orbit-bytes"
    if alias == "direct":
        assert window_path.read_bytes() == b"orbit-bytes"
    elif alias == "symlink":
        assert window_path.is_symlink()
    else:
        assert os.stat(window_path).st_ino == os.stat(external).st_ino


def test_pytomography_adaptor_accepts_external_orbit_with_an_input_like_basename(
    tmp_path: Path, monkeypatch
) -> None:
    adaptor = _make_preflight_adaptor(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    external = outside / "case01_src.smi"
    external.write_bytes(b"external-orbit")
    monkeypatch.setattr(
        adaptor.python_connector, "run", lambda runtime_operator=None, timeout=None: {}
    )

    outputs = adaptor.run(runtime_operator=RuntimeOperator(orbit_file=external))

    assert outputs == {}
    assert external.read_bytes() == b"external-orbit"
