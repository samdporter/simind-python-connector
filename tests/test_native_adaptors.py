from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import simind_python_connector.connectors.sirf_adaptor as sirf_mod
import simind_python_connector.connectors.stir_adaptor as stir_mod
from simind_python_connector.configs import get
from simind_python_connector.connectors.python_connector import (
    ProjectionResult,
    RuntimeOperator,
)
from simind_python_connector.connectors.sirf_adaptor import SirfSimindAdaptor
from simind_python_connector.connectors.stir_adaptor import StirSimindAdaptor
from simind_python_connector.core.types import ScoringRoutine, SimulationError
from simind_python_connector.utils.interfile import InterfileHeader


pytestmark = pytest.mark.unit


class _SirfLikeImage:
    """Stand-in for sirf.STIR.ImageData; sizes and offset are in (z, y, x)."""

    def __init__(self, array, voxel_sizes=(4.0, 4.0, 4.0), origin=(0.0, 0.0, 0.0)):
        self.array = np.asarray(array)
        self._voxel_sizes = voxel_sizes
        self._origin = origin

    def as_array(self):
        return self.array

    def voxel_sizes(self):
        return self._voxel_sizes

    def get_geometrical_info(self):
        return SimpleNamespace(get_offset=lambda: self._origin)


class _OneBasedCoordinate:
    """STIR coordinates are 1-based: [1] is z, [2] is y, [3] is x."""

    def __init__(self, z, y, x):
        self._values = {1: z, 2: y, 3: x}

    def __getitem__(self, index):
        return self._values[index]


class _StirLikeImage:
    """Stand-in for stir.FloatVoxelsOnCartesianGrid."""

    def __init__(self, array, spacing=(4.0, 4.0, 4.0), origin=(0.0, 0.0, 0.0)):
        self.array = np.asarray(array)
        self._spacing = _OneBasedCoordinate(*spacing)
        self._origin = _OneBasedCoordinate(*origin)

    def as_array(self):
        return self.array

    def get_grid_spacing(self):
        return self._spacing

    def get_origin(self):
        return self._origin


def _patch_stir_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyProjData:
        @staticmethod
        def read_from_file(path: str) -> str:
            return f"stir:{path}"

    class _DummyStir:
        ProjData = _DummyProjData

    monkeypatch.setattr(stir_mod, "stir", _DummyStir)


def _patch_sirf_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyAcquisitionData:
        def __init__(self, path: str) -> None:
            self.path = path

    class _DummySirf:
        AcquisitionData = _DummyAcquisitionData

    monkeypatch.setattr(sirf_mod, "sirf", _DummySirf)


_CASES = [
    (StirSimindAdaptor, _patch_stir_backend, _StirLikeImage),
    (SirfSimindAdaptor, _patch_sirf_backend, _SirfLikeImage),
]


def _make_adaptor(cls, tmp_path: Path, **kwargs):
    return cls(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
        **kwargs,
    )


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_requires_backend(cls, patch, image_cls, tmp_path, monkeypatch):
    module = stir_mod if cls is StirSimindAdaptor else sirf_mod
    monkeypatch.setattr(module, "stir" if cls is StirSimindAdaptor else "sirf", None)
    with pytest.raises(ImportError, match="requires the"):
        _make_adaptor(cls, tmp_path)


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_run_validates_required_inputs(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)

    with pytest.raises(ValueError, match="Both source and mu_map"):
        adaptor.run()

    adaptor.set_source(image_cls(np.zeros((2, 3, 4), dtype=np.float32)))
    with pytest.raises(ValueError, match="Both source and mu_map"):
        adaptor.run()


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_run_validates_shape_match(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor.set_source(image_cls(np.zeros((2, 3, 4), dtype=np.float32)))
    adaptor.set_mu_map(image_cls(np.zeros((2, 3, 5), dtype=np.float32)))

    with pytest.raises(ValueError, match="matching shapes"):
        adaptor.run()


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_run_forwards_expected_connector_inputs(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path, scoring_routine=ScoringRoutine.PENETRATE)
    source_arr = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    adaptor.set_source(image_cls(source_arr, (4.0, 4.0, 4.0)))
    adaptor.set_mu_map(image_cls(np.ones_like(source_arr) * 0.15, (4.0, 4.0, 4.0)))

    captured: dict[str, object] = {}

    def fake_configure_voxel_phantom(
        source, mu_map, voxel_size_mm, scoring_routine, mu_map_type, mu_map_energy_kev
    ):
        captured.update(
            source=source,
            mu_map=mu_map,
            voxel_size_mm=voxel_size_mm,
            scoring_routine=scoring_routine,
            mu_map_type=mu_map_type,
            mu_map_energy_kev=mu_map_energy_kev,
        )

    def fake_run(runtime_operator=None):
        captured["runtime_operator"] = runtime_operator
        return {"tot_w1": SimpleNamespace(header_path=tmp_path / "case01_tot_w1.hs")}

    monkeypatch.setattr(
        adaptor.python_connector,
        "configure_voxel_phantom",
        fake_configure_voxel_phantom,
    )
    monkeypatch.setattr(adaptor.python_connector, "run", fake_run)

    runtime_operator = RuntimeOperator(switches={"RR": 12345})
    outputs = adaptor.run(runtime_operator=runtime_operator)

    header = str(tmp_path / "case01_tot_w1.hs")
    if cls is StirSimindAdaptor:
        assert outputs["tot_w1"] == f"stir:{header}"
    else:
        assert outputs["tot_w1"].path == header
    assert np.asarray(captured["source"]).dtype == np.float32
    assert np.asarray(captured["mu_map"]).dtype == np.float32
    assert np.asarray(captured["source"]).shape == (2, 3, 4)
    assert captured["voxel_size_mm"] == pytest.approx((4.0, 4.0, 4.0))
    assert captured["scoring_routine"] == ScoringRoutine.PENETRATE
    assert captured["mu_map_type"] == "attenuation"
    assert captured["mu_map_energy_kev"] is None
    assert captured["runtime_operator"] is runtime_operator


def test_stir_voxel_sizes_use_one_based_grid_spacing(tmp_path, monkeypatch):
    _patch_stir_backend(monkeypatch)
    adaptor = _make_adaptor(StirSimindAdaptor, tmp_path)
    image = _StirLikeImage(np.zeros((1, 1, 1)), spacing=(2.0, 3.0, 4.0))
    assert adaptor._voxel_sizes_mm(image) == (2.0, 3.0, 4.0)


def test_sirf_voxel_sizes_use_voxel_sizes(tmp_path, monkeypatch):
    _patch_sirf_backend(monkeypatch)
    adaptor = _make_adaptor(SirfSimindAdaptor, tmp_path)
    image = _SirfLikeImage(np.zeros((1, 1, 1)), voxel_sizes=(2.0, 3.0, 4.0))
    assert adaptor._voxel_sizes_mm(image) == (2.0, 3.0, 4.0)


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_missing_component_errors_list_available_keys(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor._outputs = {"tot_w1": "projection"}  # type: ignore[assignment]

    with pytest.raises(KeyError, match="Available: tot_w1"):
        adaptor.get_scatter_output()
    with pytest.raises(KeyError, match="Available: tot_w1"):
        adaptor.get_penetrate_output("all_interactions")


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_clears_cached_outputs_on_failed_rerun(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    source = image_cls(np.zeros((2, 3, 4), dtype=np.float32))
    adaptor.set_source(source)
    adaptor.set_mu_map(image_cls(np.zeros((2, 3, 4), dtype=np.float32)))
    adaptor._outputs = {"stale": object()}  # type: ignore[assignment]

    def failing_run(runtime_operator=None):
        raise RuntimeError("simulated failure")

    adaptor.python_connector.run = failing_run  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated failure"):
        adaptor.run()

    assert adaptor._outputs is None
    with pytest.raises(RuntimeError, match="Run the adaptor first"):
        adaptor.get_outputs()


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_rejects_invalid_photon_multiplier(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    with pytest.raises(ValueError, match="integer >= 1"):
        _make_adaptor(cls, tmp_path, photon_multiplier=0)


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_passes_anisotropic_voxel_sizes(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(
        cls, tmp_path, mu_map_type="density", mu_map_energy_kev=None
    )
    adaptor.set_source(image_cls(np.ones((2, 3, 4)), (8.0, 4.0, 4.0)))
    adaptor.set_mu_map(image_cls(np.ones((2, 3, 4)), (8.0, 4.0, 4.0)))
    captured = {}
    monkeypatch.setattr(
        adaptor.python_connector,
        "configure_voxel_phantom",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        adaptor.python_connector, "run", lambda runtime_operator=None: {}
    )
    adaptor.run()
    assert captured["voxel_size_mm"] == (8.0, 4.0, 4.0)
    assert captured["mu_map_type"] == "density"


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_rejects_mu_map_with_different_origin(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor.set_source(image_cls(np.zeros((2, 3, 4))))
    adaptor.set_mu_map(image_cls(np.zeros((2, 3, 4)), origin=(0.0, 0.0, 2.0)))
    with pytest.raises(ValueError, match="origins differ"):
        adaptor.run()


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
@pytest.mark.parametrize("gap, matches", [(0.0009, True), (0.0011, False)])
def test_adaptor_origin_tolerance_is_one_micron(
    cls, patch, image_cls, tmp_path, monkeypatch, gap, matches
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor.set_source(image_cls(np.zeros((2, 3, 4))))
    adaptor.set_mu_map(image_cls(np.zeros((2, 3, 4)), origin=(0.0, 0.0, gap)))
    monkeypatch.setattr(
        adaptor.python_connector, "configure_voxel_phantom", lambda **kwargs: None
    )
    monkeypatch.setattr(
        adaptor.python_connector, "run", lambda runtime_operator=None: {}
    )
    if matches:
        adaptor.run()
    else:
        with pytest.raises(ValueError, match="origins differ"):
            adaptor.run()


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_rejects_mu_map_with_different_spacing(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor.set_source(image_cls(np.zeros((2, 3, 4)), (4.0, 4.0, 4.0)))
    adaptor.set_mu_map(image_cls(np.zeros((2, 3, 4)), (3.0, 3.0, 3.0)))
    with pytest.raises(ValueError, match="voxel sizes differ"):
        adaptor.run()


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_set_activity_delegates(cls, patch, image_cls, tmp_path, monkeypatch):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor.set_activity(100.0, 15.0)
    assert adaptor.get_config().get_value(25) == pytest.approx(1500.0)


_GEOMETRY_HEADER = """!INTERFILE :=
  !name of data file := {data}
  !number format := float
  !number of bytes per pixel := 4
  imagedata byte order := LITTLEENDIAN
  !number of projections := 3
  !extent of rotation := 360
  !direction of rotation := {direction}
  start angle := 180
  Radius := 250
  !matrix size [1] := 4
  scaling factor (mm/pixel) [1] := 4.42
  !matrix size [2] := 2
  scaling factor (mm/pixel) [2] := 4.42
  !END OF INTERFILE :=
"""


def _write_header(path, direction="CW"):
    data = path.with_suffix(".s")
    np.zeros((3, 2, 4), dtype=np.float32).tofile(data)
    path.write_text(_GEOMETRY_HEADER.format(data=data.name, direction=direction))
    return path


def _fake_simind_run(adaptor, tmp_path, monkeypatch, direction="CW"):
    simind_header = _write_header(tmp_path / "case01_tot_w1.hs", direction)
    projection = np.arange(24, dtype=np.float32).reshape(3, 2, 4)

    def fake_run(runtime_operator=None):
        return {
            "tot_w1": ProjectionResult(
                projection=projection,
                header_path=simind_header,
                data_path=simind_header.with_suffix(".s"),
                metadata={},
            )
        }

    monkeypatch.setattr(adaptor.python_connector, "run", fake_run)
    monkeypatch.setattr(
        adaptor.python_connector, "configure_voxel_phantom", lambda **kwargs: None
    )
    return projection


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_with_template_returns_outputs_in_template_geometry(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    template = _write_header(tmp_path / "template.hs")
    adaptor.set_template(template)
    adaptor.set_source(image_cls(np.ones((2, 3, 4), dtype=np.float32)))
    adaptor.set_mu_map(image_cls(np.zeros((2, 3, 4), dtype=np.float32)))
    configured = []
    monkeypatch.setattr(
        adaptor.python_connector, "configure_acquisition", configured.append
    )
    projection = _fake_simind_run(adaptor, tmp_path, monkeypatch)

    outputs = adaptor.run()

    assert configured[0].num_projections == 3
    tmpl = tmp_path / "case01_tot_w1_tmpl.hs"
    if cls is StirSimindAdaptor:
        assert outputs["tot_w1"] == f"stir:{tmpl}"
    else:
        assert outputs["tot_w1"].path == str(tmpl)
    expected_storage = projection.astype("<f4")
    written = np.fromfile(tmpl.with_suffix(".s"), dtype="<f4")
    assert np.array_equal(written, expected_storage.ravel())


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_with_template_rejects_mismatched_geometry(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor.set_template(_write_header(tmp_path / "template.hs", direction="CW"))
    adaptor.set_source(image_cls(np.ones((2, 3, 4), dtype=np.float32)))
    adaptor.set_mu_map(image_cls(np.zeros((2, 3, 4), dtype=np.float32)))
    monkeypatch.setattr(
        adaptor.python_connector, "configure_acquisition", lambda g: None
    )
    _fake_simind_run(adaptor, tmp_path, monkeypatch, direction="CCW")

    with pytest.raises(SimulationError, match="direction"):
        adaptor.run()


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_with_template_survives_a_second_run(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor.set_template(_write_header(tmp_path / "template.hs"))
    adaptor.set_source(image_cls(np.ones((2, 3, 4), dtype=np.float32)))
    adaptor.set_mu_map(image_cls(np.zeros((2, 3, 4), dtype=np.float32)))
    tmpl = tmp_path / "case01_tot_w1_tmpl.hs"
    values = iter([1.0, 7.0])

    # Only the executor is mocked; the connector runs for real, so
    # _clear_previous_outputs deletes the first run's files. Every call must
    # therefore write the raw header and its data again.
    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None, **kwargs
    ):
        assert not list(Path(cwd).glob(f"{output_prefix}_*_tmpl.hs"))
        raw_header = _write_header(Path(cwd) / f"{output_prefix}_tot_w1.hs")
        level = next(values)
        np.full((3, 2, 4), level, dtype=np.float32).tofile(raw_header.with_suffix(".s"))

    monkeypatch.setattr(
        adaptor.python_connector.executor, "run_simulation", fake_run_simulation
    )

    adaptor.run()

    assert np.all(np.fromfile(tmpl.with_suffix(".s"), dtype="<f4") == 1.0)

    adaptor.run()

    raw = adaptor.python_connector.get_outputs()
    assert all(not key.endswith("_tmpl") for key in raw)
    assert np.all(np.fromfile(tmpl.with_suffix(".s"), dtype="<f4") == 7.0)


@pytest.mark.parametrize("cls, patch, image_cls", _CASES)
def test_adaptor_with_template_reports_every_mismatch(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor.set_template(_write_header(tmp_path / "template.hs", direction="CW"))
    adaptor.set_source(image_cls(np.ones((2, 3, 4), dtype=np.float32)))
    adaptor.set_mu_map(image_cls(np.zeros((2, 3, 4), dtype=np.float32)))
    monkeypatch.setattr(
        adaptor.python_connector, "configure_acquisition", lambda g: None
    )
    monkeypatch.setattr(
        adaptor.python_connector, "configure_voxel_phantom", lambda **kwargs: None
    )
    projection = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    raw_header = _write_header(tmp_path / "case01_tot_w1.hs", direction="CCW")
    header = InterfileHeader.from_file(raw_header)
    header.set("!number of projections", 5)
    header.set("!matrix size [1]", 8)
    header.set("Radius", 260)
    header.write(raw_header)
    monkeypatch.setattr(
        adaptor.python_connector,
        "run",
        lambda runtime_operator=None: {
            "tot_w1": ProjectionResult(
                projection=projection,
                header_path=raw_header,
                data_path=raw_header.with_suffix(".s"),
                metadata={},
            )
        },
    )

    with pytest.raises(SimulationError) as excinfo:
        adaptor.run()

    for word in ("direction", "num_projections", "num_bins", "radii"):
        assert word in str(excinfo.value)
    with pytest.raises(RuntimeError, match="Run the adaptor first"):
        adaptor.get_outputs()
