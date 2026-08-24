from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import simind_python_connector.connectors.sirf_adaptor as sirf_mod
import simind_python_connector.connectors.stir_adaptor as stir_mod
from simind_python_connector.configs import get
from simind_python_connector.connectors.python_connector import RuntimeOperator
from simind_python_connector.connectors.sirf_adaptor import SirfSimindAdaptor
from simind_python_connector.connectors.stir_adaptor import StirSimindAdaptor
from simind_python_connector.core.types import ScoringRoutine


pytestmark = pytest.mark.unit


class _ImageWithVoxelSizes:
    def __init__(self, array: np.ndarray, voxel_sizes: tuple[float, ...]) -> None:
        self.array = array
        self._voxel_sizes = voxel_sizes

    def voxel_sizes(self) -> tuple[float, ...]:
        return self._voxel_sizes


class _ImageWithGridSpacing:
    def __init__(self, array: np.ndarray, spacing: tuple[float, ...]) -> None:
        self.array = array
        self._spacing = spacing

    def get_grid_spacing(self) -> tuple[float, ...]:
        return self._spacing


class _NonIterableFloat3Coordinate:
    def __init__(self, x: float, y: float, z: float) -> None:
        self._x = x
        self._y = y
        self._z = z

    def x(self) -> float:
        return self._x

    def y(self) -> float:
        return self._y

    def z(self) -> float:
        return self._z


class _ImageWithNonIterableGridSpacing:
    def __init__(
        self, array: np.ndarray, spacing: _NonIterableFloat3Coordinate
    ) -> None:
        self.array = array
        self._spacing = spacing

    def get_grid_spacing(self) -> _NonIterableFloat3Coordinate:
        return self._spacing


class _ImageWithoutSpacing:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array


class _AtOnlyGridSpacing:
    """Non-iterable STIR-style coordinate exposing values via at()."""

    def __init__(self, unused: float, z: float, y: float, x: float) -> None:
        self._values = (unused, z, y, x)

    def at(self, index: int) -> float:
        return self._values[index]


class _GetItemOnlyGridSpacing:
    """Non-iterable (z, y, x) coordinate exposing 1-based indexing."""

    def __init__(self, z: float, y: float, x: float) -> None:
        self._values = (z, y, x)

    def __getitem__(self, index: int) -> float:
        if index not in (1, 2, 3):
            raise IndexError(index)
        return self._values[index - 1]


def _patch_stir_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyProjData:
        @staticmethod
        def read_from_file(path: str) -> str:
            return f"stir:{path}"

    class _DummyStir:
        ProjData = _DummyProjData

    monkeypatch.setattr(stir_mod, "stir", _DummyStir)
    monkeypatch.setattr(stir_mod, "get_array", lambda image: image.array)


def _patch_sirf_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyAcquisitionData:
        def __init__(self, path: str) -> None:
            self.path = path

    class _DummySirf:
        AcquisitionData = _DummyAcquisitionData

    monkeypatch.setattr(sirf_mod, "sirf", _DummySirf)
    monkeypatch.setattr(sirf_mod, "get_array", lambda image: image.array)


def test_stir_adaptor_run_validates_required_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_stir_backend(monkeypatch)
    adaptor = StirSimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
    )

    with pytest.raises(ValueError, match="Both source and mu_map"):
        adaptor.run()

    source = _ImageWithVoxelSizes(
        np.zeros((2, 3, 4), dtype=np.float32), (1.0, 1.0, 4.0)
    )
    adaptor.set_source(source)
    with pytest.raises(ValueError, match="Both source and mu_map"):
        adaptor.run()


def test_stir_adaptor_run_validates_shape_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_stir_backend(monkeypatch)
    adaptor = StirSimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
    )

    source = _ImageWithVoxelSizes(
        np.zeros((2, 3, 4), dtype=np.float32), (1.0, 1.0, 4.0)
    )
    mu_map = _ImageWithVoxelSizes(
        np.zeros((2, 3, 5), dtype=np.float32), (1.0, 1.0, 4.0)
    )
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)

    with pytest.raises(ValueError, match="matching shapes"):
        adaptor.run()


def test_stir_adaptor_run_forwards_expected_connector_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_stir_backend(monkeypatch)
    adaptor = StirSimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
        scoring_routine=ScoringRoutine.PENETRATE,
    )

    source_arr = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    mu_arr = np.ones_like(source_arr) * 0.15
    source = _ImageWithVoxelSizes(source_arr, (1.0, 1.0, 4.25))
    mu_map = _ImageWithVoxelSizes(mu_arr, (1.0, 1.0, 4.25))
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)

    captured: dict[str, object] = {}

    def fake_configure_voxel_phantom(source, mu_map, voxel_size_mm, scoring_routine):
        captured["source"] = source
        captured["mu_map"] = mu_map
        captured["voxel_size_mm"] = voxel_size_mm
        captured["scoring_routine"] = scoring_routine
        return (tmp_path / "case01_src.smi", tmp_path / "case01_dns.dmi")

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

    assert outputs["tot_w1"] == f"stir:{tmp_path / 'case01_tot_w1.hs'}"
    assert np.asarray(captured["source"]).dtype == np.float32
    assert np.asarray(captured["mu_map"]).dtype == np.float32
    assert np.asarray(captured["source"]).shape == (2, 3, 4)
    assert np.asarray(captured["mu_map"]).shape == (2, 3, 4)
    assert captured["voxel_size_mm"] == pytest.approx(1.0)
    assert captured["scoring_routine"] == ScoringRoutine.PENETRATE
    assert captured["runtime_operator"] is runtime_operator


def test_stir_adaptor_extracts_voxel_size_from_supported_spacing_sources() -> None:
    """Package contract: images report spacing as (z, y, x); the z component
    is the first spatial element."""
    voxel_sizes_image = _ImageWithVoxelSizes(
        np.zeros((2, 3, 4), dtype=np.float32), (2.0, 1.0, 4.0)
    )
    assert StirSimindAdaptor._extract_voxel_size_mm(voxel_sizes_image) == pytest.approx(
        2.0
    )

    # Four-element raw sequences follow STIR's (unused, z, y, x) layout
    spacing_4d_image = _ImageWithGridSpacing(
        np.zeros((2, 3, 4), dtype=np.float32), (0.0, 2.0, 2.0, 5.0)
    )
    assert StirSimindAdaptor._extract_voxel_size_mm(spacing_4d_image) == pytest.approx(
        2.0
    )

    # Three-element sequences are already in (z, y, x) order
    spacing_3d_image = _ImageWithGridSpacing(
        np.zeros((2, 3, 4), dtype=np.float32), (6.0, 2.0, 1.0)
    )
    assert StirSimindAdaptor._extract_voxel_size_mm(spacing_3d_image) == pytest.approx(
        6.0
    )

    float3_spacing_image = _ImageWithNonIterableGridSpacing(
        np.zeros((2, 3, 4), dtype=np.float32),
        _NonIterableFloat3Coordinate(1.0, 2.0, 7.0),
    )
    assert StirSimindAdaptor._extract_voxel_size_mm(
        float3_spacing_image
    ) == pytest.approx(7.0)

    with pytest.raises(ValueError, match="voxel_sizes\\(\\) or get_grid_spacing\\(\\)"):
        StirSimindAdaptor._extract_voxel_size_mm(
            _ImageWithoutSpacing(np.zeros((1, 1, 1)))
        )


def test_extract_voxel_size_reads_z_from_non_iterable_fallbacks() -> None:
    """at()/index-only coordinates follow STIR's (unused, z, y, x) ordering,
    so index 1 carries the z spacing."""
    at_only_image = _ImageWithNonIterableGridSpacing(
        np.zeros((2, 3, 4), dtype=np.float32),
        _AtOnlyGridSpacing(9.0, 2.0, 3.0, 4.0),
    )
    assert StirSimindAdaptor._extract_voxel_size_mm(at_only_image) == pytest.approx(2.0)

    getitem_only_image = _ImageWithNonIterableGridSpacing(
        np.zeros((2, 3, 4), dtype=np.float32),
        _GetItemOnlyGridSpacing(2.0, 3.0, 4.0),
    )
    assert StirSimindAdaptor._extract_voxel_size_mm(
        getitem_only_image
    ) == pytest.approx(2.0)


def test_stir_adaptor_missing_component_errors_list_available_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_stir_backend(monkeypatch)
    adaptor = StirSimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
    )
    adaptor._outputs = {"tot_w1": "projection"}  # type: ignore[assignment]

    with pytest.raises(KeyError, match="Available: tot_w1"):
        adaptor.get_scatter_output()
    with pytest.raises(KeyError, match="Available: tot_w1"):
        adaptor.get_penetrate_output("all_interactions")


def test_sirf_adaptor_run_validates_required_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sirf_backend(monkeypatch)
    adaptor = SirfSimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
    )

    with pytest.raises(ValueError, match="Both source and mu_map"):
        adaptor.run()

    source = _ImageWithVoxelSizes(
        np.zeros((2, 3, 4), dtype=np.float32), (1.0, 1.0, 4.0)
    )
    adaptor.set_source(source)
    with pytest.raises(ValueError, match="Both source and mu_map"):
        adaptor.run()


def test_sirf_adaptor_run_validates_shape_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sirf_backend(monkeypatch)
    adaptor = SirfSimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
    )

    source = _ImageWithVoxelSizes(
        np.zeros((2, 3, 4), dtype=np.float32), (1.0, 1.0, 4.0)
    )
    mu_map = _ImageWithVoxelSizes(
        np.zeros((2, 3, 5), dtype=np.float32), (1.0, 1.0, 4.0)
    )
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)

    with pytest.raises(ValueError, match="matching shapes"):
        adaptor.run()


def test_sirf_adaptor_run_forwards_expected_connector_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sirf_backend(monkeypatch)
    adaptor = SirfSimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
        scoring_routine=ScoringRoutine.PENETRATE,
    )

    source_arr = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    mu_arr = np.ones_like(source_arr) * 0.2
    source = _ImageWithVoxelSizes(source_arr, (1.0, 1.0, 3.75))
    mu_map = _ImageWithVoxelSizes(mu_arr, (1.0, 1.0, 3.75))
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)

    captured: dict[str, object] = {}

    def fake_configure_voxel_phantom(source, mu_map, voxel_size_mm, scoring_routine):
        captured["source"] = source
        captured["mu_map"] = mu_map
        captured["voxel_size_mm"] = voxel_size_mm
        captured["scoring_routine"] = scoring_routine
        return (tmp_path / "case01_src.smi", tmp_path / "case01_dns.dmi")

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

    assert outputs["tot_w1"].path == str(tmp_path / "case01_tot_w1.hs")
    assert np.asarray(captured["source"]).dtype == np.float32
    assert np.asarray(captured["mu_map"]).dtype == np.float32
    assert np.asarray(captured["source"]).shape == (2, 3, 4)
    assert np.asarray(captured["mu_map"]).shape == (2, 3, 4)
    assert captured["voxel_size_mm"] == pytest.approx(1.0)
    assert captured["scoring_routine"] == ScoringRoutine.PENETRATE
    assert captured["runtime_operator"] is runtime_operator


def test_sirf_adaptor_extracts_voxel_size_from_supported_spacing_sources() -> None:
    """Package contract: images report spacing as (z, y, x); the z component
    is the first spatial element."""
    voxel_sizes_image = _ImageWithVoxelSizes(
        np.zeros((2, 3, 4), dtype=np.float32), (3.0, 1.0, 4.0)
    )
    assert SirfSimindAdaptor._extract_voxel_size_mm(voxel_sizes_image) == pytest.approx(
        3.0
    )

    # Four-element raw sequences follow STIR's (unused, z, y, x) layout
    spacing_4d_image = _ImageWithGridSpacing(
        np.zeros((2, 3, 4), dtype=np.float32), (0.0, 3.0, 2.0, 5.0)
    )
    assert SirfSimindAdaptor._extract_voxel_size_mm(spacing_4d_image) == pytest.approx(
        3.0
    )

    # Three-element sequences are already in (z, y, x) order
    spacing_3d_image = _ImageWithGridSpacing(
        np.zeros((2, 3, 4), dtype=np.float32), (6.0, 2.0, 1.0)
    )
    assert SirfSimindAdaptor._extract_voxel_size_mm(spacing_3d_image) == pytest.approx(
        6.0
    )

    float3_spacing_image = _ImageWithNonIterableGridSpacing(
        np.zeros((2, 3, 4), dtype=np.float32),
        _NonIterableFloat3Coordinate(1.0, 2.0, 7.0),
    )
    assert SirfSimindAdaptor._extract_voxel_size_mm(
        float3_spacing_image
    ) == pytest.approx(7.0)

    with pytest.raises(ValueError, match="voxel_sizes\\(\\) or get_grid_spacing\\(\\)"):
        SirfSimindAdaptor._extract_voxel_size_mm(
            _ImageWithoutSpacing(np.zeros((1, 1, 1)))
        )


def test_sirf_adaptor_missing_component_errors_list_available_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sirf_backend(monkeypatch)
    adaptor = SirfSimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
    )
    adaptor._outputs = {"tot_w1": "projection"}  # type: ignore[assignment]

    with pytest.raises(KeyError, match="Available: tot_w1"):
        adaptor.get_scatter_output()
    with pytest.raises(KeyError, match="Available: tot_w1"):
        adaptor.get_penetrate_output("all_interactions")


def test_stir_adaptor_clears_cached_outputs_on_failed_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_stir_backend(monkeypatch)
    adaptor = _make_adaptor(StirSimindAdaptor, tmp_path)
    _assert_failed_rerun_clears_cache(adaptor)


def test_sirf_adaptor_clears_cached_outputs_on_failed_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sirf_backend(monkeypatch)
    adaptor = _make_adaptor(SirfSimindAdaptor, tmp_path)
    _assert_failed_rerun_clears_cache(adaptor)


def _make_adaptor(cls, tmp_path: Path):
    adaptor = cls(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
    )
    source = _ImageWithVoxelSizes(
        np.zeros((2, 3, 4), dtype=np.float32), (2.0, 1.0, 4.0)
    )
    adaptor.set_source(source)
    adaptor.set_mu_map(
        _ImageWithVoxelSizes(np.zeros_like(source.array), (2.0, 1.0, 4.0))
    )
    return adaptor


def _assert_failed_rerun_clears_cache(adaptor) -> None:
    adaptor._outputs = {"stale": object()}  # type: ignore[assignment]

    def failing_run(runtime_operator=None):
        raise RuntimeError("simulated failure")

    adaptor.python_connector.run = failing_run  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated failure"):
        adaptor.run()

    assert adaptor._outputs is None
    with pytest.raises(RuntimeError, match="Run the adaptor first"):
        adaptor.get_outputs()
