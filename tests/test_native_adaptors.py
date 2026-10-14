from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import simind_python_connector.connectors.pytomography_adaptor as pytomography_mod
import simind_python_connector.connectors.sirf_adaptor as sirf_mod
import simind_python_connector.connectors.stir_adaptor as stir_mod
from simind_python_connector.configs import get
from simind_python_connector.connectors.python_connector import RuntimeOperator
from simind_python_connector.connectors.pytomography_adaptor import (
    PyTomographySimindAdaptor,
)
from simind_python_connector.connectors.sirf_adaptor import SirfSimindAdaptor
from simind_python_connector.connectors.stir_adaptor import StirSimindAdaptor
from simind_python_connector.core.types import ScoringRoutine, SimulationError


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


class _GeometryInfo:
    def __init__(self, offset, spacing, direction) -> None:
        self._offset = offset
        self._spacing = spacing
        self._direction = direction

    def get_offset(self):
        return self._offset

    def get_spacing(self):
        return self._spacing

    def get_direction_matrix(self):
        return self._direction


class _SIRFImageWithGeometry:
    def __init__(self, array: np.ndarray, geometry: _GeometryInfo) -> None:
        self.array = array
        self.geometry = geometry

    def voxel_sizes(self):
        return (2.0, 2.0, 2.0)

    def get_geometrical_info(self):
        return self.geometry


class _STIRImageWithGeometry:
    def __init__(
        self, array: np.ndarray, origin, spacing, min_indices=(0, 0, 0)
    ) -> None:
        self.array = array
        self._origin = origin
        self._spacing = spacing
        self._min_indices = min_indices

    def get_grid_spacing(self):
        return self._spacing

    def get_origin(self):
        return self._origin

    def get_min_z(self):
        return self._min_indices[0]

    def get_min_y(self):
        return self._min_indices[1]

    def get_min_x(self):
        return self._min_indices[2]


class _NativeImageWrapper:
    def __init__(self, native_object, array: np.ndarray) -> None:
        self.native_object = native_object
        self.array = array

    def voxel_sizes(self):
        return (2.0, 2.0, 2.0)


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
        np.zeros((2, 3, 4), dtype=np.float32), (1.0, 1.0, 1.0)
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
        np.zeros((2, 3, 4), dtype=np.float32), (1.0, 1.0, 1.0)
    )
    mu_map = _ImageWithVoxelSizes(
        np.zeros((2, 3, 5), dtype=np.float32), (1.0, 1.0, 1.0)
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
    source = _ImageWithVoxelSizes(source_arr, (1.0, 1.0, 1.0))
    mu_map = _ImageWithVoxelSizes(mu_arr, (1.0, 1.0, 1.0))
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

    runtime_operator = RuntimeOperator(switches={"RR": 12345}, timeout=42.0)
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
    """Images report spacing as (z, y, x); raw STIR sequences prepend an unused
    element. The connector takes one voxel size, so voxels must be cubic."""
    array = np.zeros((2, 3, 4), dtype=np.float32)
    for image in (
        _ImageWithVoxelSizes(array, (2.0, 2.0, 2.0)),
        _ImageWithGridSpacing(array, (0.0, 2.0, 2.0, 2.0)),
        _ImageWithGridSpacing(array, (2.0, 2.0, 2.0)),
        _ImageWithNonIterableGridSpacing(
            array, _NonIterableFloat3Coordinate(2.0, 2.0, 2.0)
        ),
    ):
        assert StirSimindAdaptor._extract_voxel_size_mm(image) == pytest.approx(2.0)

    for image in (
        _ImageWithVoxelSizes(array, (3.0, 2.0, 2.0)),
        _ImageWithGridSpacing(array, (0.0, 2.0, 2.0, 5.0)),
        _ImageWithNonIterableGridSpacing(
            array, _NonIterableFloat3Coordinate(1.0, 2.0, 2.0)
        ),
    ):
        with pytest.raises(ValueError, match="cubic voxels"):
            StirSimindAdaptor._extract_voxel_size_mm(image)

    with pytest.raises(ValueError, match=r"voxel_sizes\(\) or get_grid_spacing\(\)"):
        StirSimindAdaptor._extract_voxel_size_mm(
            _ImageWithoutSpacing(np.zeros((1, 1, 1)))
        )


def test_extract_voxel_size_reads_non_iterable_fallbacks() -> None:
    """at()/index-only coordinates follow STIR's 1-based (z, y, x) indexing."""
    array = np.zeros((2, 3, 4), dtype=np.float32)
    at_only = _ImageWithNonIterableGridSpacing(
        array, _AtOnlyGridSpacing(9.0, 2.0, 2.0, 2.0)
    )
    assert StirSimindAdaptor._extract_voxel_size_mm(at_only) == pytest.approx(2.0)

    getitem_only = _ImageWithNonIterableGridSpacing(
        array, _GetItemOnlyGridSpacing(2.0, 2.0, 2.0)
    )
    assert StirSimindAdaptor._extract_voxel_size_mm(getitem_only) == pytest.approx(2.0)

    anisotropic = _ImageWithNonIterableGridSpacing(
        array, _AtOnlyGridSpacing(9.0, 4.0, 2.0, 2.0)
    )
    with pytest.raises(ValueError, match="cubic voxels"):
        StirSimindAdaptor._extract_voxel_size_mm(anisotropic)


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
        np.zeros((2, 3, 4), dtype=np.float32), (1.0, 1.0, 1.0)
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
        np.zeros((2, 3, 4), dtype=np.float32), (1.0, 1.0, 1.0)
    )
    mu_map = _ImageWithVoxelSizes(
        np.zeros((2, 3, 5), dtype=np.float32), (1.0, 1.0, 1.0)
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
    source = _ImageWithVoxelSizes(source_arr, (1.0, 1.0, 1.0))
    mu_map = _ImageWithVoxelSizes(mu_arr, (1.0, 1.0, 1.0))
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

    runtime_operator = RuntimeOperator(switches={"RR": 12345}, timeout=42.0)
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
    """Images report spacing as (z, y, x); raw STIR sequences prepend an unused
    element. The connector takes one voxel size, so voxels must be cubic."""
    array = np.zeros((2, 3, 4), dtype=np.float32)
    for image in (
        _ImageWithVoxelSizes(array, (2.0, 2.0, 2.0)),
        _ImageWithGridSpacing(array, (0.0, 2.0, 2.0, 2.0)),
        _ImageWithGridSpacing(array, (2.0, 2.0, 2.0)),
        _ImageWithNonIterableGridSpacing(
            array, _NonIterableFloat3Coordinate(2.0, 2.0, 2.0)
        ),
    ):
        assert SirfSimindAdaptor._extract_voxel_size_mm(image) == pytest.approx(2.0)

    for image in (
        _ImageWithVoxelSizes(array, (3.0, 2.0, 2.0)),
        _ImageWithGridSpacing(array, (0.0, 2.0, 2.0, 5.0)),
        _ImageWithNonIterableGridSpacing(
            array, _NonIterableFloat3Coordinate(1.0, 2.0, 2.0)
        ),
    ):
        with pytest.raises(ValueError, match="cubic voxels"):
            SirfSimindAdaptor._extract_voxel_size_mm(image)

    with pytest.raises(ValueError, match=r"voxel_sizes\(\) or get_grid_spacing\(\)"):
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


@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
def test_adaptor_rejects_mu_map_with_different_voxel_size(
    cls, patch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor.set_mu_map(
        _ImageWithVoxelSizes(np.zeros((2, 3, 4), dtype=np.float32), (4.0, 4.0, 4.0))
    )

    with pytest.raises(ValueError, match="same voxel size"):
        adaptor.run()


@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
@pytest.mark.parametrize("changed_prefix", ["case02", "b" * 16])
def test_native_adaptor_rejects_changed_energy_window_prefix_before_writes(
    cls,
    patch,
    changed_prefix: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor.set_energy_windows([126.0], [154.0], [0])
    original_window = tmp_path / "case01.win"
    original_window_bytes = original_window.read_bytes()
    changed_window = tmp_path / f"{changed_prefix}.win"
    changed_window.write_bytes(b"existing changed window")
    destinations = [
        changed_window,
        tmp_path / f"{changed_prefix}_src.smi",
        tmp_path / f"{changed_prefix}_dns.dmi",
    ]
    for path in destinations[1:]:
        path.write_bytes(b"existing voxel input")
    adaptor.python_connector.output_prefix = changed_prefix

    configured: list[bool] = []
    launched: list[bool] = []
    monkeypatch.setattr(
        adaptor.python_connector,
        "configure_voxel_phantom",
        lambda **kwargs: configured.append(True),
    )
    monkeypatch.setattr(
        adaptor.python_connector,
        "run",
        lambda **kwargs: launched.append(True) or {},
    )

    with pytest.raises(ValueError, match="set_energy_windows"):
        adaptor.run()

    assert configured == []
    assert launched == []
    assert original_window.read_bytes() == original_window_bytes
    assert [path.read_bytes() for path in destinations] == [
        b"existing changed window",
        b"existing voxel input",
        b"existing voxel input",
    ]


@pytest.mark.parametrize(
    "cls, spacing",
    [
        (StirSimindAdaptor, (4.0, float("nan"), 4.0)),
        (SirfSimindAdaptor, (4.0, float("nan"), 4.0)),
        (StirSimindAdaptor, (0.0, 0.0, 0.0)),
        (SirfSimindAdaptor, (0.0, 0.0, 0.0)),
        (StirSimindAdaptor, (-4.0, -4.0, -4.0)),
        (SirfSimindAdaptor, (-4.0, -4.0, -4.0)),
    ],
)
def test_adaptor_rejects_nonfinite_or_nonpositive_voxel_spacing(
    cls, spacing
) -> None:
    image = _ImageWithVoxelSizes(np.zeros((2, 3, 4), dtype=np.float32), spacing)

    with pytest.raises(ValueError, match="finite and positive"):
        cls._extract_voxel_size_mm(image)


@pytest.mark.parametrize(
    "cls, patch, image_factory, geometries",
    [
        (
            StirSimindAdaptor,
            _patch_stir_backend,
            lambda array, origin, direction: _STIRImageWithGeometry(
                array, (0.0, *origin), (0.0, *(2.0 * v for v in direction))
            ),
            [
                ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ((1.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ],
        ),
        (
            SirfSimindAdaptor,
            _patch_sirf_backend,
            lambda array, origin, direction: _SIRFImageWithGeometry(
                array,
                _GeometryInfo(
                    offset=origin,
                    spacing=(2.0, 2.0, 2.0),
                    direction=direction,
                ),
            ),
            [
                ((0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
                ((1.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
            ],
        ),
    ],
)
def test_adaptor_rejects_mu_map_with_different_origin(
    cls,
    patch,
    image_factory,
    geometries,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    array = np.zeros((2, 3, 4), dtype=np.float32)
    adaptor.set_source(image_factory(array, *geometries[0]))
    adaptor.set_mu_map(image_factory(array, *geometries[1]))
    monkeypatch.setattr(
        stir_mod if cls is StirSimindAdaptor else sirf_mod,
        "get_array",
        lambda image: image.array,
    )

    with pytest.raises(ValueError, match="same spatial geometry"):
        adaptor._validate_inputs()


def test_sirf_adaptor_rejects_mu_map_with_different_axis_directions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sirf_backend(monkeypatch)
    adaptor = _make_adaptor(SirfSimindAdaptor, tmp_path)
    array = np.zeros((2, 3, 4), dtype=np.float32)
    directions = (
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    adaptor.set_source(
        _SIRFImageWithGeometry(
            array, _GeometryInfo((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), directions[0])
        )
    )
    adaptor.set_mu_map(
        _SIRFImageWithGeometry(
            array, _GeometryInfo((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), directions[1])
        )
    )
    monkeypatch.setattr(sirf_mod, "get_array", lambda image: image.array)

    with pytest.raises(ValueError, match="same spatial geometry"):
        adaptor._validate_inputs()


def test_sirf_adaptor_rejects_unreadable_present_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sirf_backend(monkeypatch)
    adaptor = _make_adaptor(SirfSimindAdaptor, tmp_path)
    array = np.zeros((2, 3, 4), dtype=np.float32)
    malformed = _GeometryInfo(
        offset=object(),
        spacing=(2.0, 2.0, 2.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    adaptor.set_source(_SIRFImageWithGeometry(array, malformed))
    adaptor.set_mu_map(_SIRFImageWithGeometry(array.copy(), malformed))
    monkeypatch.setattr(sirf_mod, "get_array", lambda image: image.array)

    with pytest.raises(ValueError, match="Unable to read SIRF geometry"):
        adaptor._validate_inputs()


@pytest.mark.parametrize(
    "cls, patch, source_native, mu_native",
    [
        (
            StirSimindAdaptor,
            _patch_stir_backend,
            lambda array: _STIRImageWithGeometry(
                array, (0.0, 0.0, 0.0, 0.0), (0.0, 2.0, 2.0, 2.0)
            ),
            lambda array: _STIRImageWithGeometry(
                array, (0.0, 1.0, 0.0, 0.0), (0.0, 2.0, 2.0, 2.0)
            ),
        ),
        (
            SirfSimindAdaptor,
            _patch_sirf_backend,
            lambda array: _SIRFImageWithGeometry(
                array,
                _GeometryInfo(
                    (0.0, 0.0, 0.0),
                    (2.0, 2.0, 2.0),
                    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                ),
            ),
            lambda array: _SIRFImageWithGeometry(
                array,
                _GeometryInfo(
                    (1.0, 0.0, 0.0),
                    (2.0, 2.0, 2.0),
                    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                ),
            ),
        ),
    ],
)
def test_adaptor_unwraps_native_images_before_comparing_geometry(
    cls,
    patch,
    source_native,
    mu_native,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    array = np.zeros((2, 3, 4), dtype=np.float32)
    source = _NativeImageWrapper(source_native(array), array)
    mu_map = _NativeImageWrapper(mu_native(array), array)
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)
    module = stir_mod if cls is StirSimindAdaptor else sirf_mod
    monkeypatch.setattr(module, "get_array", lambda image: image.array)

    with pytest.raises(ValueError, match="same spatial geometry"):
        adaptor._validate_inputs()


@pytest.mark.parametrize(
    "cls, patch, image_factory",
    [
        (
            StirSimindAdaptor,
            _patch_stir_backend,
            lambda array: _STIRImageWithGeometry(
                array, (0.0, 0.0, 0.0, 0.0), (0.0, 2.0, 2.0, 2.0)
            ),
        ),
        (
            SirfSimindAdaptor,
            _patch_sirf_backend,
            lambda array: _SIRFImageWithGeometry(
                array,
                _GeometryInfo(
                    (0.0, 0.0, 0.0),
                    (2.0, 2.0, 2.0),
                    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                ),
            ),
        ),
    ],
)
def test_adaptor_accepts_wrapped_images_with_matching_geometry(
    cls, patch, image_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    array = np.zeros((2, 3, 4), dtype=np.float32)
    adaptor.set_source(_NativeImageWrapper(image_factory(array), array))
    adaptor.set_mu_map(_NativeImageWrapper(image_factory(array.copy()), array.copy()))
    module = stir_mod if cls is StirSimindAdaptor else sirf_mod
    monkeypatch.setattr(module, "get_array", lambda image: image.array)

    adaptor._validate_inputs()


def test_stir_adaptor_compares_physical_start_using_minimum_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_stir_backend(monkeypatch)
    adaptor = _make_adaptor(StirSimindAdaptor, tmp_path)
    array = np.zeros((2, 3, 4), dtype=np.float32)
    origin = (0.0, 0.0, 0.0, 0.0)
    spacing = (0.0, 2.0, 2.0, 2.0)
    adaptor.set_source(_STIRImageWithGeometry(array, origin, spacing, (0, 0, 0)))
    adaptor.set_mu_map(_STIRImageWithGeometry(array, origin, spacing, (1, 0, 0)))
    monkeypatch.setattr(stir_mod, "get_array", lambda image: image.array)

    with pytest.raises(ValueError, match="same spatial geometry"):
        adaptor._validate_inputs()


@pytest.mark.parametrize(
    "cls, patch, module",
    [
        (StirSimindAdaptor, _patch_stir_backend, stir_mod),
        (SirfSimindAdaptor, _patch_sirf_backend, sirf_mod),
    ],
)
def test_adaptor_accepts_duck_images_with_spacing_but_no_origin(
    cls, patch, module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    array = np.zeros((2, 3, 4), dtype=np.float32)
    source = _ImageWithGridSpacing(array, (0.0, 2.0, 2.0, 2.0))
    mu_map = _ImageWithGridSpacing(array.copy(), (0.0, 2.0, 2.0, 2.0))
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)
    monkeypatch.setattr(module, "get_array", lambda image: image.array)

    adaptor._validate_inputs()


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
        np.zeros((2, 3, 4), dtype=np.float32), (2.0, 2.0, 2.0)
    )
    adaptor.set_source(source)
    adaptor.set_mu_map(
        _ImageWithVoxelSizes(np.zeros_like(source.array), (2.0, 2.0, 2.0))
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


@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
def test_adaptor_primary_output_requires_a_successful_run(
    cls, patch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)

    with pytest.raises(RuntimeError, match="Run the adaptor first"):
        adaptor.get_primary_output(window=1)

    def failing_run(runtime_operator=None):
        raise RuntimeError("simulated failure")

    adaptor.python_connector.run = failing_run  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated failure"):
        adaptor.run()

    with pytest.raises(RuntimeError, match="Run the adaptor first"):
        adaptor.get_primary_output(window=1)


@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
def test_adaptor_primary_output_unsupported_after_populated_run(
    cls, patch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    populated = {"tot_w1": "projection", "sca_w1": "scatter", "air_w1": "air"}
    adaptor._outputs = dict(populated)  # type: ignore[assignment]

    with pytest.raises(NotImplementedError, match="standalone primary projection"):
        adaptor.get_primary_output(window=1)

    assert adaptor._outputs == populated
    assert "pri_w1" not in adaptor._outputs


@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
def test_adaptor_keeps_the_config_scoring_routine_by_default(
    cls, patch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = cls(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
    )
    adaptor.add_config_value(84, ScoringRoutine.PENETRATE.value)
    image = _ImageWithVoxelSizes(np.ones((2, 3, 4), dtype=np.float32), (1.0, 1.0, 1.0))
    adaptor.set_source(image)
    adaptor.set_mu_map(image)
    monkeypatch.setattr(
        adaptor.python_connector, "run", lambda runtime_operator=None: {}
    )

    adaptor.run()

    assert adaptor.get_scoring_routine() == ScoringRoutine.PENETRATE
    assert int(adaptor.get_config().get_value(84)) == ScoringRoutine.PENETRATE.value


def _seed_existing_inputs(tmp_path: Path, prefix: str = "case01") -> dict[Path, bytes]:
    paths = [
        tmp_path / f"{prefix}_src.smi",
        tmp_path / f"{prefix}_dns.dmi",
        tmp_path / f"{prefix}.win",
        tmp_path / f"{prefix}.smc",
        tmp_path / f"{prefix}_tot_w1.hs",
        tmp_path / f"{prefix}_orbit.cor",
    ]
    for path in paths:
        path.write_bytes(f"existing:{path.name}".encode())
    return {path: path.read_bytes() for path in paths}


def _invalid_runtime_operator(
    kind: str, tmp_path: Path
) -> tuple[RuntimeOperator, type[Exception]]:
    if kind == "switch":
        return RuntimeOperator(switches={"NN": 0}), ValueError
    if kind == "string":
        return RuntimeOperator(switches={"FI": "bad name"}), SimulationError
    if kind == "orbit":
        orbit_dir = tmp_path / "orbit-dir"
        orbit_dir.mkdir(exist_ok=True)
        return RuntimeOperator(orbit_file=orbit_dir), ValueError
    if kind == "timeout":
        return RuntimeOperator(timeout=0), SimulationError
    raise AssertionError(f"unknown runtime-operator kind {kind!r}")


def _patch_pytomography_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pytomography_mod, "torch", SimpleNamespace(Tensor=object))


def _make_pytomography_adaptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> PyTomographySimindAdaptor:
    _patch_pytomography_backend(monkeypatch)
    adaptor = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
    )
    # Bypass the real torch tensors: these private values only need to satisfy
    # the input checks and the axes conversion before configure_voxel_phantom.
    adaptor._source = SimpleNamespace(shape=(2, 3, 4))  # type: ignore[assignment]
    adaptor._mu_map = SimpleNamespace(shape=(2, 3, 4))  # type: ignore[assignment]
    adaptor._energy_windows = ([126.0], [154.0], [0])
    monkeypatch.setattr(
        adaptor,
        "to_simind_image_axes",
        lambda value: SimpleNamespace(
            numpy=lambda: np.zeros((2, 3, 4), dtype=np.float32)
        ),
    )
    return adaptor


_ADAPTOR_BACKENDS = [
    (StirSimindAdaptor, _patch_stir_backend),
    (SirfSimindAdaptor, _patch_sirf_backend),
    (PyTomographySimindAdaptor, _patch_pytomography_backend),
]


@pytest.mark.parametrize("cls, patch", _ADAPTOR_BACKENDS)
@pytest.mark.parametrize("bool_value", [True, False])
def test_adaptor_constructor_rejects_boolean_scoring_routine(
    cls, patch, bool_value, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)

    with pytest.raises(ValueError, match="scoring_routine"):
        cls(
            config_source=get("AnyScan.yaml"),
            output_dir=str(tmp_path),
            output_prefix="case01",
            scoring_routine=bool_value,
        )


@pytest.mark.parametrize("cls, patch", _ADAPTOR_BACKENDS)
@pytest.mark.parametrize(
    "scoring, expected",
    [
        pytest.param(None, None, id="none"),
        pytest.param(0, ScoringRoutine.DUMMY, id="dummy-code"),
        pytest.param(1, ScoringRoutine.SCATTWIN, id="scattwin-code"),
        pytest.param(4, ScoringRoutine.PENETRATE, id="penetrate-code"),
        pytest.param(ScoringRoutine.PENETRATE, ScoringRoutine.PENETRATE, id="enum"),
    ],
)
def test_adaptor_constructor_accepts_valid_scoring_routine(
    cls, patch, scoring, expected, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)

    adaptor = cls(
        config_source=get("AnyScan.yaml"),
        output_dir=str(tmp_path),
        output_prefix="case01",
        scoring_routine=scoring,
    )

    assert adaptor._scoring_routine == expected


@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
@pytest.mark.parametrize("kind", ["switch", "string", "orbit", "timeout"])
def test_adaptor_rejects_invalid_runtime_operator_before_writes(
    cls, patch, kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    seeded = _seed_existing_inputs(tmp_path)
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


@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
def test_adaptor_accepts_valid_runtime_operator(
    cls, patch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    monkeypatch.setattr(
        adaptor.python_connector, "run", lambda runtime_operator=None: {}
    )

    outputs = adaptor.run(
        runtime_operator=RuntimeOperator(switches={"RR": 12345}, timeout=42.0)
    )

    assert outputs == {}


def _make_conflicting_orbit(destination: Path, kind: str, tmp_path: Path) -> Path:
    """Seed ``destination`` as a direct/symlink/hardlink alias of an orbit."""
    if kind == "direct":
        destination.write_bytes(b"orbit-bytes")
        return destination
    external = tmp_path / "radii.cor"
    external.write_bytes(b"orbit-bytes")
    if kind == "symlink":
        destination.symlink_to(external)
    else:
        os.link(external, destination)
    return external


def _assert_conflicting_orbit_preserved(
    destination: Path, orbit: Path, kind: str
) -> None:
    if kind == "direct":
        assert destination.read_bytes() == b"orbit-bytes"
        assert not destination.is_symlink()
    elif kind == "symlink":
        assert destination.is_symlink()
        assert orbit.read_bytes() == b"orbit-bytes"
    else:
        assert os.path.samefile(destination, orbit)
        assert orbit.read_bytes() == b"orbit-bytes"


@pytest.mark.parametrize("filename", ["case01_src.smi", "case01_dns.dmi"])
@pytest.mark.parametrize("kind", ["direct", "symlink", "hardlink"])
@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
def test_adaptor_rejects_orbit_conflicting_with_voxel_destination(
    cls,
    patch,
    kind: str,
    filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    destination = tmp_path / filename
    orbit = _make_conflicting_orbit(destination, kind, tmp_path)

    original_configure = adaptor.python_connector.configure_voxel_phantom
    configure_calls: list[int] = []

    def spy_configure(*args, **kwargs):
        configure_calls.append(1)
        return original_configure(*args, **kwargs)

    monkeypatch.setattr(
        adaptor.python_connector, "configure_voxel_phantom", spy_configure
    )
    monkeypatch.setattr(
        adaptor.python_connector, "run", lambda runtime_operator=None: {}
    )

    with pytest.raises(ValueError, match="orbit_file"):
        adaptor.run(runtime_operator=RuntimeOperator(orbit_file=orbit))

    assert configure_calls == []
    _assert_conflicting_orbit_preserved(destination, orbit, kind)


@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
def test_adaptor_accepts_external_orbit_resembling_voxel_destination(
    cls, patch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    orbit = source_dir / "case01_src.smi"
    orbit.write_bytes(b"external-orbit")
    connector = adaptor.python_connector
    monkeypatch.setattr(connector.executor, "run_simulation", lambda *a, **k: None)
    monkeypatch.setattr(connector, "_ensure_interfile_headers", lambda: [])
    monkeypatch.setattr(connector, "_load_projection_outputs", lambda h: {})

    outputs = adaptor.run(runtime_operator=RuntimeOperator(orbit_file=orbit))

    assert outputs == {}
    assert orbit.read_bytes() == b"external-orbit"
    assert (tmp_path / "case01_orbit.cor").read_bytes() == b"external-orbit"


@pytest.mark.parametrize("kind", ["switch", "string", "orbit", "timeout"])
def test_pytomography_adaptor_rejects_invalid_runtime_operator_before_writes(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adaptor = _make_pytomography_adaptor(tmp_path, monkeypatch)
    seeded = _seed_existing_inputs(tmp_path)
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


def test_pytomography_adaptor_accepts_valid_runtime_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adaptor = _make_pytomography_adaptor(tmp_path, monkeypatch)
    monkeypatch.setattr(
        adaptor.python_connector, "run", lambda runtime_operator=None: {}
    )

    outputs = adaptor.run(
        runtime_operator=RuntimeOperator(switches={"RR": 12345}, timeout=42.0)
    )

    assert outputs == {}


@pytest.mark.parametrize("filename", ["case01_src.smi", "case01_dns.dmi", "case01.win"])
@pytest.mark.parametrize("kind", ["direct", "symlink", "hardlink"])
def test_pytomography_adaptor_rejects_orbit_conflicting_with_destination(
    kind: str, filename: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adaptor = _make_pytomography_adaptor(tmp_path, monkeypatch)
    destination = tmp_path / filename
    orbit = _make_conflicting_orbit(destination, kind, tmp_path)

    original_configure = adaptor.python_connector.configure_voxel_phantom
    original_windows = adaptor.python_connector.set_energy_windows
    configure_calls: list[int] = []
    window_calls: list[int] = []

    def spy_configure(*args, **kwargs):
        configure_calls.append(1)
        return original_configure(*args, **kwargs)

    def spy_windows(*args, **kwargs):
        window_calls.append(1)
        return original_windows(*args, **kwargs)

    monkeypatch.setattr(
        adaptor.python_connector, "configure_voxel_phantom", spy_configure
    )
    monkeypatch.setattr(adaptor.python_connector, "set_energy_windows", spy_windows)
    monkeypatch.setattr(
        adaptor.python_connector, "run", lambda runtime_operator=None: {}
    )

    with pytest.raises(ValueError, match="orbit_file"):
        adaptor.run(runtime_operator=RuntimeOperator(orbit_file=orbit))

    assert configure_calls == []
    assert window_calls == []
    _assert_conflicting_orbit_preserved(destination, orbit, kind)
