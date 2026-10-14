from __future__ import annotations

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
    assert captured["voxel_size_mm"] == pytest.approx((1.0, 1.0, 1.0))
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
    ):
        assert StirSimindAdaptor._extract_voxel_size_mm(image) == pytest.approx(2.0)

    for image in (
        _ImageWithVoxelSizes(array, (3.0, 2.0, 2.0)),
        _ImageWithGridSpacing(array, (0.0, 2.0, 2.0, 5.0)),
    ):
        with pytest.raises(ValueError, match="cubic voxels|square in-plane"):
            StirSimindAdaptor._extract_voxel_size_mm(image)

    with pytest.raises(ValueError, match=r"voxel_sizes\(\) or get_grid_spacing\(\)"):
        StirSimindAdaptor._extract_voxel_size_mm(
            _ImageWithoutSpacing(np.zeros((1, 1, 1)))
        )


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
    assert captured["voxel_size_mm"] == pytest.approx((1.0, 1.0, 1.0))
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
    ):
        assert SirfSimindAdaptor._extract_voxel_size_mm(image) == pytest.approx(2.0)

    for image in (
        _ImageWithVoxelSizes(array, (3.0, 2.0, 2.0)),
        _ImageWithGridSpacing(array, (0.0, 2.0, 2.0, 5.0)),
    ):
        with pytest.raises(ValueError, match="cubic voxels|square in-plane"):
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
def test_native_adaptors_forward_anisotropic_axial_spacing(
    cls, patch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    source = _ImageWithVoxelSizes(np.ones((2, 3, 4)), (8.0, 4.0, 4.0))
    mu_map = _ImageWithVoxelSizes(np.zeros_like(source.array), (8.0, 4.0, 4.0))
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)
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


@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
def test_native_adaptors_reject_different_axial_spacing(
    cls, patch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    source = _ImageWithVoxelSizes(np.ones((2, 3, 4)), (8.0, 4.0, 4.0))
    mu_map = _ImageWithVoxelSizes(np.zeros_like(source.array), (4.0, 4.0, 4.0))
    adaptor.set_source(source)
    adaptor.set_mu_map(mu_map)

    with pytest.raises(ValueError, match="same voxel size"):
        adaptor._validate_inputs()


@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
def test_native_adaptors_reject_nonsquare_inplane_voxels(
    cls, patch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    source = _ImageWithVoxelSizes(np.ones((2, 3, 4)), (8.0, 4.0, 4.1))
    adaptor.set_source(source)
    adaptor.set_mu_map(_ImageWithVoxelSizes(np.zeros_like(source.array), (8, 4, 4.1)))

    with pytest.raises(ValueError, match="square in-plane"):
        adaptor._validate_inputs()


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
def test_adaptor_rejects_invalid_runtime_operator_before_launch(
    cls, patch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    launched: list[object] = []
    monkeypatch.setattr(
        adaptor.python_connector.executor,
        "run_simulation",
        lambda *args, **kwargs: launched.append(args),
    )

    with pytest.raises(ValueError):
        adaptor.run(runtime_operator=RuntimeOperator(switches={"NN": 0}))

    assert launched == []


@pytest.mark.parametrize(
    "cls, patch",
    [
        (StirSimindAdaptor, _patch_stir_backend),
        (SirfSimindAdaptor, _patch_sirf_backend),
    ],
)
def test_adaptor_rejects_orbit_at_voxel_input_destination(
    cls, patch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    orbit = tmp_path / "case01_src.smi"
    orbit.write_bytes(b"orbit-bytes")
    monkeypatch.setattr(
        adaptor.python_connector, "run", lambda runtime_operator=None: {}
    )

    with pytest.raises(ValueError, match="orbit_file"):
        adaptor.run(runtime_operator=RuntimeOperator(orbit_file=orbit))

    assert orbit.read_bytes() == b"orbit-bytes"


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
