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


class _SirfLikeImage:
    """Stand-in for sirf.STIR.ImageData: as_array() and voxel_sizes() in (z, y, x)."""

    def __init__(self, array, voxel_sizes=(4.0, 4.0, 4.0)):
        self.array = np.asarray(array)
        self._voxel_sizes = voxel_sizes

    def as_array(self):
        return self.array

    def voxel_sizes(self):
        return self._voxel_sizes


class _OneBasedCoordinate:
    """STIR coordinates are 1-based: [1] is z, [2] is y, [3] is x."""

    def __init__(self, z, y, x):
        self._values = {1: z, 2: y, 3: x}

    def __getitem__(self, index):
        return self._values[index]


class _StirLikeImage:
    """Stand-in for stir.FloatVoxelsOnCartesianGrid."""

    def __init__(self, array, spacing=(4.0, 4.0, 4.0)):
        self.array = np.asarray(array)
        self._spacing = _OneBasedCoordinate(*spacing)

    def as_array(self):
        return self.array

    def get_grid_spacing(self):
        return self._spacing


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

    def fake_configure_voxel_phantom(source, mu_map, voxel_size_mm, scoring_routine):
        captured.update(
            source=source,
            mu_map=mu_map,
            voxel_size_mm=voxel_size_mm,
            scoring_routine=scoring_routine,
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
    assert captured["voxel_size_mm"] == pytest.approx(4.0)
    assert captured["scoring_routine"] == ScoringRoutine.PENETRATE
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
def test_adaptor_rejects_anisotropic_voxels(
    cls, patch, image_cls, tmp_path, monkeypatch
):
    patch(monkeypatch)
    adaptor = _make_adaptor(cls, tmp_path)
    adaptor.set_source(image_cls(np.zeros((2, 3, 4)), (2.0, 4.0, 4.0)))
    adaptor.set_mu_map(image_cls(np.zeros((2, 3, 4)), (2.0, 4.0, 4.0)))
    with pytest.raises(ValueError, match="not isotropic"):
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
