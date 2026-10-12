"""Tests for Example 12: the phantomgen branch must configure its projection grid."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from simind_python_connector import SimindPythonConnector
from simind_python_connector.phantoms import AnalyticPhantom


ROOT = Path(__file__).resolve().parents[1]

# Source voxels are scaled to MAX_SOURCE * quantization_scale integer histories,
# and MAX_SOURCE is 500, so 0.001 rounds the whole map to zero. 0.01 keeps the
# run tiny while leaving the source non-zero.
_SMOKE_QUANTIZATION_SCALE = 0.01


def _load_example():
    path = ROOT / "examples" / "12_phantoms.py"
    spec = importlib.util.spec_from_file_location("example_12_phantoms", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_phantomgen(monkeypatch):
    calls = {}

    def create_nema(
        matrix_size, voxel_size_mm, nema_dict, center_offset_mm, supersample
    ):
        calls.update(matrix_size=matrix_size, voxel_size_mm=voxel_size_mm)
        activity = np.arange(np.prod(matrix_size), dtype=np.float32).reshape(
            matrix_size
        )
        mu = np.full(matrix_size, 0.096, dtype=np.float32)
        mu.flat[0] = 0.029
        background = np.ones(matrix_size, dtype=np.uint8)
        background.flat[0] = 0
        lung = np.zeros(matrix_size, dtype=np.uint8)
        lung.flat[0] = 1
        return activity, mu, {"background": background, "lung_insert": lung}

    module = types.ModuleType("phantomgen")
    module.create_nema = create_nema
    module.pet_nema_dict = {"preset": "pet"}
    module.earl_nema_dict = {"preset": "earl"}
    monkeypatch.setitem(sys.modules, "phantomgen", module)
    return calls


class _ConnectorSpy:
    def __init__(self, config_source, output_dir, output_prefix="output", **kwargs):
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.values = {}
        self.values_at_run = None
        self.phantom = None

    def set_energy_windows(self, *args):
        pass

    def add_runtime_switch(self, *args):
        pass

    def add_config_value(self, index, value):
        self.values[index] = value

    def configure_phantom(self, phantom, **kwargs):
        self.phantom = phantom

    def run(self):
        self.values_at_run = dict(self.values)
        return {"tot_w1": types.SimpleNamespace(projection=np.zeros((2, 128, 128)))}


def _spy_factory(spies):
    def factory(*args, **kwargs):
        spy = _ConnectorSpy(*args, **kwargs)
        spies.append(spy)
        return spy

    return factory


def _prepare(module, monkeypatch, tmp_path, spies):
    monkeypatch.setattr(module, "SimindPythonConnector", _spy_factory(spies))
    monkeypatch.setattr(module.shutil, "which", lambda name: "/fake/simind")
    monkeypatch.chdir(tmp_path)


@pytest.mark.unit
def test_nema_branch_configures_a_projection_grid_covering_the_phantom(
    tmp_path, monkeypatch
):
    module = _load_example()
    calls = _install_fake_phantomgen(monkeypatch)
    spies = []
    _prepare(module, monkeypatch, tmp_path, spies)

    module.main()

    assert calls["matrix_size"] == (128, 128, 128)
    assert calls["voxel_size_mm"] == (4.42, 4.42, 4.42)

    spy = spies[0]
    grid = spy.values_at_run
    assert grid is not None
    assert grid[28] == pytest.approx(0.442)
    assert grid[76] == 128
    assert grid[77] == 128

    phantom = spy.phantom
    nz, ny, nx = phantom.activity_mbq.shape
    fov_width_mm = grid[76] * grid[28] * 10.0
    fov_length_mm = grid[77] * grid[28] * 10.0
    assert fov_width_mm >= max(nx, ny) * phantom.voxel_size_mm[2]
    assert fov_length_mm >= nz * phantom.voxel_size_mm[0]


@pytest.mark.unit
def test_missing_phantomgen_keeps_the_analytic_fallback_grid(tmp_path, monkeypatch):
    module = _load_example()
    monkeypatch.setitem(sys.modules, "phantomgen", None)
    spies = []
    _prepare(module, monkeypatch, tmp_path, spies)

    module.main()

    spy = spies[0]
    assert isinstance(spy.phantom, AnalyticPhantom)
    grid = spy.values_at_run
    assert grid[28] == pytest.approx(0.4)
    assert grid[76] == 64
    assert grid[77] == 64


@pytest.mark.integration
@pytest.mark.ci_skip
@pytest.mark.requires_simind
@pytest.mark.requires_phantomgen
def test_example_12_nema_branch_runs_with_real_phantomgen_and_simind(
    tmp_path, monkeypatch, caplog
):
    module = _load_example()
    captured = []

    class _FastConnector(SimindPythonConnector):
        def __init__(self, *args, **kwargs):
            kwargs["quantization_scale"] = _SMOKE_QUANTIZATION_SCALE
            super().__init__(*args, **kwargs)
            self.add_config_value(29, 2)
            self.add_runtime_switch("NN", 1)

        def run(self, *args, **kwargs):
            outputs = super().run(*args, **kwargs)
            captured.append(self)
            return outputs

    monkeypatch.setattr(module, "SimindPythonConnector", _FastConnector)
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        module.main()

    connector = captured[0]
    result = connector.get_outputs()["tot_w1"]
    projection = result.projection
    print(f"smoke: projection {projection.shape}, total counts {projection.sum():.4g}")

    assert projection.shape == (2, 128, 128)
    assert np.isfinite(projection).all()
    assert projection.sum() > 0

    assert float(result.metadata["scaling factor (mm/pixel) [1]"]) == pytest.approx(
        4.42
    )
    assert float(result.metadata["scaling factor (mm/pixel) [2]"]) == pytest.approx(
        4.42
    )

    overrun = [
        record
        for record in caplog.records
        if "projection field of view" in record.getMessage()
    ]
    assert overrun == []
