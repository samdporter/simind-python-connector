import sys
import types

import numpy as np
import pytest

from simind_python_connector.phantoms import VoxelPhantom, nema_iec_phantom


def _fake_phantomgen(monkeypatch):
    calls = {}

    def create_nema(
        matrix_size, voxel_size_mm, nema_dict, center_offset_mm, supersample
    ):
        calls.update(
            matrix_size=matrix_size,
            voxel_size_mm=voxel_size_mm,
            nema_dict=nema_dict,
            center_offset_mm=center_offset_mm,
            supersample=supersample,
        )
        mu = np.full(matrix_size, 0.096, dtype=np.float32)
        mu[0] = 0.029  # lung
        activity = np.arange(np.prod(matrix_size), dtype=np.float32).reshape(
            matrix_size
        )
        background = np.zeros(matrix_size, dtype=np.uint8)
        background[1:] = 1
        lung = np.zeros(matrix_size, dtype=np.uint8)
        lung[0] = 1
        calls["activity"] = activity
        return activity, mu, {"background": background, "lung_insert": lung}

    module = types.ModuleType("phantomgen")
    module.create_nema = create_nema
    module.pet_nema_dict = {"preset": "pet"}
    module.earl_nema_dict = {"preset": "earl"}
    monkeypatch.setitem(sys.modules, "phantomgen", module)
    return calls


@pytest.mark.unit
def test_nema_phantom_converts_mu_to_water_equivalent_density(monkeypatch):
    calls = _fake_phantomgen(monkeypatch)
    phantom = nema_iec_phantom((4, 3, 2), (2.5, 2.5, 2.5), supersample=2)

    assert isinstance(phantom, VoxelPhantom)
    assert calls["nema_dict"] == {"preset": "pet"}
    assert calls["matrix_size"] == (4, 3, 2)
    assert calls["voxel_size_mm"] == (2.5, 2.5, 2.5)
    assert calls["center_offset_mm"] == (0.0, 0.0, 0.0)
    assert calls["supersample"] == 2
    np.testing.assert_allclose(phantom.density_g_cm3[1:], 1.0)
    np.testing.assert_allclose(phantom.density_g_cm3[0], 0.029 / 0.096, rtol=1e-6)
    assert phantom.activity_mbq is calls["activity"]
    assert sorted(phantom.masks) == ["background", "lung_insert"]
    assert phantom.voxel_size_mm == (2.5, 2.5, 2.5)


@pytest.mark.unit
def test_nema_phantom_presets_and_dicts(monkeypatch):
    calls = _fake_phantomgen(monkeypatch)
    nema_iec_phantom((4, 3, 2), (2.5, 2.5, 2.5), preset="earl")
    assert calls["nema_dict"] == {"preset": "earl"}
    custom = {"preset": "mine"}
    nema_iec_phantom((4, 3, 2), (2.5, 2.5, 2.5), preset=custom)
    assert calls["nema_dict"] is custom


@pytest.mark.unit
def test_nema_phantom_rejects_unknown_preset_names(monkeypatch):
    _fake_phantomgen(monkeypatch)
    with pytest.raises(ValueError, match="'pet' or 'earl'"):
        nema_iec_phantom((4, 3, 2), (2.5, 2.5, 2.5), preset="spect")


@pytest.mark.unit
def test_missing_phantomgen_explains_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "phantomgen", None)
    with pytest.raises(
        ImportError, match="git\\+https://github.com/varzakis/phantomgen"
    ):
        nema_iec_phantom((4, 3, 2), (2.5, 2.5, 2.5))


@pytest.mark.requires_phantomgen
def test_real_pet_preset_densities():
    phantom = nema_iec_phantom((96, 128, 128), (2.5, 2.5, 2.5), preset="pet")
    density, masks = phantom.density_g_cm3, phantom.masks
    assert np.median(density[masks["background"] > 0]) == pytest.approx(1.0, abs=0.01)
    assert np.median(density[masks["lung_insert"] > 0]) == pytest.approx(0.30, abs=0.02)
    assert phantom.activity_mbq.dtype == np.float32
    assert phantom.activity_mbq.sum() > 0
    assert {f"sphere_{i}" for i in range(1, 7)} <= set(masks)
