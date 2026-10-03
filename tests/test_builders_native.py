import importlib
import sys
import types

import numpy as np
import pytest

from simind_python_connector.builders import _native


pytestmark = pytest.mark.unit

_BACKEND_MODULES = ("sirf.STIR", "stir", "stirextra")


def _fake_import_module(available):
    real_import_module = importlib.import_module

    def fake(name, *args, **kwargs):
        if name in _BACKEND_MODULES:
            if name in available:
                return types.ModuleType(name)
            raise ImportError(name)
        return real_import_module(name, *args, **kwargs)

    return fake


def test_resolve_backend_prefers_sirf(monkeypatch):
    monkeypatch.setattr(
        importlib, "import_module", _fake_import_module(set(_BACKEND_MODULES))
    )
    assert _native.resolve_backend(None) == "sirf"


def test_resolve_backend_falls_back_to_stir(monkeypatch):
    monkeypatch.setattr(
        importlib, "import_module", _fake_import_module({"stir", "stirextra"})
    )
    assert _native.resolve_backend(None) == "stir"


def test_resolve_backend_requires_stirextra_for_stir(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", _fake_import_module({"stir"}))
    with pytest.raises(ImportError, match="Neither SIRF nor STIR"):
        _native.resolve_backend(None)


def test_resolve_backend_explicit_backend_must_be_installed(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", _fake_import_module(set()))
    with pytest.raises(ImportError, match="SIRF is not available"):
        _native.resolve_backend("sirf")
    with pytest.raises(ImportError, match="STIR Python is not available"):
        _native.resolve_backend("stir")


def test_resolve_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="backend must be one of"):
        _native.resolve_backend("pet")


def _install_fake_stir(monkeypatch, calls):
    class FakeProjData:
        @staticmethod
        def read_from_file(path):
            calls.append(("read", path))
            return "proj-from-file"

    class FakeProjDataInMemory:
        def __init__(self, source):
            calls.append(("in_memory", source))
            self.source = source

        def fill(self, values):
            calls.append(("fill", list(values)))

        def write_to_file(self, path):
            calls.append(("write", path))

    fake_stir = types.ModuleType("stir")
    fake_stir.ProjData = FakeProjData
    fake_stir.ProjDataInMemory = FakeProjDataInMemory
    monkeypatch.setitem(sys.modules, "stir", fake_stir)
    monkeypatch.setitem(sys.modules, "stirextra", types.ModuleType("stirextra"))
    return FakeProjDataInMemory


def test_load_acquisition_stir_returns_in_memory_copy(monkeypatch):
    calls = []
    in_memory_type = _install_fake_stir(monkeypatch, calls)

    result = _native.load_acquisition("x.hs", "stir")

    assert isinstance(result, in_memory_type)
    assert calls == [("read", "x.hs"), ("in_memory", "proj-from-file")]


def test_refill_acquisition_stir_fills_copy_and_writes(monkeypatch):
    calls = []
    in_memory_type = _install_fake_stir(monkeypatch, calls)
    original = in_memory_type("original")
    calls.clear()

    result = _native.refill_acquisition(
        original, np.array([1.0, 2.0], dtype=np.float32), "out.hs", "stir"
    )

    assert result is not original
    assert calls == [
        ("in_memory", original),
        ("fill", [1.0, 2.0]),
        ("write", "out.hs"),
    ]


def test_refill_acquisition_sirf_clones_fills_and_writes():
    class FakeSirfAcquisition:
        def __init__(self):
            self.events = []

        def clone(self):
            self.events.append("clone")
            return self

        def fill(self, array):
            self.events.append(("fill", array.shape))

        def write(self, path):
            self.events.append(("write", path))

    acquisition = FakeSirfAcquisition()
    result = _native.refill_acquisition(
        acquisition, np.zeros((1, 2, 3, 4), dtype=np.float32), "out.hs", "sirf"
    )

    assert result is acquisition
    assert acquisition.events == ["clone", ("fill", (1, 2, 3, 4)), ("write", "out.hs")]
