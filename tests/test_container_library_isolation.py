import importlib
import os

import pytest


EXPECTATIONS = {
    "python": {
        "allowed": (),
        "blocked": ("sirf", "stir", "stirextra", "pytomography"),
    },
    "stir": {
        "allowed": ("stir",),
        "blocked": ("sirf", "pytomography"),
    },
    "sirf": {
        "allowed": ("sirf.STIR",),
        "blocked": ("stir", "stirextra", "pytomography"),
    },
    "pytomography": {
        "allowed": ("pytomography",),
        "blocked": ("sirf", "stir", "stirextra"),
    },
}


def _stir_conversion_module():
    """Return the STIR array-conversion module, preferring the modern layout."""
    from simind_python_connector.utils.import_helpers import _import_stir_extra

    return _import_stir_extra()


@pytest.mark.unit
def test_container_backend_library_isolation():
    backend = os.environ.get("SIMIND_CONNECTOR_BACKEND")
    if backend not in EXPECTATIONS:
        pytest.skip("Container backend isolation check only runs in docker services")

    expected = EXPECTATIONS[backend]

    for module in expected["allowed"]:
        importlib.import_module(module)

    for module in expected["blocked"]:
        with pytest.raises(ImportError):
            importlib.import_module(module)

    if backend == "stir":
        # Importing ``stir`` alone is not enough; STIR is only usable with one
        # of the conversion modules, and the modern ``stir.extra`` is preferred.
        _stir_conversion_module()


@pytest.mark.unit
def test_stir_isolation_requires_conversion_module(monkeypatch):
    """A bare ``stir`` import must not satisfy the STIR isolation check."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "stir", types.ModuleType("stir"))
    monkeypatch.delitem(sys.modules, "stir.extra", raising=False)
    # ``None`` in sys.modules makes ``import stirextra`` fail even when the
    # legacy module is installed on disk.
    monkeypatch.setitem(sys.modules, "stirextra", None)

    with pytest.raises(ImportError):
        _stir_conversion_module()


@pytest.mark.unit
def test_stir_isolation_accepts_modern_layout(monkeypatch):
    """The modern ``stir.extra`` layout satisfies the STIR isolation check."""
    import sys
    import types

    extra = types.ModuleType("stir.extra")
    fake_stir = types.ModuleType("stir")
    fake_stir.extra = extra
    monkeypatch.setitem(sys.modules, "stir", fake_stir)
    monkeypatch.setitem(sys.modules, "stir.extra", extra)
    monkeypatch.delitem(sys.modules, "stirextra", raising=False)

    assert _stir_conversion_module() is extra
