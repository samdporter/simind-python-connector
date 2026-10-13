import os
import shutil

import pytest


def _importable(*modules: str) -> bool:
    try:
        for module in modules:
            __import__(module)
    except ImportError:
        return False
    return True


def _stir_available() -> bool:
    """True when STIR Python and its array-conversion module are importable.

    Reuses the production modern/legacy detection so a broken ``stir.extra``
    dependency is not masked by the legacy fallback.
    """
    try:
        import stir  # noqa: F401

        from simind_python_connector.utils import _import_stir_extra

        _import_stir_extra()
    except ImportError:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    """Skip tests whose optional dependencies are not available."""
    available = {
        "requires_sirf": _importable("sirf.STIR"),
        "requires_stir": _stir_available(),
        "requires_cil": _importable("cil"),
        "requires_pytomography": _importable("pytomography"),
        "requires_simind": shutil.which("simind") is not None,
        "requires_phantomgen": _importable("phantomgen"),
    }
    in_ci = (
        os.getenv("CI", "false").lower() == "true"
        or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
    )

    for item in items:
        for marker, is_available in available.items():
            if marker in item.keywords and not is_available:
                item.add_marker(pytest.mark.skip(reason=f"{marker}: not available"))
        if "ci_skip" in item.keywords and in_ci:
            item.add_marker(pytest.mark.skip(reason="Skipped in CI environment"))
