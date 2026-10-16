"""Skip tests whose external runtime is unavailable."""

import importlib
import os
import shutil

import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    requirements = {
        "requires_sirf": ("sirf.STIR", "SIRF"),
        "requires_stir": ("stir", "STIR Python"),
        "requires_cil": ("cil", "CIL"),
        "requires_pytomography": ("pytomography", "PyTomography"),
        "requires_setr": ("setr", "SETR"),
    }
    # Run after marker deselection so unit tests do not load unused backends.
    for marker, (module, label) in requirements.items():
        affected = [item for item in items if marker in item.keywords]
        if not affected:
            continue
        try:
            importlib.import_module(module)
            if marker == "requires_stir":
                from simind_python_connector.utils.import_helpers import (
                    _import_stir_extra,
                )

                _import_stir_extra()
        except ImportError:
            skip = pytest.mark.skip(reason=f"{label} not available")
            for item in affected:
                item.add_marker(skip)

    simind_available = (
        shutil.which(os.environ.get("SIMIND_BIN") or "simind") is not None
    )
    in_ci = any(
        os.getenv(name, "false").lower() == "true" for name in ("CI", "GITHUB_ACTIONS")
    )
    for item in items:
        if "requires_simind" in item.keywords and not simind_available:
            item.add_marker(pytest.mark.skip(reason="SIMIND not available"))
        if "ci_skip" in item.keywords and in_ci:
            item.add_marker(pytest.mark.skip(reason="Skipped in CI environment"))
