"""Keep core examples backend-free and adaptor examples isolated."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gallery", ROOT / "docs" / "_gallery.py")
gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        f"{number:02d}_{suffix}.py"
        for number, suffix in enumerate(
            [
                "basic_simulation",
                "runtime_switch_comparison",
                "multi_window",
                "custom_config",
                "scattwin_vs_penetrate_comparison",
                "schneider_density_conversion",
            ],
            start=1,
        )
    ],
)
def test_core_examples_do_not_import_optional_backends(name: str):
    assert gallery.example_backend(ROOT / "examples" / name) == "python"


@pytest.mark.parametrize(
    "name,backend",
    [
        ("07A_stir_adaptor_osem.py", "stir"),
        ("07B_sirf_adaptor_osem.py", "sirf"),
        ("07C_pytomography_adaptor_osem.py", "pytomography"),
    ],
)
@pytest.mark.unit
def test_adaptor_examples_import_only_their_backend(name: str, backend: str):
    assert gallery.example_backend(ROOT / "examples" / name) == backend
