"""Keep core examples backend-free and adaptor examples isolated."""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


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
    roots = _import_roots(ROOT / "examples" / name)
    assert not roots & {"sirf", "stir", "pytomography"}


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
    roots = _import_roots(ROOT / "examples" / name)
    assert backend in roots
    assert not roots & ({"sirf", "stir", "pytomography"} - {backend})
