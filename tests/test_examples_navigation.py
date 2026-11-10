"""The public examples hierarchy has one root, not duplicated section content."""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("real_data",)


def _toctree_entries(path: Path) -> list[str]:
    entries = []
    in_tree = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(".. toctree::"):
            in_tree = True
        elif in_tree and not stripped:
            continue
        elif in_tree and line.startswith("   ") and not stripped.startswith(":"):
            entries.append(stripped)
        elif in_tree and not line.startswith(" "):
            in_tree = False
    return entries


@pytest.mark.unit
def test_examples_navigation_has_one_duplication_free_root():
    root = ROOT / "docs" / "example_pages" / "index.rst"
    assert root.is_file()
    entries = _toctree_entries(root)
    assert entries[-len(SECTIONS) :] == [
        f"/auto_examples/{name}/index" for name in SECTIONS
    ]
    assert entries[:15] == [
        "basic_simulation",
        "multi_window",
        "custom_configuration",
        "scattwin_vs_penetrate",
        "density_conversion",
        "phantom_library",
        "mc_scatter_osem",
        "scatter_normalisation",
        "residual_correction",
        "cil_scatter",
        "cil_rdp_svrg",
        "stir_osem",
        "sirf_osem",
        "pytomography_osem",
        "dicom_recipe",
    ]
    assert (ROOT / "docs" / "example_pages" / "_template.rst.txt").is_file()
    assert (ROOT / "docs" / "_static" / "examples").is_dir()

    index_entries = _toctree_entries(ROOT / "docs" / "index.rst")
    assert index_entries.count("example_pages/index") == 1
    assert "auto_examples/index" not in index_entries
    assert not (ROOT / "docs" / "examples.rst").exists()
