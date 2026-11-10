"""The public examples hierarchy has one root, not duplicated section content."""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SECTIONS = (
    "phantoms",
    "mc_scatter",
    "cil",
    "real_data",
)


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
    authored = ROOT / "examples" / "index.rst"
    generated = ROOT / "docs" / "auto_examples" / "index.rst"
    assert authored.is_file()
    assert _toctree_entries(authored) == [
        "/example_pages/index",
        *(f"{name}/index" for name in SECTIONS),
    ]
    assert (ROOT / "docs" / "example_pages" / "index.rst").is_file()
    assert (ROOT / "docs" / "example_pages" / "_template.rst.txt").is_file()
    assert (ROOT / "docs" / "_static" / "examples").is_dir()
    # The gallery copies the authored root verbatim instead of generating an
    # aggregate page that repeats every section's prose and thumbnails.
    assert generated.read_text() == authored.read_text()
    assert "sphx-glr-thumbnail" not in generated.read_text()

    index_entries = _toctree_entries(ROOT / "docs" / "index.rst")
    assert index_entries.count("auto_examples/index") == 1
    assert "examples" not in index_entries
    assert not (ROOT / "docs" / "examples.rst").exists()
