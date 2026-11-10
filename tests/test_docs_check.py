"""Authored documentation inventory, figures, links and source checks."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("docs_check", ROOT / "docs" / "_docs.py")
docs_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs_check)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _png(path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), (200, 40, 40)).save(path)
    return path


def _codes(problems) -> set[str]:
    return {problem.code for problem in problems}


def _fixture(tmp_path: Path, *, figure: bool = True):
    root = tmp_path
    docs = root / "docs"
    _write(root / "examples" / "s" / "plot_01_demo.py", "x = 1\n")
    docs.mkdir()
    (docs / "examples").symlink_to(root / "examples", target_is_directory=True)
    page = _write(
        docs / "example_pages" / "demo.rst",
        "\n".join(
            [
                "Demo",
                "====",
                "",
                "Download the script: :download:`plot_01_demo.py",
                "</examples/s/plot_01_demo.py>`.",
                "",
                ".. figure:: /_static/examples/demo/plot.png",
                "",
            ]
        ),
    )
    if figure:
        _png(docs / "_static" / "examples" / "demo" / "plot.png")
    _write(
        docs / "example_pages.json",
        json.dumps(
            {
                "authored_pages": {
                    "example_pages/demo": {
                        "source": "examples/s/plot_01_demo.py",
                        "figures": ["_static/examples/demo/plot.png"],
                    }
                }
            }
        ),
    )
    return root, docs, page


@pytest.mark.unit
def test_authored_page_passes(tmp_path):
    _, docs, _ = _fixture(tmp_path)
    assert docs_check.check_docs(docs) == []


@pytest.mark.unit
def test_missing_or_corrupt_figure_fails(tmp_path):
    _, docs, _ = _fixture(tmp_path)
    figure = docs / "_static" / "examples" / "demo" / "plot.png"
    figure.write_bytes(b"not a png")
    assert "corrupt_image" in _codes(docs_check.check_docs(docs))
    figure.unlink()
    assert "missing_image" in _codes(docs_check.check_docs(docs))


@pytest.mark.unit
def test_missing_source_fails(tmp_path):
    _, docs, _ = _fixture(tmp_path)
    (docs.parent / "examples" / "s" / "plot_01_demo.py").unlink()
    assert "source_missing" in _codes(docs_check.check_docs(docs))


@pytest.mark.unit
def test_page_without_inventory_entry_fails(tmp_path):
    _, docs, _ = _fixture(tmp_path)
    _write(docs / "example_pages" / "other.rst", "Other\n=====\n")
    assert "manifest_missing" in _codes(docs_check.check_docs(docs))


@pytest.mark.unit
def test_manifest_entry_without_page_fails(tmp_path):
    _, docs, page = _fixture(tmp_path)
    page.unlink()
    assert "page_missing" in _codes(docs_check.check_docs(docs))


@pytest.mark.unit
@pytest.mark.parametrize(
    "replacement",
    [
        "\nThe file plot.png is described here.\n",
        "\n.. code-block:: rst\n\n   .. figure:: /_static/examples/demo/plot.png\n",
        "\n.. comment\n   .. figure:: /_static/examples/demo/plot.png\n",
        "\n.. raw:: text\n\n   <img src='/_static/examples/demo/plot.png'>\n",
        "\n.. figure:: /_static/examples/other/plot.png\n",
    ],
)
def test_unreferenced_or_decoy_figures_fail(tmp_path, replacement):
    _, docs, page = _fixture(tmp_path)
    if "other" in replacement:
        _png(docs / "_static" / "examples" / "other" / "plot.png")
    page.write_text(
        "Demo\n====\n\n"
        "Download the script: :download:`plot_01_demo.py "
        "</examples/s/plot_01_demo.py>`.\n" + replacement
    )
    assert "figure_not_referenced" in _codes(docs_check.check_docs(docs))


@pytest.mark.unit
def test_single_quoted_raw_html_image_passes(tmp_path):
    _, docs, page = _fixture(tmp_path)
    page.write_text(
        "Demo\n====\n\n"
        "Download the script: :download:`plot_01_demo.py "
        "</examples/s/plot_01_demo.py>`.\n\n"
        ".. raw:: html\n\n"
        "   <img src='/_static/examples/demo/plot.png'>\n"
    )
    assert docs_check.check_docs(docs) == []


@pytest.mark.unit
def test_broken_html_links_and_images_fail(tmp_path):
    _, docs, _ = _fixture(tmp_path)
    html = tmp_path / "html"
    _write(
        html / "example_pages" / "demo.html",
        '<a href="missing.html">next</a><img src="missing.png">',
    )
    codes = _codes(docs_check.check_docs(docs, html_dir=html))
    assert {"broken_html_link", "broken_html_image"} <= codes


@pytest.mark.unit
def test_missing_summary_field_fails(tmp_path):
    _, docs, _ = _fixture(tmp_path)
    manifest = json.loads((docs / "example_pages.json").read_text())
    manifest["authored_pages"]["example_pages/demo"]["run_summary"] = {
        "path": "example_pages/demo.json",
        "required_fields": ["metric.value"],
    }
    _write(docs / "example_pages.json", json.dumps(manifest))
    _write(docs / "example_pages" / "demo.json", json.dumps({"metric": {}}))
    assert "missing_summary_field" in _codes(docs_check.check_docs(docs))


@pytest.mark.unit
def test_backend_separation_violation_fails(tmp_path):
    _, docs, _ = _fixture(tmp_path)
    _write(
        docs.parent / "examples" / "s" / "plot_01_demo.py",
        "import sirf\nimport stir\n",
    )
    with pytest.raises(ValueError, match="more than one backend"):
        docs_check.check_docs(docs)
