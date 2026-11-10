"""Gallery backend classification, inventory, figure and link checks."""

import importlib.util
import json
import re
import zipfile
from pathlib import Path

import pytest
from sphinx_gallery.utils import get_md5sum


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gallery", ROOT / "docs" / "_gallery.py")
gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _png(path: Path, color: tuple[int, int, int] = (200, 40, 40)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color).save(path)
    return path


def _codes(problems) -> set[str]:
    return {problem.code for problem in problems}


def _gallery_fixture(tmp_path: Path, *, figure: bool = True):
    """Build one generated example, its page, download and inventory entry."""

    root = tmp_path
    docs = root / "docs"
    base = docs / "auto_examples" / "s"
    source = "x = 1\n"
    example = _write(root / "examples" / "s" / "plot_01_demo.py", source)
    _write(base / "plot_01_demo.py.md5", get_md5sum(example, mode="t"))
    _write(base / "plot_01_demo.py", source)
    with zipfile.ZipFile(base / "plot_01_demo.zip", "w") as archive:
        archive.writestr("plot_01_demo.py", source)
    name = "sphx_glr_plot_01_demo_001.png"
    _write(
        base / "plot_01_demo.rst",
        f"Demo\n====\n\n.. image-sg:: /auto_examples/s/images/{name}\n",
    )
    if figure:
        _png(base / "images" / name)
    _write(
        docs / "gallery_figures.json",
        json.dumps(
            {
                "schema": 2,
                "examples": {
                    "s/plot_01_demo": {"figures": [name]},
                },
                "authored_pages": {},
            }
        ),
    )
    return root, example, docs, docs / "auto_examples"


def _add_authored_fixture(root: Path, docs: Path, base: Path) -> None:
    slug = "demo"
    link = docs / "examples"
    if not link.exists():
        link.symlink_to(root / "examples", target_is_directory=True)
    page = docs / "example_pages" / f"{slug}.rst"
    _write(
        page,
        "\n".join(
            [
                "Demo tutorial",
                "=============",
                "",
                ".. literalinclude:: /examples/s/plot_01_demo.py",
                "   :language: python",
                "",
                ".. figure:: /_static/examples/demo/plot.png",
                "",
            ]
        ),
    )
    _png(docs / "_static" / "examples" / slug / "plot.png")
    manifest = json.loads((docs / "gallery_figures.json").read_text())
    manifest["authored_pages"]["example_pages/demo"] = {
        "source": "examples/s/plot_01_demo.py",
        "figures": ["_static/examples/demo/plot.png"],
    }
    _write(docs / "gallery_figures.json", json.dumps(manifest))
    del base


@pytest.mark.unit
@pytest.mark.parametrize(
    "source,backend",
    [
        ("import sirf.STIR\n", "sirf"),
        ("from cil.optimisation import x\n", "sirf"),
        ("import stir\n", "stir"),
        ("import numpy\n", "python"),
    ],
)
def test_backend_is_taken_from_the_imports(tmp_path, source, backend):
    assert gallery.example_backend(_write(tmp_path / "x.py", source)) == backend


@pytest.mark.unit
def test_two_backends_are_rejected(tmp_path):
    path = _write(tmp_path / "mixed.py", "import sirf\nimport stir\n")
    with pytest.raises(ValueError, match="mixed.py"):
        gallery.example_backend(path)


@pytest.mark.unit
def test_only_plot_files_in_sections_are_gallery_examples(tmp_path):
    examples = tmp_path / "examples"
    _write(examples / "a.py", "")
    _write(examples / "s" / "_h.py", "")
    _write(examples / "s" / "01_x.py", "")
    example = _write(examples / "s" / "plot_01_x.py", "")
    assert gallery.gallery_examples(examples) == [example]


@pytest.mark.unit
def test_execution_pattern_matches_only_that_backend(tmp_path):
    examples = tmp_path / "examples"
    python_example = _write(examples / "one" / "plot_01_a.py", "import numpy\n")
    sirf_example = _write(examples / "two" / "plot_02_b.py", "import sirf\n")
    pattern = gallery.execution_pattern(examples, "sirf")
    assert re.search(pattern, str(sirf_example))
    assert not re.search(pattern, str(python_example))
    assert gallery.execution_pattern(examples, "stir") == r"^$"


@pytest.mark.unit
def test_stale_examples_detects_edited_sources(tmp_path):
    examples = tmp_path / "examples"
    built = tmp_path / "built"
    edited = _write(examples / "s" / "plot_01_edited.py", "x = 1\n")
    kept = _write(examples / "s" / "plot_02_kept.py", "y = 1\n")
    for path in (edited, kept):
        _write(built / "s" / f"{path.name}.md5", get_md5sum(path, mode="t"))
    edited.write_text("x = 2\n")
    assert gallery.stale_examples(examples, built) == [edited]


@pytest.mark.unit
def test_stale_examples_only_checks_requested_backends(tmp_path):
    examples = tmp_path / "examples"
    python_example = _write(examples / "s" / "plot_01_core.py", "x = 1\n")
    stir_example = _write(examples / "s" / "plot_02_stir.py", "import stir\n")
    built = tmp_path / "built"
    assert gallery.stale_examples(examples, built, ("python",)) == [python_example]
    assert gallery.stale_examples(examples, built, ("stir",)) == [stir_example]
    assert gallery.stale_examples(examples, built) == [python_example, stir_example]


@pytest.mark.unit
def test_manifest_inventory_matches_generated_sources():
    entries = gallery.load_manifest(ROOT / "docs")["examples"]
    for key in entries:
        assert (ROOT / "examples" / f"{key}.py").is_file(), key


@pytest.mark.unit
def test_gate_rejects_a_generated_source_without_inventory(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _write(root / "examples" / "phantoms" / "plot_20_missing.py", "x = 1\n")
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert any(
        problem.code == "manifest_missing"
        and "examples/phantoms/plot_20_missing.py" in problem.detail
        for problem in problems
    )


@pytest.mark.unit
def test_generated_coverage_is_checked_for_other_backends(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _write(root / "examples" / "phantoms" / "plot_20_missing.py", "x = 1\n")
    problems = gallery.check_gallery(
        root / "examples", gallery_dir, docs, ("stir",), root=root
    )
    assert any(
        problem.code == "manifest_missing"
        and "examples/phantoms/plot_20_missing.py" in problem.detail
        for problem in problems
    )


@pytest.mark.unit
def test_generated_coverage_finds_nested_sources(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _write(
        root / "examples" / "phantoms" / "nested" / "plot_20_missing.py",
        "x = 1\n",
    )
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert any(
        problem.code == "manifest_missing"
        and "examples/phantoms/nested/plot_20_missing.py" in problem.detail
        for problem in problems
    )


@pytest.mark.unit
def test_authored_entry_cannot_exempt_a_source_that_is_still_generated(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _add_authored_fixture(root, docs, gallery_dir)
    _write(root / "examples" / "phantoms" / "plot_20_authored.py", "x = 1\n")
    _write(
        docs / "example_pages" / "excluded.rst",
        "Excluded\n========\n\n"
        ".. literalinclude:: /examples/phantoms/plot_20_authored.py\n",
    )
    manifest = json.loads((docs / "gallery_figures.json").read_text())
    manifest["authored_pages"]["example_pages/excluded"] = {
        "source": "examples/phantoms/plot_20_authored.py",
        "figures": [],
    }
    _write(docs / "gallery_figures.json", json.dumps(manifest))
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert any(
        problem.code == "manifest_missing"
        and "examples/phantoms/plot_20_authored.py" in problem.detail
        for problem in problems
    )


@pytest.mark.unit
def test_gate_exempts_an_authored_source_without_inventory(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _add_authored_fixture(root, docs, gallery_dir)
    _write(root / "examples" / "phantoms" / "plot_09_authored.py", "x = 1\n")
    _write(
        docs / "example_pages" / "excluded.rst",
        "Excluded\n========\n\n"
        ".. literalinclude:: /examples/phantoms/plot_09_authored.py\n",
    )
    manifest = json.loads((docs / "gallery_figures.json").read_text())
    manifest["authored_pages"]["example_pages/excluded"] = {
        "source": "examples/phantoms/plot_09_authored.py",
        "figures": [],
    }
    _write(docs / "gallery_figures.json", json.dumps(manifest))
    assert gallery.check_gallery(root / "examples", gallery_dir, docs, root=root) == []


@pytest.mark.unit
def test_gate_accepts_a_current_generated_page(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    assert gallery.check_gallery(root / "examples", gallery_dir, docs, root=root) == []


@pytest.mark.unit
def test_gate_rejects_a_missing_or_corrupt_figure(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    image = gallery_dir / "s" / "images" / "sphx_glr_plot_01_demo_001.png"
    image.write_bytes(b"not a png")
    assert "corrupt_image" in _codes(
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    )
    image.unlink()
    assert "missing_image" in _codes(
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    )


@pytest.mark.unit
def test_gate_rejects_an_example_that_imports_two_backends(tmp_path):
    root, example, docs, gallery_dir = _gallery_fixture(tmp_path)
    example.write_text("import sirf\nimport stir\n")
    with pytest.raises(ValueError, match="more than one backend"):
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)


@pytest.mark.unit
def test_gate_rejects_a_stale_generated_page(tmp_path):
    root, example, docs, gallery_dir = _gallery_fixture(tmp_path)
    example.write_text("x = 2\n")
    assert "source_stale" in _codes(
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    )
    assert gallery.stale_examples(root / "examples", gallery_dir, root=root) == [
        example
    ]


@pytest.mark.unit
def test_gate_rejects_a_mismatched_downloaded_script(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    (gallery_dir / "s" / "plot_01_demo.py").write_text("x = 2\n")
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "download_source_mismatch" in _codes(problems)


@pytest.mark.unit
def test_gate_rejects_broken_html_links_and_images(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    html = tmp_path / "html"
    page = html / "auto_examples" / "s" / "plot_01_demo.html"
    _write(page, '<a href="missing.html">next</a><img src="missing.png">')
    assert "broken_html_link" in _codes(
        gallery.check_gallery(
            root / "examples", gallery_dir, docs, root=root, html_dir=html
        )
    )
    assert "broken_html_image" in _codes(
        gallery.check_gallery(
            root / "examples", gallery_dir, docs, root=root, html_dir=html
        )
    )


@pytest.mark.unit
def test_gate_rejects_a_figure_name_that_only_appears_in_prose(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    page = gallery_dir / "s" / "plot_01_demo.rst"
    page.write_text(
        "Demo\n====\n\nThe file sphx_glr_plot_01_demo_001.png is described here.\n"
    )
    assert "figure_not_referenced" in _codes(
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    )


@pytest.mark.unit
def test_gate_ignores_image_directives_inside_code_blocks(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    page = gallery_dir / "s" / "plot_01_demo.rst"
    page.write_text(
        "\n".join(
            [
                "Demo",
                "====",
                "",
                ".. code-block:: rst",
                "",
                "   .. image-sg:: /auto_examples/s/images/"
                "sphx_glr_plot_01_demo_001.png",
                "",
            ]
        )
    )
    assert "figure_not_referenced" in _codes(
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    )


@pytest.mark.unit
def test_gate_ignores_raw_html_images_inside_code_blocks(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _add_authored_fixture(root, docs, gallery_dir)
    page = docs / "example_pages" / "demo.rst"
    page.write_text(
        "\n".join(
            [
                "Demo tutorial",
                "=============",
                "",
                ".. code-block:: html",
                "",
                "   <img src='/_static/examples/demo/plot.png'>",
                "",
            ]
        )
    )
    assert "figure_not_referenced" in _codes(
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    )


@pytest.mark.unit
def test_gate_ignores_images_inside_rst_comments(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    page = gallery_dir / "s" / "plot_01_demo.rst"
    page.write_text(
        "\n".join(
            [
                "Demo",
                "====",
                "",
                ".. this comment hides an image",
                "   .. image-sg:: /auto_examples/s/images/"
                "sphx_glr_plot_01_demo_001.png",
                "",
            ]
        )
    )
    assert "figure_not_referenced" in _codes(
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    )


@pytest.mark.unit
def test_gate_ignores_html_in_non_html_raw_blocks(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _add_authored_fixture(root, docs, gallery_dir)
    page = docs / "example_pages" / "demo.rst"
    page.write_text(
        "\n".join(
            [
                "Demo tutorial",
                "=============",
                "",
                ".. literalinclude:: /examples/s/plot_01_demo.py",
                "",
                ".. raw:: text",
                "",
                "   <img src='/_static/examples/demo/plot.png'>",
                "",
            ]
        )
    )
    assert "figure_not_referenced" in _codes(
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    )


@pytest.mark.unit
def test_gate_accepts_a_single_quoted_raw_html_image(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _add_authored_fixture(root, docs, gallery_dir)
    page = docs / "example_pages" / "demo.rst"
    page.write_text(
        "\n".join(
            [
                "Demo tutorial",
                "=============",
                "",
                ".. literalinclude:: /examples/s/plot_01_demo.py",
                "",
                ".. raw:: html",
                "",
                "   <img src='/_static/examples/demo/plot.png'>",
                "",
            ]
        )
    )
    assert gallery.check_gallery(root / "examples", gallery_dir, docs, root=root) == []


@pytest.mark.unit
def test_gate_rejects_a_same_basename_figure_in_another_directory(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _add_authored_fixture(root, docs, gallery_dir)
    _png(docs / "_static" / "examples" / "other" / "plot.png")
    page = docs / "example_pages" / "demo.rst"
    page.write_text(
        page.read_text().replace(
            "/_static/examples/demo/plot.png", "/_static/examples/other/plot.png"
        )
    )
    assert "figure_not_referenced" in _codes(
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    )


@pytest.mark.unit
def test_gate_accepts_a_current_authored_page(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _add_authored_fixture(root, docs, gallery_dir)
    assert gallery.check_gallery(root / "examples", gallery_dir, docs, root=root) == []


@pytest.mark.unit
def test_gate_rejects_a_mismatched_included_script(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _add_authored_fixture(root, docs, gallery_dir)
    page = docs / "example_pages" / "demo.rst"
    page.write_text(
        page.read_text().replace("/examples/s/plot_01_demo.py", "/examples/s/other.py")
    )
    assert "included_source_mismatch" in _codes(
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    )


@pytest.mark.unit
def test_gate_rejects_a_missing_summary_field(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    manifest = json.loads((docs / "gallery_figures.json").read_text())
    manifest["examples"]["s/plot_01_demo"]["run_summary"] = {
        "path": "gallery_provenance/demo.json",
        "required_fields": ["metric.value"],
    }
    _write(docs / "gallery_figures.json", json.dumps(manifest))
    _write(docs / "gallery_provenance" / "demo.json", json.dumps({"metric": {}}))
    assert "missing_summary_field" in _codes(
        gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    )
    _write(
        docs / "gallery_provenance" / "demo.json",
        json.dumps({"metric": {"value": 1}}),
    )
    assert gallery.check_gallery(root / "examples", gallery_dir, docs, root=root) == []
