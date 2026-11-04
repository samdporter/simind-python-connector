"""Gallery backend classification, execution patterns and staleness check."""

import importlib.util
import re
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
def test_every_real_gallery_example_has_one_backend():
    for path in gallery.gallery_examples(ROOT / "examples"):
        backend = gallery.example_backend(path)
        if path.parent.name == "getting_started":
            assert backend == "python"


@pytest.mark.unit
def test_real_data_examples_run_in_exactly_one_backend_build():
    examples = ROOT / "examples"
    patterns = {
        backend: gallery.execution_pattern(examples, backend)
        for backend in gallery.BACKENDS
    }
    paths = [
        path
        for path in gallery.gallery_examples(examples)
        if path.parent.name == "real_data"
    ]
    assert paths
    for path in paths:
        path_str = str(path)
        matched = [
            name for name, pattern in patterns.items() if re.search(pattern, path_str)
        ]
        assert matched == [gallery.example_backend(path)], path.name


@pytest.mark.unit
def test_sphinx_gallery_publishes_figures_that_are_not_closed(tmp_path):
    """A saved-but-open figure must become a page image, not only a thumbnail."""
    pytest.importorskip("sphinx")
    pytest.importorskip("sphinx_gallery")
    from sphinx.application import Sphinx

    src = tmp_path / "docs"
    examples = src / "examples"
    _write(examples / "GALLERY_HEADER.rst", "Gallery examples\n================\n")
    _write(
        examples / "plot_demo.py",
        "\n".join(
            [
                '"""Synthetic gallery figure lifecycle check."""',
                "",
                "import numpy as np",
                "import matplotlib",
                "matplotlib.use('Agg')",
                "import matplotlib.pyplot as plt",
                "figure, axis = plt.subplots()",
                "axis.imshow(np.arange(16).reshape(4, 4))",
                "figure.savefig('demo_panel.png')",
                "figure, axis = plt.subplots()",
                "axis.plot(np.arange(8))",
                "figure.savefig('demo_curve.png')",
            ]
        ),
    )
    _write(
        src / "conf.py",
        "\n".join(
            [
                "extensions = ['sphinx_gallery.gen_gallery']",
                "sphinx_gallery_conf = {",
                "    'examples_dirs': ['examples'],",
                "    'gallery_dirs': ['auto_examples'],",
                "    'filename_pattern': r'.*',",
                "    'plot_gallery': True,",
                "    'image_scrapers': ('matplotlib',),",
                "    'abort_on_example_error': True,",
                "}",
                "project = 'gallery regression'",
                "master_doc = 'index'",
            ]
        ),
    )
    _write(src / "index.rst", "Gallery\n=======\n")
    app = Sphinx(
        srcdir=str(src),
        confdir=str(src),
        outdir=str(tmp_path / "html"),
        doctreedir=str(tmp_path / "doctrees"),
        buildername="html",
        freshenv=True,
        warningiserror=False,
        status=None,
    )
    app.build()
    page = (src / "auto_examples" / "plot_demo.rst").read_text()
    assert ".. image-sg::" in page
    assert "sphx_glr_plot_demo_001.png" in page
    assert "sphx_glr_plot_demo_002.png" in page
    assert "_thumb.png" not in page


@pytest.mark.unit
def test_y90_scatter_page_publishes_non_thumbnail_reconstruction_figures():
    """The committed page must reference the figures, not only the thumbnail."""
    page = ROOT / "docs" / "auto_examples" / "real_data" / "plot_18_y90_mc_scatter.rst"
    images = re.findall(r"image-sg:: (\S+)", page.read_text())
    assert len(images) >= 6, images
    assert all("thumb" not in image for image in images)
