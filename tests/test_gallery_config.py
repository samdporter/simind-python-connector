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
