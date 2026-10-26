"""Gallery backend classification, execution patterns and staleness check."""

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
def test_manifest_covers_every_example_and_only_04_has_no_figures():
    manifest = gallery.load_manifest(ROOT / "docs")
    entries = manifest["examples"]
    keys = {
        gallery.example_key(path)
        for path in gallery.gallery_examples(ROOT / "examples")
    }
    assert set(entries) == keys
    zero_figure = {
        key for key, entry in entries.items() if entry.get("intentional_zero_figures")
    }
    assert zero_figure == {"getting_started/plot_04_custom_config"}


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


def _png(path: Path, color: tuple[int, int, int] = (200, 40, 40)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color).save(path)
    return path


def _placeholder(path: Path) -> Path:
    from sphinx_gallery.gen_gallery import glr_path_static
    from sphinx_gallery.utils import scale_image

    path.parent.mkdir(parents=True, exist_ok=True)
    scale_image(Path(glr_path_static()) / "no_image.png", path, 400, 280)
    return path


def _gallery_fixture(
    tmp_path: Path,
    *,
    figure_count: int = 1,
    embed: bool = True,
    create_images: bool = True,
    thumbnail: str = "real",
    manifest_extra: dict | None = None,
    summary: dict | None = None,
    page_extra: str = "",
):
    """Build one synthetic example, gallery page, manifest and provenance."""

    root = tmp_path
    section = "s"
    stem = "plot_01_demo"
    example = _write(root / "examples" / section / f"{stem}.py", "x = 1\n")
    docs = root / "docs"
    base = docs / "auto_examples" / section
    _write(base / f"{stem}.py.md5", get_md5sum(example, mode="t"))

    blocks = []
    for number in range(1, figure_count + 1):
        name = f"sphx_glr_{stem}_{number:03d}.png"
        if create_images:
            _png(base / "images" / name)
        if embed:
            blocks.append(
                f".. image-sg:: /auto_examples/{section}/images/{name}\n"
                f"   :alt: panel {number}\n"
            )
    _write(base / f"{stem}.rst", f"Title\n=====\n\n{''.join(blocks)}{page_extra}")
    _write(base / f"{stem}.py", "x = 1\n")
    _write(base / f"{stem}.ipynb", "{}")
    with zipfile.ZipFile(base / f"{stem}.zip", "w") as archive:
        archive.writestr(f"{stem}.py", "x = 1\n")

    thumb = base / "images" / "thumb" / f"sphx_glr_{stem}_thumb.png"
    if thumbnail == "real":
        _png(thumb)
    elif thumbnail == "placeholder":
        _placeholder(thumb)

    roles = [
        {
            "role": f"panel{number}",
            "accepts": [f"sphx_glr_{stem}_{number:03d}.png"],
        }
        for number in range(1, figure_count + 1)
    ]
    entry = {
        "figures": roles,
        "figure_count": figure_count,
        "thumbnail": (
            "placeholder"
            if manifest_extra and manifest_extra.get("intentional_zero_figures")
            else "real"
        ),
    }
    if manifest_extra:
        entry.update(manifest_extra)
    _write(
        docs / "gallery_figures.json",
        json.dumps({"schema": 1, "examples": {f"{section}/{stem}": entry}}),
    )

    if summary is not None:
        _write(docs / "gallery_provenance" / "demo.json", json.dumps(summary))

    gallery.record_provenance(root)
    return root, example, docs, docs / "auto_examples"


def _codes(problems) -> set[str]:
    return {problem.code for problem in problems}


@pytest.mark.unit
def test_gate_rejects_current_md5_with_a_missing_image(tmp_path):
    root, example, docs, gallery_dir = _gallery_fixture(tmp_path, create_images=False)
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "missing_image" in _codes(problems)
    assert gallery.stale_examples(root / "examples", gallery_dir, root=root) == []


@pytest.mark.unit
def test_gate_rejects_figures_that_are_saved_but_not_embedded(tmp_path):
    from PIL import Image

    root, _, docs, gallery_dir = _gallery_fixture(tmp_path, embed=False)
    saved = next((gallery_dir / "s" / "images").glob("sphx_glr_*_001.png"))
    assert Image.open(saved).size
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "no_embedded_figures" in _codes(problems)
    assert "unembedded_figure" in _codes(problems)


@pytest.mark.unit
def test_gate_rejects_a_corrupt_png(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    corrupt = gallery_dir / "s" / "images" / "sphx_glr_plot_01_demo_001.png"
    corrupt.write_bytes(b"this is not a png")
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "corrupt_image" in _codes(problems)


@pytest.mark.unit
def test_gate_rejects_a_placeholder_thumbnail(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path, thumbnail="placeholder")
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "placeholder_thumbnail" in _codes(problems)


@pytest.mark.unit
def test_gate_accepts_an_intentional_zero_figure_page(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(
        tmp_path,
        figure_count=0,
        thumbnail="placeholder",
        manifest_extra={"intentional_zero_figures": True},
    )
    assert gallery.check_gallery(root / "examples", gallery_dir, docs, root=root) == []


@pytest.mark.unit
def test_gate_rejects_a_changed_shared_dependency(tmp_path):
    root, example, docs, gallery_dir = _gallery_fixture(tmp_path)
    _write(root / "simind_python_connector" / "helpers.py", "VALUE = 1\n")
    assert gallery.stale_examples(root / "examples", gallery_dir, root=root) == [
        example
    ]
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "dependency_changed" in _codes(problems)


@pytest.mark.unit
def test_gate_rejects_a_failed_backend_build(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    gallery.record_provenance(
        root, build_failed=True, error="RuntimeError: SIMIND executable not found"
    )
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "build_failed" in _codes(problems)


@pytest.mark.unit
def test_gate_rejects_missing_run_summary_metadata(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(
        tmp_path,
        manifest_extra={
            "run_summary": {
                "path": "gallery_provenance/demo.json",
                "required_fields": ["engine.version"],
            }
        },
        summary={"status": "accepted", "gates": {}},
    )
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "missing_summary_field" in _codes(problems)


@pytest.mark.unit
def test_gate_rejects_a_quantitative_claim_with_a_failed_native_gate(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(
        tmp_path,
        manifest_extra={
            "run_summary": {
                "path": "gallery_provenance/demo.json",
                "required_fields": ["gates.native_precision_passed"],
                "statistical_claims": {
                    "quantitative": True,
                    "required_gates": ["native_precision_passed"],
                },
            }
        },
        summary={
            "status": "accepted",
            "gates": {"native_precision_passed": False},
        },
    )
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "statistical_gate_failed" in _codes(problems)


@pytest.mark.unit
def test_gate_accepts_a_valid_explicit_negative_result(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(
        tmp_path,
        page_extra="This page reports an audited negative result.\n",
        manifest_extra={
            "run_summary": {
                "path": "gallery_provenance/demo.json",
                "required_fields": [
                    "status",
                    "gates.native_precision_passed",
                ],
                "statistical_claims": {
                    "quantitative": False,
                    "negative_result": True,
                    "required_gates": ["native_precision_passed"],
                    "page_markers": ["audited negative result"],
                },
            }
        },
        summary={
            "status": "quality_not_established",
            "gates": {"native_precision_passed": False},
        },
    )
    assert gallery.check_gallery(root / "examples", gallery_dir, docs, root=root) == []


@pytest.mark.unit
def test_gate_rejects_a_negative_result_relabelled_as_accepted(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(
        tmp_path,
        page_extra="This page reports an audited negative result.\n",
        manifest_extra={
            "run_summary": {
                "path": "gallery_provenance/demo.json",
                "required_fields": ["status"],
                "statistical_claims": {
                    "quantitative": False,
                    "negative_result": True,
                    "required_gates": ["native_precision_passed"],
                    "page_markers": ["audited negative result"],
                },
            }
        },
        summary={
            "status": "accepted",
            "gates": {"native_precision_passed": True},
        },
    )
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "negative_result_mislabelled" in _codes(problems)
    assert "negative_result_without_failure" in _codes(problems)


@pytest.mark.unit
def test_gate_rejects_a_negative_page_relabelled_quantitatively(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(
        tmp_path,
        page_extra=(
            "This page reports an audited negative result. "
            "The result is quantitatively validated.\n"
        ),
        manifest_extra={
            "run_summary": {
                "path": "gallery_provenance/demo.json",
                "required_fields": ["status"],
                "statistical_claims": {
                    "quantitative": False,
                    "negative_result": True,
                    "required_gates": ["native_precision_passed"],
                    "page_markers": ["audited negative result"],
                    "forbidden_page_markers": ["quantitatively validated"],
                },
            }
        },
        summary={
            "status": "quality_not_established",
            "gates": {"native_precision_passed": False},
        },
    )
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "negative_result_mislabelled" in _codes(problems)


@pytest.mark.unit
def test_html_gate_requires_expected_figures_and_working_links(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    html_root = tmp_path / "html"
    html_page = html_root / "auto_examples" / "s" / "plot_01_demo.html"
    html_page.parent.mkdir(parents=True, exist_ok=True)
    image = gallery_dir / "s" / "images" / "sphx_glr_plot_01_demo_001.png"
    html_page.write_text(
        f'<img src="../../../docs/auto_examples/s/images/{image.name}">'
    )
    assert (
        gallery.check_gallery(
            root / "examples",
            gallery_dir,
            docs,
            root=root,
            html_dir=html_root,
        )
        == []
    )
    html_page.write_text('<img src="images/missing.png">')
    problems = gallery.check_gallery(
        root / "examples",
        gallery_dir,
        docs,
        root=root,
        html_dir=html_root,
    )
    assert "broken_html_image" in _codes(problems)
    assert "html_missing_figure" in _codes(problems)
