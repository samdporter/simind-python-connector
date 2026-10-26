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


def _notebook(source: str) -> str:
    """A structurally valid notebook holding the example's code as one cell."""

    return json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [source.strip()],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


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
    section: str = "s",
    stem: str = "plot_01_demo",
    source: str = "x = 1\n",
    panels: list[tuple[int, str]] | None = None,
    roles: list[dict] | None = None,
    manifest_count: int | list[int] | None = None,
):
    """Build one synthetic example, gallery page, manifest and provenance.

    ``panels`` lists ``(figure number, alt text)`` pairs to embed; ``roles``
    and ``manifest_count`` override the generated figure expectations.
    """

    root = tmp_path
    if panels is None:
        panels = [(number, f"panel {number}") for number in range(1, figure_count + 1)]
    docs = root / "docs"
    base = docs / "auto_examples" / section
    example = _write(root / "examples" / section / f"{stem}.py", source)
    _write(base / f"{stem}.py.md5", get_md5sum(example, mode="t"))

    blocks = []
    for number, alt in panels:
        name = f"sphx_glr_{stem}_{number:03d}.png"
        if create_images:
            _png(base / "images" / name)
        if embed:
            blocks.append(
                f".. image-sg:: /auto_examples/{section}/images/{name}\n"
                f"   :alt: {alt}\n"
            )
    _write(base / f"{stem}.rst", f"Title\n=====\n\n{''.join(blocks)}{page_extra}")
    _write(base / f"{stem}.py", source)
    _write(base / f"{stem}.ipynb", _notebook(source))
    with zipfile.ZipFile(base / f"{stem}.zip", "w") as archive:
        archive.writestr(f"{stem}.py", source)

    thumb = base / "images" / "thumb" / f"sphx_glr_{stem}_thumb.png"
    if thumbnail == "real":
        _png(thumb)
    elif thumbnail == "placeholder":
        _placeholder(thumb)

    if roles is None:
        roles = [
            {
                "role": f"panel{number}",
                "accepts": [f"sphx_glr_{stem}_{number:03d}.png"],
            }
            for number, _ in panels
        ]
    entry = {
        "figures": roles,
        "figure_count": figure_count if manifest_count is None else manifest_count,
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


@pytest.mark.unit
def test_cached_backend_build_keeps_the_stale_dependency_digest(tmp_path):
    root, example, docs, gallery_dir = _gallery_fixture(tmp_path)
    _write(root / "simind_python_connector" / "helpers.py", "VALUE = 1\n")
    # Sphinx-gallery reused its cached output, so nothing was executed: the
    # recorded digest must stay stale and the gate must keep rejecting.
    gallery.record_gallery_build(root, docs, "python", None, executed=set())
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "dependency_changed" in _codes(problems)


@pytest.mark.unit
def test_prepared_backend_build_owns_the_examples_it_reexecutes(tmp_path):
    root, example, docs, gallery_dir = _gallery_fixture(tmp_path)
    sidecar = gallery_dir / "s" / f"{example.name}.md5"
    assert gallery.prepare_backend_build(root, docs, ("python",)) == set()
    assert sidecar.exists()

    _write(root / "simind_python_connector" / "helpers.py", "VALUE = 1\n")
    executed = gallery.prepare_backend_build(root, docs, ("python",))
    assert executed == {gallery.example_key(example)}
    assert not sidecar.exists()
    assert gallery.load_build_plan(root, docs, "python") == executed

    # The build executed it (sphinx-gallery rewrites the sidecar), so the
    # provenance the plan refreshes lets the gate pass again.
    _write(sidecar, get_md5sum(example, mode="t"))
    gallery.record_gallery_build(root, docs, "python", None)
    assert not gallery.build_plan_path(docs).exists()
    assert gallery.check_gallery(root / "examples", gallery_dir, docs, root=root) == []


@pytest.mark.unit
def test_backend_build_without_a_prepare_plan_records_nothing(tmp_path):
    root, example, docs, gallery_dir = _gallery_fixture(tmp_path)
    _write(root / "simind_python_connector" / "helpers.py", "VALUE = 1\n")
    # No prepare step ran, so nothing proves the cached figures were rebuilt:
    # the build-finished hook must not stamp them with the new digest.
    gallery.record_gallery_build(root, docs, "python", None)
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "dependency_changed" in _codes(problems)


@pytest.mark.unit
def test_record_gallery_build_ignores_a_plan_moved_under_it(tmp_path):
    root, example, docs, gallery_dir = _gallery_fixture(tmp_path)
    _write(root / "simind_python_connector" / "helpers.py", "VALUE = 1\n")
    gallery.prepare_backend_build(root, docs, ("python",))
    # The shared dependency changes again while the build is running: the plan
    # no longer describes this tree, so it may not refresh anything.
    _write(root / "simind_python_connector" / "helpers.py", "VALUE = 2\n")
    gallery.record_gallery_build(root, docs, "python", None)
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "dependency_changed" in _codes(problems)


@pytest.mark.unit
@pytest.mark.parametrize(
    "content",
    [
        "{}",
        "[1, 2, 3]",
        '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
        '{"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# hi"]}],'
        ' "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
        '{"cells": [{"cell_type": "code", "metadata": {}, "source": ["x = 1"]}],'
        ' "metadata": {}, "nbformat": 3}',
        '{"cells": "x = 1", "metadata": {}, "nbformat": 4}',
    ],
)
def test_gate_rejects_an_invalid_notebook(tmp_path, content):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    assert gallery.check_gallery(root / "examples", gallery_dir, docs, root=root) == []
    _write(gallery_dir / "s" / "plot_01_demo.ipynb", content)
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "notebook_invalid" in _codes(problems)


@pytest.mark.unit
def test_gate_rejects_a_corrupt_notebook(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _write(gallery_dir / "s" / "plot_01_demo.ipynb", "not json at all")
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "download_corrupt" in _codes(problems)


@pytest.mark.unit
def test_gate_rejects_a_notebook_of_another_source(tmp_path):
    root, _, docs, gallery_dir = _gallery_fixture(tmp_path)
    _write(gallery_dir / "s" / "plot_01_demo.ipynb", _notebook("x = 2\n"))
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert "notebook_stale" in _codes(problems)


_18_KEY = "real_data/plot_18_y90_mc_scatter"

# The two layouts example 18 can produce: with or without the collapsed
# global-arm reconstruction panel.  The alt texts follow the expectations the
# committed manifest already declares for each role.
_18_LAYOUTS = {
    "global_arm_panel": {
        "panels": [
            (
                1,
                "CT attenuation (red) and vendor umap (green): 0.000 voxels "
                "(PASSED), axial k=64, coronal j=64, sagittal i=64",
            ),
            (
                2,
                "Frozen liver (green) and sphere (red) VOIs, axial k=64, "
                "coronal j=54, sagittal i=46",
            ),
            (
                3,
                "Reference, sum, Reference, trimmed, Corrected, CT-based, "
                "Corrected, SPECT-based",
            ),
            (
                4,
                "Global scaling — collapsed; diagnostic only, not "
                "quantitatively interpretable, Reference, sum",
            ),
            (5, "plot 18 y90 mc scatter"),
            (6, "plot 18 y90 mc scatter"),
            (7, "Profiles through the hot-sphere column, Reference, sum"),
        ],
        "required": {
            "registration": 1,
            "vois": 2,
            "reconstructions": 3,
            "ratios": 5,
            "scatter": 6,
            "profiles": 7,
        },
    },
    "no_global_arm_panel": {
        "panels": [
            (
                1,
                "CT attenuation (red) and vendor umap (green): 0.000 voxels "
                "(PASSED), axial k=64, coronal j=64, sagittal i=64",
            ),
            (
                2,
                "Frozen liver (green) and sphere (red) VOIs, axial k=64, "
                "coronal j=54, sagittal i=46",
            ),
            (
                3,
                "Reference, sum, Reference, trimmed, Corrected, CT-based, "
                "Corrected, SPECT-based",
            ),
            (4, "plot 18 y90 mc scatter"),
            (5, "plot 18 y90 mc scatter"),
            (6, "Profiles through the hot-sphere column, Reference, sum"),
        ],
        "required": {
            "registration": 1,
            "vois": 2,
            "reconstructions": 3,
            "ratios": 4,
            "scatter": 5,
            "profiles": 6,
        },
    },
}


def _gallery18(tmp_path: Path, layout: str, *, drop: int | None = None):
    """Build example 18 in one supported layout, optionally without a panel."""

    spec = _18_LAYOUTS[layout]
    manifest = gallery.load_manifest(ROOT / "docs")["examples"][_18_KEY]
    return _gallery_fixture(
        tmp_path,
        section="real_data",
        stem="plot_18_y90_mc_scatter",
        panels=[pair for pair in spec["panels"] if pair[0] != drop],
        roles=manifest["figures"],
        manifest_count=manifest["figure_count"],
    )


@pytest.mark.unit
@pytest.mark.parametrize("layout", sorted(_18_LAYOUTS))
def test_example18_supported_layouts_pass_the_gate(tmp_path, layout):
    root, _, docs, gallery_dir = _gallery18(tmp_path, layout)
    assert gallery.check_gallery(root / "examples", gallery_dir, docs, root=root) == []


@pytest.mark.unit
@pytest.mark.parametrize("layout", sorted(_18_LAYOUTS))
@pytest.mark.parametrize(
    "role",
    ["registration", "vois", "reconstructions", "ratios", "scatter", "profiles"],
)
def test_example18_gate_rejects_a_missing_required_panel(tmp_path, layout, role):
    drop = _18_LAYOUTS[layout]["required"][role]
    root, _, docs, gallery_dir = _gallery18(tmp_path, layout, drop=drop)
    problems = gallery.check_gallery(root / "examples", gallery_dir, docs, root=root)
    assert problems, f"the gate must reject layout {layout} without {role}"
    assert _codes(problems) & {
        "missing_required_figure",
        "ambiguous_figure_roles",
        "figure_count",
    }, _codes(problems)


def _html18(tmp_path: Path, layout: str, *, drop: int | None = None) -> Path:
    """Render example 18's HTML page with one optional panel dropped."""

    spec = _18_LAYOUTS[layout]
    page = (
        tmp_path
        / "html"
        / "auto_examples"
        / "real_data"
        / "plot_18_y90_mc_scatter.html"
    )
    tags = [
        f'<img src="../../../docs/auto_examples/real_data/images/'
        f'sphx_glr_plot_18_y90_mc_scatter_{number:03d}.png" alt="{alt}"/>'
        for number, alt in spec["panels"]
        if number != drop
    ]
    _write(page, "".join(tags))
    return page


@pytest.mark.unit
def test_html_gate_accepts_the_seven_panel_example18_page(tmp_path):
    root, _, docs, gallery_dir = _gallery18(tmp_path, "global_arm_panel")
    _html18(tmp_path, "global_arm_panel")
    assert (
        gallery.check_gallery(
            root / "examples",
            gallery_dir,
            docs,
            root=root,
            html_dir=tmp_path / "html",
        )
        == []
    )


@pytest.mark.unit
def test_html_gate_rejects_a_ratios_panel_missing_from_example18(tmp_path):
    root, _, docs, gallery_dir = _gallery18(tmp_path, "global_arm_panel")
    # Only the ratios image disappears from the rendered page: the optional
    # global reconstruction must not stand in for it.
    _html18(tmp_path, "global_arm_panel", drop=5)
    problems = gallery.check_gallery(
        root / "examples",
        gallery_dir,
        docs,
        root=root,
        html_dir=tmp_path / "html",
    )
    assert _codes(problems) & {
        "html_missing_figure",
        "html_ambiguous_figure_roles",
    }, _codes(problems)
    assert any("ratios" in problem.detail for problem in problems)
