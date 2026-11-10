"""Helpers for the sphinx-gallery configuration and the documentation gate.

The gate is deliberately static: it reads the committed gallery output and the
explicit page inventory, then checks the rendered HTML when a build is
available.  It never executes an example, so the offline documentation build
can run it without SIMIND, STIR, SIRF, PyTomography or the restricted Y-90
data.
"""

from __future__ import annotations

import ast
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple

from docutils import nodes
from docutils.core import publish_doctree
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives.images import Image
from sphinx_gallery.utils import get_md5sum


BACKENDS = ("python", "stir", "sirf", "pytomography")
GENERATED_SECTIONS = ("mc_scatter", "cil", "real_data")
GALLERY_IGNORE_PATTERN = r"^(?!plot_)[^/]*\.py$|^plot_0[1-9].*\.py$|^plot_11_.*\.py$"
MANIFEST_NAME = "gallery_figures.json"

_ADAPTOR_ROOTS = {
    "sirf": "sirf",
    "cil": "sirf",
    "stir": "stir",
    "pytomography": "pytomography",
}
_ANCHOR_TAG = re.compile(r"<a\b([^>]*)>")
_IMAGE_TAG = re.compile(r"<img\b([^>]*)>")
_ATTR = re.compile(r"""([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_LITERALINCLUDE = re.compile(r"^\s*\.\.\s+literalinclude::\s+(\S+)\s*$", re.M)
_DIRECTIVES_REGISTERED = False


class _OnlyDirective(Directive):
    """Pass-through for Sphinx's ``only`` wrapper in generated pages."""

    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True

    def run(self):
        container = nodes.container()
        self.state.nested_parse(self.content, self.content_offset, container)
        return container.children


class _ImageSGDirective(Image):
    """Sphinx-Gallery's image directive, including its ``srcset`` option."""

    option_spec = dict(Image.option_spec, srcset=directives.unchanged)


class _ImgCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            source = dict(attrs).get("src")
            if source:
                self.sources.append(source)


def _register_directives() -> None:
    global _DIRECTIVES_REGISTERED
    if _DIRECTIVES_REGISTERED:
        return
    directives.register_directive("only", _OnlyDirective)
    directives.register_directive("image-sg", _ImageSGDirective)
    _DIRECTIVES_REGISTERED = True


def _attributes(raw: str) -> dict[str, str]:
    return {
        match.group(1): (
            match.group(2) if match.group(2) is not None else match.group(3)
        )
        for match in _ATTR.finditer(raw)
    }


def _resolved_image_target(raw: str, page: Path, docs_dir: Path) -> Path | None:
    target = raw.split("?")[0].split("#")[0]
    if not target or target.startswith(("http:", "https:", "data:")):
        return None
    if target.startswith("/"):
        return (docs_dir / target.lstrip("/")).resolve()
    return (page.parent / target).resolve()


def _raw_html_images(raw_text: str) -> list[str]:
    parser = _ImgCollector()
    parser.feed(raw_text)
    parser.close()
    return parser.sources


def _referenced_images(page: Path, docs_dir: Path) -> set[Path]:
    """Return resolved targets of image nodes and HTML raw image nodes."""

    _register_directives()
    document = publish_doctree(
        page.read_text(encoding="utf-8"),
        settings_overrides={
            "report_level": 5,
            "halt_level": 5,
            "file_insertion_enabled": False,
            "raw_enabled": True,
        },
    )
    targets = set()
    for node in document.findall(nodes.image):
        target = _resolved_image_target(node.get("uri", ""), page, docs_dir)
        if target is not None:
            targets.add(target)
    for node in document.findall(nodes.raw):
        formats = str(node.get("format", "")).lower().split()
        if "html" not in formats:
            continue
        for source in _raw_html_images(node.astext()):
            target = _resolved_image_target(source, page, docs_dir)
            if target is not None:
                targets.add(target)
    return targets


class GalleryProblem(NamedTuple):
    """A single reason the documentation inventory is not consistent."""

    example: str
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.example}: {self.code}: {self.detail}"


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def example_backend(path: Path) -> str:
    roots = _import_roots(path)
    backends = {_ADAPTOR_ROOTS[root] for root in roots & _ADAPTOR_ROOTS.keys()}
    if len(backends) > 1:
        raise ValueError(f"{path.name} imports more than one backend: {sorted(roots)}")
    return backends.pop() if backends else "python"


def _walk_plot_sources(directory: Path, seen: set[tuple[int, int]]):
    """Walk directories, following symlinks once and pruning cycles."""

    try:
        stat = directory.stat()
    except OSError:
        return
    identity = (stat.st_dev, stat.st_ino)
    if identity in seen:
        return
    seen.add(identity)
    try:
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
    except OSError:
        return
    for entry in entries:
        if entry.is_dir():
            yield from _walk_plot_sources(entry, seen)
        elif (
            entry.is_file() and entry.name.startswith("plot_") and entry.suffix == ".py"
        ):
            yield entry


def gallery_examples(examples_dir: Path) -> list[Path]:
    return sorted(_walk_plot_sources(examples_dir, set()))


def example_key(path: Path) -> str:
    """Return the manifest key for a gallery source file."""

    return f"{path.parent.name}/{path.stem}"


def source_key(path: Path, examples_dir: Path) -> str:
    """Return the repository-relative key for a gallery source."""

    return path.relative_to(examples_dir).with_suffix("").as_posix()


def gallery_ignores(path: Path) -> bool:
    """Match Sphinx-Gallery's filename-only ignore rule."""

    return re.search(GALLERY_IGNORE_PATTERN, path.name) is not None


def execution_pattern(examples_dir: Path, backend: str) -> str:
    paths = [
        path
        for path in gallery_examples(examples_dir)
        if example_backend(path) == backend
    ]
    if not paths:
        return r"^$"
    names = "|".join(
        path.relative_to(examples_dir).as_posix().replace(".", r"\.") for path in paths
    )
    return f"/examples/({names})$"


def load_manifest(docs_dir: Path) -> dict:
    path = docs_dir / MANIFEST_NAME
    if not path.exists():
        return {"examples": {}, "authored_pages": {}}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("examples", {})
    manifest.setdefault("authored_pages", {})
    return manifest


def _declared_figures(entry: dict) -> list[str]:
    """Return the image filenames an inventory entry declares.

    The current schema stores plain filenames.  Role dictionaries from the old
    gallery manifest are accepted so a page can still be checked while its
    entry is being migrated.
    """

    figures = []
    for item in entry.get("figures", []):
        if isinstance(item, str):
            figures.append(item)
        elif isinstance(item, dict):
            figures.extend(item.get("accepts", []))
    return figures


def _image_problem(path: Path) -> str | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except Exception as error:  # pragma: no cover - message is asserted in tests
        return f"{type(error).__name__}: {error}"
    return None


def _check_figure(path: Path, key: str, label: str) -> list[GalleryProblem]:
    if not path.exists():
        return [GalleryProblem(key, "missing_image", label)]
    problem = _image_problem(path)
    if problem:
        return [GalleryProblem(key, "corrupt_image", f"{label}: {problem}")]
    return []


def _check_downloads(
    example: Path, base: Path, stem: str, key: str
) -> list[GalleryProblem]:
    """Check the generated script download and archive against the source."""

    problems = []
    source = example.read_bytes()
    python = base / f"{stem}.py"
    if not python.exists():
        problems.append(GalleryProblem(key, "download_missing", python.name))
    elif python.read_bytes() != source:
        problems.append(GalleryProblem(key, "download_source_mismatch", python.name))

    archive_path = base / f"{stem}.zip"
    if not archive_path.exists():
        problems.append(GalleryProblem(key, "download_missing", archive_path.name))
    else:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                if f"{stem}.py" not in names:
                    problems.append(
                        GalleryProblem(
                            key,
                            "download_corrupt",
                            f"{archive_path.name} has no source",
                        )
                    )
                elif archive.read(f"{stem}.py") != source:
                    problems.append(
                        GalleryProblem(
                            key,
                            "download_source_mismatch",
                            f"{archive_path.name}:{stem}.py",
                        )
                    )
        except zipfile.BadZipFile as error:
            problems.append(
                GalleryProblem(key, "download_corrupt", f"{archive_path.name}: {error}")
            )
    return problems


def _get_field(payload: dict, dotted: str):
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return _MISSING if value is None else value


_MISSING = object()


def _check_run_summary(entry: dict, docs_dir: Path, key: str) -> list[GalleryProblem]:
    """Check the small numerical table a retained page reports."""

    block = entry.get("run_summary")
    if not block:
        return []
    summary_path = docs_dir / block["path"]
    if not summary_path.exists():
        return [
            GalleryProblem(
                key,
                "missing_run_summary",
                summary_path.relative_to(docs_dir).as_posix(),
            )
        ]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as error:
        return [GalleryProblem(key, "corrupt_run_summary", str(error))]

    problems = []
    for field in block.get("required_fields", []):
        if _get_field(summary, field) is _MISSING:
            problems.append(GalleryProblem(key, "missing_summary_field", field))

    for field, expected in block.get("expected_values", {}).items():
        actual = _get_field(summary, field)
        if actual is _MISSING:
            problems.append(GalleryProblem(key, "missing_summary_field", field))
        elif actual != expected:
            problems.append(
                GalleryProblem(
                    key,
                    "contradictory_run_summary",
                    f"{field} is {actual!r}, expected {expected!r}",
                )
            )

    for left, right in block.get("consistent_fields", []):
        left_value = _get_field(summary, left)
        right_value = _get_field(summary, right)
        if left_value is _MISSING:
            problems.append(GalleryProblem(key, "missing_summary_field", left))
        elif right_value is _MISSING:
            problems.append(GalleryProblem(key, "missing_summary_field", right))
        elif left_value != right_value:
            problems.append(
                GalleryProblem(
                    key,
                    "contradictory_run_summary",
                    f"{left} is {left_value!r}, {right} is {right_value!r}",
                )
            )
    return problems


def _html_targets(page: Path) -> list[tuple[str, str]]:
    """Return ``(kind, target)`` for each local link and image on a page."""

    text = page.read_text(encoding="utf-8", errors="replace")
    targets = []
    for tag, attributes in (
        ("link", _ANCHOR_TAG.findall(text)),
        ("image", _IMAGE_TAG.findall(text)),
    ):
        for raw in attributes:
            attrs = _attributes(raw)
            target = attrs.get("href") if tag == "link" else attrs.get("src")
            if target:
                targets.append((tag, target))
    return targets


def _broken_html_targets(page: Path, label: str) -> list[GalleryProblem]:
    problems = []
    for kind, source in _html_targets(page):
        if source.startswith(
            ("http:", "https:", "data:", "mailto:", "javascript:", "tel:", "#")
        ):
            continue
        target = (page.parent / source.split("?")[0].split("#")[0]).resolve()
        if not target.exists():
            code = "broken_html_link" if kind == "link" else "broken_html_image"
            problems.append(GalleryProblem(label, code, source))
    return problems


def _check_generated_page(
    example: Path,
    entry: dict,
    gallery_dir: Path,
) -> list[GalleryProblem]:
    key = example_key(example)
    section, stem = key.split("/", 1)
    page_dir = gallery_dir / section
    problems = []

    md5_file = page_dir / f"{example.name}.md5"
    current_md5 = get_md5sum(example, mode="t")
    if not md5_file.exists() or md5_file.read_text() != current_md5:
        problems.append(
            GalleryProblem(
                key, "source_stale", "gallery page does not match the current source"
            )
        )

    rst = page_dir / f"{stem}.rst"
    if not rst.exists():
        problems.append(GalleryProblem(key, "page_missing", rst.name))
    for name in _declared_figures(entry):
        problems.extend(
            _check_figure(page_dir / "images" / name, key, f"images/{name}")
        )

    if rst.exists():
        referenced = _referenced_images(rst, gallery_dir.parent)
        for name in _declared_figures(entry):
            expected = (page_dir / "images" / name).resolve()
            if expected not in referenced:
                problems.append(GalleryProblem(key, "figure_not_referenced", name))
        problems.extend(_check_downloads(example, page_dir, stem, key))
        problems.extend(_check_run_summary(entry, gallery_dir.parent, key))
    return problems


def _literalinclude_paths(page: Path, docs_dir: Path) -> list[Path]:
    paths = []
    for raw in _LITERALINCLUDE.findall(page.read_text(encoding="utf-8")):
        if raw.startswith("/"):
            paths.append((docs_dir / raw.lstrip("/")).resolve())
        else:
            paths.append((page.parent / raw).resolve())
    return paths


def _check_authored_page(key: str, entry: dict, docs_dir: Path) -> list[GalleryProblem]:
    problems = []
    page = docs_dir / f"{key}.rst"
    if not page.exists():
        return [GalleryProblem(key, "page_missing", f"{key}.rst")]

    source = entry.get("source")
    if source is not None:
        source_path = (docs_dir / ".." / source).resolve()
        if not source_path.exists():
            problems.append(GalleryProblem(key, "source_missing", source))
        else:
            includes = _literalinclude_paths(page, docs_dir)
            if source_path not in includes:
                problems.append(
                    GalleryProblem(
                        key,
                        "included_source_mismatch",
                        f"literalinclude does not use {source}",
                    )
                )

    referenced = _referenced_images(page, docs_dir)
    for name in _declared_figures(entry):
        figure = (docs_dir / name).resolve()
        problems.extend(_check_figure(figure, key, name))
        if figure not in referenced:
            problems.append(GalleryProblem(key, "figure_not_referenced", name))
    return problems


def _check_html(
    html_dir: Path,
    examples: list[Path],
    authored: dict,
) -> list[GalleryProblem]:
    problems = []
    gallery_root = html_dir / "auto_examples"
    for example in examples:
        key = example_key(example)
        section, stem = key.split("/", 1)
        page = gallery_root / section / f"{stem}.html"
        if not page.exists():
            problems.append(GalleryProblem(key, "html_page_missing", page.name))
        else:
            problems.extend(_broken_html_targets(page, key))

    for key in sorted(authored):
        page = html_dir / f"{key}.html"
        if not page.exists():
            problems.append(GalleryProblem(key, "html_page_missing", f"{key}.html"))
        else:
            problems.extend(_broken_html_targets(page, key))

    for page in sorted(gallery_root.rglob("index.html")):
        problems.extend(
            _broken_html_targets(page, page.relative_to(html_dir).as_posix())
        )
    return problems


def check_gallery(
    examples_dir: Path,
    gallery_dir: Path,
    docs_dir: Path,
    backends: tuple[str, ...] = BACKENDS,
    *,
    root: Path | None = None,
    html_dir: Path | None = None,
) -> list[GalleryProblem]:
    """Check inventory, figures, downloads and rendered links without running."""

    del root  # retained for callers that still pass the repository root
    manifest = load_manifest(docs_dir)
    entries = manifest["examples"]
    authored = manifest["authored_pages"]
    all_examples = gallery_examples(examples_dir)
    problems: list[GalleryProblem] = []
    checked_examples = []

    # Coverage is independent of the requested backend: every source that
    # Sphinx-Gallery would generate needs an inventory entry.
    for example in all_examples:
        generated = example.relative_to(examples_dir).parts[
            0
        ] in GENERATED_SECTIONS and not gallery_ignores(example)
        key = source_key(example, examples_dir)
        entry = entries.get(key)
        if generated and entry is None:
            source = example.relative_to(examples_dir.parent).as_posix()
            problems.append(
                GalleryProblem(
                    key,
                    "manifest_missing",
                    f"{source} has no inventory entry",
                )
            )
            continue
        if entry is None:
            continue
        if example_backend(example) not in backends:
            continue
        checked_examples.append(example)
        problems.extend(_check_generated_page(example, entry, gallery_dir))

    all_keys = {source_key(path, examples_dir) for path in all_examples}
    for key in sorted(entries):
        if key not in all_keys:
            problems.append(
                GalleryProblem(key, "manifest_orphan", "no matching example source")
            )

    for key, entry in sorted(authored.items()):
        problems.extend(_check_authored_page(key, entry, docs_dir))

    authored_dir = docs_dir / "example_pages"
    for page in sorted(authored_dir.glob("*.rst")):
        if page.stem == "index":
            continue
        key = f"example_pages/{page.stem}"
        if key not in authored:
            problems.append(
                GalleryProblem(key, "manifest_missing", "no inventory entry")
            )

    if html_dir is not None:
        problems.extend(_check_html(html_dir, checked_examples, authored))
    return problems


def stale_examples(
    examples_dir: Path,
    gallery_dir: Path,
    backends: tuple[str, ...] = BACKENDS,
    *,
    root: Path | None = None,
) -> list[Path]:
    """Return examples whose source changed after their gallery page was built."""

    del root  # retained for callers that still pass the repository root
    entries = load_manifest(gallery_dir.parent).get("examples", {})
    stale = []
    for path in gallery_examples(examples_dir):
        if entries and example_key(path) not in entries:
            continue
        if example_backend(path) not in backends:
            continue
        md5_file = gallery_dir / path.parent.name / f"{path.name}.md5"
        source_md5 = get_md5sum(path, mode="t")
        if not md5_file.exists() or md5_file.read_text() != source_md5:
            stale.append(path)
    return stale
