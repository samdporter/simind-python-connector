"""Helpers for the authored documentation inventory and its static check.

The check reads the committed inventory and the rendered HTML when available.
It never executes an example, so it needs no SIMIND, backend or restricted
Y-90 data.
"""

from __future__ import annotations

import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple

from docutils import nodes
from docutils.core import publish_doctree


MANIFEST_NAME = "example_pages.json"
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
_MISSING = object()


class DocsProblem(NamedTuple):
    """A single reason the authored inventory is inconsistent."""

    page: str
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.page}: {self.code}: {self.detail}"


def example_backend(path: Path) -> str:
    """Return the single backend an example imports, or ``python``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    backends = {_ADAPTOR_ROOTS[root] for root in roots & _ADAPTOR_ROOTS.keys()}
    if len(backends) > 1:
        raise ValueError(f"{path.name} imports more than one backend: {sorted(roots)}")
    return backends.pop() if backends else "python"


def load_manifest(docs_dir: Path) -> dict:
    path = docs_dir / MANIFEST_NAME
    if not path.exists():
        return {"authored_pages": {}}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("authored_pages", {})
    return manifest


def _declared_figures(entry: dict) -> list[str]:
    return [item for item in entry.get("figures", []) if isinstance(item, str)]


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


def _check_figure(path: Path, key: str, label: str) -> list[DocsProblem]:
    if not path.exists():
        return [DocsProblem(key, "missing_image", label)]
    problem = _image_problem(path)
    if problem:
        return [DocsProblem(key, "corrupt_image", f"{label}: {problem}")]
    return []


class _ImgCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            source = dict(attrs).get("src")
            if source:
                self.sources.append(source)


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


def _referenced_images(page: Path, docs_dir: Path) -> set[Path]:
    """Return targets of real image/figure directives and HTML raw images."""
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
        if "html" not in str(node.get("format", "")).lower().split():
            continue
        parser = _ImgCollector()
        parser.feed(node.astext())
        parser.close()
        for source in parser.sources:
            target = _resolved_image_target(source, page, docs_dir)
            if target is not None:
                targets.add(target)
    return targets


def _get_field(payload: dict, dotted: str):
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return _MISSING if value is None else value


def _check_run_summary(entry: dict, docs_dir: Path, key: str) -> list[DocsProblem]:
    block = entry.get("run_summary")
    if not block:
        return []
    summary_path = docs_dir / block["path"]
    if not summary_path.exists():
        return [
            DocsProblem(
                key,
                "missing_run_summary",
                summary_path.relative_to(docs_dir).as_posix(),
            )
        ]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as error:
        return [DocsProblem(key, "corrupt_run_summary", str(error))]

    problems = []
    for field in block.get("required_fields", []):
        if _get_field(summary, field) is _MISSING:
            problems.append(DocsProblem(key, "missing_summary_field", field))
    for field, expected in block.get("expected_values", {}).items():
        actual = _get_field(summary, field)
        if actual is _MISSING:
            problems.append(DocsProblem(key, "missing_summary_field", field))
        elif actual != expected:
            problems.append(
                DocsProblem(
                    key,
                    "contradictory_run_summary",
                    f"{field} is {actual!r}, expected {expected!r}",
                )
            )
    for left, right in block.get("consistent_fields", []):
        left_value = _get_field(summary, left)
        right_value = _get_field(summary, right)
        if left_value is _MISSING:
            problems.append(DocsProblem(key, "missing_summary_field", left))
        elif right_value is _MISSING:
            problems.append(DocsProblem(key, "missing_summary_field", right))
        elif left_value != right_value:
            problems.append(
                DocsProblem(
                    key,
                    "contradictory_run_summary",
                    f"{left} is {left_value!r}, {right} is {right_value!r}",
                )
            )
    return problems


def _canonical_source(path: Path, docs_dir: Path) -> Path:
    """Map a copied ``docs/examples`` tree back to the shipped root examples."""
    mirror = docs_dir / "examples"
    try:
        relative = path.relative_to(mirror)
    except ValueError:
        return path
    return (docs_dir.parent / "examples" / relative).resolve()


def _literalinclude_paths(page: Path, docs_dir: Path) -> list[Path]:
    paths = []
    for raw in _LITERALINCLUDE.findall(page.read_text(encoding="utf-8")):
        if raw.startswith("/"):
            target = (docs_dir / raw.lstrip("/")).resolve()
        else:
            target = (page.parent / raw).resolve()
        paths.append(_canonical_source(target, docs_dir))
    return paths


def _check_authored_page(key: str, entry: dict, docs_dir: Path) -> list[DocsProblem]:
    problems = []
    page = docs_dir / f"{key}.rst"
    if not page.exists():
        return [DocsProblem(key, "page_missing", f"{key}.rst")]

    source = entry.get("source")
    if source is not None:
        source_path = (docs_dir / ".." / source).resolve()
        if not source_path.exists():
            problems.append(DocsProblem(key, "source_missing", source))
        else:
            example_backend(source_path)
            includes = _literalinclude_paths(page, docs_dir)
            if source_path not in includes:
                problems.append(
                    DocsProblem(
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
            problems.append(DocsProblem(key, "figure_not_referenced", name))
    problems.extend(_check_run_summary(entry, docs_dir, key))
    return problems


def _html_targets(page: Path) -> list[tuple[str, str]]:
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


def _broken_html_targets(page: Path, label: str) -> list[DocsProblem]:
    problems = []
    for kind, source in _html_targets(page):
        if source.startswith(
            ("http:", "https:", "data:", "mailto:", "javascript:", "tel:", "#")
        ):
            continue
        target = (page.parent / source.split("?")[0].split("#")[0]).resolve()
        if not target.exists():
            code = "broken_html_link" if kind == "link" else "broken_html_image"
            problems.append(DocsProblem(label, code, source))
    return problems


def check_docs(docs_dir: Path, *, html_dir: Path | None = None) -> list[DocsProblem]:
    """Check the authored inventory, sources, figures and rendered links."""
    authored = load_manifest(docs_dir)["authored_pages"]
    problems: list[DocsProblem] = []

    for key, entry in sorted(authored.items()):
        problems.extend(_check_authored_page(key, entry, docs_dir))

    pages = docs_dir / "example_pages"
    for page in sorted(pages.glob("*.rst")):
        if page.stem == "index":
            continue
        key = f"example_pages/{page.stem}"
        if key not in authored:
            problems.append(DocsProblem(key, "manifest_missing", "no inventory entry"))

    if html_dir is not None:
        for key in sorted(authored):
            page = html_dir / f"{key}.html"
            if not page.exists():
                problems.append(DocsProblem(key, "html_page_missing", f"{key}.html"))
            else:
                problems.extend(_broken_html_targets(page, key))
    return problems
