"""Helpers for the sphinx-gallery configuration and its staleness check."""

import ast
from pathlib import Path

from sphinx_gallery.utils import get_md5sum


BACKENDS = ("python", "stir", "sirf", "pytomography")
_ADAPTOR_ROOTS = {
    "sirf": "sirf",
    "cil": "sirf",
    "stir": "stir",
    "pytomography": "pytomography",
}


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


def gallery_examples(examples_dir: Path) -> list[Path]:
    return sorted(examples_dir.glob("*/plot_*.py"))


def execution_pattern(examples_dir: Path, backend: str) -> str:
    paths = [
        path
        for path in gallery_examples(examples_dir)
        if example_backend(path) == backend
    ]
    if not paths:
        return r"^$"
    names = "|".join(
        f"{path.parent.name}/{path.name}".replace(".", r"\.") for path in paths
    )
    return f"/examples/({names})$"


def stale_examples(examples_dir: Path, gallery_dir: Path) -> list[Path]:
    stale = []
    for path in gallery_examples(examples_dir):
        md5_file = gallery_dir / path.parent.name / f"{path.name}.md5"
        if not md5_file.exists() or md5_file.read_text() != get_md5sum(path, mode="t"):
            stale.append(path)
    return stale
