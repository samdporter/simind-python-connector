"""Require every test to declare its suite or dependency marker."""

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parent
ALLOWED = {
    "unit",
    "integration",
    "ci_skip",
    "requires_sirf",
    "requires_stir",
    "requires_simind",
    "requires_pytomography",
    "requires_cil",
    "requires_phantomgen",
}


def _marked(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute) and child.attr in ALLOWED
        for child in ast.walk(node)
    )


def test_all_tests_have_dependency_or_category_marker():
    missing = []
    for path in sorted(ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        module_marked = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
            and _marked(node.value)
            for node in tree.body
        )
        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test_"):
                if not module_marked and not _marked_decorators(node):
                    missing.append(f"{path.name}:{node.name}")
            if isinstance(node, ast.ClassDef):
                class_marked = _marked_decorators(node)
                for child in node.body:
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name.startswith("test_")
                        and not module_marked
                        and not class_marked
                        and not _marked_decorators(child)
                    ):
                        missing.append(f"{path.name}:{node.name}.{child.name}")
    assert not missing, "Tests without a recognized marker: " + ", ".join(missing)


def _marked_decorators(node: ast.AST) -> bool:
    return any(_marked(decorator) for decorator in getattr(node, "decorator_list", ()))
