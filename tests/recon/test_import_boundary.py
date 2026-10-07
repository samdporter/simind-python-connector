import ast
import subprocess
import sys
from pathlib import Path

import pytest

import simind_python_connector


pytestmark = pytest.mark.unit

PACKAGE_DIR = Path(simind_python_connector.__file__).resolve().parent


def _is_recon(name: str) -> bool:
    parts = name.lstrip(".").split(".")
    in_package = name.startswith(".") or parts[0] == "simind_python_connector"
    return in_package and "recon" in parts


def _recon_imports(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            names = [base] + [f"{base}.{alias.name}" for alias in node.names]
        else:
            continue
        if any(_is_recon(name) for name in names):
            found.append(f"{path.name}:{node.lineno}")
    return found


def test_scanner_finds_every_form_of_recon_import(tmp_path):
    path = tmp_path / "module.py"
    path.write_text(
        "import simind_python_connector.recon\n"
        "from simind_python_connector.recon.updates import UpdateSchedule\n"
        "from simind_python_connector import recon\n"
        "from .recon import cil\n"
        "from cil.recon import FBP\n"
    )
    assert _recon_imports(path) == [
        "module.py:1",
        "module.py:2",
        "module.py:3",
        "module.py:4",
    ]


def test_core_modules_do_not_import_recon():
    offenders = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if "recon" not in path.relative_to(PACKAGE_DIR).parts:
            offenders += _recon_imports(path)
    assert offenders == []


def test_importing_the_core_does_not_load_recon():
    code = (
        "import sys\n"
        "import simind_python_connector as package\n"
        "import simind_python_connector.connectors\n"
        "import simind_python_connector.normalisation\n"
        "import simind_python_connector.utils.interfile\n"
        "package.SimindPythonConnector\n"
        "prefix = 'simind_python_connector.recon'\n"
        "print(sorted(m for m in sys.modules if m.startswith(prefix)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_importing_recon_does_not_load_backends_or_scipy():
    code = (
        "import sys\n"
        "import simind_python_connector.recon as recon\n"
        "recon.SimindProjector, recon.ScatterCorrection, recon.AdditiveUpdater\n"
        "heavy = ('cil', 'sirf', 'stir', 'scipy')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
