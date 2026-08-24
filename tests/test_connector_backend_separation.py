import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONNECTORS_DIR = ROOT / "simind_python_connector" / "connectors"

pytestmark = pytest.mark.unit


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.unit
def test_stir_adaptor_module_is_backend_specific():
    imports = _import_roots(CONNECTORS_DIR / "stir_adaptor.py")
    assert "sirf" not in imports
    assert "pytomography" not in imports


@pytest.mark.unit
def test_sirf_adaptor_module_is_backend_specific():
    imports = _import_roots(CONNECTORS_DIR / "sirf_adaptor.py")
    assert "stir" not in imports
    assert "pytomography" not in imports


@pytest.mark.unit
def test_pytomography_adaptor_module_is_backend_specific():
    imports = _import_roots(CONNECTORS_DIR / "pytomography_adaptor.py")
    assert "sirf" not in imports
    assert "stir" not in imports


def test_importing_connectors_package_does_not_import_backends():
    """Pure-core import must succeed with every optional backend blocked."""
    code = (
        "import simind_python_connector.connectors as c\n"
        "from simind_python_connector.core.config import SimulationConfig\n"
        "blocked = ('sirf', 'stir', 'stirextra', 'pytomography')\n"
        "for name in blocked:\n"
        "    assert name not in __import__('sys').modules, name\n"
        "print('OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT / "docker" / "import_guard") + os.pathsep + env.get("PYTHONPATH", "")
    )
    env["SIMIND_PYTHON_CONNECTOR_BLOCK_IMPORTS"] = "sirf,stir,stirextra,pytomography"

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_lazy_adaptor_raises_construction_import_error_when_blocked():
    """Accessing an adaptor lazily is fine; constructing it reports the missing
    backend instead of failing at pure-core import time."""
    code = (
        "import simind_python_connector.connectors as c\n"
        "cls = getattr(c, 'StirSimindAdaptor')\n"
        "try:\n"
        "    cls('x', '.', 'p')\n"
        "except ImportError as exc:\n"
        "    print('IMPORT_ERROR:', exc)\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT / "docker" / "import_guard") + os.pathsep + env.get("PYTHONPATH", "")
    )
    env["SIMIND_PYTHON_CONNECTOR_BLOCK_IMPORTS"] = "sirf,stir,stirextra,pytomography"

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert "IMPORT_ERROR:" in result.stdout


def test_get_stir_types_requires_both_stir_and_stirextra():

    code = (
        "import sys, types\n"
        "fake = types.ModuleType('stir')\n"
        "class P: pass\n"
        "fake.FloatVoxelsOnCartesianGrid = P\n"
        "fake.ProjData = P\n"
        "sys.modules['stir'] = fake\n"
        "assert 'stirextra' not in sys.modules\n"
        "from simind_python_connector.utils.import_helpers import get_stir_types\n"
        "img, proj, available = get_stir_types()\n"
        "assert available is False and img is type(None) and proj is type(None)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
