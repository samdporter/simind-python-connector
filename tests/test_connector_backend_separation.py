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


def _write_fake_stir(base: Path, *, extra: str) -> Path:
    """Write a fake ``stir`` package with the requested conversion module."""
    pkg = base / "fakes"
    stir_dir = pkg / "stir"
    stir_dir.mkdir(parents=True)
    (stir_dir / "__init__.py").write_text(
        "class FloatVoxelsOnCartesianGrid:\n    pass\n\nclass ProjData:\n    pass\n"
    )
    if extra == "modern":
        (stir_dir / "extra.py").write_text("def to_numpy(obj):\n    return obj\n")
        (pkg / "stirextra.py").write_text(
            "raise AssertionError('stirextra must not be imported')\n"
        )
    elif extra == "legacy":
        (pkg / "stirextra.py").write_text(
            "import warnings\n"
            "warnings.warn('Please replace `stirextra` with `stir.extra`',"
            " DeprecationWarning, stacklevel=2)\n"
            "def to_numpy(obj):\n    return obj\n"
        )
    elif extra == "broken":
        (stir_dir / "extra.py").write_text("import missing_dep_xyz\n")
        (pkg / "stirextra.py").write_text(
            "raise AssertionError('stirextra must not be imported')\n"
        )
    return pkg


def _run_in_subprocess(
    code: str,
    *,
    extra_path: Path | None = None,
    deprecations_as_errors: bool = False,
):
    """Run ``code`` in a fresh interpreter with the fake package on the path."""
    env = dict(os.environ)
    paths = [str(ROOT)]
    if extra_path is not None:
        paths.insert(0, str(extra_path))
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    args = [sys.executable]
    if deprecations_as_errors:
        args += ["-W", "error::DeprecationWarning"]
    args += ["-c", code]
    return subprocess.run(args, capture_output=True, text=True, env=env)


def test_get_stir_types_accepts_modern_stir_extra(tmp_path):
    """A modern ``stir.extra`` install reports STIR as available."""
    pkg = _write_fake_stir(tmp_path, extra="modern")

    code = (
        "import sys\n"
        "from simind_python_connector.utils.import_helpers import (\n"
        "    _import_stir_extra,\n"
        "    get_stir_types,\n"
        ")\n"
        "extra = _import_stir_extra()\n"
        "assert extra.__name__ == 'stir.extra', extra.__name__\n"
        "assert 'stirextra' not in sys.modules\n"
        "img, proj, available = get_stir_types()\n"
        "assert available is True\n"
        "print('OK')\n"
    )
    result = _run_in_subprocess(code, extra_path=pkg, deprecations_as_errors=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_import_stir_extra_falls_back_to_legacy(tmp_path):
    """A legacy-only install imports ``stirextra`` with its warning suppressed."""
    pkg = _write_fake_stir(tmp_path, extra="legacy")

    code = (
        "from simind_python_connector.utils.import_helpers import (\n"
        "    _import_stir_extra,\n"
        "    get_stir_types,\n"
        ")\n"
        "extra = _import_stir_extra()\n"
        "assert extra.__name__ == 'stirextra', extra.__name__\n"
        "img, proj, available = get_stir_types()\n"
        "assert available is True\n"
        "print('OK')\n"
    )
    result = _run_in_subprocess(code, extra_path=pkg, deprecations_as_errors=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_get_stir_types_unavailable_without_extra_module(tmp_path):
    """``stir`` without either conversion module is still unavailable."""
    pkg = _write_fake_stir(tmp_path, extra="none")

    code = (
        "from simind_python_connector.utils.import_helpers import (\n"
        "    _import_stir_extra,\n"
        "    get_stir_types,\n"
        ")\n"
        "try:\n"
        "    _import_stir_extra()\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')\n"
        "img, proj, available = get_stir_types()\n"
        "assert available is False and img is type(None) and proj is type(None)\n"
        "print('OK')\n"
    )
    result = _run_in_subprocess(code, extra_path=pkg)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_import_stir_extra_propagates_unrelated_import_error(tmp_path):
    """A broken ``stir.extra`` dependency is not masked by the legacy fallback."""
    pkg = _write_fake_stir(tmp_path, extra="broken")

    code = (
        "from simind_python_connector.utils.import_helpers import (\n"
        "    _import_stir_extra,\n"
        "    get_stir_types,\n"
        ")\n"
        "try:\n"
        "    _import_stir_extra()\n"
        "except ModuleNotFoundError as exc:\n"
        "    assert exc.name == 'missing_dep_xyz', exc.name\n"
        "else:\n"
        "    raise SystemExit('expected ModuleNotFoundError')\n"
        "img, proj, available = get_stir_types()\n"
        "assert available is False, 'broken modern must not report STIR available'\n"
        "print('OK')\n"
    )
    result = _run_in_subprocess(code, extra_path=pkg)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
