import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_IMPORTS = (
    "import simind_python_connector\n"
    "from simind_python_connector import SimindPythonConnector\n"
    "import simind_python_connector.connectors\n"
    "import simind_python_connector.converters.simind_to_stir\n"
)


def _run(code: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


def test_importing_the_core_does_not_configure_logging():
    code = (
        "import logging\n"
        "root = logging.getLogger()\n"
        "before = (len(root.handlers), root.level)\n"
        + _IMPORTS
        + "print(before == (len(root.handlers), root.level))\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_importing_the_core_never_imports_backends_even_if_installed(tmp_path: Path):
    for name in ("sirf", "stir", "stirextra", "torch", "pytomography"):
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("")
    (tmp_path / "sirf" / "STIR.py").write_text("")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        _IMPORTS
        + "import sys\n"
        + "names = ('sirf', 'stir', 'stirextra', 'torch', 'pytomography')\n"
        + "print(sorted(name for name in names if name in sys.modules))\n"
    )
    result = _run(code, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
