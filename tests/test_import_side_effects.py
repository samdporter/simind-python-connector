import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit

_IMPORTS = (
    "import simind_python_connector\n"
    "from simind_python_connector import SimindPythonConnector\n"
    "import simind_python_connector.connectors\n"
    "import simind_python_connector.converters.simind_to_stir\n"
)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


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
