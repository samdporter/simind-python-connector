import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit


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
