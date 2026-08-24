"""Policy test: the removed SimindSimulator API must not reappear."""

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("scripts", "examples", "simind_python_connector")


def test_removed_simind_simulator_symbol_is_absent():
    hits = []
    for root in SCAN_ROOTS:
        for path in (ROOT / root).rglob("*.py"):
            if "SimindSimulator" in path.read_text():
                hits.append(str(path.relative_to(ROOT)))
    assert not hits, "Removed SimindSimulator referenced in: " + ", ".join(hits)
