"""Exit non-zero when a gallery example has no up-to-date executed output."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

from _gallery import BACKENDS, stale_examples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    args = parser.parse_args()
    stale = stale_examples(
        ROOT / "examples",
        ROOT / "docs" / "auto_examples",
        tuple(args.backend or BACKENDS),
    )
    for path in stale:
        print(f"stale: {path.relative_to(ROOT)}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
