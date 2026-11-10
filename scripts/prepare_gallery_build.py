"""Prepare one backend's gallery build and write its execution plan.

Run immediately before the backend's Sphinx build (``scripts/build_gallery.sh``
does).  Sphinx-Gallery re-executes an example only when its source changes, so
a changed shared helper, config or package would otherwise leave cached figures
in place while the build-finished hook stamped them with the current dependency
digest.  Dropping the ``.md5`` sidecar of every stale example forces the gallery
to re-run it, and the written plan records which examples this build owns so
provenance is refreshed only for examples that actually executed.
"""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

from _gallery import BACKENDS, build_plan_path, prepare_backend_build  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    args = parser.parse_args()

    executed = sorted(prepare_backend_build(ROOT, ROOT / "docs", (args.backend,)))
    for key in executed:
        print(f"rebuild: {key}")
    print(
        f"gallery plan for {args.backend}: {len(executed)} stale example(s), "
        f"written to {build_plan_path(ROOT / 'docs').relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
