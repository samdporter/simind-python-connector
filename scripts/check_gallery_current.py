"""Exit non-zero when the committed gallery is not release-ready.

Checks source freshness, shared-helper/config provenance, required embedded
figures, thumbnails, download artifacts, reported statistical gates and, when
an HTML build is available, the final rendered figure links.  It never executes
an example.
"""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

from _gallery import BACKENDS, check_gallery, stale_examples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument(
        "--html-dir",
        type=Path,
        help="check the rendered HTML in this directory as well",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="skip the rendered-HTML check even when docs/_build/html exists",
    )
    args = parser.parse_args()

    backends = tuple(args.backend or BACKENDS)
    examples_dir = ROOT / "examples"
    docs_dir = ROOT / "docs"
    gallery_dir = docs_dir / "auto_examples"

    html_dir = args.html_dir
    if html_dir is None and not args.no_html:
        candidate = docs_dir / "_build" / "html"
        if candidate.exists():
            html_dir = candidate

    stale = stale_examples(examples_dir, gallery_dir, backends, root=ROOT)
    for path in stale:
        print(f"stale: {path.relative_to(ROOT)}")

    problems = check_gallery(
        examples_dir,
        gallery_dir,
        docs_dir,
        backends,
        root=ROOT,
        html_dir=html_dir,
    )
    for problem in problems:
        print(problem)

    if stale or problems:
        print(
            f"gallery gate failed: {len(stale)} stale source(s), "
            f"{len(problems)} artifact/provenance problem(s)"
        )
        return 1
    print("gallery gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
