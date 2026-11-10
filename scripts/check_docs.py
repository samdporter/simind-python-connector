"""Exit non-zero when the authored documentation inventory is inconsistent.

Checks every authored page has an inventory entry, its figures exist and are
referenced, its literalinclude uses the declared source, and, when an HTML
build is available, local links and images resolve.  It never executes an
example.
"""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

from _docs import check_docs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
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

    docs_dir = ROOT / "docs"
    html_dir = args.html_dir
    if html_dir is None and not args.no_html:
        candidate = docs_dir / "_build" / "html"
        if candidate.exists():
            html_dir = candidate

    problems = check_docs(docs_dir, html_dir=html_dir)
    for problem in problems:
        print(problem)
    if problems:
        print(f"docs check failed: {len(problems)} problem(s)")
        return 1
    print("docs check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
