#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYPROJECT="${PROJECT_ROOT}/pyproject.toml"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [ $# -eq 0 ]; then
    log_error "No version specified. Usage: $0 [patch|minor|major|<version>]"
    exit 1
fi

VERSION_TYPE="$1"

CURRENT_VERSION=$(grep "^version = " "$PYPROJECT" | sed 's/.*"\([^"]*\)".*/\1/')

if [ -z "$CURRENT_VERSION" ]; then
    log_error "Could not extract version from $PYPROJECT"
    exit 1
fi

log_info "Current version: ${CURRENT_VERSION}"

bump_version() {
    local version="$1"
    local type="$2"
    local major
    local minor
    local patch
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    patch=$(echo "$version" | cut -d. -f3)

    case "$type" in
        patch) echo "${major}.${minor}.$((patch + 1))" ;;
        minor) echo "${major}.$((minor + 1)).0" ;;
        major) echo "$((major + 1)).0.0" ;;
        *)
            if [[ "$type" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                echo "$type"
            else
                echo ""
            fi
            ;;
    esac
}

NEW_VERSION=$(bump_version "$CURRENT_VERSION" "$VERSION_TYPE")

if [ -z "$NEW_VERSION" ]; then
    log_error "Invalid version type: $VERSION_TYPE. Use patch, minor, major, or <exact_version>"
    exit 1
fi

RELEASE_DATE=$(date +%Y-%m-%d)

log_info "Preparing release ${NEW_VERSION} (${RELEASE_DATE})..."
python3 "${SCRIPT_DIR}/prepare_release.py" "$NEW_VERSION" --date "$RELEASE_DATE"

log_warn "Release metadata prepared. Nothing has been committed, tagged or pushed."
echo ""
echo "Next steps:"
echo "  1. Inspect the working tree and the index before staging anything:"
echo "     git status"
echo "     git diff --cached"
echo "     Commit implementation or test corrections separately, in their own"
echo "     commit, before the metadata commit below."
echo "  2. Stage exactly the eight version surfaces, then read back what is staged:"
echo "     git add pyproject.toml uv.lock CHANGELOG.md docs/changelog.rst CITATION.cff .zenodo.json simind_python_connector/__init__.py docs/conf.py"
echo "     git diff --cached --stat"
echo "     git diff --cached"
echo "     The staged diff must contain only those eight paths; unrelated"
echo "     pre-staged files must not be included."
echo "  3. Commit the staged metadata:"
echo "     git commit -m \"chore: release ${NEW_VERSION}\""
echo "  4. Only after validation passes and the final reviewer approves, tag that"
echo "     commit and push the branch and the tag:"
echo "     git tag -a v${NEW_VERSION} -m \"Release ${NEW_VERSION}\""
echo "     git push origin HEAD v${NEW_VERSION}"
echo "  5. Once validation has passed and the final reviewer approves, publish the"
echo "     GitHub release from the pushed tag:"
echo "     gh release create v${NEW_VERSION} --verify-tag --title \"${NEW_VERSION}\" --notes-file <notes.md>"
echo "     Publishing the GitHub release starts publish.yml, which reruns the tests,"
echo "     checks v${NEW_VERSION} against pyproject.toml and uploads to PyPI."
echo "     Pushing the tag alone does not publish to PyPI."
