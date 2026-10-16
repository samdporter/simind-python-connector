#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION_FILE="${PROJECT_ROOT}/simind_python_connector/_version.py"
CHANGELOG_MD="${PROJECT_ROOT}/CHANGELOG.md"
CHANGELOG_RST="${PROJECT_ROOT}/docs/changelog.rst"

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

# Read the same version used by the package, documentation and build backend.
CURRENT_VERSION=$(sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' "$VERSION_FILE")

if [ -z "$CURRENT_VERSION" ]; then
    log_error "Could not extract version from $VERSION_FILE"
    exit 1
fi

log_info "Current version: ${CURRENT_VERSION}"

# Determine new version
bump_version() {
    local version="$1"
    local type="$2"
    local major=$(echo "$version" | cut -d. -f1)
    local minor=$(echo "$version" | cut -d. -f2)
    local patch=$(echo "$version" | cut -d. -f3)
    
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

log_info "Bumping version from ${CURRENT_VERSION} to ${NEW_VERSION}"

RELEASE_DATE=$(date +%Y-%m-%d)
TAG="v${NEW_VERSION}"
cd "$PROJECT_ROOT"

# Release notes are written by hand before running this script.
if ! grep -q "^## \[${NEW_VERSION}\]" "$CHANGELOG_MD"; then
    log_error "CHANGELOG.md has no '## [${NEW_VERSION}]' entry. Write the release notes first."
    exit 1
fi
if ! grep -q "^${NEW_VERSION} - " "$CHANGELOG_RST"; then
    log_error "docs/changelog.rst has no '${NEW_VERSION} - <date>' entry. Write the release notes first."
    exit 1
fi
if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
    log_error "Tag ${TAG} already exists."
    exit 1
fi

# Replace the version in one file and check that it changed.
set_version() {
    local file="$1"
    local expression="$2"
    sed -i.bak "$expression" "$file"
    if cmp -s "$file" "$file.bak"; then
        rm -f "$file.bak"
        log_error "Could not update the version in $file"
        exit 1
    fi
    rm -f "$file.bak"
}

log_info "Updating version sources..."
set_version simind_python_connector/_version.py "s/^__version__ = \"${CURRENT_VERSION}\"/__version__ = \"${NEW_VERSION}\"/"
set_version .zenodo.json "s/\"version\": \"${CURRENT_VERSION}\"/\"version\": \"${NEW_VERSION}\"/"
set_version CITATION.cff "s/^version: ${CURRENT_VERSION}$/version: ${NEW_VERSION}/;s/^date-released: .*/date-released: ${RELEASE_DATE}/"

log_info "Committing and tagging..."
git add simind_python_connector/_version.py .zenodo.json \
    CITATION.cff CHANGELOG.md docs/changelog.rst
git commit -m "chore: release ${NEW_VERSION}"
git tag -a "$TAG" -m "Release ${NEW_VERSION}"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
log_info "Release ${NEW_VERSION} committed and tagged as ${TAG} on ${BRANCH}."
echo ""
echo "Next steps:"
echo "  1. Push the branch and the tag: git push origin ${BRANCH} ${TAG}"
echo "  2. Publish the release from that tag:"
echo "     gh release create ${TAG} --verify-tag --title \"${NEW_VERSION}\" --notes-file <notes.md>"
echo "  3. publish.yml runs the tests, checks ${TAG} against _version.py and publishes to PyPI."
