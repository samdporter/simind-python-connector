#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYPROJECT="${PROJECT_ROOT}/pyproject.toml"
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

# Extract current version from pyproject.toml
CURRENT_VERSION=$(grep "^version = " "$PYPROJECT" | sed 's/.*"\([^"]*\)".*/\1/')

if [ -z "$CURRENT_VERSION" ]; then
    log_error "Could not extract version from $PYPROJECT"
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

# Update pyproject.toml version
log_info "Updating pyproject.toml version..."
sed -i.bak "s/^version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" "$PYPROJECT"
rm -f "$PYPROJECT.bak"

# Update CHANGELOG.md
log_info "Adding version entry to CHANGELOG.md..."
CURRENT_DATE=$(date +%Y-%m-%d)

if ! grep -q "^## \[${NEW_VERSION}\]" "$CHANGELOG_MD"; then
    # Find the line number of the last version entry
    LAST_LINE=$(grep -n "^## \[" "$CHANGELOG_MD" | tail -1 | cut -d: -f1)
    if [ -n "$LAST_LINE" ]; then
        # Insert new version entry after the last one
        sed -i.bak "${LAST_LINE}a\\
\\
## [${NEW_VERSION}] - ${CURRENT_DATE}\\
\\
### Added\\
- Version bumped to ${NEW_VERSION}\\
\\
### Changed\\
- Automated release preparation\\
\\
### Fixed\\
- Updated test to use dynamic version" "$CHANGELOG_MD"
        rm -f "$CHANGELOG_MD.bak"
    fi
fi

# Update docs/changelog.rst
log_info "Adding version entry to docs/changelog.rst..."

# The RST changelog uses format: "1.0.1 - 2026-04-16" followed by "------------------"
if ! grep -q "^${NEW_VERSION} - " "$CHANGELOG_RST"; then
    # Find the line number of the first version entry (format: X.Y.Z - YYYY-MM-DD)
    FIRST_VERSION_LINE=$(grep -n "^[0-9]\+\.[0-9]\+\.[0-9]\+ - " "$CHANGELOG_RST" | head -1 | cut -d: -f1)
    if [ -n "$FIRST_VERSION_LINE" ]; then
        # Insert new version entry before the first one (at the top)
        sed -i.bak "${FIRST_VERSION_LINE}i\\
${NEW_VERSION} - ${CURRENT_DATE}\\
------------------------\\
\\
### Added\\
- Version bumped to ${NEW_VERSION}\\
\\
### Changed\\
- Automated release preparation\\
\\
### Fixed\\
- Updated test to use dynamic version\\
" "$CHANGELOG_RST"
        rm -f "$CHANGELOG_RST.bak"
    fi
fi

# Commit changes
log_info "Committing changes..."
cd "$PROJECT_ROOT"
git add pyproject.toml CHANGELOG.md docs/changelog.rst

if git diff --staged --quiet; then
    log_warn "No changes staged. Nothing to commit."
    exit 0
else
    git commit -m "chore(release): bump version to ${NEW_VERSION}"
    log_info "Commit created: chore(release): bump version to ${NEW_VERSION}"
fi

log_info "Release preparation complete!"
echo ""
echo "Next steps:"
echo "  1. Run: gh release create ${NEW_VERSION} --title \"Release ${NEW_VERSION}\" --notes \"Automated release via script\""
echo "  2. Push tags: git push origin main --tags"
echo "  3. CI (publish.yml) will detect new tag and publish to PyPI"