#!/bin/bash
#
# Publish macula_tweann to Hex.pm
#
# Usage:
#   ./scripts/publish-to-hex.sh [--dry-run]
#
# Requirements:
#   - rebar3_hex plugin (configured in rebar.config)
#   - HEX_API_KEY environment variable or ~/.hex/hex.config
#
# Options:
#   --dry-run    Show what would be published without actually publishing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse arguments
DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== DRY RUN MODE ==="
fi

# Extract version from app.src
VERSION=$(grep -oP '{vsn,\s*"\K[^"]+' src/macula_tweann.app.src)
echo "Publishing macula_tweann version $VERSION to Hex.pm"

# Check for uncommitted changes
if ! git diff --quiet HEAD; then
    echo "Warning: You have uncommitted changes"
    echo "It's recommended to commit all changes before publishing"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 1
    fi
fi

# Clean previous builds
echo "Cleaning previous builds..."
rebar3 clean

# Run tests first
echo "Running tests..."
if ! rebar3 eunit; then
    echo "Tests failed! Fix tests before publishing."
    exit 1
fi

# Run dialyzer
echo "Running dialyzer..."
if ! rebar3 dialyzer; then
    echo "Dialyzer found issues! Fix them before publishing."
    exit 1
fi

# Compile
echo "Compiling..."
rebar3 compile

# Generate documentation
echo "Generating documentation..."
rebar3 edoc

# Build the package
echo "Building hex package..."
if [[ -n "$DRY_RUN" ]]; then
    rebar3 hex build
    echo ""
    echo "=== DRY RUN COMPLETE ==="
    echo "Package would be published as macula_tweann v$VERSION"
    echo "Run without --dry-run to actually publish"
else
    # Publish to Hex.pm
    echo "Publishing to Hex.pm..."
    rebar3 hex publish

    echo ""
    echo "=== SUCCESS ==="
    echo "Published macula_tweann v$VERSION to Hex.pm"
    echo "View at: https://hex.pm/packages/macula_tweann"
fi
