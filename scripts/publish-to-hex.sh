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
# Note: rebar3 eunit exits non-zero if tests are cancelled (e.g., NIF skip test)
# We check the actual test output for "Failed: 0" to determine success
TEST_OUTPUT=$(rebar3 eunit 2>&1) || true
echo "$TEST_OUTPUT" | tail -20

# Check for actual failures (not just cancelled tests)
if echo "$TEST_OUTPUT" | grep -q "Failed: [1-9]"; then
    echo "Tests failed! Fix tests before publishing."
    exit 1
fi

# Verify we have passing tests
if ! echo "$TEST_OUTPUT" | grep -q "Passed:"; then
    echo "Could not verify test results. Check test output above."
    exit 1
fi

echo "Tests passed (cancelled NIF tests are expected in Community Edition)"

# Run dialyzer
echo "Running dialyzer..."
if ! rebar3 dialyzer; then
    echo "Dialyzer found issues! Fix them before publishing."
    exit 1
fi

# Build the package
echo "Building hex package..."
rebar3 hex build
echo ""

if [[ -n "$DRY_RUN" ]]; then
    echo "=== DRY RUN COMPLETE ==="
    echo "Package would be published as macula_tweann v$VERSION"
    echo "Run without --dry-run to actually publish"
else
    # Publish to Hex.pm
    echo "Publishing to Hex.pm..."
    rebar3 hex publish --yes

    echo ""
    echo "=== SUCCESS ==="
    echo "Published macula_tweann v$VERSION to Hex.pm"
    echo "View at: https://hex.pm/packages/macula_tweann"
fi
