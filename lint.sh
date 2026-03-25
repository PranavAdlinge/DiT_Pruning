#!/usr/bin/env bash
# Lint and format with ruff. Run from repo root.
#
# Usage:
#   ./lint.sh          # check only
#   ./lint.sh --fix    # auto-fix
#   ./lint.sh format   # format code

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v ruff &> /dev/null; then
    echo "ruff not found. Install with: pip install ruff"
    exit 1
fi

case "${1:-check}" in
    format)
        ruff format src tests datasets
        ;;
    fix)
        ruff check src tests datasets --fix
        ruff format src tests datasets
        ;;
    *)
        ruff check src tests datasets
        ruff format src tests datasets --check
        ;;
esac
