#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/validate_release.py
python -m analysis.reproduce_results
python -m pytest -q
python -m ruff check analysis data_processing environments experiments tests
