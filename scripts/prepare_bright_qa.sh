#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m data_processing.prepare_bright_qa50 \
  --output-root data/bright \
  --reference-manifest data/splits/bright_qa150.json \
  --extension-per-domain 0 \
  --max-documents-per-domain 50 \
  --seed 20260822 \
  --overwrite
