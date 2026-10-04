#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${BROWSECOMP_SOURCE:?Set BROWSECOMP_SOURCE to an officially obtained BrowseComp+ JSONL file}"

python -m data_processing.prepare_browsecomp \
  --source "$BROWSECOMP_SOURCE" \
  --output-root data/browsecomp_plus \
  --heldout-ids data/splits/heldout_ids.json \
  --excluded-ids data/splits/excluded_ids.jsonl \
  --noise-seed 20260523 \
  --overwrite

python -m data_processing.validate_browsecomp data/browsecomp_plus
