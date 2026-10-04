#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in the environment}"
: "${BRIGHT_SNAPSHOTS_ROOT:?Set BRIGHT_SNAPSHOTS_ROOT to aops/biology/stackoverflow snapshots}"

mkdir -p outputs/bright
for domain in aops biology stackoverflow; do
  python -m experiments.inference.bright_frozen_filesystem \
    --shared-root data/bright \
    --domain "$domain" \
    --snapshot "$BRIGHT_SNAPSHOTS_ROOT/$domain.json" \
    --output-path "outputs/bright/$domain.jsonl" \
    --answerer-model gemini-3.1-flash-lite-preview \
    --answerer-base-url https://generativelanguage.googleapis.com/v1beta \
    --answerer-api-key-env GEMINI_API_KEY \
    --judge-model gemini-3.1-flash-lite-preview \
    --judge-base-url https://generativelanguage.googleapis.com/v1beta \
    --judge-api-key-env GEMINI_API_KEY \
    --samples-per-question 4 \
    --answerer-temperature 0.7 \
    --answerer-top-p 1.0 \
    --answerer-max-turns 32 \
    --resume
done

python -m analysis.bright_qa --outputs-root outputs/bright --method librarian
