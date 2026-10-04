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
: "${BROWSECOMP_EVAL_INDEX:=data/splits/browsecomp_amortized50.jsonl}"
: "${LIBRARIAN_SNAPSHOTS_ROOT:?Set LIBRARIAN_SNAPSHOTS_ROOT}"
: "${BASE_SNAPSHOTS_ROOT:?Set BASE_SNAPSHOTS_ROOT}"

mkdir -p outputs/browsecomp
for arm in librarian base; do
  if [[ "$arm" == librarian ]]; then
    snapshots_root="$LIBRARIAN_SNAPSHOTS_ROOT"
  else
    snapshots_root="$BASE_SNAPSHOTS_ROOT"
  fi
  variants_root="outputs/browsecomp/variants/$arm"
  if [[ -d "$variants_root" ]]; then
    python -m analysis.hierarchy_ablation validate-hierarchy-tree \
      --snapshots-root "$snapshots_root" \
      --index "$BROWSECOMP_EVAL_INDEX" \
      --output-dir "$variants_root" \
      --seed 19105
  else
    python -m analysis.hierarchy_ablation hierarchy-tree \
      --snapshots-root "$snapshots_root" \
      --index "$BROWSECOMP_EVAL_INDEX" \
      --output-dir "$variants_root" \
      --seed 19105 \
      --skip-missing
  fi
  python -m experiments.inference.browsecomp_hierarchy_ablation \
    --index "$BROWSECOMP_EVAL_INDEX" \
    --variants-root "$variants_root" \
    --variants original \
    --output "outputs/browsecomp/$arm.jsonl" \
    --answerer-model gemini-3.1-flash-lite-preview \
    --judge-model gemini-3.1-flash-lite-preview \
    --concurrency 8 \
    --missing-snapshot-score-zero \
    --resume
done

python -m analysis.browsecomp \
  --librarian outputs/browsecomp/librarian.jsonl \
  --base outputs/browsecomp/base.jsonl \
  --manifest "$BROWSECOMP_EVAL_INDEX" \
  --output outputs/browsecomp/report.json
