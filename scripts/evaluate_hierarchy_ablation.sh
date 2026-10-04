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
: "${BROWSECOMP_SNAPSHOTS_ROOT:?Set BROWSECOMP_SNAPSHOTS_ROOT}"
BROWSECOMP_EVAL_INDEX=data/splits/browsecomp_hierarchy49.jsonl

OUT=outputs/hierarchy_ablation
mkdir -p "$OUT"

if [[ -d "$OUT/variants" ]]; then
  python -m analysis.hierarchy_ablation validate-hierarchy-tree \
    --snapshots-root "$BROWSECOMP_SNAPSHOTS_ROOT" \
    --index "$BROWSECOMP_EVAL_INDEX" \
    --output-dir "$OUT/variants" \
    --seed 20260924
else
  python -m analysis.hierarchy_ablation hierarchy-tree \
    --snapshots-root "$BROWSECOMP_SNAPSHOTS_ROOT" \
    --index "$BROWSECOMP_EVAL_INDEX" \
    --output-dir "$OUT/variants" \
    --seed 20260924
fi

python -m experiments.inference.browsecomp_hierarchy_ablation \
  --index "$BROWSECOMP_EVAL_INDEX" \
  --variants-root "$OUT/variants" \
  --output "$OUT/execution.jsonl" \
  --answerer-model gemini-3.1-flash-lite-preview \
  --judge-model gemini-3.1-flash-lite-preview \
  --concurrency 8 \
  --resume

python -m analysis.hierarchy_ablation hierarchy-report \
  --results "$OUT/execution.jsonl" \
  --index "$BROWSECOMP_EVAL_INDEX" \
  --variants-root "$OUT/variants" \
  --output "$OUT/report.json" \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20260924

python -m analysis.hierarchy_ablation compact-ledger \
  --results "$OUT/execution.jsonl" \
  --output "$OUT/per_question.jsonl"

python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("outputs/hierarchy_ablation/report.json").read_text())
print("variant     accuracy  navigation tokens")
for variant in ("original", "flattened", "shuffled"):
    row = report["variants"][variant]
    print(
        f"{variant:<11} {row['question_micro_accuracy']:>7.2%}"
        f" {row['mean_answerer_tokens_read']:>18,.2f}"
    )
PY
