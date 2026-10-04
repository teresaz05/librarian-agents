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

QUESTIONS="${BRIGHT_QUESTIONS:-data/bright/heldout_ids.json}"
DOCS="${BRIGHT_DOCS:-data/bright/train}"
PRIVILEGED="${BRIGHT_PRIVILEGED:-data/bright/train_privileged}"
OUT="${BRIGHT_BASELINE_OUTPUT:-outputs/bright/baselines}"
mkdir -p "$OUT/retrieval" "$OUT/answers"

for strategy in bm25_chunked embed_chunked hybrid_rrf_chunked lateinter_chunked rerank_fixed_chunked; do
  python -m experiments.inference.strong_retrieval \
    --strategy "$strategy" \
    --questions-path "$QUESTIONS" \
    --docs-dir "$DOCS" \
    --privileged-dir "$PRIVILEGED" \
    --output-path "$OUT/retrieval/$strategy.jsonl" \
    --resume
  python -m experiments.inference.answer \
    --input-path "$OUT/retrieval/$strategy.jsonl" \
    --output-path "$OUT/answers/$strategy.jsonl" \
    --docs-dir "$DOCS" \
    --privileged-dir "$PRIVILEGED" \
    --non-local-answerer \
    --provider gemini \
    --model-name-or-path gemini-3.1-flash-lite-preview \
    --max-new-tokens 128 \
    --temperature 0.7 \
    --top-p 1.0 \
    --n 4 \
    --judge \
    --judge-provider gemini \
    --judge-model-name gemini-3.1-flash-lite-preview \
    --resume
done

python -m experiments.inference.graph_rag \
  --questions-path "$QUESTIONS" --docs-dir "$DOCS" --privileged-dir "$PRIVILEGED" \
  --retrieval-output-path "$OUT/retrieval/graph.jsonl" \
  --output-path "$OUT/answers/graph.jsonl" \
  --non-local-graph --graph-provider gemini --graph-model-name gemini-3.1-flash-lite-preview \
  --non-local-answerer --provider gemini --model-name-or-path gemini-3.1-flash-lite-preview \
  --max-new-tokens 128 --temperature 0.7 --top-p 1.0 --n 4 \
  --judge --judge-provider gemini --judge-model-name gemini-3.1-flash-lite-preview --resume

python -m experiments.inference.agentic_rag \
  --questions-path "$QUESTIONS" --docs-dir "$DOCS" --privileged-dir "$PRIVILEGED" \
  --retrieval-output-path "$OUT/retrieval/agentic.jsonl" \
  --output-path "$OUT/answers/agentic.jsonl" \
  --non-local-agent \
  --agent-provider gemini --agent-model-name gemini-3.1-flash-lite-preview \
  --non-local-answerer --provider gemini --model-name-or-path gemini-3.1-flash-lite-preview \
  --max-new-tokens 128 --temperature 0.7 --top-p 1.0 --n 4 \
  --judge --judge-provider gemini --judge-model-name gemini-3.1-flash-lite-preview --resume

python -m experiments.inference.autocompaction_rag \
  --questions-path "$QUESTIONS" --docs-dir "$DOCS" --privileged-dir "$PRIVILEGED" \
  --retrieval-output-path "$OUT/retrieval/autocompaction.jsonl" \
  --output-path "$OUT/answers/autocompaction.jsonl" \
  --non-local-generator \
  --generator-provider gemini --generator-model-name gemini-3.1-flash-lite-preview \
  --non-local-answerer --provider gemini --model-name-or-path gemini-3.1-flash-lite-preview \
  --max-new-tokens 128 --temperature 0.7 --top-p 1.0 --n 4 \
  --judge --judge-provider gemini --judge-model-name gemini-3.1-flash-lite-preview --resume

python -m analysis.bright_qa \
  --answers-dir "$OUT/answers" \
  --manifest data/splits/bright_qa150.json \
  --output "$OUT/results.json"
