#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${TINKER_API_KEY:?Set TINKER_API_KEY in the environment}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in the environment}"
: "${BUILDER_EXECUTOR_BASE_URL:?Set BUILDER_EXECUTOR_BASE_URL}"
: "${BRIGHT_LIBRARIAN_CHECKPOINT:?Set BRIGHT_LIBRARIAN_CHECKPOINT}"

SNAPSHOT_ROOT="${BRIGHT_SNAPSHOTS_ROOT:-outputs/bright/snapshots}"
mkdir -p "$SNAPSHOT_ROOT"

for domain in aops biology stackoverflow; do
  construction_root="data/bright/construction/$domain"
  python -m data_processing.prepare_bright_frozen_construction \
    --shared-root data/bright \
    --domain "$domain" \
    --output-root "$construction_root"

  stage="outputs/bright/construction/$domain"
  python -m experiments.training.train \
    model_name=Qwen/Qwen3.5-4B \
    lora_rank=32 \
    load_checkpoint_path="$BRIGHT_LIBRARIAN_CHECKPOINT" \
    learning_rate=0.0 \
    batch_size=1 \
    group_size=1 \
    max_steps=1 \
    eval_only=true \
    eval_every=1 \
    seed=2 \
    max_turns=32 \
    max_tokens=1024 \
    max_trajectory_tokens=60000 \
    temperature=0.0 \
    max_concurrent_rollout_groups=1 \
    index_jsonl="$construction_root/seed_index/index.jsonl" \
    eval_index_jsonl="$construction_root/evaluation_index/index.jsonl" \
    eval_size=1 \
    answerer_workspace_mode=synthetic_only \
    answerer_max_turns=32 \
    answerer_final_answer_max_tokens=128 \
    terminal_answerer_repeats=1 \
    answerability_delta_reward_scale=0.0 \
    builder_compaction_enabled=true \
    builder_compaction_model=gemini-3.1-flash-lite-preview \
    builder_compaction_trigger_tokens=3000 \
    builder_compaction_keep_recent_turns=1 \
    builder_compaction_max_output_tokens=500 \
    builder_compaction_input_max_chars=30000 \
    builder_executor_enabled=true \
    builder_executor_backend=vllm \
    builder_executor_model=Qwen/Qwen3.5-35B-A3B \
    builder_executor_base_url="$BUILDER_EXECUTOR_BASE_URL" \
    builder_executor_api_key_env=BUILDER_EXECUTOR_API_KEY \
    builder_executor_max_source_chars=16000 \
    builder_executor_max_output_tokens=512 \
    builder_executor_temperature=0.0 \
    step_penalty=0.0 \
    termination_penalty=0.0 \
    empty_synthetic_penalty=0.0 \
    filesystem_maturity_scale=0.0 \
    step_construction_action_bonus=0.0 \
    step_filesystem_maturity_delta_scale=0.0 \
    step_non_construction_turn_penalty=0.0 \
    step_non_construction_streak_penalty=0.0 \
    step_tool_error_penalty=0.0 \
    filesystem_snapshot_dir="$stage/snapshots" \
    log_path="$stage/logs" \
    behavior_if_log_dir_exists=raise \
    disable_sample_trajectory_printing=true \
    rollout_json_export=false

  construction_id="bright_${domain}_shared_filesystem"
  cp "$stage/snapshots/$construction_id/trajectory_00.json" "$SNAPSHOT_ROOT/$domain.json"
done
