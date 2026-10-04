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

python -m experiments.training.train \
  model_name=Qwen/Qwen3.5-4B \
  lora_rank=32 \
  learning_rate=4e-5 \
  batch_size=8 \
  group_size=4 \
  train_epochs=10 \
  reshuffle_each_epoch=true \
  max_steps=850 \
  stream_minibatch_groups_per_batch=8 \
  stream_minibatch_num_minibatches=8 \
  max_concurrent_rollout_groups=1 \
  loss_fn=importance_sampling \
  seed=2 \
  max_tokens=1024 \
  temperature=1.0 \
  save_every=85 \
  rolling_save_every=1 \
  index_jsonl=data/browsecomp_plus/train/index.jsonl \
  excluded_qids_jsonl=data/splits/excluded_ids.jsonl \
  eval_index_jsonl=data/browsecomp_plus/heldout50/index.jsonl \
  reward_mode=hybrid \
  max_turns=32 \
  max_trajectory_tokens=60000 \
  step_penalty=0.0 \
  termination_penalty=0.0 \
  empty_synthetic_penalty=0.0 \
  synthetic_success_bonus=0.0 \
  synthetic_usage_bonus=0.0 \
  raw_usage_ratio_penalty=0.0 \
  filesystem_maturity_scale=0.0 \
  mature_stop_bonus=0.0 \
  answerer_workspace_mode=synthetic_only \
  answerer_max_turns=32 \
  answerer_final_answer_max_tokens=128 \
  answerer_retrieval_cost_scale=0.0 \
  answerer_synthetic_read_cost_scale=0.0 \
  terminal_answerer_repeats=4 \
  answerability_delta_reward_scale=1.0 \
  answerability_delta_min_abs=0.25 \
  answerability_delta_allow_negative=true \
  answerability_probe_max_per_episode=4 \
  answerability_probe_interval_turns=8 \
  answerability_probe_min_maturity=0.45 \
  answerability_probe_repeats=4 \
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
  step_construction_action_bonus=0.0 \
  step_filesystem_maturity_delta_scale=0.0 \
  step_non_construction_turn_penalty=0.0 \
  step_non_construction_streak_penalty=0.0 \
  step_tool_error_penalty=0.0 \
  log_path=outputs/training/browsecomp_plus \
  wandb_project="${WANDB_PROJECT:-librarian-agents}" \
  wandb_name=browsecomp-plus-paper
