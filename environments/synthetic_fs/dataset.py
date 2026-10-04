from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import chz
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder

from environments.synthetic_fs.common import (
    DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS,
    DEFAULT_BUILDER_EXECUTOR_BASE_URL,
    LOGGER,
    AnswererWorkspaceMode,
    JudgeBackend,
)
from environments.synthetic_fs.message_env import (
    build_redacting_agent_tool_env,
    initial_messages_builder,
)
from environments.synthetic_fs.reward import SyntheticFilesystemReward
from environments.synthetic_fs.workspace import (
    BuilderFilesystemTools,
    FrozenSyntheticFileExecutor,
    SyntheticFilesystemDatum,
    SyntheticFilesystemState,
)


@dataclass(frozen=True)
class SyntheticFilesystemEnvGroupBuilder(EnvGroupBuilder):
    datum: SyntheticFilesystemDatum
    model_name: str
    renderer_name: str | None
    group_size: int
    reward_mode: str = "hybrid"
    answerer_backend: JudgeBackend = "gemini"
    answerer_model: str = "gemini-3.1-flash-lite-preview"
    answerer_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    answerer_api_key_env: str = "GEMINI_API_KEY"
    judge_backend: JudgeBackend = "gemini"
    judge_model: str = "gemini-3.1-flash-lite-preview"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    judge_api_key_env: str = "GEMINI_API_KEY"
    judge_score_mode: str = "binary"
    max_turns: int = 32
    max_trajectory_tokens: int | None = 140000
    max_generation_tokens: int | None = None
    step_penalty: float = 0.0
    termination_penalty: float = 0.1
    raw_docs_penalty: float = 0.0
    empty_synthetic_penalty: float = 1.0
    synthetic_success_bonus: float = 0.0
    synthetic_usage_bonus: float = 0.0
    raw_usage_ratio_penalty: float = 0.0
    filesystem_maturity_scale: float = 0.5
    filesystem_coverage_weight: float = 0.35
    filesystem_direct_coverage_weight: float = 0.0
    filesystem_expansion_weight: float = 0.3
    filesystem_organization_weight: float = 0.35
    filesystem_stop_weight: float = 0.0
    mature_stop_bonus: float = 0.0
    mature_stop_min_score: float = 0.8
    terminal_reward_clip_min: float = -1.0
    terminal_reward_clip_max: float = 3.0
    answerer_max_turns: int = 32
    answerer_max_output_tokens: int = 512
    answerer_temperature: float = 0.0
    answerer_top_p: float | None = None
    answerer_workspace_mode: AnswererWorkspaceMode = "synthetic_only"
    answerer_final_answer_max_tokens: int = 512
    answerer_retrieval_cost_scale: float = 0.0
    answerer_retrieval_cost_token_unit: float = 1000.0
    answerer_retrieval_cost_correct_only: bool = True
    answerer_synthetic_read_cost_scale: float = 0.0
    answerer_synthetic_read_cost_unit: float = 10.0
    terminal_answerer_repeats: int = 4
    terminal_answerer_enabled: bool = True
    answerability_delta_reward_scale: float = 0.0
    answerability_delta_min_abs: float = 0.25
    answerability_delta_allow_negative: bool = True
    answerability_probe_max_per_episode: int = 4
    answerability_probe_interval_turns: int = 8
    answerability_probe_min_maturity: float = 0.45
    answerability_probe_repeats: int = 4
    answerer_repeat_concurrency: int = 1
    judge_max_output_tokens: int = 64
    log_step_details: bool = False
    log_compaction_summaries: bool = False
    retain_reward_tool_messages: bool = False
    trim_terminal_history_for_memory: bool = True
    return_empty_terminal_observation: bool = True
    clear_state_on_terminal_for_memory: bool = True
    builder_compaction_enabled: bool = True
    builder_compaction_backend: JudgeBackend = "gemini"
    builder_compaction_model: str = "gemini-3.1-flash-lite-preview"
    builder_compaction_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    builder_compaction_api_key_env: str = "GEMINI_API_KEY"
    builder_compaction_trigger_tokens: int = 3000
    builder_compaction_keep_recent_turns: int = 1
    builder_compaction_max_output_tokens: int = 800
    builder_compaction_input_max_chars: int = DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS
    builder_compaction_target_tokens: int = 0
    builder_executor_enabled: bool = True
    builder_batch_tools_enabled: bool = True
    builder_coverage_first: bool = False
    builder_coverage_leaf_max_browse_turns: int = 0
    builder_enforce_datum_file_budget: bool = False
    builder_task_conditioned: bool = False
    builder_include_source_inventory: bool = False
    builder_plan_once_prompt_aligned: bool = False
    builder_raw_read_max_chars: int = 12000
    builder_executor_backend: JudgeBackend = "vllm"
    builder_executor_model: str = "Qwen/Qwen3.5-35B-A3B"
    builder_executor_base_url: str = DEFAULT_BUILDER_EXECUTOR_BASE_URL
    builder_executor_api_key_env: str = "BUILDER_EXECUTOR_API_KEY"
    builder_executor_max_source_chars: int = 16000
    builder_executor_max_output_tokens: int = 512
    builder_executor_temperature: float = 0.0
    builder_executor_top_k: int = 0
    builder_executor_deterministic: bool = False
    step_construction_action_bonus: float = 0.05
    step_filesystem_maturity_delta_scale: float = 0.5
    step_non_construction_turn_penalty: float = 0.005
    step_non_construction_streak_penalty: float = 0.0
    step_non_construction_streak_free: int = 3
    step_tool_error_penalty: float = 0.05
    filesystem_snapshot_dir: str = ""
    training_progress_metrics: tuple[tuple[str, float], ...] = ()

    async def make_envs(self) -> Sequence[Env]:
        if (
            self.builder_plan_once_prompt_aligned
            and self.builder_executor_deterministic
            and self.builder_executor_model != "local-extractive-v1"
        ):
            raise ValueError(
                "Deterministic plan-once construction must identify its executor as "
                "local-extractive-v1, not as a provider model"
            )
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)
        envs: list[Env] = []
        raw_doc_root = Path(self.datum["agent_query_dir"])
        visible_raw_doc_paths = [
            str(file_info.get("relative_path", ""))
            for file_info in self.datum.get("files", [])
            if str(file_info.get("relative_path", ""))
        ]
        construction_budget_active_files = (
            int(self.datum.get("construction_budget_active_files", 0) or 0)
            if self.builder_enforce_datum_file_budget
            else 0
        )
        if construction_budget_active_files < 0:
            raise ValueError("construction_budget_active_files must be non-negative")
        for trajectory_idx in range(self.group_size):
            state = SyntheticFilesystemState(
                raw_doc_root=raw_doc_root,
                visible_raw_doc_paths=visible_raw_doc_paths,
                question_id=self.datum["question_id"],
                raw_read_max_chars=self.builder_raw_read_max_chars,
            )
            executor = (
                FrozenSyntheticFileExecutor(
                    backend=self.builder_executor_backend,
                    model=self.builder_executor_model,
                    base_url=self.builder_executor_base_url,
                    api_key_env=self.builder_executor_api_key_env,
                    max_source_chars=self.builder_executor_max_source_chars,
                    max_output_tokens=self.builder_executor_max_output_tokens,
                    temperature=self.builder_executor_temperature,
                    top_k=self.builder_executor_top_k or None,
                    deterministic=self.builder_executor_deterministic,
                )
                if self.builder_executor_enabled
                else None
            )
            fs_tools = BuilderFilesystemTools(state, executor=executor)
            tools = [
                fs_tools.list_files,
                fs_tools.read_file,
            ]
            if self.builder_batch_tools_enabled or self.builder_executor_enabled:
                tools.append(fs_tools.read_many)
            tools.extend(
                [
                    fs_tools.create_cluster,
                    fs_tools.merge_clusters,
                ]
            )
            if self.builder_batch_tools_enabled or self.builder_executor_enabled:
                tools.extend(
                    [
                        fs_tools.create_clusters,
                        fs_tools.merge_many,
                    ]
                )
            tools.append(fs_tools.delete)
            reward_fn = SyntheticFilesystemReward(
                state=state,
                gold_answer=self.datum["gold_answer"],
                question=self.datum["question"],
                answerer_backend=self.answerer_backend,
                answerer_model=self.answerer_model,
                answerer_base_url=self.answerer_base_url,
                answerer_api_key_env=self.answerer_api_key_env,
                reward_mode=self.reward_mode,
                judge_backend=self.judge_backend,
                judge_model=self.judge_model,
                judge_base_url=self.judge_base_url,
                judge_api_key_env=self.judge_api_key_env,
                judge_score_mode=self.judge_score_mode,
                step_penalty=self.step_penalty,
                termination_penalty=self.termination_penalty,
                raw_docs_penalty=self.raw_docs_penalty,
                empty_synthetic_penalty=self.empty_synthetic_penalty,
                synthetic_success_bonus=self.synthetic_success_bonus,
                synthetic_usage_bonus=self.synthetic_usage_bonus,
                raw_usage_ratio_penalty=self.raw_usage_ratio_penalty,
                filesystem_maturity_scale=self.filesystem_maturity_scale,
                filesystem_coverage_weight=self.filesystem_coverage_weight,
                filesystem_direct_coverage_weight=self.filesystem_direct_coverage_weight,
                filesystem_expansion_weight=self.filesystem_expansion_weight,
                filesystem_organization_weight=self.filesystem_organization_weight,
                filesystem_stop_weight=self.filesystem_stop_weight,
                mature_stop_bonus=self.mature_stop_bonus,
                mature_stop_min_score=self.mature_stop_min_score,
                terminal_reward_clip_min=self.terminal_reward_clip_min,
                terminal_reward_clip_max=self.terminal_reward_clip_max,
                builder_max_turns=self.max_turns,
                answerer_max_turns=self.answerer_max_turns,
                answerer_max_output_tokens=self.answerer_max_output_tokens,
                answerer_temperature=self.answerer_temperature,
                answerer_top_p=self.answerer_top_p,
                answerer_workspace_mode=self.answerer_workspace_mode,
                answerer_final_answer_max_tokens=self.answerer_final_answer_max_tokens,
                answerer_retrieval_cost_scale=self.answerer_retrieval_cost_scale,
                answerer_retrieval_cost_token_unit=self.answerer_retrieval_cost_token_unit,
                answerer_retrieval_cost_correct_only=self.answerer_retrieval_cost_correct_only,
                answerer_synthetic_read_cost_scale=self.answerer_synthetic_read_cost_scale,
                answerer_synthetic_read_cost_unit=self.answerer_synthetic_read_cost_unit,
                terminal_answerer_repeats=self.terminal_answerer_repeats,
                terminal_answerer_enabled=self.terminal_answerer_enabled,
                answerability_delta_reward_scale=self.answerability_delta_reward_scale,
                answerability_delta_min_abs=self.answerability_delta_min_abs,
                answerability_delta_allow_negative=self.answerability_delta_allow_negative,
                answerability_probe_repeats=self.answerability_probe_repeats,
                answerer_repeat_concurrency=self.answerer_repeat_concurrency,
                judge_max_output_tokens=self.judge_max_output_tokens,
                snapshot_path=(
                    Path(self.filesystem_snapshot_dir)
                    / self.datum["question_id"]
                    / f"trajectory_{trajectory_idx:02d}.json"
                    if self.filesystem_snapshot_dir
                    else None
                ),
            )
            envs.append(
                build_redacting_agent_tool_env(
                    renderer=renderer,
                    tools=tools,
                    initial_messages=initial_messages_builder(
                        self.datum,
                        renderer,
                        fs_tools,
                        executor_enabled=self.builder_executor_enabled,
                        batch_tools_enabled=self.builder_batch_tools_enabled
                        or self.builder_executor_enabled,
                        coverage_first=self.builder_coverage_first,
                        coverage_leaf_max_browse_turns=self.builder_coverage_leaf_max_browse_turns,
                        task_question=self.datum["question"] if self.builder_task_conditioned else None,
                        source_inventory=(
                            visible_raw_doc_paths if self.builder_include_source_inventory else None
                        ),
                        construction_budget_active_files=construction_budget_active_files,
                        plan_once_prompt_aligned=self.builder_plan_once_prompt_aligned,
                        explicit_construction_instruction=(
                            self.datum["question"] if self.builder_plan_once_prompt_aligned else None
                        ),
                    ),
                    reward_fn=reward_fn,
                    max_turns=self.max_turns,
                    state=state,
                    filesystem_maturity_score_fn=reward_fn._current_filesystem_maturity_score,
                    step_construction_action_bonus=self.step_construction_action_bonus,
                    step_filesystem_maturity_delta_scale=self.step_filesystem_maturity_delta_scale,
                    step_non_construction_turn_penalty=self.step_non_construction_turn_penalty,
                    step_non_construction_streak_penalty=self.step_non_construction_streak_penalty,
                    step_non_construction_streak_free=self.step_non_construction_streak_free,
                    step_tool_error_penalty=self.step_tool_error_penalty,
                    coverage_first=self.builder_coverage_first,
                    coverage_leaf_max_browse_turns=self.builder_coverage_leaf_max_browse_turns,
                    minimum_active_files_before_stop=construction_budget_active_files,
                    answerability_delta_reward_scale=self.answerability_delta_reward_scale,
                    answerability_probe_max_per_episode=self.answerability_probe_max_per_episode,
                    answerability_probe_interval_turns=self.answerability_probe_interval_turns,
                    answerability_probe_min_maturity=self.answerability_probe_min_maturity,
                    log_step_details=self.log_step_details,
                    log_compaction_summaries=self.log_compaction_summaries,
                    retain_reward_tool_messages=self.retain_reward_tool_messages,
                    trim_terminal_history_for_memory=self.trim_terminal_history_for_memory,
                    return_empty_terminal_observation=self.return_empty_terminal_observation,
                    clear_state_on_terminal_for_memory=self.clear_state_on_terminal_for_memory,
                    compaction_enabled=self.builder_compaction_enabled,
                    compaction_backend=self.builder_compaction_backend,
                    compaction_model=self.builder_compaction_model,
                    compaction_base_url=self.builder_compaction_base_url,
                    compaction_api_key_env=self.builder_compaction_api_key_env,
                    compaction_trigger_tokens=self.builder_compaction_trigger_tokens,
                    compaction_keep_recent_turns=self.builder_compaction_keep_recent_turns,
                    compaction_max_output_tokens=self.builder_compaction_max_output_tokens,
                    compaction_input_max_chars=self.builder_compaction_input_max_chars,
                    compaction_target_tokens=self.builder_compaction_target_tokens,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    progress_metrics=dict(self.training_progress_metrics),
                    # Leave per-turn generation uncapped inside the environment.
                )
            )
        return envs

    def logging_tags(self) -> list[str]:
        return [self.datum.get("dataset_type", "support_only"), "synthetic_filesystem_builder"]


class SyntheticFilesystemRLDataset(RLDataset):
    def __init__(
        self,
        env_group_builders: list[SyntheticFilesystemEnvGroupBuilder],
        batch_size: int,
        *,
        num_epochs: int = 1,
        reshuffle_each_epoch: bool = True,
        seed: int = 0,
    ):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size
        self.num_epochs = max(1, num_epochs)
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.seed = seed
        self._epoch_cache: dict[int, list[SyntheticFilesystemEnvGroupBuilder]] = {}

    def _batches_per_epoch(self) -> int:
        return (len(self.env_group_builders) + self.batch_size - 1) // self.batch_size

    def _builders_for_epoch(self, epoch_index: int) -> list[SyntheticFilesystemEnvGroupBuilder]:
        if not self.reshuffle_each_epoch:
            return self.env_group_builders
        if epoch_index not in self._epoch_cache:
            builders = list(self.env_group_builders)
            random.Random(self.seed + epoch_index).shuffle(builders)
            self._epoch_cache[epoch_index] = builders
        return self._epoch_cache[epoch_index]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batches_per_epoch = max(1, self._batches_per_epoch())
        epoch_index = index // batches_per_epoch
        batch_index_in_epoch = index % batches_per_epoch
        epoch_builders = self._builders_for_epoch(epoch_index)
        start = batch_index_in_epoch * self.batch_size
        end = start + self.batch_size
        batch_builders = epoch_builders[start:end]
        examples_this_batch = len(batch_builders)
        examples_seen = epoch_index * len(self.env_group_builders) + start + examples_this_batch
        progress_metrics = tuple(
            sorted(
                {
                    "training_batch_index": float(index + 1),
                    "training_batch_in_epoch": float(batch_index_in_epoch + 1),
                    "training_batches_per_epoch": float(batches_per_epoch),
                    "training_epoch": float(epoch_index)
                    + float(batch_index_in_epoch + 1) / float(batches_per_epoch),
                    "training_epoch_index": float(epoch_index),
                    "training_epochs_total": float(self.num_epochs),
                    "training_examples_per_batch": float(self.batch_size),
                    "training_examples_seen": float(examples_seen),
                    "training_rollouts_seen": float(
                        examples_seen * self.env_group_builders[0].group_size
                    )
                    if self.env_group_builders
                    else 0.0,
                    "training_train_rows": float(len(self.env_group_builders)),
                }.items()
            )
        )
        return [
            replace(builder, training_progress_metrics=progress_metrics)
            for builder in batch_builders
        ]

    def __len__(self) -> int:
        return self._batches_per_epoch() * self.num_epochs


@chz.chz
class SyntheticFilesystemDatasetBuilder(RLDatasetBuilder):
    index_jsonl: str
    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    train_epochs: int = 1
    reshuffle_each_epoch: bool = True
    renderer_name: str | None = None
    reward_mode: str = "hybrid"
    answerer_backend: JudgeBackend = "gemini"
    answerer_model: str = "gemini-3.1-flash-lite-preview"
    answerer_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    answerer_api_key_env: str = "GEMINI_API_KEY"
    judge_backend: JudgeBackend = "gemini"
    judge_model: str = "gemini-3.1-flash-lite-preview"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    judge_api_key_env: str = "GEMINI_API_KEY"
    judge_score_mode: str = "binary"
    max_turns: int = 32
    max_trajectory_tokens: int | None = 140000
    max_generation_tokens: int | None = None
    step_penalty: float = 0.0
    termination_penalty: float = 0.1
    raw_docs_penalty: float = 0.0
    empty_synthetic_penalty: float = 1.0
    synthetic_success_bonus: float = 0.0
    synthetic_usage_bonus: float = 0.0
    raw_usage_ratio_penalty: float = 0.0
    filesystem_maturity_scale: float = 0.5
    filesystem_coverage_weight: float = 0.35
    filesystem_direct_coverage_weight: float = 0.0
    filesystem_expansion_weight: float = 0.3
    filesystem_organization_weight: float = 0.35
    filesystem_stop_weight: float = 0.0
    mature_stop_bonus: float = 0.0
    mature_stop_min_score: float = 0.8
    terminal_reward_clip_min: float = -1.0
    terminal_reward_clip_max: float = 3.0
    answerer_max_turns: int = 32
    answerer_max_output_tokens: int = 512
    answerer_temperature: float = 0.0
    answerer_top_p: float | None = None
    answerer_workspace_mode: AnswererWorkspaceMode = "synthetic_only"
    answerer_final_answer_max_tokens: int = 512
    answerer_retrieval_cost_scale: float = 0.0
    answerer_retrieval_cost_token_unit: float = 1000.0
    answerer_retrieval_cost_correct_only: bool = True
    answerer_synthetic_read_cost_scale: float = 0.0
    answerer_synthetic_read_cost_unit: float = 10.0
    terminal_answerer_repeats: int = 4
    terminal_answerer_enabled: bool = True
    answerability_delta_reward_scale: float = 0.0
    answerability_delta_min_abs: float = 0.25
    answerability_delta_allow_negative: bool = True
    answerability_probe_max_per_episode: int = 4
    answerability_probe_interval_turns: int = 8
    answerability_probe_min_maturity: float = 0.45
    answerability_probe_repeats: int = 4
    answerer_repeat_concurrency: int = 1
    judge_max_output_tokens: int = 64
    log_step_details: bool = False
    log_compaction_summaries: bool = False
    retain_reward_tool_messages: bool = False
    trim_terminal_history_for_memory: bool = True
    return_empty_terminal_observation: bool = True
    clear_state_on_terminal_for_memory: bool = True
    builder_compaction_enabled: bool = True
    builder_compaction_backend: JudgeBackend = "gemini"
    builder_compaction_model: str = "gemini-3.1-flash-lite-preview"
    builder_compaction_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    builder_compaction_api_key_env: str = "GEMINI_API_KEY"
    builder_compaction_trigger_tokens: int = 3000
    builder_compaction_keep_recent_turns: int = 1
    builder_compaction_max_output_tokens: int = 800
    builder_compaction_input_max_chars: int = DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS
    builder_compaction_target_tokens: int = 0
    builder_executor_enabled: bool = True
    builder_batch_tools_enabled: bool = True
    builder_coverage_first: bool = False
    builder_coverage_leaf_max_browse_turns: int = 0
    builder_enforce_datum_file_budget: bool = False
    builder_task_conditioned: bool = False
    builder_include_source_inventory: bool = False
    builder_plan_once_prompt_aligned: bool = False
    builder_raw_read_max_chars: int = 12000
    builder_executor_backend: JudgeBackend = "vllm"
    builder_executor_model: str = "Qwen/Qwen3.5-35B-A3B"
    builder_executor_base_url: str = DEFAULT_BUILDER_EXECUTOR_BASE_URL
    builder_executor_api_key_env: str = "BUILDER_EXECUTOR_API_KEY"
    builder_executor_max_source_chars: int = 16000
    builder_executor_max_output_tokens: int = 512
    builder_executor_temperature: float = 0.0
    builder_executor_top_k: int = 0
    builder_executor_deterministic: bool = False
    step_construction_action_bonus: float = 0.05
    step_filesystem_maturity_delta_scale: float = 0.5
    step_non_construction_turn_penalty: float = 0.005
    step_non_construction_streak_penalty: float = 0.0
    step_non_construction_streak_free: int = 3
    step_tool_error_penalty: float = 0.05
    filesystem_snapshot_dir: str = ""
    excluded_qids_jsonl: str = ""
    eval_index_jsonl: str = ""
    seed: int = 0
    eval_size: int = 0
    limit: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        data = load_index(Path(self.index_jsonl))
        excluded = (
            load_excluded_qids(Path(self.excluded_qids_jsonl))
            if self.excluded_qids_jsonl
            else set()
        )
        if excluded:
            data = [datum for datum in data if datum["question_id"] not in excluded]

        eval_rows: list[SyntheticFilesystemDatum] = []
        if self.eval_index_jsonl:
            eval_rows = load_index(Path(self.eval_index_jsonl))
            if excluded:
                eval_rows = [datum for datum in eval_rows if datum["question_id"] not in excluded]
            eval_qids = {datum["question_id"] for datum in eval_rows}
            data = [datum for datum in data if datum["question_id"] not in eval_qids]

        rng = random.Random(self.seed)
        rng.shuffle(data)
        if self.limit > 0:
            data = data[: self.limit]

        train_rows = data
        if not eval_rows and self.eval_size > 0:
            eval_rows = data[: self.eval_size]
            train_rows = data[self.eval_size :]
        if not train_rows:
            raise ValueError(
                "No training rows remain after applying excluded_qids_jsonl/eval_index_jsonl/limit. "
                "Use a training index that does not fully overlap the held-out eval index."
            )
        LOGGER.info(
            "Synthetic FS dataset split: train_rows=%d eval_rows=%d excluded_qids=%d eval_index_jsonl=%s",
            len(train_rows),
            len(eval_rows),
            len(excluded),
            self.eval_index_jsonl or "(none; eval_size split)"
            if self.eval_size > 0
            else self.eval_index_jsonl or "(none)",
        )

        train_builders = [
            SyntheticFilesystemEnvGroupBuilder(
                datum=datum,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
                reward_mode=self.reward_mode,
                answerer_backend=self.answerer_backend,
                answerer_model=self.answerer_model,
                answerer_base_url=self.answerer_base_url,
                answerer_api_key_env=self.answerer_api_key_env,
                judge_backend=self.judge_backend,
                judge_model=self.judge_model,
                judge_base_url=self.judge_base_url,
                judge_api_key_env=self.judge_api_key_env,
                judge_score_mode=self.judge_score_mode,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                step_penalty=self.step_penalty,
                termination_penalty=self.termination_penalty,
                raw_docs_penalty=self.raw_docs_penalty,
                empty_synthetic_penalty=self.empty_synthetic_penalty,
                synthetic_success_bonus=self.synthetic_success_bonus,
                synthetic_usage_bonus=self.synthetic_usage_bonus,
                raw_usage_ratio_penalty=self.raw_usage_ratio_penalty,
                filesystem_maturity_scale=self.filesystem_maturity_scale,
                filesystem_coverage_weight=self.filesystem_coverage_weight,
                filesystem_direct_coverage_weight=self.filesystem_direct_coverage_weight,
                filesystem_expansion_weight=self.filesystem_expansion_weight,
                filesystem_organization_weight=self.filesystem_organization_weight,
                filesystem_stop_weight=self.filesystem_stop_weight,
                mature_stop_bonus=self.mature_stop_bonus,
                mature_stop_min_score=self.mature_stop_min_score,
                terminal_reward_clip_min=self.terminal_reward_clip_min,
                terminal_reward_clip_max=self.terminal_reward_clip_max,
                answerer_max_turns=self.answerer_max_turns,
                answerer_max_output_tokens=self.answerer_max_output_tokens,
                answerer_temperature=self.answerer_temperature,
                answerer_top_p=self.answerer_top_p,
                answerer_workspace_mode=self.answerer_workspace_mode,
                answerer_final_answer_max_tokens=self.answerer_final_answer_max_tokens,
                answerer_retrieval_cost_scale=self.answerer_retrieval_cost_scale,
                answerer_retrieval_cost_token_unit=self.answerer_retrieval_cost_token_unit,
                answerer_retrieval_cost_correct_only=self.answerer_retrieval_cost_correct_only,
                answerer_synthetic_read_cost_scale=self.answerer_synthetic_read_cost_scale,
                answerer_synthetic_read_cost_unit=self.answerer_synthetic_read_cost_unit,
                terminal_answerer_repeats=self.terminal_answerer_repeats,
                terminal_answerer_enabled=self.terminal_answerer_enabled,
                answerability_delta_reward_scale=self.answerability_delta_reward_scale,
                answerability_delta_min_abs=self.answerability_delta_min_abs,
                answerability_delta_allow_negative=self.answerability_delta_allow_negative,
                answerability_probe_max_per_episode=self.answerability_probe_max_per_episode,
                answerability_probe_interval_turns=self.answerability_probe_interval_turns,
                answerability_probe_min_maturity=self.answerability_probe_min_maturity,
                answerability_probe_repeats=self.answerability_probe_repeats,
                answerer_repeat_concurrency=self.answerer_repeat_concurrency,
                judge_max_output_tokens=self.judge_max_output_tokens,
                log_step_details=self.log_step_details,
                log_compaction_summaries=self.log_compaction_summaries,
                retain_reward_tool_messages=self.retain_reward_tool_messages,
                trim_terminal_history_for_memory=self.trim_terminal_history_for_memory,
                return_empty_terminal_observation=self.return_empty_terminal_observation,
                clear_state_on_terminal_for_memory=self.clear_state_on_terminal_for_memory,
                builder_compaction_enabled=self.builder_compaction_enabled,
                builder_compaction_backend=self.builder_compaction_backend,
                builder_compaction_model=self.builder_compaction_model,
                builder_compaction_base_url=self.builder_compaction_base_url,
                builder_compaction_api_key_env=self.builder_compaction_api_key_env,
                builder_compaction_trigger_tokens=self.builder_compaction_trigger_tokens,
                builder_compaction_keep_recent_turns=self.builder_compaction_keep_recent_turns,
                builder_compaction_max_output_tokens=self.builder_compaction_max_output_tokens,
                builder_compaction_input_max_chars=self.builder_compaction_input_max_chars,
                builder_compaction_target_tokens=self.builder_compaction_target_tokens,
                builder_executor_enabled=self.builder_executor_enabled,
                builder_batch_tools_enabled=self.builder_batch_tools_enabled,
                builder_coverage_first=self.builder_coverage_first,
                builder_coverage_leaf_max_browse_turns=self.builder_coverage_leaf_max_browse_turns,
                builder_enforce_datum_file_budget=self.builder_enforce_datum_file_budget,
                builder_task_conditioned=self.builder_task_conditioned,
                builder_include_source_inventory=self.builder_include_source_inventory,
                builder_plan_once_prompt_aligned=self.builder_plan_once_prompt_aligned,
                builder_raw_read_max_chars=self.builder_raw_read_max_chars,
                builder_executor_backend=self.builder_executor_backend,
                builder_executor_model=self.builder_executor_model,
                builder_executor_base_url=self.builder_executor_base_url,
                builder_executor_api_key_env=self.builder_executor_api_key_env,
                builder_executor_max_source_chars=self.builder_executor_max_source_chars,
                builder_executor_max_output_tokens=self.builder_executor_max_output_tokens,
                builder_executor_temperature=self.builder_executor_temperature,
                builder_executor_top_k=self.builder_executor_top_k,
                builder_executor_deterministic=self.builder_executor_deterministic,
                step_construction_action_bonus=self.step_construction_action_bonus,
                step_filesystem_maturity_delta_scale=self.step_filesystem_maturity_delta_scale,
                step_non_construction_turn_penalty=self.step_non_construction_turn_penalty,
                step_non_construction_streak_penalty=self.step_non_construction_streak_penalty,
                step_non_construction_streak_free=self.step_non_construction_streak_free,
                step_tool_error_penalty=self.step_tool_error_penalty,
                filesystem_snapshot_dir=self.filesystem_snapshot_dir,
            )
            for datum in train_rows
        ]
        train_dataset = SyntheticFilesystemRLDataset(
            train_builders,
            batch_size=self.batch_size,
            num_epochs=self.train_epochs,
            reshuffle_each_epoch=self.reshuffle_each_epoch,
            seed=self.seed,
        )

        if not eval_rows:
            return train_dataset, None

        eval_builders = [
            SyntheticFilesystemEnvGroupBuilder(
                datum=datum,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
                reward_mode=self.reward_mode,
                answerer_backend=self.answerer_backend,
                answerer_model=self.answerer_model,
                answerer_base_url=self.answerer_base_url,
                answerer_api_key_env=self.answerer_api_key_env,
                judge_backend=self.judge_backend,
                judge_model=self.judge_model,
                judge_base_url=self.judge_base_url,
                judge_api_key_env=self.judge_api_key_env,
                judge_score_mode=self.judge_score_mode,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                step_penalty=self.step_penalty,
                termination_penalty=self.termination_penalty,
                raw_docs_penalty=self.raw_docs_penalty,
                empty_synthetic_penalty=self.empty_synthetic_penalty,
                synthetic_success_bonus=self.synthetic_success_bonus,
                synthetic_usage_bonus=self.synthetic_usage_bonus,
                raw_usage_ratio_penalty=self.raw_usage_ratio_penalty,
                filesystem_maturity_scale=self.filesystem_maturity_scale,
                filesystem_coverage_weight=self.filesystem_coverage_weight,
                filesystem_direct_coverage_weight=self.filesystem_direct_coverage_weight,
                filesystem_expansion_weight=self.filesystem_expansion_weight,
                filesystem_organization_weight=self.filesystem_organization_weight,
                filesystem_stop_weight=self.filesystem_stop_weight,
                mature_stop_bonus=self.mature_stop_bonus,
                mature_stop_min_score=self.mature_stop_min_score,
                terminal_reward_clip_min=self.terminal_reward_clip_min,
                terminal_reward_clip_max=self.terminal_reward_clip_max,
                answerer_max_turns=self.answerer_max_turns,
                answerer_max_output_tokens=self.answerer_max_output_tokens,
                answerer_temperature=self.answerer_temperature,
                answerer_top_p=self.answerer_top_p,
                answerer_workspace_mode=self.answerer_workspace_mode,
                answerer_final_answer_max_tokens=self.answerer_final_answer_max_tokens,
                answerer_retrieval_cost_scale=self.answerer_retrieval_cost_scale,
                answerer_retrieval_cost_token_unit=self.answerer_retrieval_cost_token_unit,
                answerer_retrieval_cost_correct_only=self.answerer_retrieval_cost_correct_only,
                answerer_synthetic_read_cost_scale=self.answerer_synthetic_read_cost_scale,
                answerer_synthetic_read_cost_unit=self.answerer_synthetic_read_cost_unit,
                terminal_answerer_repeats=self.terminal_answerer_repeats,
                terminal_answerer_enabled=self.terminal_answerer_enabled,
                answerability_delta_reward_scale=self.answerability_delta_reward_scale,
                answerability_delta_min_abs=self.answerability_delta_min_abs,
                answerability_delta_allow_negative=self.answerability_delta_allow_negative,
                answerability_probe_max_per_episode=self.answerability_probe_max_per_episode,
                answerability_probe_interval_turns=self.answerability_probe_interval_turns,
                answerability_probe_min_maturity=self.answerability_probe_min_maturity,
                answerability_probe_repeats=self.answerability_probe_repeats,
                answerer_repeat_concurrency=self.answerer_repeat_concurrency,
                judge_max_output_tokens=self.judge_max_output_tokens,
                log_step_details=self.log_step_details,
                log_compaction_summaries=self.log_compaction_summaries,
                retain_reward_tool_messages=self.retain_reward_tool_messages,
                trim_terminal_history_for_memory=self.trim_terminal_history_for_memory,
                return_empty_terminal_observation=self.return_empty_terminal_observation,
                clear_state_on_terminal_for_memory=self.clear_state_on_terminal_for_memory,
                builder_compaction_enabled=self.builder_compaction_enabled,
                builder_compaction_backend=self.builder_compaction_backend,
                builder_compaction_model=self.builder_compaction_model,
                builder_compaction_base_url=self.builder_compaction_base_url,
                builder_compaction_api_key_env=self.builder_compaction_api_key_env,
                builder_compaction_trigger_tokens=self.builder_compaction_trigger_tokens,
                builder_compaction_keep_recent_turns=self.builder_compaction_keep_recent_turns,
                builder_compaction_max_output_tokens=self.builder_compaction_max_output_tokens,
                builder_compaction_input_max_chars=self.builder_compaction_input_max_chars,
                builder_compaction_target_tokens=self.builder_compaction_target_tokens,
                builder_executor_enabled=self.builder_executor_enabled,
                builder_batch_tools_enabled=self.builder_batch_tools_enabled,
                builder_coverage_first=self.builder_coverage_first,
                builder_coverage_leaf_max_browse_turns=self.builder_coverage_leaf_max_browse_turns,
                builder_enforce_datum_file_budget=self.builder_enforce_datum_file_budget,
                builder_task_conditioned=self.builder_task_conditioned,
                builder_include_source_inventory=self.builder_include_source_inventory,
                builder_plan_once_prompt_aligned=self.builder_plan_once_prompt_aligned,
                builder_raw_read_max_chars=self.builder_raw_read_max_chars,
                builder_executor_backend=self.builder_executor_backend,
                builder_executor_model=self.builder_executor_model,
                builder_executor_base_url=self.builder_executor_base_url,
                builder_executor_api_key_env=self.builder_executor_api_key_env,
                builder_executor_max_source_chars=self.builder_executor_max_source_chars,
                builder_executor_max_output_tokens=self.builder_executor_max_output_tokens,
                builder_executor_temperature=self.builder_executor_temperature,
                builder_executor_top_k=self.builder_executor_top_k,
                builder_executor_deterministic=self.builder_executor_deterministic,
                step_construction_action_bonus=self.step_construction_action_bonus,
                step_filesystem_maturity_delta_scale=self.step_filesystem_maturity_delta_scale,
                step_non_construction_turn_penalty=self.step_non_construction_turn_penalty,
                step_non_construction_streak_penalty=self.step_non_construction_streak_penalty,
                step_non_construction_streak_free=self.step_non_construction_streak_free,
                step_tool_error_penalty=self.step_tool_error_penalty,
                filesystem_snapshot_dir=self.filesystem_snapshot_dir,
            )
            for datum in eval_rows
        ]
        eval_dataset = SyntheticFilesystemRLDataset(
            eval_builders,
            batch_size=self.batch_size,
            num_epochs=1,
            reshuffle_each_epoch=False,
            seed=self.seed,
        )
        return train_dataset, eval_dataset


def load_excluded_qids(path: Path) -> set[str]:
    qids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", "")).strip()
            if qid:
                qids.add(qid)
    return qids


def load_index(path: Path) -> list[SyntheticFilesystemDatum]:
    path = path.resolve()
    rows: list[SyntheticFilesystemDatum] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ["question_id", "agent_query_dir", "privileged_query_dir", "num_docs"]
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{path}:{line_num} missing required keys: {missing}")
            rows.append(hydrate_privileged_record(row, base_dir=path.parent))
    if not rows:
        raise ValueError(f"No rows found in index: {path}")
    return rows


def hydrate_privileged_record(
    row: dict[str, Any], *, base_dir: Path | None = None
) -> SyntheticFilesystemDatum:
    qid = str(row["question_id"])
    privileged_query_dir = Path(row["privileged_query_dir"])
    agent_query_dir = Path(row["agent_query_dir"])
    if base_dir is not None:
        if not privileged_query_dir.is_absolute():
            privileged_query_dir = (base_dir / privileged_query_dir).resolve()
        if not agent_query_dir.is_absolute():
            agent_query_dir = (base_dir / agent_query_dir).resolve()
    query_path = privileged_query_dir / "query.txt"
    answer_path = privileged_query_dir / "answer.txt"
    if not query_path.exists():
        raise ValueError(f"Missing privileged query file for {qid}: {query_path}")
    if not answer_path.exists():
        raise ValueError(f"Missing privileged answer file for {qid}: {answer_path}")
    hydrated: SyntheticFilesystemDatum = {
        "question_id": qid,
        "question": query_path.read_text(encoding="utf-8").strip(),
        "gold_answer": answer_path.read_text(encoding="utf-8").strip(),
        "agent_query_dir": str(agent_query_dir),
        "privileged_query_dir": str(privileged_query_dir),
        "num_docs": int(row["num_docs"]),
        "dataset_type": str(row.get("dataset_type", "support_only")),
        "files": list(row.get("files", [])),
    }
    if "amortization_questions" in row:
        questions = row["amortization_questions"]
        if not isinstance(questions, list) or not all(
            isinstance(question, dict) for question in questions
        ):
            raise ValueError(f"Invalid amortization_questions for {qid}")
        hydrated["amortization_questions"] = questions
        hydrated["num_questions"] = int(row.get("num_questions", len(questions)))
    for key in (
        "construction_budget_active_files",
        "construction_budget_reference_builder_turns",
        "construction_budget_reference_builder_tokens",
    ):
        if key in row:
            hydrated[key] = int(row[key])  # type: ignore[literal-required]
    if "domain" in row:
        hydrated["domain"] = str(row["domain"])
    if "cluster_id" in row:
        hydrated["cluster_id"] = str(row["cluster_id"])
    return hydrated
