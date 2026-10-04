from __future__ import annotations

import asyncio
import json
import string
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any

from tinker_cookbook.renderers.base import Message
from tinker_cookbook.utils import trace

from environments.synthetic_fs.clients import (
    _api_key_from_env,
    _call_openai_compatible_chat_model_messages,
    _normalize_model_http_url,
    context_limited_completion_tokens,
    context_limited_prompt_tokens,
)
from environments.synthetic_fs.common import (
    ANSWER_PREFIX,
    APPROX_CHARS_PER_TOKEN,
    LOGGER,
    MODEL_HTTP_MAX_ATTEMPTS,
    MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS,
    MODEL_HTTP_TIMEOUT_SECONDS,
    READ_FILE_MAX_LINES,
    STOP_MESSAGE,
    AnswererWorkspaceMode,
    JudgeBackend,
)
from environments.synthetic_fs.prompts import (
    build_answerer_system_prompt,
    build_answerer_user_prompt,
)
from environments.synthetic_fs.workspace import (
    ReadOnlyAnswererWorkspaceTools,
    SyntheticFileRecord,
    SyntheticFilesystemState,
)

ANSWERER_CONTEXT_BOUNDARY_MAX_ATTEMPTS = 4


@dataclass
class SyntheticFilesystemReward:
    state: SyntheticFilesystemState
    gold_answer: str
    question: str
    answerer_backend: JudgeBackend = "gemini"
    answerer_model: str = "gemini-3.1-flash-lite-preview"
    answerer_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    answerer_api_key_env: str = "GEMINI_API_KEY"
    reward_mode: str = "hybrid"
    judge_backend: JudgeBackend = "gemini"
    judge_model: str = "gemini-3.1-flash-lite-preview"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    judge_api_key_env: str = "GEMINI_API_KEY"
    judge_score_mode: str = "binary"
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
    builder_max_turns: int = 32
    answerer_max_turns: int = 32
    answerer_max_output_tokens: int = 512
    answerer_max_total_read_chars: int = 0
    answerer_max_file_reads: int = 0
    answerer_max_total_named_reads: int = 0
    answerer_temperature: float = 0
    answerer_top_p: float | None = None
    answerer_top_k: int | None = None
    answerer_min_p: float | None = None
    answerer_presence_penalty: float | None = None
    answerer_repetition_penalty: float | None = None
    answerer_enable_thinking: bool | None = None
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
    answerability_probe_repeats: int = 4
    answerer_repeat_concurrency: int = 1
    judge_max_output_tokens: int = 64
    snapshot_path: Path | None = None
    last_answerer_trace: dict[str, Any] = field(default_factory=dict, init=False)

    def _answerer_usage_metrics(self, answerer_result: dict[str, Any]) -> dict[str, float]:
        read_paths = [str(path) for path in answerer_result.get("read_paths", [])]
        synthetic_read_count = sum(
            1 for path in read_paths if path != "README.md" and not path.startswith("raw_docs/")
        )
        raw_doc_read_count = sum(1 for path in read_paths if path.startswith("raw_docs/"))
        synthetic_chars_read = float(answerer_result.get("synthetic_chars_read", 0) or 0)
        raw_chars_read = float(answerer_result.get("raw_chars_read", 0) or 0)
        readme_chars_read = float(answerer_result.get("readme_chars_read", 0) or 0)
        synthetic_tokens_read = synthetic_chars_read / float(APPROX_CHARS_PER_TOKEN)
        raw_tokens_read = raw_chars_read / float(APPROX_CHARS_PER_TOKEN)
        readme_tokens_read = readme_chars_read / float(APPROX_CHARS_PER_TOKEN)
        total_answerer_tokens_read = synthetic_tokens_read + raw_tokens_read + readme_tokens_read
        retrieval_token_unit = max(1.0, float(self.answerer_retrieval_cost_token_unit))
        retrieval_cost_score = (
            1.0 - (1.0 / ((1.0 + synthetic_tokens_read / retrieval_token_unit) ** 0.5))
            if synthetic_tokens_read > 0.0
            else 0.0
        )
        synthetic_read_unit = max(1.0, float(self.answerer_synthetic_read_cost_unit))
        synthetic_read_cost_score = (
            1.0 - (1.0 / ((1.0 + float(synthetic_read_count) / synthetic_read_unit) ** 0.5))
            if synthetic_read_count > 0
            else 0.0
        )
        return {
            "synthetic_read_count": float(synthetic_read_count),
            "raw_doc_read_count": float(raw_doc_read_count),
            "answerer_synthetic_chars_read": synthetic_chars_read,
            "answerer_raw_chars_read": raw_chars_read,
            "answerer_readme_chars_read": readme_chars_read,
            "answerer_synthetic_tokens_read": synthetic_tokens_read,
            "answerer_raw_tokens_read": raw_tokens_read,
            "answerer_readme_tokens_read": readme_tokens_read,
            "answerer_total_tokens_read": total_answerer_tokens_read,
            "answerer_retrieval_cost_score": retrieval_cost_score,
            "answerer_synthetic_read_cost_score": synthetic_read_cost_score,
        }

    async def _score_answerer_output(
        self,
        *,
        answerer_output: str,
        synthetic_read_count: float,
        answerer_evaluated: float,
        terminal_step: float,
    ) -> dict[str, float]:
        extracted = self._extract_answer(answerer_output)
        candidate = extracted if extracted is not None else answerer_output.strip()
        # Judge noncanonical answers semantically.
        judge_candidate = candidate if candidate else "Unknown"
        exact_match = 0.0
        judge_used = 0.0
        judge_score = 0.0
        judge_error = 0.0
        judge_skipped_no_candidate = 0.0
        judge_skipped_no_synthetic_reads = 0.0
        if terminal_step < 1.0:
            correct = 0.0
        elif self.answerer_workspace_mode == "synthetic_only" and synthetic_read_count <= 0:
            correct = 0.0
            judge_skipped_no_synthetic_reads = float(answerer_evaluated >= 1.0)
        elif answerer_evaluated >= 1.0:
            exact_match = float(self._check_answer(candidate))
            if self.reward_mode == "exact":
                correct = exact_match
            elif self.reward_mode == "llm":
                judge_used = 1.0
                try:
                    judge_score = await self._llm_judge_score(judge_candidate)
                except Exception:
                    LOGGER.warning(
                        "LLM judge call failed; scoring candidate as incorrect.", exc_info=True
                    )
                    judge_error = 1.0
                    judge_score = 0.0
                correct = judge_score
            elif self.reward_mode == "hybrid":
                if exact_match >= 1.0:
                    correct = 1.0
                else:
                    judge_used = 1.0
                    try:
                        judge_score = await self._llm_judge_score(judge_candidate)
                    except Exception:
                        LOGGER.warning(
                            "LLM judge call failed; scoring candidate as incorrect.", exc_info=True
                        )
                        judge_error = 1.0
                        judge_score = 0.0
                    correct = judge_score
            else:
                raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
        else:
            correct = 0.0
            judge_skipped_no_candidate = float(answerer_evaluated >= 1.0)
        return {
            "correct": correct,
            "exact_match": exact_match,
            "judge_used": judge_used,
            "judge_score": judge_score,
            "judge_error": judge_error,
            "answerer_error": 0.0,
            "judge_skipped_no_candidate": judge_skipped_no_candidate,
            "judge_skipped_no_synthetic_reads": judge_skipped_no_synthetic_reads,
        }

    def _empty_answerer_result(self) -> dict[str, Any]:
        return {
            "answer_text": "",
            "steps": 0.0,
            "read_paths": [],
            "used_raw_docs": False,
            "used_readme": False,
            "freeform_final_answer": False,
            "prefixed_final_answer": False,
            "synthetic_chars_read": 0,
            "raw_chars_read": 0,
            "readme_chars_read": 0,
        }

    async def _run_scored_answerer_repeats(
        self,
        *,
        repeats: int,
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, float], int]:
        """Run the answerer one or more times and average score/usage metrics.

        Repeats are independent reward samples. Run them with bounded concurrency
        to keep the robust averaged reward while avoiding unnecessary serial
        remote-call latency.
        """
        repeat_count = max(1, int(repeats))
        repeat_concurrency = max(1, min(int(self.answerer_repeat_concurrency), repeat_count))

        async def run_one_repeat(
            repeat_idx: int,
        ) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
            async with trace.scope_span("answerer_repeat"):
                try:
                    answerer_result = await self._run_answerer()
                    usage = self._answerer_usage_metrics(answerer_result)
                    score_metrics = await self._score_answerer_output(
                        answerer_output=str(answerer_result.get("answer_text", "")),
                        synthetic_read_count=usage["synthetic_read_count"],
                        answerer_evaluated=1.0,
                        terminal_step=1.0,
                    )
                    score_metrics.setdefault("answerer_error", 0.0)
                    score_metrics.setdefault("judge_error", 0.0)
                except Exception:
                    LOGGER.warning(
                        "Answerer repeat %s/%s failed; scoring this repeat as incorrect.",
                        repeat_idx + 1,
                        repeat_count,
                        exc_info=True,
                    )
                    answerer_result = self._empty_answerer_result()
                    usage = self._answerer_usage_metrics(answerer_result)
                    score_metrics = {
                        "correct": 0.0,
                        "exact_match": 0.0,
                        "judge_used": 0.0,
                        "judge_score": 0.0,
                        "judge_error": 0.0,
                        "answerer_error": 1.0,
                        "judge_skipped_no_candidate": 0.0,
                        "judge_skipped_no_synthetic_reads": 0.0,
                    }
            return answerer_result, usage, score_metrics

        if repeat_concurrency <= 1:
            scored = [await run_one_repeat(repeat_idx) for repeat_idx in range(repeat_count)]
        else:
            semaphore = asyncio.Semaphore(repeat_concurrency)

            async def run_bounded_repeat(
                repeat_idx: int,
            ) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
                async with semaphore:
                    return await run_one_repeat(repeat_idx)

            scored = list(
                await asyncio.gather(
                    *(run_bounded_repeat(repeat_idx) for repeat_idx in range(repeat_count))
                )
            )

        primary_result = dict(
            next(
                (
                    result
                    for result, _, metrics in scored
                    if float(metrics.get("answerer_error", 0.0)) <= 0.0
                ),
                scored[0][0],
            )
        )

        def mean_float(values: Sequence[float]) -> float:
            return sum(float(v) for v in values) / float(len(values)) if values else 0.0

        usage_keys = sorted(set().union(*(usage.keys() for _, usage, _ in scored)))
        avg_usage = {
            key: mean_float([usage.get(key, 0.0) for _, usage, _ in scored]) for key in usage_keys
        }
        score_keys = sorted(set().union(*(metrics.keys() for _, _, metrics in scored)))
        avg_score_metrics = {
            key: mean_float([score_metrics.get(key, 0.0) for _, _, score_metrics in scored])
            for key in score_keys
        }

        primary_result["steps"] = mean_float(
            [float(result.get("steps", 0.0) or 0.0) for result, _, _ in scored]
        )
        for key in [
            "used_raw_docs",
            "used_readme",
            "freeform_final_answer",
            "prefixed_final_answer",
        ]:
            primary_result[key] = mean_float(
                [float(bool(result.get(key, False))) for result, _, _ in scored]
            )
        return primary_result, avg_usage, avg_score_metrics, repeat_count

    async def answerability_progress_probe(
        self,
        *,
        previous_score: float,
        previous_best_score: float,
    ) -> tuple[float, float, dict[str, float]]:
        """Answerer-as-progress reward for the current partial filesystem.

        This is intentionally bounded by the caller. It rewards local progress
        over the previous probe score while also logging the best-so-far score
        for diagnostics.
        """
        active_records = self.state.active_records()
        if not active_records:
            return (
                previous_best_score,
                0.0,
                {
                    "answerability_probe_evaluated": 0.0,
                    "answerability_probe_reward": 0.0,
                },
            )

        _, usage, score_metrics, repeat_count = await self._run_scored_answerer_repeats(
            repeats=self.answerability_probe_repeats,
        )
        score = score_metrics["correct"]
        raw_delta = score - previous_score
        if abs(raw_delta) < self.answerability_delta_min_abs:
            delta = 0.0
        elif self.answerability_delta_allow_negative:
            delta = raw_delta
        else:
            delta = max(0.0, raw_delta)
        new_best = max(previous_best_score, score)
        reward = self.answerability_delta_reward_scale * delta

        retrieval_penalty = 0.0
        synthetic_read_penalty = 0.0
        if score > 0.0:
            retrieval_penalty = (
                self.answerer_retrieval_cost_scale * usage["answerer_retrieval_cost_score"]
            )
            synthetic_read_penalty = (
                self.answerer_synthetic_read_cost_scale
                * usage["answerer_synthetic_read_cost_score"]
            )
            reward -= retrieval_penalty + synthetic_read_penalty

        metrics = {
            "answerability_probe_evaluated": 1.0,
            "answerability_probe_score": score,
            "answerability_probe_previous_score": previous_score,
            "answerability_probe_best_score": new_best,
            "answerability_probe_raw_delta": raw_delta,
            "answerability_probe_delta": delta,
            "answerability_probe_min_abs_delta": self.answerability_delta_min_abs,
            "answerability_probe_reward": reward,
            "answerability_probe_repeats": float(repeat_count),
            "answerability_probe_repeat_concurrency": float(
                max(1, min(int(self.answerer_repeat_concurrency), int(repeat_count)))
            ),
            "answerability_probe_retrieval_cost_penalty": retrieval_penalty,
            "answerability_probe_synthetic_read_cost_penalty": synthetic_read_penalty,
            "answerability_probe_synthetic_read_count": usage["synthetic_read_count"],
            "answerability_probe_total_tokens_read": usage["answerer_total_tokens_read"],
            "answerability_probe_judge_used": score_metrics["judge_used"],
            "answerability_probe_exact_match": score_metrics["exact_match"],
            "answerability_probe_answerer_error_rate": score_metrics.get("answerer_error", 0.0),
            "answerability_probe_judge_error_rate": score_metrics.get("judge_error", 0.0),
        }
        return new_best, reward, metrics

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        self.last_answerer_trace = {}
        assistant_messages = [msg for msg in history if msg.get("role") == "assistant"]
        assistant_turns = len(assistant_messages)
        final_message = assistant_messages[-1] if assistant_messages else None

        assistant_text = (
            (get_text_content(final_message) or "").strip() if final_message is not None else ""
        )
        stop_called = float(assistant_text == STOP_MESSAGE)
        final_has_tool_calls = bool(final_message is not None and final_message.get("tool_calls"))
        no_tool_terminal = float(final_message is not None and not final_has_tool_calls)
        max_turn_terminal = float(assistant_turns >= self.builder_max_turns)
        terminal_step = float(
            stop_called >= 1.0 or max_turn_terminal >= 1.0 or no_tool_terminal >= 1.0
        )
        active_records = self.state.active_records()

        answerer_result = self._empty_answerer_result()
        usage = self._answerer_usage_metrics(answerer_result)
        score_metrics = await self._score_answerer_output(
            answerer_output="",
            synthetic_read_count=usage["synthetic_read_count"],
            answerer_evaluated=0.0,
            terminal_step=terminal_step,
        )
        answerer_evaluated = 0.0
        answerer_repeats_used = 0
        if (
            terminal_step >= 1.0
            and self.terminal_answerer_enabled
            and not (self.answerer_workspace_mode == "synthetic_only" and not active_records)
        ):
            (
                answerer_result,
                usage,
                score_metrics,
                answerer_repeats_used,
            ) = await self._run_scored_answerer_repeats(
                repeats=self.terminal_answerer_repeats,
            )
            answerer_evaluated = 1.0

        synthetic_read_count = usage["synthetic_read_count"]
        raw_doc_read_count = usage["raw_doc_read_count"]
        synthetic_chars_read = usage["answerer_synthetic_chars_read"]
        raw_chars_read = usage["answerer_raw_chars_read"]
        readme_chars_read = usage["answerer_readme_chars_read"]
        synthetic_tokens_read = usage["answerer_synthetic_tokens_read"]
        raw_tokens_read = usage["answerer_raw_tokens_read"]
        readme_tokens_read = usage["answerer_readme_tokens_read"]
        total_answerer_tokens_read = usage["answerer_total_tokens_read"]
        retrieval_cost_score = usage["answerer_retrieval_cost_score"]
        synthetic_read_cost_score = usage["answerer_synthetic_read_cost_score"]

        correct = score_metrics["correct"]
        exact_match = score_metrics["exact_match"]
        judge_used = score_metrics["judge_used"]
        judge_score = score_metrics["judge_score"]
        answerer_error_rate = score_metrics.get("answerer_error", 0.0)
        judge_error_rate = score_metrics.get("judge_error", 0.0)
        judge_skipped_no_candidate = score_metrics["judge_skipped_no_candidate"]
        judge_skipped_no_synthetic_reads = score_metrics["judge_skipped_no_synthetic_reads"]

        terminal_answer_score = correct if terminal_step >= 1.0 else 0.0
        reward = terminal_answer_score
        step_penalty_value = 0.0
        if self.step_penalty > 0.0:
            step_penalty_value = self.step_penalty * float(assistant_turns)
            reward -= step_penalty_value
        used_raw_docs = float(answerer_result["used_raw_docs"])
        num_active_files = float(len(active_records))
        empty_synthetic = float(num_active_files <= 0.0)
        raw_docs_penalty_value = 0.0
        if terminal_step >= 1.0 and used_raw_docs > 0.0:
            raw_docs_penalty_value = self.raw_docs_penalty
            reward -= raw_docs_penalty_value
        empty_synthetic_penalty_value = 0.0
        if terminal_step >= 1.0 and empty_synthetic > 0.0:
            empty_synthetic_penalty_value = self.empty_synthetic_penalty
            reward -= empty_synthetic_penalty_value
        synthetic_success_bonus_applied = 0.0
        synthetic_success_bonus_value = 0.0
        if (
            terminal_step >= 1.0
            and correct > 0.0
            and used_raw_docs <= 0.0
            and num_active_files > 0.0
        ):
            synthetic_success_bonus_value = self.synthetic_success_bonus
            reward += synthetic_success_bonus_value
            synthetic_success_bonus_applied = 1.0
        num_active_clusters = sum(1 for rec in active_records if rec.kind == "cluster")
        num_active_merges = sum(1 for rec in active_records if rec.kind == "merge_summary")
        num_root_entries = sum(1 for rec in active_records if rec.parent_id is None)
        num_provenance_files = sum(1 for rec in active_records if rec.direct_raw_doc_paths)
        total_content_reads = synthetic_read_count + raw_doc_read_count
        synthetic_usage_ratio = (
            float(synthetic_read_count) / float(total_content_reads)
            if total_content_reads > 0
            else 0.0
        )
        raw_usage_ratio = (
            float(raw_doc_read_count) / float(total_content_reads)
            if total_content_reads > 0
            else 0.0
        )
        raw_usage_ratio_penalty_value = 0.0
        synthetic_usage_bonus_value = 0.0
        if terminal_step >= 1.0:
            raw_usage_ratio_penalty_value = self.raw_usage_ratio_penalty * raw_usage_ratio
            synthetic_usage_bonus_value = self.synthetic_usage_bonus * synthetic_usage_ratio
            reward -= raw_usage_ratio_penalty_value
            reward += synthetic_usage_bonus_value
        retrieval_cost_penalty = 0.0
        if terminal_step >= 1.0 and self.answerer_retrieval_cost_scale > 0.0:
            should_charge_retrieval = correct > 0.0 or not self.answerer_retrieval_cost_correct_only
            if should_charge_retrieval:
                retrieval_cost_penalty = self.answerer_retrieval_cost_scale * retrieval_cost_score
                reward -= retrieval_cost_penalty
        synthetic_read_cost_penalty = 0.0
        if terminal_step >= 1.0 and self.answerer_synthetic_read_cost_scale > 0.0:
            should_charge_synthetic_reads = (
                correct > 0.0 or not self.answerer_retrieval_cost_correct_only
            )
            if should_charge_synthetic_reads:
                synthetic_read_cost_penalty = (
                    self.answerer_synthetic_read_cost_scale * synthetic_read_cost_score
                )
                reward -= synthetic_read_cost_penalty
        maturity = self._filesystem_maturity_scores(
            active_records=active_records,
            num_active_files=num_active_files,
            num_active_merges=float(num_active_merges),
            stop_called=stop_called,
        )
        filesystem_maturity_reward = 0.0
        if terminal_step >= 1.0 and self.filesystem_maturity_scale > 0.0:
            filesystem_maturity_reward = (
                self.filesystem_maturity_scale * maturity["filesystem_maturity_score"]
            )
            reward += filesystem_maturity_reward
        mature_stop_bonus_applied = 0.0
        if (
            terminal_step >= 1.0
            and stop_called >= 1.0
            and self.mature_stop_bonus > 0.0
            and maturity["filesystem_maturity_score"] >= self.mature_stop_min_score
        ):
            reward += self.mature_stop_bonus
            mature_stop_bonus_applied = 1.0
        termination_penalty_value = 0.0
        if terminal_step >= 1.0 and stop_called < 1.0:
            termination_penalty_value = self.termination_penalty
            reward -= termination_penalty_value
        terminal_reward_unclipped = reward
        reward = max(self.terminal_reward_clip_min, min(self.terminal_reward_clip_max, reward))

        metrics = self._metrics_template(
            stop_called=stop_called,
            correct=correct,
            exact_match=exact_match,
            judge_used=judge_used,
            judge_score=judge_score,
            num_active_files=num_active_files,
            num_raw_docs_in_fallback=0.0,
            total_synthetic_files=float(len(self.state.files_by_id)),
            total_operations=float(self.state.operation_round),
            builder_turns_seen=float(assistant_turns),
            terminal_builder_step=terminal_step,
            max_turn_terminal=max_turn_terminal,
            no_tool_terminal=no_tool_terminal,
            answerer_evaluated=answerer_evaluated,
            answerer_steps=float(answerer_result["steps"]),
            answerer_used_raw_docs=used_raw_docs,
            answerer_used_readme=float(answerer_result["used_readme"]),
            answerer_freeform_final_answer=float(
                answerer_result.get("freeform_final_answer", False)
            ),
            answerer_prefixed_final_answer=float(
                answerer_result.get("prefixed_final_answer", False)
            ),
            judge_skipped_no_candidate=judge_skipped_no_candidate,
            judge_skipped_no_synthetic_reads=judge_skipped_no_synthetic_reads,
            step_penalty_applied=(
                self.step_penalty * float(assistant_turns) if self.step_penalty > 0.0 else 0.0
            ),
            termination_penalty_applied=float(terminal_step >= 1.0 and stop_called < 1.0),
            raw_docs_penalty_applied=float(terminal_step >= 1.0 and used_raw_docs > 0.0),
            empty_synthetic_penalty_applied=float(terminal_step >= 1.0 and empty_synthetic > 0.0),
            synthetic_success_bonus_applied=synthetic_success_bonus_applied,
            num_active_clusters=float(num_active_clusters),
            num_active_merges=float(num_active_merges),
            num_root_entries=float(num_root_entries),
            num_provenance_files=float(num_provenance_files),
            synthetic_read_count=float(synthetic_read_count),
            raw_doc_read_count=float(raw_doc_read_count),
            answerer_synthetic_chars_read=synthetic_chars_read,
            answerer_raw_chars_read=raw_chars_read,
            answerer_readme_chars_read=readme_chars_read,
            answerer_synthetic_tokens_read=synthetic_tokens_read,
            answerer_raw_tokens_read=raw_tokens_read,
            answerer_readme_tokens_read=readme_tokens_read,
            answerer_total_tokens_read=total_answerer_tokens_read,
            answerer_retrieval_cost_score=retrieval_cost_score,
            answerer_retrieval_cost_penalty=retrieval_cost_penalty,
            answerer_synthetic_read_cost_score=synthetic_read_cost_score,
            answerer_synthetic_read_cost_penalty=synthetic_read_cost_penalty,
            synthetic_usage_ratio=synthetic_usage_ratio,
            raw_usage_ratio=raw_usage_ratio,
            filesystem_maturity_reward=filesystem_maturity_reward,
            mature_stop_bonus_applied=mature_stop_bonus_applied,
            **maturity,
        )
        metrics["answerer_repeats"] = float(answerer_repeats_used)
        metrics["terminal_answerer_enabled"] = float(self.terminal_answerer_enabled)
        metrics["answerer_repeat_concurrency"] = (
            float(max(1, min(int(self.answerer_repeat_concurrency), int(answerer_repeats_used))))
            if answerer_repeats_used > 0
            else 0.0
        )
        metrics["answerer_error_rate"] = answerer_error_rate
        metrics["judge_error_rate"] = judge_error_rate
        if terminal_step >= 1.0 and answerer_evaluated >= 1.0:
            self.last_answerer_trace = {
                "answer_text": str(answerer_result.get("answer_text", ""))[:4000],
                "read_paths": [str(path) for path in answerer_result.get("read_paths", [])],
                "steps": float(answerer_result.get("steps", 0.0) or 0.0),
                "used_raw_docs": answerer_result.get("used_raw_docs", False),
                "used_readme": answerer_result.get("used_readme", False),
                "freeform_final_answer": answerer_result.get("freeform_final_answer", False),
                "prefixed_final_answer": answerer_result.get("prefixed_final_answer", False),
                "answerer_repeats": int(answerer_repeats_used),
            }
        metrics.update(
            {
                "terminal_answer_score": terminal_answer_score,
                "terminal_filesystem_structure_reward": filesystem_maturity_reward,
                "terminal_retrieval_cost_penalty": retrieval_cost_penalty,
                "terminal_file_read_cost_penalty": synthetic_read_cost_penalty,
                "terminal_step_penalty": step_penalty_value,
                "terminal_raw_docs_penalty": raw_docs_penalty_value,
                "terminal_raw_usage_ratio_penalty": raw_usage_ratio_penalty_value,
                "terminal_empty_filesystem_penalty": empty_synthetic_penalty_value,
                "terminal_synthetic_success_bonus": synthetic_success_bonus_value,
                "terminal_synthetic_usage_bonus": synthetic_usage_bonus_value,
                "terminal_mature_stop_bonus": (self.mature_stop_bonus * mature_stop_bonus_applied),
                "terminal_termination_penalty": termination_penalty_value,
                "terminal_reward_unclipped": terminal_reward_unclipped,
                "terminal_reward": reward,
            }
        )
        if terminal_step >= 1.0 and self.snapshot_path is not None:
            self.state.save_snapshot(self.snapshot_path)
        return reward, metrics

    def _current_filesystem_maturity_score(self) -> float:
        active_records = self.state.active_records()
        num_active_merges = sum(1 for rec in active_records if rec.kind == "merge_summary")
        return self._filesystem_maturity_scores(
            active_records=active_records,
            num_active_files=float(len(active_records)),
            num_active_merges=float(num_active_merges),
            stop_called=0.0,
        )["filesystem_maturity_score"]

    def _filesystem_maturity_scores(
        self,
        *,
        active_records: Sequence[SyntheticFileRecord],
        num_active_files: float,
        num_active_merges: float,
        stop_called: float,
    ) -> dict[str, float]:
        visible_raw_docs = max(1, len(self.state.visible_raw_doc_paths))
        covered_raw_docs: set[str] = set()
        directly_covered_raw_docs: set[str] = set()
        memo: dict[str, set[str]] = {}
        per_file_raw_counts: list[int] = []
        for rec in active_records:
            transitive = self.state._transitive_raw_docs_for_node(rec.node_id, memo)
            covered_raw_docs.update(transitive)
            directly_covered_raw_docs.update(rec.direct_raw_doc_paths)
            if rec.kind == "cluster":
                per_file_raw_counts.append(len(transitive))

        covered_count = len(covered_raw_docs)
        coverage_score = min(1.0, float(covered_count) / float(visible_raw_docs))
        direct_covered_count = len(directly_covered_raw_docs)
        direct_coverage_score = min(1.0, float(direct_covered_count) / float(visible_raw_docs))
        raw_doc_count = len(self.state.visible_raw_doc_paths)
        active_count = max(0.0, num_active_files)
        expansion_score = 1.0 - (1.0 / ((1.0 + active_count) ** 0.5)) if active_count > 0.0 else 0.0
        root_entries = sum(1 for rec in active_records if rec.parent_id is None)
        root_ratio = float(root_entries) / active_count if active_count > 0.0 else 1.0
        branching_score = max(0.0, 1.0 - root_ratio) if active_count > 1.0 else 0.0
        merge_score = (
            1.0 - (1.0 / ((1.0 + num_active_merges) ** 0.5)) if num_active_merges > 0.0 else 0.0
        )
        organization_score = 0.5 * branching_score + 0.5 * merge_score
        stop_maturity_bonus = float(
            stop_called >= 1.0
            and coverage_score >= 0.85
            and expansion_score >= 0.75
            and organization_score >= 0.25
        )
        maturity_score = (
            self.filesystem_coverage_weight * coverage_score
            + self.filesystem_direct_coverage_weight * direct_coverage_score
            + self.filesystem_expansion_weight * expansion_score
            + self.filesystem_organization_weight * organization_score
            + self.filesystem_stop_weight * stop_maturity_bonus
        )
        return {
            "filesystem_maturity_score": maturity_score,
            "filesystem_coverage_score": coverage_score,
            "filesystem_direct_coverage_score": direct_coverage_score,
            "filesystem_expansion_score": expansion_score,
            "filesystem_organization_score": organization_score,
            "filesystem_merge_score": merge_score,
            "filesystem_stop_maturity_bonus": stop_maturity_bonus,
            "filesystem_covered_raw_docs": float(covered_count),
            "filesystem_direct_covered_raw_docs": float(direct_covered_count),
            "filesystem_visible_raw_docs": float(raw_doc_count),
            "filesystem_root_ratio": root_ratio,
            "filesystem_branching_score": branching_score,
        }

    def _metrics_template(
        self,
        *,
        stop_called: float,
        correct: float = 0.0,
        exact_match: float = 0.0,
        judge_used: float = 0.0,
        judge_score: float = 0.0,
        num_active_files: float = 0.0,
        num_raw_docs_in_fallback: float = 0.0,
        total_synthetic_files: float = 0.0,
        total_operations: float = 0.0,
        builder_turns_seen: float = 0.0,
        terminal_builder_step: float = 0.0,
        max_turn_terminal: float = 0.0,
        no_tool_terminal: float = 0.0,
        answerer_evaluated: float = 0.0,
        answerer_steps: float = 0.0,
        answerer_used_raw_docs: float = 0.0,
        answerer_used_readme: float = 0.0,
        answerer_freeform_final_answer: float = 0.0,
        answerer_prefixed_final_answer: float = 0.0,
        judge_skipped_no_candidate: float = 0.0,
        judge_skipped_no_synthetic_reads: float = 0.0,
        step_penalty_applied: float = 0.0,
        termination_penalty_applied: float = 0.0,
        raw_docs_penalty_applied: float = 0.0,
        empty_synthetic_penalty_applied: float = 0.0,
        synthetic_success_bonus_applied: float = 0.0,
        num_active_clusters: float = 0.0,
        num_active_merges: float = 0.0,
        num_root_entries: float = 0.0,
        num_provenance_files: float = 0.0,
        synthetic_read_count: float = 0.0,
        raw_doc_read_count: float = 0.0,
        answerer_synthetic_chars_read: float = 0.0,
        answerer_raw_chars_read: float = 0.0,
        answerer_readme_chars_read: float = 0.0,
        answerer_synthetic_tokens_read: float = 0.0,
        answerer_raw_tokens_read: float = 0.0,
        answerer_readme_tokens_read: float = 0.0,
        answerer_total_tokens_read: float = 0.0,
        answerer_retrieval_cost_score: float = 0.0,
        answerer_retrieval_cost_penalty: float = 0.0,
        answerer_synthetic_read_cost_score: float = 0.0,
        answerer_synthetic_read_cost_penalty: float = 0.0,
        synthetic_usage_ratio: float = 0.0,
        raw_usage_ratio: float = 0.0,
        filesystem_maturity_reward: float = 0.0,
        mature_stop_bonus_applied: float = 0.0,
        filesystem_maturity_score: float = 0.0,
        filesystem_coverage_score: float = 0.0,
        filesystem_direct_coverage_score: float = 0.0,
        filesystem_expansion_score: float = 0.0,
        filesystem_organization_score: float = 0.0,
        filesystem_merge_score: float = 0.0,
        filesystem_stop_maturity_bonus: float = 0.0,
        filesystem_covered_raw_docs: float = 0.0,
        filesystem_direct_covered_raw_docs: float = 0.0,
        filesystem_visible_raw_docs: float = 0.0,
        filesystem_root_ratio: float = 0.0,
        filesystem_branching_score: float = 0.0,
    ) -> dict[str, float]:
        return {
            "stop_called": stop_called,
            "correct": correct,
            "exact_match": exact_match,
            "judge_used": judge_used,
            "judge_score": judge_score,
            "num_active_files": num_active_files,
            "num_raw_docs_in_fallback": num_raw_docs_in_fallback,
            "total_synthetic_files": total_synthetic_files,
            "total_operations": total_operations,
            "builder_turns_seen": builder_turns_seen,
            "terminal_builder_step": terminal_builder_step,
            "max_turn_terminal": max_turn_terminal,
            "no_tool_terminal": no_tool_terminal,
            "answerer_evaluated": answerer_evaluated,
            "answerer_steps": answerer_steps,
            "answerer_used_raw_docs": answerer_used_raw_docs,
            "answerer_used_readme": answerer_used_readme,
            "answerer_freeform_final_answer": answerer_freeform_final_answer,
            "answerer_prefixed_final_answer": answerer_prefixed_final_answer,
            "judge_skipped_no_candidate": judge_skipped_no_candidate,
            "judge_skipped_no_synthetic_reads": judge_skipped_no_synthetic_reads,
            "step_penalty_applied": step_penalty_applied,
            "termination_penalty_applied": termination_penalty_applied,
            "raw_docs_penalty_applied": raw_docs_penalty_applied,
            "empty_synthetic_penalty_applied": empty_synthetic_penalty_applied,
            "synthetic_success_bonus_applied": synthetic_success_bonus_applied,
            "num_active_clusters": num_active_clusters,
            "num_active_merges": num_active_merges,
            "num_root_entries": num_root_entries,
            "num_provenance_files": num_provenance_files,
            "synthetic_read_count": synthetic_read_count,
            "raw_doc_read_count": raw_doc_read_count,
            "answerer_synthetic_chars_read": answerer_synthetic_chars_read,
            "answerer_raw_chars_read": answerer_raw_chars_read,
            "answerer_readme_chars_read": answerer_readme_chars_read,
            "answerer_synthetic_tokens_read": answerer_synthetic_tokens_read,
            "answerer_raw_tokens_read": answerer_raw_tokens_read,
            "answerer_readme_tokens_read": answerer_readme_tokens_read,
            "answerer_total_tokens_read": answerer_total_tokens_read,
            "answerer_retrieval_cost_score": answerer_retrieval_cost_score,
            "answerer_retrieval_cost_penalty": answerer_retrieval_cost_penalty,
            "answerer_synthetic_read_cost_score": answerer_synthetic_read_cost_score,
            "answerer_synthetic_read_cost_penalty": answerer_synthetic_read_cost_penalty,
            "synthetic_usage_ratio": synthetic_usage_ratio,
            "raw_usage_ratio": raw_usage_ratio,
            "filesystem_maturity_reward": filesystem_maturity_reward,
            "mature_stop_bonus_applied": mature_stop_bonus_applied,
            "filesystem_maturity_score": filesystem_maturity_score,
            "filesystem_coverage_score": filesystem_coverage_score,
            "filesystem_direct_coverage_score": filesystem_direct_coverage_score,
            "filesystem_expansion_score": filesystem_expansion_score,
            "filesystem_organization_score": filesystem_organization_score,
            "filesystem_merge_score": filesystem_merge_score,
            "filesystem_stop_maturity_bonus": filesystem_stop_maturity_bonus,
            "filesystem_covered_raw_docs": filesystem_covered_raw_docs,
            "filesystem_direct_covered_raw_docs": filesystem_direct_covered_raw_docs,
            "filesystem_visible_raw_docs": filesystem_visible_raw_docs,
            "filesystem_root_ratio": filesystem_root_ratio,
            "filesystem_branching_score": filesystem_branching_score,
        }

    def _extract_answer(self, text: str) -> str | None:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len(lines) != 1:
            return None
        if not lines[0].startswith(ANSWER_PREFIX):
            return None
        answer = lines[0][len(ANSWER_PREFIX) :].strip()
        return answer or None

    def _check_answer(self, model_answer: str) -> bool:
        return normalize_answer(model_answer) == normalize_answer(self.gold_answer)

    async def _llm_judge_score(self, model_answer: str) -> float:
        if self.judge_score_mode == "binary":
            prompt = (
                "You are judging whether a predicted answer should count as correct for a question. "
                'Return only JSON in the form {"correct": 0 or 1}.\n\n'
                f"Question: {self.question}\n"
                f"Gold answer: {self.gold_answer}\n"
                f"Predicted answer: {model_answer}\n\n"
                "Count semantically equivalent answers as correct even if formatting differs. "
                "Be strict about factual mismatch."
            )
        elif self.judge_score_mode == "graded":
            prompt = (
                "You are grading how well a predicted answer answers a question using a gold answer. "
                'Return only JSON in the form {"score": 0, 1, 2, 3, or 4}.\n\n'
                f"Question: {self.question}\n"
                f"Gold answer: {self.gold_answer}\n"
                f"Predicted answer: {model_answer}\n\n"
                "Use 4 for semantically correct, 3 for mostly correct with a minor omission, "
                "2 for partially correct, 1 for relevant but materially incomplete, and 0 for incorrect or unsupported."
            )
        else:
            raise ValueError(f"Unknown judge_score_mode: {self.judge_score_mode}")
        async with trace.scope_span("remote/judge_model_call"):
            response_text = await asyncio.to_thread(
                self._call_chat_model,
                backend=self.judge_backend,
                model=self.judge_model,
                base_url=self.judge_base_url,
                api_key_env=self.judge_api_key_env,
                prompt=prompt,
                response_json=True,
                max_output_tokens=self.judge_max_output_tokens,
                enable_thinking=False,
            )
        try:
            parsed = json.loads(response_text)
            if not isinstance(parsed, dict):
                raise ValueError("Judge response must be a JSON object")
            if self.judge_score_mode == "graded":
                value = parsed.get("score")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("Graded judge score must be numeric")
                if float(value) not in {0.0, 1.0, 2.0, 3.0, 4.0}:
                    raise ValueError("Graded judge score must be an integer from 0 to 4")
                return float(value) / 4.0
            value = parsed.get("correct")
            if isinstance(value, bool) or value not in (0, 1, 0.0, 1.0):
                raise ValueError("Binary judge score must be exactly 0 or 1")
            return float(value)
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Malformed judge response: {response_text[:200]}") from exc

    async def run_answerer(self) -> dict[str, Any]:
        """Run one frozen-answerer pass over the current final workspace."""
        return await self._run_answerer()

    def _call_answerer_model_messages(
        self,
        messages: Sequence[dict[str, str]],
    ) -> str:
        kwargs = {
            "backend": self.answerer_backend,
            "model": self.answerer_model,
            "base_url": self.answerer_base_url,
            "api_key_env": self.answerer_api_key_env,
            "messages": messages,
            "response_json": False,
            "temperature": self.answerer_temperature,
            "top_p": self.answerer_top_p,
            "top_k": self.answerer_top_k,
            "min_p": self.answerer_min_p,
            "presence_penalty": self.answerer_presence_penalty,
            "repetition_penalty": self.answerer_repetition_penalty,
            "enable_thinking": self.answerer_enable_thinking,
        }
        if self.answerer_backend == "vllm":
            try:
                return self._call_chat_model_messages(
                    **kwargs,
                    max_output_tokens=self.answerer_max_output_tokens,
                )
            except RuntimeError as exc:
                prompt_tokens = context_limited_prompt_tokens(
                    exc,
                    self.answerer_max_output_tokens,
                )
                if prompt_tokens is None or prompt_tokens < 1:
                    raise
                retry_messages = [
                    *messages,
                    {"role": "user", "content": self._context_tail_reminder()},
                ]
                LOGGER.warning(
                    "Answerer prompt reached its context boundary; retrying with the prompt "
                    "left-truncated to %d tokens.",
                    prompt_tokens,
                )
                return self._call_chat_model_messages(
                    **(kwargs | {"messages": retry_messages}),
                    max_output_tokens=self.answerer_max_output_tokens,
                    truncate_prompt_tokens=prompt_tokens,
                    truncation_side="left",
                )

        completion_tokens = self.answerer_max_output_tokens
        for attempt in range(ANSWERER_CONTEXT_BOUNDARY_MAX_ATTEMPTS):
            try:
                return self._call_chat_model_messages(
                    **kwargs,
                    max_output_tokens=completion_tokens,
                )
            except RuntimeError as exc:
                available_tokens = context_limited_completion_tokens(
                    exc,
                    completion_tokens,
                )
                if (
                    available_tokens is None
                    or available_tokens < 1
                    or attempt == ANSWERER_CONTEXT_BOUNDARY_MAX_ATTEMPTS - 1
                ):
                    raise
                LOGGER.warning(
                    "Answerer prompt reached its context boundary; reducing the completion "
                    "cap from %d to %d tokens.",
                    completion_tokens,
                    available_tokens,
                )
                completion_tokens = available_tokens
        raise RuntimeError("Answerer context-boundary retry loop exited unexpectedly")

    def _context_tail_reminder(self) -> str:
        return (
            "Context-boundary reminder:\n"
            f"Question: {self.question}\n"
            "Use only the retained workspace evidence. Return exactly one compact JSON action "
            '({"action":"list_files","path":"."} or '
            '{"action":"read_file","path":"some/path.txt","start_line":1,'
            f'"num_lines":{READ_FILE_MAX_LINES}}}), or exactly one final answer line: '
            f"{ANSWER_PREFIX} <final answer>"
        )

    async def _run_answerer(self) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="synthetic_fs_answerer_") as tmpdir:
            root_bootstrap_entries = self.state.answerer_bootstrap_entries()
            workspace_path_to_node_id = self.state.workspace_path_to_node_id()
            provenance_memo: dict[str, set[str]] = {}
            allowed_raw_docs: set[str] = set()
            if self.answerer_workspace_mode == "synthetic_plus_raw":
                for record in self.state.root_active_records():
                    allowed_raw_docs.update(
                        self.state._transitive_raw_docs_for_node(record.node_id, provenance_memo)
                    )
            workspace_root = self.state.materialize_final_workspace(
                Path(tmpdir) / "final_workspace",
                mode=self.answerer_workspace_mode,
            )
            tools = ReadOnlyAnswererWorkspaceTools(workspace_root)
            full_readme_text = (workspace_root / "README.md").read_text(encoding="utf-8").strip()
            remaining_read_chars = max(0, int(self.answerer_max_total_read_chars))

            def limit_read_content(content: str) -> str:
                nonlocal remaining_read_chars
                if self.answerer_max_total_read_chars <= 0:
                    return content
                visible = content[:remaining_read_chars]
                remaining_read_chars -= len(visible)
                return visible

            if self.answerer_max_total_read_chars > 0:
                readme_allowance = min(
                    len(full_readme_text),
                    max(1, int(self.answerer_max_total_read_chars) // 4),
                )
                readme_text = full_readme_text[:readme_allowance]
                remaining_read_chars -= len(readme_text)
            else:
                readme_text = full_readme_text
            max_bootstrap_entries = len(root_bootstrap_entries)
            if self.answerer_max_total_named_reads > 0:
                # Reserve one read for README.
                max_bootstrap_entries = max(0, self.answerer_max_total_named_reads - 1)
            visible_root_entries: list[dict[str, str]] = []
            for entry in root_bootstrap_entries[:max_bootstrap_entries]:
                content = limit_read_content(str(entry["content"]).strip())
                if content:
                    visible_root_entries.append({"path": str(entry["path"]), "content": content})

            bootstrap_blocks = [
                "[README.md]",
                readme_text,
            ]
            for entry in visible_root_entries:
                bootstrap_blocks.extend(
                    [
                        "",
                        f"[ROOT_SYNTHETIC_ENTRY] {entry['path']}",
                        entry["content"].strip(),
                    ]
                )

            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": build_answerer_system_prompt(self.answerer_workspace_mode),
                },
                {
                    "role": "user",
                    "content": build_answerer_user_prompt(
                        self.question, self.answerer_workspace_mode
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Workspace bootstrap:\n"
                        "README.md and the current root-level synthetic entries are already loaded below.\n\n"
                        + "\n".join(bootstrap_blocks)
                    ),
                },
            ]
            read_paths: list[str] = [
                "README.md",
                *[entry["path"] for entry in visible_root_entries],
            ]
            used_readme = True
            used_raw_docs = False
            readme_chars_read = len(readme_text)
            synthetic_chars_read = sum(len(entry["content"]) for entry in visible_root_entries)
            raw_chars_read = 0
            tool_read_actions = 0
            empty_model_outputs = 0

            max_tool_reads = self.answerer_max_file_reads
            remaining_named_reads: int | None = None
            if self.answerer_max_total_named_reads > 0:
                remaining_named_reads = max(0, self.answerer_max_total_named_reads - len(read_paths))

            def answerer_result(
                *,
                answer_text: str,
                steps: int,
                freeform_final_answer: bool,
                prefixed_final_answer: bool,
            ) -> dict[str, Any]:
                return {
                    "answer_text": answer_text,
                    "steps": steps,
                    "read_paths": read_paths,
                    "used_raw_docs": used_raw_docs,
                    "used_readme": used_readme,
                    "freeform_final_answer": freeform_final_answer,
                    "prefixed_final_answer": prefixed_final_answer,
                    "synthetic_chars_read": synthetic_chars_read,
                    "raw_chars_read": raw_chars_read,
                    "readme_chars_read": readme_chars_read,
                    "tool_read_actions": tool_read_actions,
                    "empty_model_outputs": empty_model_outputs,
                    "answerer_context_char_budget": int(self.answerer_max_total_read_chars),
                    "answerer_total_named_read_budget": int(self.answerer_max_total_named_reads),
                }

            def list_files_payload(path: str) -> dict[str, Any]:
                clean = path.strip() or "."
                if self.answerer_workspace_mode == "synthetic_plus_raw" and (
                    clean == "raw_docs" or clean.startswith("raw_docs/")
                ):
                    listing = tools.list_files(path=clean)
                    allowed = set(allowed_raw_docs)
                    listing["entries"] = [
                        entry
                        for entry in listing["entries"]
                        if entry["type"] == "dir" or entry["path"] in allowed
                    ]
                    return listing
                return tools.list_files(path=clean)

            for step in range(1, self.answerer_max_turns + 1):
                async with trace.scope_span("remote/answerer_model_call"):
                    response_text = await asyncio.to_thread(
                        self._call_answerer_model_messages,
                        messages,
                    )
                response_text = response_text.strip()
                messages.append({"role": "assistant", "content": response_text})

                if not response_text:
                    # Retry an empty provider turn without changing state or budgets.
                    empty_model_outputs += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous turn was empty. Continue from the workspace evidence "
                                "already shown. Return one valid tool action or one non-empty final answer."
                            ),
                        }
                    )
                    continue

                if response_text.startswith(ANSWER_PREFIX):
                    response_text = self._truncate_final_answer(response_text)
                    return answerer_result(
                        answer_text=response_text,
                        steps=step,
                        freeform_final_answer=False,
                        prefixed_final_answer=True,
                    )

                action = self._parse_answerer_action(response_text)
                if action is None:
                    return answerer_result(
                        answer_text=self._truncate_final_answer(response_text),
                        steps=step,
                        freeform_final_answer=True,
                        prefixed_final_answer=False,
                    )

                try:
                    if action["action"] == "list_files":
                        payload = list_files_payload(str(action.get("path", ".")))
                    elif action["action"] == "read_file":
                        if (
                            (max_tool_reads > 0 and tool_read_actions >= max_tool_reads)
                            or (
                                remaining_named_reads is not None
                                and tool_read_actions >= remaining_named_reads
                            )
                        ):
                            raise ValueError("Answerer file-read budget exhausted")
                        path = str(action.get("path", ""))
                        if (
                            self.answerer_workspace_mode == "synthetic_plus_raw"
                            and path.startswith("raw_docs/")
                            and path not in allowed_raw_docs
                        ):
                            raise ValueError(
                                "Raw document is not available yet. Only use raw_docs/ through provenance "
                                "from synthetic files already in context or read earlier."
                            )
                        payload = tools.read_file(
                            path=path,
                            start_line=int(action.get("start_line", 1)),
                            num_lines=int(action.get("num_lines", READ_FILE_MAX_LINES)),
                        )
                        content = limit_read_content(str(payload.get("content", "")))
                        if content != str(payload.get("content", "")):
                            payload = payload | {"content": content, "truncated_by_answer_budget": True}
                        content_chars = len(content)
                        tool_read_actions += 1
                        read_paths.append(path)
                        if path != "README.md" and not path.startswith("raw_docs/"):
                            synthetic_chars_read += content_chars
                            node_id = workspace_path_to_node_id.get(path)
                            if node_id is not None:
                                allowed_raw_docs.update(
                                    self.state._transitive_raw_docs_for_node(
                                        node_id, provenance_memo
                                    )
                                )
                        if path.startswith("raw_docs/"):
                            used_raw_docs = True
                            raw_chars_read += content_chars
                        if path == "README.md":
                            readme_chars_read += content_chars
                    else:
                        raise ValueError(f"Unknown action: {action['action']}")
                except Exception as e:
                    payload = {"error": str(e)}

                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}",
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached the browsing/tool-call budget. "
                        "Do not call more tools. Based only on the workspace evidence already shown in this conversation, "
                        f"return exactly one non-empty final answer line in this format:\n{ANSWER_PREFIX} <final answer>"
                    ),
                }
            )
            async with trace.scope_span("remote/answerer_model_call"):
                response_text = await asyncio.to_thread(
                    self._call_answerer_model_messages,
                    messages,
                )
            response_text = self._truncate_final_answer(response_text.strip())
            if not response_text:
                response_text = f"{ANSWER_PREFIX} Unknown"
            return answerer_result(
                answer_text=response_text,
                steps=self.answerer_max_turns + 1,
                freeform_final_answer=not response_text.startswith(ANSWER_PREFIX),
                prefixed_final_answer=response_text.startswith(ANSWER_PREFIX),
            )

    def _truncate_final_answer(self, response_text: str) -> str:
        if self.answerer_final_answer_max_tokens <= 0:
            return response_text.strip()
        has_prefix = response_text.startswith(ANSWER_PREFIX)
        answer_body = (
            response_text[len(ANSWER_PREFIX) :].strip() if has_prefix else response_text.strip()
        )
        tokens = answer_body.split()
        if len(tokens) <= self.answerer_final_answer_max_tokens:
            return response_text.strip()
        truncated = " ".join(tokens[: self.answerer_final_answer_max_tokens]).strip()
        return f"{ANSWER_PREFIX} {truncated}".strip() if has_prefix else truncated

    def _parse_answerer_action(self, text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        if not cleaned:
            return None
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except Exception:
            return None
        action = parsed.get("action")
        if action not in {"list_files", "read_file"}:
            return None
        return parsed

    def _call_chat_model(
        self,
        *,
        backend: JudgeBackend,
        model: str,
        base_url: str,
        api_key_env: str,
        prompt: str,
        response_json: bool,
        max_output_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        return self._call_chat_model_messages(
            backend=backend,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_json=response_json,
            max_output_tokens=max_output_tokens,
            enable_thinking=enable_thinking,
        )

    def _call_chat_model_messages(
        self,
        *,
        backend: JudgeBackend,
        model: str,
        base_url: str,
        api_key_env: str,
        messages: Sequence[dict[str, str]],
        response_json: bool,
        max_output_tokens: int | None = None,
        temperature: float = 0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        enable_thinking: bool | None = None,
        truncate_prompt_tokens: int | None = None,
        truncation_side: str | None = None,
    ) -> str:
        if backend in {"openrouter", "vllm"}:
            return _call_openai_compatible_chat_model_messages(
                backend=backend,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
                messages=messages,
                response_json=response_json,
                max_output_tokens=max_output_tokens,
                open_json=self._urlopen_json_with_retry,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                enable_thinking=enable_thinking,
                truncate_prompt_tokens=truncate_prompt_tokens,
                truncation_side=truncation_side,
            )

        if backend == "gemini":
            api_key = _api_key_from_env(api_key_env, required=True)
            model_path = model if model.startswith("models/") else f"models/{model}"
            url = f"{_normalize_model_http_url(base_url).rstrip('/')}/{model_path}:generateContent"
            system_texts = [
                m["content"] for m in messages if m.get("role") == "system" and m.get("content")
            ]
            contents = []
            for message in messages:
                role = message.get("role", "user")
                if role == "system":
                    continue
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(
                    {"role": gemini_role, "parts": [{"text": message.get("content", "")}]}
                )
            payload: dict[str, Any] = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                },
            }
            if max_output_tokens is not None and max_output_tokens > 0:
                payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
            if top_p is not None:
                payload["generationConfig"]["topP"] = top_p
            if top_k is not None:
                payload["generationConfig"]["topK"] = top_k
            if presence_penalty is not None:
                payload["generationConfig"]["presencePenalty"] = presence_penalty
            if system_texts:
                payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}
            if response_json:
                payload["generationConfig"]["responseMimeType"] = "application/json"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                method="POST",
            )
            body = self._urlopen_json_with_retry(req)

            candidates = body.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"Response missing candidates: {body}")
            content = (candidates[0].get("content") or {}).get("parts") or []
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("text")
            ]
            return "".join(text_parts).strip()

        raise ValueError(f"Unknown backend: {backend}")

    def _urlopen_json_with_retry(self, req: urllib.request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        retryable_http_codes = {429, 500, 502, 503, 504}

        for attempt in range(1, MODEL_HTTP_MAX_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(req, timeout=MODEL_HTTP_TIMEOUT_SECONDS) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code in retryable_http_codes and attempt < MODEL_HTTP_MAX_ATTEMPTS:
                    time.sleep(min(2 ** (attempt - 1), MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS))
                    last_error = RuntimeError(f"HTTP error {e.code}: {body[:500]}")
                    continue
                raise RuntimeError(f"HTTP error {e.code}: {body[:500]}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < MODEL_HTTP_MAX_ATTEMPTS:
                    time.sleep(min(2 ** (attempt - 1), MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS))
                    last_error = RuntimeError(f"URL/timeout error: {e}")
                    continue
                raise RuntimeError(f"URL/timeout error: {e}") from e

        if last_error is not None:
            raise last_error
        raise RuntimeError("Request failed without a captured error")


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        import re

        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    transformations = [lower, remove_punc, remove_articles, white_space_fix]
    return reduce(lambda text, fn: fn(text), transformations, s)


def get_text_content(message: Message) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
        return "".join(parts)
    return str(content)
