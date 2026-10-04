from __future__ import annotations

import asyncio
import gc
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env
from tinker_cookbook.tool_use.tools import handle_tool_call
from tinker_cookbook.tool_use.types import Tool, ToolResult
from tinker_cookbook.utils import trace

from environments.synthetic_fs.clients import call_chat_model_messages
from environments.synthetic_fs.common import (
    APPROX_CHARS_PER_TOKEN,
    BUILDER_COMPACTION_SUMMARY_MARKER,
    DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS,
    DEFAULT_BUILDER_COMPACTION_KEEP_RECENT_TURNS,
    DEFAULT_BUILDER_COMPACTION_MAX_OUTPUT_TOKENS,
    DEFAULT_BUILDER_COMPACTION_TRIGGER_TOKENS,
    LOGGER,
    JudgeBackend,
)
from environments.synthetic_fs.coverage import coverage_leaf_creation_required
from environments.synthetic_fs.prompts import (
    build_builder_system_prompt,
    build_builder_user_prompt,
)
from environments.synthetic_fs.reward import get_text_content
from environments.synthetic_fs.workspace import (
    BuilderFilesystemTools,
    SyntheticFilesystemDatum,
    SyntheticFilesystemState,
)

RewardResult = tuple[float, dict[str, float]]
RewardFn = Callable[[list[Message]], Awaitable[RewardResult]]
ProxyScoreFn = Callable[[], float]


def _tool_result_payload(tool_result: ToolResult) -> dict[str, Any]:
    if not getattr(tool_result, "messages", None):
        return {}
    content = tool_result.messages[0].get("content", "")
    if not isinstance(content, str):
        return {}
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tool_result_text(tool_result: ToolResult, max_chars: int = 700) -> str:
    if not getattr(tool_result, "messages", None):
        return ""
    text = get_text_content(tool_result.messages[0])
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated {len(text) - max_chars} chars]"


def _tool_result_succeeded(tool_result: ToolResult) -> bool:
    metadata = getattr(tool_result, "metadata", {}) or {}
    if "error" in metadata:
        return False
    payload = _tool_result_payload(tool_result)
    if payload.get("ok") is False:
        return False
    if "error" in payload:
        return False
    return True


def _tool_result_error_text(tool_result: ToolResult) -> str:
    metadata = getattr(tool_result, "metadata", {}) or {}
    error = metadata.get("error", "")
    if error:
        return str(error)
    payload = _tool_result_payload(tool_result)
    payload_error = payload.get("error")
    if payload_error:
        error_code = payload.get("error_code", "")
        prefix = f"{error_code}: " if error_code else ""
        return prefix + str(payload_error)
    text = _tool_result_text(tool_result, max_chars=1200)
    if "error" in text.lower():
        return text
    return ""


def _tool_result_error_code(tool_result: ToolResult) -> str:
    metadata = getattr(tool_result, "metadata", {}) or {}
    metadata_error = metadata.get("error", "")
    if metadata_error:
        return "metadata_error"
    payload = _tool_result_payload(tool_result)
    error_code = payload.get("error_code", "")
    return str(error_code) if error_code else "unknown"


def _metric_key_part(text: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower()).strip("_")
    return cleaned or "unknown"


def _tool_call_args(tool_call: Any) -> dict[str, Any] | None:
    try:
        args = json.loads(tool_call.function.arguments)
    except Exception:
        return None
    return args if isinstance(args, dict) else None


def _with_tool_call_args(tool_call: Any, args: dict[str, Any]) -> Any:
    encoded_args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    try:
        function = tool_call.function.model_copy(update={"arguments": encoded_args})
        return tool_call.model_copy(update={"function": function})
    except AttributeError:
        function = tool_call.function.__class__(
            name=tool_call.function.name,
            arguments=encoded_args,
        )
        return tool_call.__class__(
            function=function,
            id=getattr(tool_call, "id", None),
        )


def _redacted_tool_call(tool_call: Any, tool_result: ToolResult) -> Any:
    if not _tool_result_succeeded(tool_result):
        return tool_call

    tool_name = tool_call.function.name
    args = _tool_call_args(tool_call)
    if args is None:
        return tool_call

    payload = _tool_result_payload(tool_result)
    created = payload.get("created") if isinstance(payload.get("created"), dict) else {}
    active_path = str(created.get("active_path") or created.get("history_path") or "")
    node_id = str(created.get("node_id") or "")
    handle = active_path or node_id or "created synthetic file"

    if tool_name == "create_cluster" and "cluster_text" in args:
        original_chars = len(str(args.get("cluster_text", "")))
        args = dict(args)
        args["cluster_text"] = (
            f"[redacted after successful create_cluster; stored at {handle}; "
            f"original_chars={original_chars}; use read_file on that path if needed]"
        )
        return _with_tool_call_args(tool_call, args)

    if tool_name == "merge_clusters" and "merge_summary_text" in args:
        original_chars = len(str(args.get("merge_summary_text", "")))
        args = dict(args)
        args["merge_summary_text"] = (
            f"[redacted after successful merge_clusters; stored at {handle}; "
            f"original_chars={original_chars}; use read_file on that path if needed]"
        )
        return _with_tool_call_args(tool_call, args)

    return tool_call


def _redacted_tool_call_log(tool_call: Any, tool_result: ToolResult | None = None) -> str:
    if tool_result is not None:
        tool_call = _redacted_tool_call(tool_call, tool_result)
    args = getattr(tool_call.function, "arguments", "")
    return f"{tool_call.function.name}({args})"


def _message_debug_text(message: Message) -> str:
    role = str(message.get("role", ""))
    chunks = [f"role={role}"]
    text = get_text_content(message).strip()
    if text:
        chunks.append(text)
    tool_calls = list(message.get("tool_calls") or [])
    for tool_call in tool_calls:
        chunks.append(_redacted_tool_call_log(tool_call))
    return "\n".join(chunks).strip()


def _history_approx_tokens(history: Sequence[Message]) -> int:
    total_chars = 0
    for message in history:
        total_chars += len(_message_debug_text(message)) + 12
    return max(1, total_chars // APPROX_CHARS_PER_TOKEN)


def _trim_text_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    marker = "\n\n[... content omitted from middle before compaction ...]\n\n"
    if len(marker) >= max_chars:
        return text[:max_chars]
    retained_chars = max_chars - len(marker)
    keep_head = retained_chars // 2
    keep_tail = retained_chars - keep_head
    return text[:keep_head].rstrip() + marker + text[-keep_tail:].lstrip()


def _render_history_for_compaction(history: Sequence[Message], max_chars: int) -> str:
    blocks: list[str] = []
    for idx, message in enumerate(history, start=1):
        blocks.append(f"### MESSAGE {idx}\n{_message_debug_text(message)}")
    return _trim_text_middle("\n\n".join(blocks), max_chars)


def _is_compaction_summary_message(message: Message) -> bool:
    return message.get("role") == "user" and get_text_content(message).lstrip().startswith(
        BUILDER_COMPACTION_SUMMARY_MARKER
    )


def _compaction_summary_message(summary: str) -> Message:
    return {
        "role": "user",
        "content": (
            f"{BUILDER_COMPACTION_SUMMARY_MARKER}\n"
            "Older builder context has been summarized to keep the visible builder context small. "
            "Use this memory plus the recent verbatim turns below. Re-open exact files with read_file when needed.\n\n"
            f"{summary.strip()}"
        ),
    }


@dataclass
class RedactingAgentToolMessageEnv(MessageEnv):
    """Tool-use env that stores full filesystem state but compacts visible history."""

    tools: list[Tool]
    initial_messages: list[Message]
    max_turns: int
    reward_fn: RewardFn
    state: SyntheticFilesystemState | None = None
    filesystem_maturity_score_fn: ProxyScoreFn | None = None
    step_construction_action_bonus: float = 0.05
    step_filesystem_maturity_delta_scale: float = 0.5
    step_non_construction_turn_penalty: float = 0.005
    step_non_construction_streak_penalty: float = 0.0
    step_non_construction_streak_free: int = 3
    step_tool_error_penalty: float = 0.05
    coverage_first: bool = False
    coverage_leaf_max_browse_turns: int = 0
    minimum_active_files_before_stop: int = 0
    answerability_delta_reward_scale: float = 0.0
    answerability_probe_max_per_episode: int = 4
    answerability_probe_interval_turns: int = 8
    answerability_probe_min_maturity: float = 0.45
    log_step_details: bool = False
    log_compaction_summaries: bool = False
    retain_reward_tool_messages: bool = False
    trim_terminal_history_for_memory: bool = True
    return_empty_terminal_observation: bool = True
    clear_state_on_terminal_for_memory: bool = True
    compaction_enabled: bool = True
    compaction_backend: JudgeBackend = "gemini"
    compaction_model: str = "gemini-3.1-flash-lite-preview"
    compaction_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    compaction_api_key_env: str = "GEMINI_API_KEY"
    compaction_trigger_tokens: int = DEFAULT_BUILDER_COMPACTION_TRIGGER_TOKENS
    compaction_keep_recent_turns: int = DEFAULT_BUILDER_COMPACTION_KEEP_RECENT_TURNS
    compaction_max_output_tokens: int = DEFAULT_BUILDER_COMPACTION_MAX_OUTPUT_TOKENS
    compaction_input_max_chars: int = DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS
    compaction_target_tokens: int = 0
    history: list[Message] = field(default_factory=list)
    reward_history: list[Message] = field(default_factory=list)
    progress_metrics: dict[str, float] = field(default_factory=dict)

    _turn_count: int = 0
    _tool_dict: dict[str, Tool] = field(default_factory=dict, init=False)
    _should_stop: bool = field(default=False, init=False)
    _compacted_summary: str = field(default="", init=False)
    _compaction_count: int = field(default=0, init=False)
    _compaction_error_count: int = field(default=0, init=False)
    _last_step_filesystem_maturity_score: float = field(default=0.0, init=False)
    _non_construction_streak: int = field(default=0, init=False)
    _answerability_probe_count: int = field(default=0, init=False)
    _best_answerability_score: float = field(default=0.0, init=False)
    _last_answerability_score: float = field(default=0.0, init=False)
    _last_answerability_probe_turn: int = field(default=0, init=False)
    _coverage_browse_only_turns: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._tool_dict = {tool_obj.name: tool_obj for tool_obj in self.tools}
        if self.filesystem_maturity_score_fn is not None:
            self._last_step_filesystem_maturity_score = self.filesystem_maturity_score_fn()

    async def initial_observation(self) -> list[Message]:
        if not self.history:
            self.history = list(self.initial_messages)
        if not self.reward_history:
            self.reward_history = list(self.initial_messages)
        return self.history

    async def step(self, message: Message) -> MessageStepResult:
        self._turn_count += 1
        metrics: dict[str, float] = dict(self.progress_metrics)
        logs: dict[str, str] = {}

        assistant_idx = len(self.history)
        self.history.append(message)
        reward_assistant_idx = len(self.reward_history)
        self.reward_history.append(message)

        assistant_text = get_text_content(message)
        if assistant_text and self.log_step_details:
            logs["assistant_content"] = assistant_text

        tool_calls = list(message.get("tool_calls") or [])
        direct_coverage_before = 0
        coverage_leaf_creation_due = False
        if self.state is not None:
            direct_coverage_before = len(
                {
                    path
                    for record in self.state.active_records()
                    for path in record.direct_raw_doc_paths
                }
            )
            coverage_leaf_creation_due = coverage_leaf_creation_required(
                enabled=self.coverage_first and self.coverage_leaf_max_browse_turns > 0,
                visible_raw_doc_paths=self.state.raw_doc_rel_paths(),
                active_direct_raw_doc_paths=(
                    path
                    for record in self.state.active_records()
                    for path in record.direct_raw_doc_paths
                ),
                consecutive_browse_only_turns=self._coverage_browse_only_turns,
                max_browse_only_turns=max(1, self.coverage_leaf_max_browse_turns),
            )
        tool_results: list[ToolResult] = []
        if tool_calls:
            tool_results = await asyncio.gather(
                *[handle_tool_call(self._tool_dict, tool_call) for tool_call in tool_calls]
            )

            redacted_tool_calls = [
                _redacted_tool_call(tool_call, tool_result)
                for tool_call, tool_result in zip(tool_calls, tool_results, strict=True)
            ]
            if redacted_tool_calls != tool_calls:
                redacted_message = dict(message)
                redacted_message["tool_calls"] = redacted_tool_calls
                self.history[assistant_idx] = redacted_message
                self.reward_history[reward_assistant_idx] = redacted_message

            for idx, (tool_call, tool_result) in enumerate(
                zip(tool_calls, tool_results, strict=True)
            ):
                if self.log_step_details:
                    logs[f"tool_call_{idx}"] = _redacted_tool_call_log(tool_call, tool_result)
                    logs[f"tool_result_{idx}"] = _tool_result_text(tool_result)
                if not _tool_result_succeeded(tool_result):
                    tool_name = getattr(tool_call.function, "name", "unknown_tool")
                    error_text = _tool_result_error_text(tool_result)
                    logs[f"tool_error_{idx}"] = f"{tool_name}: {error_text[:1000]}"
                for result_msg in tool_result.messages:
                    self.history.append(result_msg)
                    if self.retain_reward_tool_messages:
                        self.reward_history.append(result_msg)
                if tool_result.should_stop:
                    self._should_stop = True

        coverage_leaf_gate_violation = False
        if self.state is not None and tool_calls:
            direct_coverage_after = len(
                {
                    path
                    for record in self.state.active_records()
                    for path in record.direct_raw_doc_paths
                }
            )
            if direct_coverage_after > direct_coverage_before:
                self._coverage_browse_only_turns = 0
            else:
                self._coverage_browse_only_turns += 1
                if coverage_leaf_creation_due:
                    coverage_leaf_gate_violation = True
                    metrics["coverage_leaf_gate_violation"] = 1.0
                    metrics["coverage_leaf_gate_browse_only_turns"] = float(
                        self._coverage_browse_only_turns
                    )
        no_tool_calls = len(tool_calls) == 0
        max_turns_reached = self._turn_count >= self.max_turns
        active_file_count = len(self.state.active_records()) if self.state is not None else 0
        if (
            self.minimum_active_files_before_stop > 0
            and no_tool_calls
            and not max_turns_reached
            and active_file_count < self.minimum_active_files_before_stop
        ):
            budget_reminder: Message = {
                "role": "user",
                "content": (
                    "Matched construction budget not yet satisfied: the active filesystem has "
                    f"{active_file_count} files and the frozen target is "
                    f"{self.minimum_active_files_before_stop}. Continue constructing; do not STOP "
                    "before reaching the target, and avoid exceeding it."
                ),
            }
            self.history.append(budget_reminder)
            self.reward_history.append(budget_reminder)
            metrics["construction_budget_deferred_stop"] = 1.0
            metrics["construction_budget_active_files"] = float(active_file_count)
            metrics["construction_budget_target_active_files"] = float(
                self.minimum_active_files_before_stop
            )
            no_tool_calls = False
        if self.coverage_first and no_tool_calls and not max_turns_reached and self.state is not None:
            visible_raw_docs = set(self.state.raw_doc_rel_paths())
            covered_raw_docs = {
                path
                for record in self.state.active_records()
                for path in record.direct_raw_doc_paths
            }
            if not visible_raw_docs.issubset(covered_raw_docs):
                missing = len(visible_raw_docs - covered_raw_docs)
                coverage_reminder: Message = {
                    "role": "user",
                    "content": (
                        "Coverage gate not yet satisfied: "
                        f"{missing} visible raw document(s) still lack direct provenance in an active leaf. "
                        "Continue with filesystem tools; do not STOP until coverage is complete or the turn budget is reached."
                    ),
                }
                self.history.append(coverage_reminder)
                self.reward_history.append(coverage_reminder)
                metrics["coverage_gate_deferred_stop"] = 1.0
                metrics["coverage_gate_missing_raw_docs"] = float(missing)
                no_tool_calls = False
        done = no_tool_calls or max_turns_reached or self._should_stop or coverage_leaf_gate_violation

        reward, step_metrics = self._step_construction_reward(tool_calls, tool_results)
        metrics.update(step_metrics)
        (
            answerability_reward,
            answerability_metrics,
        ) = await self._maybe_answerability_progress_reward(
            done=done,
            step_metrics=step_metrics,
        )
        reward += answerability_reward
        metrics.update(answerability_metrics)
        if "step_reward" in metrics:
            step_process_reward = step_metrics.get("step_reward", 0.0)
            metrics["step_process_reward"] = step_process_reward
            metrics["step_reward"] = reward
            metrics.update(
                {
                    "intermediate_answerability_reward": answerability_reward,
                    "intermediate_process_reward": step_process_reward,
                    "intermediate_structure_reward": step_metrics.get(
                        "step_filesystem_maturity_reward", 0.0
                    ),
                    "intermediate_create_merge_action_bonus": step_metrics.get(
                        "step_construction_bonus", 0.0
                    ),
                    "intermediate_tool_error_penalty": step_metrics.get(
                        "step_tool_error_penalty", 0.0
                    ),
                    "intermediate_non_construction_penalty": (
                        step_metrics.get("step_non_construction_penalty", 0.0)
                        + step_metrics.get("step_non_construction_streak_penalty", 0.0)
                    ),
                    "intermediate_total_reward": reward,
                }
            )

        if not done:
            compaction_metrics, compaction_logs = await self._maybe_compact_visible_history()
            metrics.update(compaction_metrics)
            logs.update(compaction_logs)

        if max_turns_reached and not no_tool_calls:
            metrics["max_turns"] = 1.0
        if self._should_stop:
            metrics["tool_stopped"] = 1.0
        if coverage_leaf_gate_violation:
            metrics["coverage_leaf_gate_invalid_construction"] = 1.0
        if done and self.minimum_active_files_before_stop > 0:
            final_active_files = len(self.state.active_records()) if self.state is not None else 0
            metrics["construction_budget_target_active_files"] = float(
                self.minimum_active_files_before_stop
            )
            metrics["construction_budget_final_active_files"] = float(final_active_files)
            metrics["construction_budget_exact_match"] = float(
                final_active_files == self.minimum_active_files_before_stop
            )

        if done:
            terminal_reward, reward_metrics = await self.reward_fn(self.reward_history)
            reward += terminal_reward
            metrics.update(reward_metrics)
            answerer_trace = getattr(self.reward_fn, "last_answerer_trace", None)
            if self.log_step_details and answerer_trace:
                logs["answerer_trace"] = json.dumps(answerer_trace, ensure_ascii=False)
            metrics["builder_context_compactions"] = float(self._compaction_count)
            metrics["builder_context_compaction_errors"] = float(self._compaction_error_count)
            if self.trim_terminal_history_for_memory:
                self._trim_terminal_history()
            if not self.retain_reward_tool_messages:
                self.reward_history.clear()
            next_messages = [] if self.return_empty_terminal_observation else self.history
            if self.clear_state_on_terminal_for_memory and self.state is not None:
                self.state.files_by_id.clear()
                self.tools.clear()
                self._tool_dict.clear()
                self.initial_messages.clear()
            if self.return_empty_terminal_observation:
                self.history.clear()
            gc.collect()
        else:
            next_messages = self.history

        return MessageStepResult(
            reward=reward,
            episode_done=done,
            next_messages=next_messages,
            metrics=metrics,
            logs=logs,
        )

    async def _maybe_answerability_progress_reward(
        self,
        *,
        done: bool,
        step_metrics: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        metrics = {
            "answerability_probe_attempted": 0.0,
            "answerability_probe_evaluated": 0.0,
            "answerability_probe_reward": 0.0,
            "answerability_probe_interval_turns": float(self.answerability_probe_interval_turns),
        }
        if done:
            return 0.0, metrics
        if self.answerability_delta_reward_scale <= 0.0:
            return 0.0, metrics
        if self._answerability_probe_count >= max(0, self.answerability_probe_max_per_episode):
            return 0.0, metrics
        interval_turns = max(0, self.answerability_probe_interval_turns)
        if interval_turns > 0:
            turns_since_probe = self._turn_count - self._last_answerability_probe_turn
            if turns_since_probe < interval_turns:
                metrics["answerability_probe_skipped_interval"] = 1.0
                metrics["answerability_probe_turns_since_last"] = float(turns_since_probe)
                return 0.0, metrics
        if step_metrics.get("step_construction_actions", 0.0) <= 0.0:
            return 0.0, metrics
        maturity = step_metrics.get("step_filesystem_maturity_score", 0.0)
        if maturity < self.answerability_probe_min_maturity:
            return 0.0, metrics
        probe_fn = getattr(self.reward_fn, "answerability_progress_probe", None)
        if probe_fn is None:
            return 0.0, metrics

        metrics["answerability_probe_attempted"] = 1.0
        try:
            new_best, reward, probe_metrics = await probe_fn(
                previous_score=self._last_answerability_score,
                previous_best_score=self._best_answerability_score,
            )
        except Exception as exc:
            metrics["answerability_probe_error"] = 1.0
            metrics["answerability_probe_reward"] = 0.0
            LOGGER.warning("Answerability progress probe failed: %s", exc)
            return 0.0, metrics

        self._answerability_probe_count += 1
        self._last_answerability_probe_turn = self._turn_count
        current_score = float(
            probe_metrics.get("answerability_probe_score", self._last_answerability_score)
        )
        self._best_answerability_score = max(self._best_answerability_score, float(new_best))
        self._last_answerability_score = current_score
        metrics.update(probe_metrics)
        return reward, metrics

    def _trim_terminal_history(self) -> None:
        last_assistant: Message | None = None
        for message in reversed(self.history):
            if message.get("role") == "assistant":
                last_assistant = message
                break
        trimmed = list(self.initial_messages)
        if self._compacted_summary.strip():
            trimmed.append(_compaction_summary_message(self._compacted_summary))
        if last_assistant is not None:
            trimmed.append(last_assistant)
        self.history = trimmed

    def _step_construction_reward(
        self,
        tool_calls: Sequence[Any],
        tool_results: Sequence[ToolResult],
    ) -> tuple[float, dict[str, float]]:
        if not tool_calls:
            return 0.0, {}

        successful_create = 0
        successful_merge = 0
        successful_read_file = 0
        successful_list_files = 0
        tool_errors = 0
        executor_generated_actions = 0
        tool_error_counts: dict[str, int] = {}
        tool_error_code_counts: dict[str, int] = {}
        for tool_call, tool_result in zip(tool_calls, tool_results, strict=True):
            tool_name = getattr(tool_call.function, "name", "")
            if not _tool_result_succeeded(tool_result):
                tool_errors += 1
                tool_error_counts[tool_name] = tool_error_counts.get(tool_name, 0) + 1
                error_code_key = f"{_metric_key_part(tool_name)}_{_metric_key_part(_tool_result_error_code(tool_result))}"
                tool_error_code_counts[error_code_key] = (
                    tool_error_code_counts.get(error_code_key, 0) + 1
                )
                continue
            payload = _tool_result_payload(tool_result)
            if tool_name == "create_cluster":
                successful_create += 1
                if payload.get("executor_generated"):
                    executor_generated_actions += 1
            elif tool_name == "create_clusters":
                created_count = int(float(payload.get("created_count", 0.0) or 0.0))
                successful_create += created_count
                if payload.get("executor_generated"):
                    executor_generated_actions += created_count
            elif tool_name == "merge_clusters":
                successful_merge += 1
                if payload.get("executor_generated"):
                    executor_generated_actions += 1
            elif tool_name == "merge_many":
                created_count = int(float(payload.get("created_count", 0.0) or 0.0))
                successful_merge += created_count
                if payload.get("executor_generated"):
                    executor_generated_actions += created_count
            elif tool_name == "read_file":
                successful_read_file += 1
            elif tool_name == "read_many":
                reads = payload.get("reads", [])
                successful_read_file += len(reads) if isinstance(reads, list) else 1
            elif tool_name == "list_files":
                successful_list_files += 1

        construction_actions = successful_create + successful_merge
        non_construction_tool_calls = successful_read_file + successful_list_files
        reward = 0.0
        construction_bonus = self.step_construction_action_bonus * float(construction_actions)
        reward += construction_bonus

        filesystem_maturity_delta = 0.0
        filesystem_maturity_reward = 0.0
        filesystem_maturity_score = self._last_step_filesystem_maturity_score
        if construction_actions > 0 and self.filesystem_maturity_score_fn is not None:
            filesystem_maturity_score = self.filesystem_maturity_score_fn()
            filesystem_maturity_delta = max(
                0.0,
                filesystem_maturity_score - self._last_step_filesystem_maturity_score,
            )
            filesystem_maturity_reward = (
                self.step_filesystem_maturity_delta_scale * filesystem_maturity_delta
            )
            reward += filesystem_maturity_reward
            self._last_step_filesystem_maturity_score = filesystem_maturity_score

        non_construction_penalty = 0.0
        if non_construction_tool_calls > 0 and self.step_non_construction_turn_penalty > 0.0:
            non_construction_penalty = self.step_non_construction_turn_penalty * float(
                non_construction_tool_calls
            )
            reward -= non_construction_penalty

        if construction_actions > 0:
            self._non_construction_streak = 0
        elif non_construction_tool_calls > 0:
            self._non_construction_streak += non_construction_tool_calls

        non_construction_streak_penalty = 0.0
        if self.step_non_construction_streak_penalty > 0.0:
            chargeable_streak = max(
                0,
                self._non_construction_streak - max(0, self.step_non_construction_streak_free),
            )
            non_construction_streak_penalty = self.step_non_construction_streak_penalty * float(
                chargeable_streak
            )
            reward -= non_construction_streak_penalty

        tool_error_penalty = self.step_tool_error_penalty * float(tool_errors)
        reward -= tool_error_penalty

        metrics = {
            "step_successful_create_clusters": float(successful_create),
            "step_successful_merge_clusters": float(successful_merge),
            "step_successful_read_files": float(successful_read_file),
            "step_successful_list_files": float(successful_list_files),
            "step_construction_actions": float(construction_actions),
            "step_executor_generated_actions": float(executor_generated_actions),
            "step_construction_bonus": construction_bonus,
            "step_non_construction_tool_calls": float(non_construction_tool_calls),
            "step_filesystem_maturity_score": filesystem_maturity_score,
            "step_filesystem_maturity_delta": filesystem_maturity_delta,
            "step_filesystem_maturity_reward": filesystem_maturity_reward,
            "step_non_construction_penalty": non_construction_penalty,
            "step_non_construction_streak": float(self._non_construction_streak),
            "step_non_construction_streak_penalty": non_construction_streak_penalty,
            "step_tool_errors": float(tool_errors),
            "step_tool_errors_read_file": float(tool_error_counts.get("read_file", 0)),
            "step_tool_errors_read_many": float(tool_error_counts.get("read_many", 0)),
            "step_tool_errors_list_files": float(tool_error_counts.get("list_files", 0)),
            "step_tool_errors_create_cluster": float(tool_error_counts.get("create_cluster", 0)),
            "step_tool_errors_create_clusters": float(tool_error_counts.get("create_clusters", 0)),
            "step_tool_errors_merge_clusters": float(tool_error_counts.get("merge_clusters", 0)),
            "step_tool_errors_merge_many": float(tool_error_counts.get("merge_many", 0)),
            "step_tool_errors_delete": float(tool_error_counts.get("delete", 0)),
            "step_tool_error_penalty": tool_error_penalty,
            "step_reward": reward,
        }
        for error_code_key, count in tool_error_code_counts.items():
            metrics[f"step_tool_error_code_{error_code_key}"] = float(count)
        return reward, metrics

    async def _maybe_compact_visible_history(self) -> tuple[dict[str, float], dict[str, str]]:
        metrics: dict[str, float] = {}
        logs: dict[str, str] = {}
        if not self.compaction_enabled:
            return metrics, logs
        if self.compaction_trigger_tokens <= 0:
            return metrics, logs

        visible_tokens_before = _history_approx_tokens(self.history)
        metrics["builder_context_visible_tokens"] = float(visible_tokens_before)
        if visible_tokens_before < self.compaction_trigger_tokens:
            return metrics, logs

        prefix_len = len(self.initial_messages)
        if len(self.history) <= prefix_len:
            return metrics, logs

        mutable_history = list(self.history[prefix_len:])
        if mutable_history and _is_compaction_summary_message(mutable_history[0]):
            mutable_history = mutable_history[1:]

        assistant_indices = [
            idx for idx, item in enumerate(mutable_history) if item.get("role") == "assistant"
        ]
        keep_recent_turns = max(1, self.compaction_keep_recent_turns)
        if len(assistant_indices) <= keep_recent_turns:
            return metrics, logs

        tail_start = assistant_indices[-keep_recent_turns]
        history_to_compact = mutable_history[:tail_start]
        recent_history = mutable_history[tail_start:]
        if not history_to_compact:
            return metrics, logs

        transcript = _render_history_for_compaction(
            history_to_compact,
            max_chars=self.compaction_input_max_chars,
        )
        try:
            summary = await self._summarize_builder_context(transcript)
        except Exception as exc:
            self._compaction_error_count += 1
            summary = self._fallback_compaction_summary(transcript, error=str(exc))
            logs["builder_context_compaction_error"] = str(exc)[:500]

        self._compacted_summary = summary.strip()
        self.history = (
            list(self.initial_messages)
            + [_compaction_summary_message(self._compacted_summary)]
            + recent_history
        )
        self._compaction_count += 1

        visible_tokens_after = _history_approx_tokens(self.history)
        hard_compaction_applied = False
        if self.compaction_target_tokens > 0 and visible_tokens_after > self.compaction_target_tokens:
            # Compact all mutable history if preserved turns exceed the reserve.
            transcript = _render_history_for_compaction(
                self.history[prefix_len:],
                max_chars=self.compaction_input_max_chars,
            )
            try:
                summary = await self._summarize_builder_context(transcript)
            except Exception as exc:
                self._compaction_error_count += 1
                summary = self._fallback_compaction_summary(transcript, error=str(exc))
                logs["builder_context_hard_compaction_error"] = str(exc)[:500]
            self._compacted_summary = summary.strip()
            self.history = list(self.initial_messages) + [
                _compaction_summary_message(self._compacted_summary)
            ]
            self._compaction_count += 1
            visible_tokens_after = _history_approx_tokens(self.history)
            hard_compaction_applied = True
        metrics.update(
            {
                "builder_context_compaction_applied": 1.0,
                "builder_context_compactions": float(self._compaction_count),
                "builder_context_tokens_before_compaction": float(visible_tokens_before),
                "builder_context_tokens_after_compaction": float(visible_tokens_after),
                "builder_context_messages_compacted": float(len(history_to_compact)),
                "builder_context_recent_messages_kept": float(len(recent_history)),
                "builder_context_hard_compaction_applied": float(hard_compaction_applied),
                "builder_context_compaction_target_tokens": float(self.compaction_target_tokens),
                "builder_context_reserve_safe": float(
                    self.compaction_target_tokens <= 0
                    or visible_tokens_after <= self.compaction_target_tokens
                ),
            }
        )
        if self.log_compaction_summaries:
            logs["builder_context_compaction_summary"] = self._compacted_summary[:2000]
        return metrics, logs

    async def _summarize_builder_context(self, latest_transcript: str) -> str:
        filesystem_snapshot = self._filesystem_snapshot()
        existing_summary = self._compacted_summary.strip() or "(none yet)"
        system_prompt = (
            "You compact the visible context of a tool-using agent that is constructing a synthetic filesystem. "
            "Write an updated memory summary only from the provided transcript, previous summary, and filesystem snapshot. "
            "Do not add facts that are not present. Preserve concrete source paths, line ranges, names, dates, aliases, "
            "relationships, and any evidence-to-synthetic-file links. The builder is optimizing for broad corpus "
            "coverage and reusable structure, not for a particular query, so do not infer answers or introduce "
            "query-specific conclusions."
        )
        user_prompt = (
            "Update the rolling builder memory for filesystem construction.\n\n"
            "The summary should be concise but useful for the next builder turns. Include:\n"
            "- raw files/chunks already read and the important evidence extracted from them\n"
            "- synthetic files or merge summaries already created, with paths/titles and what evidence they incorporated\n"
            "- current active filesystem structure and any deleted/obsolete state if visible\n"
            "- unresolved useful next actions, such as files/chunks that may still need reading or clusters that may need merging\n"
            "- warnings about uncertainty or incomplete coverage\n\n"
            "Keep enough detail that the builder can keep constructing without re-reading every old chunk. "
            "If exact wording is needed later, say which path/line range should be re-opened with read_file.\n\n"
            f"### EXISTING ROLLING SUMMARY\n{existing_summary}\n\n"
            f"### CURRENT FILESYSTEM SNAPSHOT\n{filesystem_snapshot}\n\n"
            f"### LATEST BUILDER TRANSCRIPT TO INCORPORATE\n{latest_transcript}\n\n"
            "Return only the updated summary. Do not include preamble."
        )
        async with trace.scope_span("remote/builder_compaction_call"):
            response = await asyncio.to_thread(
                call_chat_model_messages,
                backend=self.compaction_backend,
                model=self.compaction_model,
                base_url=self.compaction_base_url,
                api_key_env=self.compaction_api_key_env,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_json=False,
                max_output_tokens=self.compaction_max_output_tokens,
                enable_thinking=False,
            )
        return response.strip()

    def _filesystem_snapshot(self) -> str:
        if self.state is None:
            return "(filesystem state unavailable)"
        active_records = self.state.active_records()
        lines = [
            f"operation_round: {self.state.operation_round}",
            "active_tree:",
            self.state.active_tree_string(),
            "active_files:",
        ]
        if not active_records:
            lines.append("- (none)")
        for rec in active_records:
            lines.append(
                "- "
                + json.dumps(
                    {
                        "path": self.state.active_path(rec.node_id),
                        "history_path": self.state.history_path(rec.node_id),
                        "title": rec.title,
                        "kind": rec.kind,
                        "direct_raw_doc_paths": rec.direct_raw_doc_paths,
                        "direct_synthetic_ids": rec.direct_synthetic_ids,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        return "\n".join(lines)

    def _fallback_compaction_summary(self, latest_transcript: str, *, error: str) -> str:
        transcript_excerpt = _trim_text_middle(latest_transcript, 6000)
        existing = self._compacted_summary.strip()
        pieces = []
        if existing:
            pieces.append(existing)
        pieces.append(
            "Automatic model compaction failed, so this is a conservative fallback summary. "
            f"Compaction error: {error[:300]}"
        )
        pieces.append("Current filesystem snapshot:\n" + self._filesystem_snapshot())
        pieces.append(
            "Recent older transcript excerpt that was compacted; re-open exact files with read_file if needed:\n"
            + transcript_excerpt
        )
        rendered = "\n\n".join(pieces).strip()
        max_chars = int(self.compaction_max_output_tokens) * APPROX_CHARS_PER_TOKEN
        return _trim_text_middle(rendered, max_chars).strip()


def build_redacting_agent_tool_env(
    *,
    renderer: Renderer,
    tools: list[Tool],
    initial_messages: list[Message],
    reward_fn: RewardFn,
    max_turns: int,
    state: SyntheticFilesystemState | None = None,
    filesystem_maturity_score_fn: ProxyScoreFn | None = None,
    step_construction_action_bonus: float = 0.05,
    step_filesystem_maturity_delta_scale: float = 0.5,
    step_non_construction_turn_penalty: float = 0.005,
    step_non_construction_streak_penalty: float = 0.0,
    step_non_construction_streak_free: int = 3,
    step_tool_error_penalty: float = 0.05,
    coverage_first: bool = False,
    coverage_leaf_max_browse_turns: int = 0,
    minimum_active_files_before_stop: int = 0,
    answerability_delta_reward_scale: float = 0.0,
    answerability_probe_max_per_episode: int = 4,
    answerability_probe_interval_turns: int = 8,
    answerability_probe_min_maturity: float = 0.45,
    log_step_details: bool = False,
    log_compaction_summaries: bool = False,
    retain_reward_tool_messages: bool = False,
    trim_terminal_history_for_memory: bool = True,
    return_empty_terminal_observation: bool = True,
    clear_state_on_terminal_for_memory: bool = True,
    compaction_enabled: bool = True,
    compaction_backend: JudgeBackend = "gemini",
    compaction_model: str = "gemini-3.1-flash-lite-preview",
    compaction_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    compaction_api_key_env: str = "GEMINI_API_KEY",
    compaction_trigger_tokens: int = DEFAULT_BUILDER_COMPACTION_TRIGGER_TOKENS,
    compaction_keep_recent_turns: int = DEFAULT_BUILDER_COMPACTION_KEEP_RECENT_TURNS,
    compaction_max_output_tokens: int = DEFAULT_BUILDER_COMPACTION_MAX_OUTPUT_TOKENS,
    compaction_input_max_chars: int = DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS,
    compaction_target_tokens: int = 0,
    failed_parse_reward: float = -0.1,
    max_trajectory_tokens: int | None = 140000,
    context_overflow_reward: float = -0.1,
    progress_metrics: dict[str, float] | None = None,
) -> Env:
    message_env = RedactingAgentToolMessageEnv(
        tools=tools,
        initial_messages=initial_messages,
        max_turns=max_turns,
        reward_fn=reward_fn,
        state=state,
        filesystem_maturity_score_fn=filesystem_maturity_score_fn,
        step_construction_action_bonus=step_construction_action_bonus,
        step_filesystem_maturity_delta_scale=step_filesystem_maturity_delta_scale,
        step_non_construction_turn_penalty=step_non_construction_turn_penalty,
        step_non_construction_streak_penalty=step_non_construction_streak_penalty,
        step_non_construction_streak_free=step_non_construction_streak_free,
        step_tool_error_penalty=step_tool_error_penalty,
        coverage_first=coverage_first,
        coverage_leaf_max_browse_turns=coverage_leaf_max_browse_turns,
        minimum_active_files_before_stop=minimum_active_files_before_stop,
        answerability_delta_reward_scale=answerability_delta_reward_scale,
        answerability_probe_max_per_episode=answerability_probe_max_per_episode,
        answerability_probe_interval_turns=answerability_probe_interval_turns,
        answerability_probe_min_maturity=answerability_probe_min_maturity,
        log_step_details=log_step_details,
        log_compaction_summaries=log_compaction_summaries,
        retain_reward_tool_messages=retain_reward_tool_messages,
        trim_terminal_history_for_memory=trim_terminal_history_for_memory,
        return_empty_terminal_observation=return_empty_terminal_observation,
        clear_state_on_terminal_for_memory=clear_state_on_terminal_for_memory,
        compaction_enabled=compaction_enabled,
        compaction_backend=compaction_backend,
        compaction_model=compaction_model,
        compaction_base_url=compaction_base_url,
        compaction_api_key_env=compaction_api_key_env,
        compaction_trigger_tokens=compaction_trigger_tokens,
        compaction_keep_recent_turns=compaction_keep_recent_turns,
        compaction_max_output_tokens=compaction_max_output_tokens,
        compaction_input_max_chars=compaction_input_max_chars,
        compaction_target_tokens=compaction_target_tokens,
        progress_metrics=dict(progress_metrics or {}),
    )
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=message_env,
        failed_parse_reward=failed_parse_reward,
        max_trajectory_tokens=max_trajectory_tokens,
        context_overflow_reward=context_overflow_reward,
    )


def initial_messages_builder(
    datum: SyntheticFilesystemDatum,
    renderer: Renderer,
    fs_tools: BuilderFilesystemTools,
    *,
    executor_enabled: bool = False,
    batch_tools_enabled: bool = False,
    coverage_first: bool = False,
    coverage_leaf_max_browse_turns: int = 0,
    task_question: str | None = None,
    source_inventory: Sequence[str] | None = None,
    plan_once_prompt_aligned: bool = False,
    explicit_construction_instruction: str | None = None,
    construction_budget_active_files: int = 0,
) -> list[Message]:
    tool_schemas = [
        fs_tools.list_files.to_spec(),
        fs_tools.read_file.to_spec(),
    ]
    if batch_tools_enabled:
        tool_schemas.append(fs_tools.read_many.to_spec())
    tool_schemas.extend(
        [
            fs_tools.create_cluster.to_spec(),
            fs_tools.merge_clusters.to_spec(),
        ]
    )
    if batch_tools_enabled:
        tool_schemas.extend(
            [
                fs_tools.create_clusters.to_spec(),
                fs_tools.merge_many.to_spec(),
            ]
        )
    tool_schemas.append(fs_tools.delete.to_spec())
    if plan_once_prompt_aligned:
        if (
            not executor_enabled
            or task_question is not None
            or not source_inventory
            or not explicit_construction_instruction
        ):
            raise ValueError(
                "Explicit prompt alignment requires question-blind executor mode, a source inventory, "
                "and a construction instruction"
            )
        system_prompt = (
            "You are a filesystem construction policy. Follow the explicit question-blind construction "
            "instruction and exact visible source handles supplied in the user message. Do not answer a "
            "downstream task. Preserve direct provenance in focused evidence cards."
        )
        user_prompt = explicit_construction_instruction.strip()
    else:
        system_prompt = build_builder_system_prompt(
            executor_enabled=executor_enabled,
            batch_tools_enabled=batch_tools_enabled,
            coverage_first=coverage_first,
            coverage_leaf_max_browse_turns=coverage_leaf_max_browse_turns,
            task_question=task_question,
        )
        user_prompt = build_builder_user_prompt(
            executor_enabled=executor_enabled,
            batch_tools_enabled=batch_tools_enabled,
            coverage_first=coverage_first,
            coverage_leaf_max_browse_turns=coverage_leaf_max_browse_turns,
            task_question=task_question,
        )
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=system_prompt,
    )
    if source_inventory:
        normalized_inventory = [
            path if path.startswith("raw_docs/") else f"raw_docs/{path}"
            for path in source_inventory
        ]
        if len(normalized_inventory) != len(set(normalized_inventory)):
            raise ValueError("Source inventory contains duplicate paths")
        if plan_once_prompt_aligned:
            user_prompt += "\n\nVisible sources:\n" + "\n".join(normalized_inventory)
        else:
            user_prompt += (
                "\n\nQuestion-blind visible source inventory (paths only):\n"
                + "\n".join(normalized_inventory)
                + "\nUse these exact paths directly; do not rediscover them with repeated directory listings."
            )
    if construction_budget_active_files > 0:
        user_prompt += (
            "\n\nMatched construction budget: leave exactly "
            f"{construction_budget_active_files} active synthetic files. This target is fixed from "
            "the paired RL rollout for this corpus. Do not stop below it or create files beyond it."
        )
    return prefix + [{"role": "user", "content": user_prompt}]
