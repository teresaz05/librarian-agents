from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from experiments.inference.answer import (
    AnswerPrompt,
    DEFAULT_ANSWER_MAX_TOKENS,
    DEFAULT_ANSWER_MIN_P,
    DEFAULT_ANSWER_PRESENCE_PENALTY,
    DEFAULT_ANSWER_REPETITION_PENALTY,
    DEFAULT_ANSWER_TEMPERATURE,
    DEFAULT_ANSWER_TOP_K,
    DEFAULT_ANSWER_TOP_P,
    DEFAULT_LOCAL_MODEL,
    QwenAnswerAgent,
    answer_records,
    clean_response,
    load_env_file,
    resolve_reasoning_arg,
)
from experiments.inference.llm import (
    ReasoningEffort,
    create_async_client,
    generate_responses,
    get_provider_config,
)
from experiments.inference.retrieve import (
    DEFAULT_DOCS_DIR,
    DEFAULT_PRIVILEGED_DIR,
    DEFAULT_QUESTIONS_PATH,
    CandidateDocument,
    ensure_no_record_errors,
    load_candidate_documents,
    load_heldout_questions,
    select_questions,
    write_jsonl,
)


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVAL_OUTPUT_PATH = REPO_ROOT / "outputs" / "inference" / "autocompaction_retrieval.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "inference" / "autocompaction_answers.jsonl"
DEFAULT_GOLD_AND_SUPPORT_ONLY = False
FORCE_FULL_DOCUMENT_CORPUS = True
DEFAULT_NON_LOCAL_GENERATOR = True
DEFAULT_NON_LOCAL_ANSWERER = True
DEFAULT_LOCAL_GENERATOR_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_LOCAL_ANSWER_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_LOCAL_GENERATOR_MAX_MODEL_LEN = 16384
DEFAULT_SUMMARY_MAX_TOKENS = 2048
DEFAULT_GENERATOR_REASONING_EFFORT = "none"
DEFAULT_ANSWER_REASONING_EFFORT = "none"
DEFAULT_COMPACTION_BATCH_SIZE = 8
DEFAULT_GENERATOR_MAX_CONCURRENT = 8

AUTOCOMPACTION_PROMPT_TEMPLATE = """Compress a document stream into query-independent notes for later QA.

Rules:
- The future question is hidden. Do not tailor the notes to any query.
- Keep only grounded facts: names, dates, numbers, identifiers, titles, events, relationships, claims, and caveats.
- Prefer dense bullet notes or compact clauses. Remove filler, repeated wording, and low-value prose.
- If facts conflict, keep the conflict and source path instead of resolving it.
- Use at most {summary_max_tokens} tokens.
- Return only the updated notes. Do not include reasoning, analysis, preambles, or markdown fences.

CURRENT SUMMARY:
{current_summary}

NEXT DOCUMENT:
path: {path}
stream_position: {stream_position}
fragment: {fragment_index}/{num_fragments}
content:
{document_text}
"""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    raw_response: str
    generated_token_count: int | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class StreamUnit:
    document: CandidateDocument
    fragment_index: int
    num_fragments: int
    text: str


@dataclass
class CompactionState:
    record_index: int
    question_entry: dict[str, str]
    documents: list[CandidateDocument]
    stream_units: list[StreamUnit]
    current_summary: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    next_stream_index: int = 0
    start_time: float = field(default_factory=time.monotonic)


class NonLocalAutocompactionGenerator:
    def __init__(
        self,
        provider: str,
        model_name: str,
        *,
        max_completion_tokens: int,
        reasoning_effort: ReasoningEffort,
        use_cache: bool,
        max_concurrent: int,
    ) -> None:
        self.config = get_provider_config(provider)
        self.provider_name = self.config.name
        self.model_name = model_name or self.config.default_model
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.use_cache = use_cache
        self.max_concurrent = max_concurrent
        self.client = create_async_client(self.config)
        self.tensor_parallel_size = None

    def generate(self, prompt: str) -> GenerationResult:
        results = self.generate_batch([prompt])
        if not results:
            return GenerationResult(text="", raw_response="")
        return results[0]

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        if not prompts:
            return []
        response_groups = asyncio.run(
            generate_responses(
                self.client,
                prompts,
                N=1,
                max_completion_tokens=self.max_completion_tokens,
                reasoning_effort=self.reasoning_effort,
                use_cache=self.use_cache,
                max_concurrent=self.max_concurrent,
                model_name=self.model_name,
                provider=self.config,
            )
        )
        results: list[GenerationResult] = []
        for response_group in response_groups:
            raw_response = response_group[0] if response_group else ""
            results.append(
                GenerationResult(
                    text=clean_compaction_response(raw_response),
                    raw_response=raw_response,
                )
            )
        return results

    def count_prompt_tokens(self, prompt: str) -> int:
        return max(1, (len(prompt) + 3) // 4)

    def stop(self) -> None:
        try:
            asyncio.run(self.client.close())
        except Exception:
            pass


class LocalAutocompactionGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        tensor_parallel_size: int | None,
        max_model_len: int,
        gpu_memory_utilization: float,
        dtype: str,
        trust_remote_code: bool,
        distributed_executor_backend: str | None,
        disable_thinking: bool,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> None:
        from vllm import SamplingParams

        self.provider_name = "local"
        self.model_name = model_name_or_path
        self.agent = QwenAnswerAgent(
            model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            distributed_executor_backend=distributed_executor_backend,
            disable_thinking=disable_thinking,
        )
        self.tensor_parallel_size = self.agent.tensor_parallel_size
        sampling_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_completion_tokens,
        }
        if top_k > 0:
            sampling_kwargs["top_k"] = top_k
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def generate(self, prompt: str) -> GenerationResult:
        results = self.generate_batch([prompt])
        if not results:
            return GenerationResult(text="", raw_response="")
        return results[0]

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        prompt_records = [
            AnswerPrompt(
                messages=[{"role": "user", "content": prompt}],
                prompt_documents=[],
                prompt_token_count=0,
            )
            for prompt in prompts
        ]
        generation_groups = self.agent.generate(prompt_records, self.sampling_params)
        results: list[GenerationResult] = []
        for group in generation_groups:
            generation = group[0] if group else {}
            raw_response = str(generation.get("raw_response") or generation.get("response") or "")
            results.append(
                GenerationResult(
                    text=clean_compaction_response(raw_response),
                    raw_response=raw_response,
                    generated_token_count=generation.get("generated_token_count"),
                    finish_reason=generation.get("finish_reason"),
                )
            )
        return results

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.agent.apply_chat_template([{"role": "user", "content": prompt}]))

    def stop(self) -> None:
        try:
            self.agent.stop()
        except Exception:
            pass


AutocompactionGenerator = NonLocalAutocompactionGenerator | LocalAutocompactionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a query-blind autocompaction baseline over heldout documents, "
            "then answer from the generated summary and judge."
        )
    )
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--privileged-dir", type=Path, default=DEFAULT_PRIVILEGED_DIR)
    parser.add_argument("--retrieval-output-path", type=Path, default=DEFAULT_RETRIEVAL_OUTPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)

    parser.add_argument(
        "--non-local-generator",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_NON_LOCAL_GENERATOR,
        help="Use experiments.inference.llm for query-blind summary generation.",
    )
    parser.add_argument("--generator-provider", default="gemini")
    parser.add_argument(
        "--generator-model-name",
        default="",
        help=(
            "Defaults to the selected provider's model for --non-local-generator, "
            f"or {DEFAULT_LOCAL_MODEL} for --no-non-local-generator."
        ),
    )
    parser.add_argument("--summary-max-tokens", type=int, default=DEFAULT_SUMMARY_MAX_TOKENS)
    parser.add_argument("--generator-reasoning-effort", default=DEFAULT_GENERATOR_REASONING_EFFORT)
    parser.add_argument("--generator-use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--generator-max-concurrent",
        type=int,
        default=DEFAULT_GENERATOR_MAX_CONCURRENT,
        help="For --non-local-generator: maximum concurrent API prompts per compaction batch.",
    )
    parser.add_argument("--generator-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--generator-distributed-executor-backend", default=None)
    parser.add_argument("--generator-max-model-len", type=int, default=DEFAULT_LOCAL_GENERATOR_MAX_MODEL_LEN)
    parser.add_argument(
        "--generator-gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "GPU memory fraction for local summary-generator vLLM. Defaults to "
            f"{DEFAULT_LOCAL_GENERATOR_GPU_MEMORY_UTILIZATION}."
        ),
    )
    parser.add_argument("--generator-dtype", default="auto")
    parser.add_argument("--generator-trust-remote-code", action="store_true")
    parser.add_argument("--generator-temperature", type=float, default=0.0)
    parser.add_argument("--generator-top-p", type=float, default=1.0)
    parser.add_argument("--generator-top-k", type=int, default=-1)
    parser.add_argument("--generator-disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-doc-chars", type=int, default=30_000)
    parser.add_argument(
        "--stream-chunk-chars",
        type=int,
        default=18_000,
        help="Split large documents into this many characters per stream update. Use 0 to disable chunking.",
    )
    parser.add_argument(
        "--include-intermediate-summaries",
        action="store_true",
        help="Store the summary text after each stream unit in the retrieval JSONL.",
    )
    parser.add_argument(
        "--stream-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a progress bar for document stream units across active batched questions.",
    )
    parser.add_argument(
        "--compaction-batch-size",
        type=int,
        default=DEFAULT_COMPACTION_BATCH_SIZE,
        help="Number of questions to advance together in each autocompaction generator batch.",
    )
    parser.add_argument(
        "--gold-and-support-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GOLD_AND_SUPPORT_ONLY,
        help="Compaction always uses all candidate documents.",
    )
    parser.add_argument("--no-text", action="store_true", help="Omit summary text from retrieval JSONL. Requires --skip-answer.")

    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--non-local-answerer", action=argparse.BooleanOptionalAction, default=DEFAULT_NON_LOCAL_ANSWERER)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument(
        "--model-name-or-path",
        default="",
        help="Defaults to Qwen locally, or the selected provider's default for --non-local-answerer.",
    )
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--answer-reasoning-effort", default=DEFAULT_ANSWER_REASONING_EFFORT)
    parser.add_argument("--answer-use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--distributed-executor-backend", default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-prompt-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_ANSWER_MAX_TOKENS)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "GPU memory fraction for the local answerer vLLM. Defaults to "
            f"{DEFAULT_LOCAL_ANSWER_GPU_MEMORY_UTILIZATION}."
        ),
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--temperature", type=float, default=DEFAULT_ANSWER_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_ANSWER_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_ANSWER_TOP_K)
    parser.add_argument("--min-p", type=float, default=DEFAULT_ANSWER_MIN_P)
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=DEFAULT_ANSWER_PRESENCE_PENALTY,
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_ANSWER_REPETITION_PENALTY,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-docs", type=int, default=1)
    parser.add_argument("--answer-max-doc-chars", type=int, default=65_536)
    parser.add_argument("--min-doc-chars", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-provider", default="gemini")
    parser.add_argument("--judge-model-name", default="")
    parser.add_argument("--judge-max-concurrent", type=int, default=4)
    parser.add_argument("--judge-max-completion-tokens", type=int, default=64)
    parser.add_argument("--judge-reasoning-effort", default="none")
    parser.add_argument("--judge-use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-id", action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.summary_max_tokens < 1:
        parser.error("--summary-max-tokens must be at least 1")
    if args.compaction_batch_size < 1:
        parser.error("--compaction-batch-size must be at least 1")
    if args.generator_max_concurrent < 1:
        parser.error("--generator-max-concurrent must be at least 1")
    if not args.non_local_generator and args.generator_max_model_len <= args.summary_max_tokens:
        parser.error("--generator-max-model-len must be greater than --summary-max-tokens")
    if args.generator_tensor_parallel_size is not None and args.generator_tensor_parallel_size < 1:
        parser.error("--generator-tensor-parallel-size must be at least 1")
    for option_name in ("generator_gpu_memory_utilization", "gpu_memory_utilization"):
        value = getattr(args, option_name)
        if value is not None and not (0.0 < value <= 1.0):
            parser.error(f"--{option_name.replace('_', '-')} must be in (0, 1]")
    if args.max_doc_chars < 0:
        parser.error("--max-doc-chars must be >= 0")
    if args.stream_chunk_chars < 0:
        parser.error("--stream-chunk-chars must be >= 0")
    if args.no_text and not args.skip_answer:
        parser.error("--no-text requires --skip-answer because the answerer reads the generated summary text")
    if FORCE_FULL_DOCUMENT_CORPUS:
        args.gold_and_support_only = False
    elif args.gold_and_support_only and args.privileged_dir is None:
        parser.error("--gold-and-support-only requires --privileged-dir")
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.max_model_len <= args.max_new_tokens:
        parser.error("--max-model-len must be greater than --max-new-tokens")
    if args.max_prompt_tokens < 1:
        parser.error("--max-prompt-tokens must be at least 1")
    if args.top_docs < 1:
        parser.error("--top-docs must be at least 1")
    if args.answer_max_doc_chars < 1:
        parser.error("--answer-max-doc-chars must be at least 1")
    if args.min_doc_chars < 0:
        parser.error("--min-doc-chars must be >= 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.judge_max_concurrent < 1:
        parser.error("--judge-max-concurrent must be at least 1")
    if args.judge_max_completion_tokens < 1:
        parser.error("--judge-max-completion-tokens must be at least 1")
    return args


def enforce_full_document_corpus(args: argparse.Namespace) -> None:
    if FORCE_FULL_DOCUMENT_CORPUS:
        args.gold_and_support_only = False


def effective_generator_gpu_memory_utilization(args: argparse.Namespace) -> float:
    if args.non_local_generator:
        raise ValueError("generator GPU memory utilization only applies to local generators")
    if args.generator_gpu_memory_utilization is not None:
        return args.generator_gpu_memory_utilization
    return DEFAULT_LOCAL_GENERATOR_GPU_MEMORY_UTILIZATION


def effective_answer_gpu_memory_utilization(args: argparse.Namespace) -> float:
    if args.gpu_memory_utilization is not None:
        return args.gpu_memory_utilization
    return DEFAULT_LOCAL_ANSWER_GPU_MEMORY_UTILIZATION


def release_cuda_memory() -> None:
    try:
        import torch
    except ImportError:
        return

    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        logger.debug("Could not run torch.cuda.ipc_collect()", exc_info=True)


def clean_compaction_response(text: str) -> str:
    cleaned = clean_response(text)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def split_text_for_stream(text: str, *, chunk_chars: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if chunk_chars <= 0 or len(stripped) <= chunk_chars:
        return [stripped]

    chunks: list[str] = []
    start = 0
    while start < len(stripped):
        end = min(len(stripped), start + chunk_chars)
        if end < len(stripped):
            paragraph = stripped.rfind("\n\n", start + chunk_chars // 2, end)
            newline = stripped.rfind("\n", start + chunk_chars // 2, end)
            sentence = stripped.rfind(". ", start + chunk_chars // 2, end)
            boundary = max(paragraph, newline, sentence)
            if boundary > start:
                end = boundary + (2 if boundary == paragraph else 1)
        chunk = stripped[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(stripped):
            break
        start = end
    return chunks


def iter_stream_units(documents: list[CandidateDocument], *, chunk_chars: int) -> Iterable[StreamUnit]:
    for document in documents:
        chunks = split_text_for_stream(document.text, chunk_chars=chunk_chars)
        num_fragments = len(chunks)
        for fragment_index, chunk in enumerate(chunks, start=1):
            yield StreamUnit(
                document=document,
                fragment_index=fragment_index,
                num_fragments=num_fragments,
                text=chunk,
            )


def render_compaction_prompt(
    *,
    current_summary: str,
    stream_unit: StreamUnit,
    stream_position: int,
    args: argparse.Namespace,
) -> str:
    summary_text = current_summary.strip() or "<empty>"
    return AUTOCOMPACTION_PROMPT_TEMPLATE.format(
        summary_max_tokens=args.summary_max_tokens,
        current_summary=summary_text,
        path=stream_unit.document.rel_path,
        stream_position=stream_position,
        fragment_index=stream_unit.fragment_index,
        num_fragments=stream_unit.num_fragments,
        document_text=stream_unit.text.strip(),
    ).strip()


def resolve_generator_model(args: argparse.Namespace) -> str:
    if args.generator_model_name:
        return args.generator_model_name
    if args.non_local_generator:
        return get_provider_config(args.generator_provider).default_model
    return DEFAULT_LOCAL_MODEL


def build_autocompaction_generator(args: argparse.Namespace) -> AutocompactionGenerator:
    generator_model = resolve_generator_model(args)
    if args.non_local_generator:
        return NonLocalAutocompactionGenerator(
            args.generator_provider,
            generator_model,
            max_completion_tokens=args.summary_max_tokens,
            reasoning_effort=resolve_reasoning_arg(args.generator_reasoning_effort),
            use_cache=args.generator_use_cache,
            max_concurrent=args.generator_max_concurrent,
        )

    return LocalAutocompactionGenerator(
        generator_model,
        tensor_parallel_size=args.generator_tensor_parallel_size,
        max_model_len=args.generator_max_model_len,
        gpu_memory_utilization=effective_generator_gpu_memory_utilization(args),
        dtype=args.generator_dtype,
        trust_remote_code=args.generator_trust_remote_code,
        distributed_executor_backend=args.generator_distributed_executor_backend,
        disable_thinking=args.generator_disable_thinking,
        max_completion_tokens=args.summary_max_tokens,
        temperature=args.generator_temperature,
        top_p=args.generator_top_p,
        top_k=args.generator_top_k,
    )


def stop_component(name: str, component: Any) -> None:
    try:
        component.stop()
    except Exception:
        logger.exception("Failed to stop %s cleanly", name)


def build_empty_summary_record(
    *,
    question_entry: dict[str, str],
    documents: list[CandidateDocument],
    generator: AutocompactionGenerator,
    args: argparse.Namespace,
    reason: str,
) -> dict[str, Any]:
    return {
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "num_candidate_docs": len(documents),
        "top_k": 0,
        "autocompaction": True,
        "autocompaction_fallback_reason": reason,
        "autocompaction_generation_uses_question": False,
        "autocompaction_non_local": bool(args.non_local_generator),
        "autocompaction_provider": generator.provider_name,
        "autocompaction_model": generator.model_name,
        "autocompaction_summary_max_tokens": args.summary_max_tokens,
        "autocompaction_stream_chunk_chars": args.stream_chunk_chars,
        "autocompaction_batch_size": args.compaction_batch_size,
        "gold_and_support_only": bool(args.gold_and_support_only),
        "retrieval_model": None,
        "embedding_dim": None,
        "normalized": False,
        "documents": [],
    }


def summary_document_to_record(
    *,
    question_id: str,
    summary: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    filename = "summary.txt"
    record: dict[str, Any] = {
        "rank": 1,
        "retrieval_rank": 1,
        "retrieval_score": 0.0,
        "path": f"{question_id}/_autocompaction/{filename}",
        "filename": filename,
        "truncated": False,
        "autocompaction_kind": "query_blind_summary",
        "summary_max_tokens": args.summary_max_tokens,
    }
    if not args.no_text:
        record["text"] = summary
    return record


def build_error_summary_record(
    *,
    question_entry: dict[str, str],
    generator: AutocompactionGenerator,
    args: argparse.Namespace,
    error: str,
) -> dict[str, Any]:
    return {
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "autocompaction": True,
        "autocompaction_generation_uses_question": False,
        "autocompaction_non_local": bool(args.non_local_generator),
        "autocompaction_provider": generator.provider_name,
        "autocompaction_model": generator.model_name,
        "autocompaction_gpu_memory_utilization": (
            None if args.non_local_generator else effective_generator_gpu_memory_utilization(args)
        ),
        "autocompaction_summary_max_tokens": args.summary_max_tokens,
        "autocompaction_stream_chunk_chars": args.stream_chunk_chars,
        "autocompaction_batch_size": args.compaction_batch_size,
        "gold_and_support_only": bool(args.gold_and_support_only),
        "retrieval_model": None,
        "embedding_dim": None,
        "normalized": False,
        "error": error,
        "documents": [],
    }


def finalize_compaction_state(
    state: CompactionState,
    *,
    generator: AutocompactionGenerator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = state.current_summary.strip()
    elapsed_seconds = time.monotonic() - state.start_time
    final_document = summary_document_to_record(
        question_id=state.question_entry["question_id"],
        summary=summary,
        args=args,
    )
    return {
        "question_id": state.question_entry["question_id"],
        "question": state.question_entry["question"],
        "num_candidate_docs": len(state.documents),
        "top_k": 1,
        "autocompaction": True,
        "autocompaction_generation_uses_question": False,
        "autocompaction_summary_scope": "question_document_corpus",
        "autocompaction_stream_order": "filename",
        "autocompaction_non_local": bool(args.non_local_generator),
        "autocompaction_provider": generator.provider_name,
        "autocompaction_model": generator.model_name,
        "autocompaction_gpu_memory_utilization": (
            None if args.non_local_generator else effective_generator_gpu_memory_utilization(args)
        ),
        "autocompaction_tensor_parallel_size": getattr(generator, "tensor_parallel_size", None),
        "autocompaction_summary_max_tokens": args.summary_max_tokens,
        "autocompaction_summary_char_count": len(summary),
        "autocompaction_summary_token_estimate": max(1, (len(summary) + 3) // 4) if summary else 0,
        "autocompaction_stream_chunk_chars": args.stream_chunk_chars,
        "autocompaction_batch_size": args.compaction_batch_size,
        "autocompaction_num_stream_units": len(state.stream_units),
        "autocompaction_num_steps": len(state.steps),
        "autocompaction_elapsed_seconds": round(elapsed_seconds, 3),
        "autocompaction_avg_seconds_per_stream_unit": (
            round(elapsed_seconds / len(state.stream_units), 3) if state.stream_units else None
        ),
        "autocompaction_steps": state.steps,
        "gold_and_support_only": bool(args.gold_and_support_only),
        "retrieval_model": None,
        "embedding_dim": None,
        "normalized": False,
        "documents": [final_document] if summary else [],
    }


def run_autocompaction_for_question(
    *,
    question_entry: dict[str, str],
    documents: list[CandidateDocument],
    generator: AutocompactionGenerator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    enforce_full_document_corpus(args)
    if not documents:
        return build_empty_summary_record(
            question_entry=question_entry,
            documents=documents,
            generator=generator,
            args=args,
            reason="no_candidate_documents",
        )

    current_summary = ""
    steps: list[dict[str, Any]] = []
    stream_units = list(iter_stream_units(documents, chunk_chars=args.stream_chunk_chars))
    if not stream_units:
        return build_empty_summary_record(
            question_entry=question_entry,
            documents=documents,
            generator=generator,
            args=args,
            reason="no_stream_units",
        )

    question_start_time = time.monotonic()
    stream_progress = tqdm(
        enumerate(stream_units, start=1),
        total=len(stream_units),
        desc=f"Compacting {question_entry['question_id']}",
        unit="unit",
        leave=False,
        dynamic_ncols=True,
        disable=not args.stream_progress,
    )
    for stream_position, stream_unit in stream_progress:
        prompt = render_compaction_prompt(
            current_summary=current_summary,
            stream_unit=stream_unit,
            stream_position=stream_position,
            args=args,
        )
        previous_summary = current_summary
        step_start_time = time.monotonic()
        result = generator.generate(prompt)
        generation_seconds = time.monotonic() - step_start_time
        if result.text:
            current_summary = result.text
        summary_token_estimate = max(1, (len(current_summary) + 3) // 4) if current_summary else 0
        step_record: dict[str, Any] = {
            "stream_position": stream_position,
            "path": stream_unit.document.rel_path,
            "fragment_index": stream_unit.fragment_index,
            "num_fragments": stream_unit.num_fragments,
            "document_char_count": len(stream_unit.text),
            "prompt_token_count": generator.count_prompt_tokens(prompt),
            "summary_char_count": len(current_summary),
            "summary_token_estimate": summary_token_estimate,
            "generation_seconds": round(generation_seconds, 3),
            "elapsed_seconds": round(time.monotonic() - question_start_time, 3),
            "generated_token_count": result.generated_token_count,
            "finish_reason": result.finish_reason,
        }
        if not result.text:
            step_record["empty_generation"] = True
            current_summary = previous_summary
        if args.include_intermediate_summaries:
            step_record["summary"] = current_summary
        steps.append(step_record)
        stream_progress.set_postfix(
            doc=Path(stream_unit.document.rel_path).name[:24],
            summary_tokens=summary_token_estimate,
            gen_s=f"{generation_seconds:.1f}",
        )

    summary = current_summary.strip()
    elapsed_seconds = time.monotonic() - question_start_time
    final_document = summary_document_to_record(
        question_id=question_entry["question_id"],
        summary=summary,
        args=args,
    )
    return {
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "num_candidate_docs": len(documents),
        "top_k": 1,
        "autocompaction": True,
        "autocompaction_generation_uses_question": False,
        "autocompaction_summary_scope": "question_document_corpus",
        "autocompaction_stream_order": "filename",
        "autocompaction_non_local": bool(args.non_local_generator),
        "autocompaction_provider": generator.provider_name,
        "autocompaction_model": generator.model_name,
        "autocompaction_gpu_memory_utilization": (
            None if args.non_local_generator else effective_generator_gpu_memory_utilization(args)
        ),
        "autocompaction_tensor_parallel_size": getattr(generator, "tensor_parallel_size", None),
        "autocompaction_summary_max_tokens": args.summary_max_tokens,
        "autocompaction_summary_char_count": len(summary),
        "autocompaction_summary_token_estimate": max(1, (len(summary) + 3) // 4) if summary else 0,
        "autocompaction_stream_chunk_chars": args.stream_chunk_chars,
        "autocompaction_batch_size": args.compaction_batch_size,
        "autocompaction_num_stream_units": len(stream_units),
        "autocompaction_num_steps": len(steps),
        "autocompaction_elapsed_seconds": round(elapsed_seconds, 3),
        "autocompaction_avg_seconds_per_stream_unit": (
            round(elapsed_seconds / len(stream_units), 3) if stream_units else None
        ),
        "autocompaction_steps": steps,
        "gold_and_support_only": bool(args.gold_and_support_only),
        "retrieval_model": None,
        "embedding_dim": None,
        "normalized": False,
        "documents": [final_document] if summary else [],
    }


def initialize_compaction_state(
    *,
    record_index: int,
    question_entry: dict[str, str],
    generator: AutocompactionGenerator,
    args: argparse.Namespace,
) -> CompactionState | dict[str, Any]:
    documents = load_candidate_documents(
        args.docs_dir,
        args.privileged_dir,
        question_entry["question_id"],
        args.max_doc_chars,
        gold_and_support_only=False,
    )
    if not documents:
        return build_empty_summary_record(
            question_entry=question_entry,
            documents=documents,
            generator=generator,
            args=args,
            reason="no_candidate_documents",
        )

    stream_units = list(iter_stream_units(documents, chunk_chars=args.stream_chunk_chars))
    if not stream_units:
        return build_empty_summary_record(
            question_entry=question_entry,
            documents=documents,
            generator=generator,
            args=args,
            reason="no_stream_units",
        )

    return CompactionState(
        record_index=record_index,
        question_entry=question_entry,
        documents=documents,
        stream_units=stream_units,
    )


def record_finished_question(
    *,
    record: dict[str, Any],
    question_progress: tqdm,
) -> None:
    question_id = str(record.get("question_id", ""))
    question_progress.update(1)
    question_progress.set_postfix(
        qid=question_id,
        units=record.get("autocompaction_num_stream_units", 0),
        summary_tokens=record.get("autocompaction_summary_token_estimate", 0),
        seconds=record.get("autocompaction_elapsed_seconds", 0),
    )
    logger.info(
        "Finished autocompaction for question %s: docs=%s stream_units=%s summary_tokens=%s elapsed=%.3fs",
        question_id,
        record.get("num_candidate_docs", 0),
        record.get("autocompaction_num_stream_units", 0),
        record.get("autocompaction_summary_token_estimate", 0),
        float(record.get("autocompaction_elapsed_seconds") or 0.0),
    )


def run_batched_autocompaction(
    *,
    questions: list[dict[str, str]],
    generator: AutocompactionGenerator,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any] | None] = [None] * len(questions)
    active: list[CompactionState] = []
    next_question_index = 0
    total_stream_units = 0

    with tqdm(
        total=len(questions),
        desc="Autocompaction",
        unit="question",
        dynamic_ncols=True,
    ) as question_progress, tqdm(
        total=0,
        desc="Stream units",
        unit="unit",
        leave=False,
        dynamic_ncols=True,
        disable=not args.stream_progress,
    ) as stream_progress:

        def finish_record(index: int, record: dict[str, Any]) -> None:
            records[index] = record
            record_finished_question(record=record, question_progress=question_progress)

        def fill_active_batch() -> None:
            nonlocal next_question_index, total_stream_units
            while len(active) < args.compaction_batch_size and next_question_index < len(questions):
                record_index = next_question_index
                question_entry = questions[record_index]
                next_question_index += 1
                question_id = question_entry["question_id"]
                question_progress.set_postfix(qid=question_id, active=len(active))
                try:
                    initialized = initialize_compaction_state(
                        record_index=record_index,
                        question_entry=question_entry,
                        generator=generator,
                        args=args,
                    )
                except Exception as exc:
                    logger.exception("Failed to initialize autocompaction for question %s", question_id)
                    finish_record(
                        record_index,
                        build_error_summary_record(
                            question_entry=question_entry,
                            generator=generator,
                            args=args,
                            error=str(exc),
                        ),
                    )
                    continue

                if isinstance(initialized, dict):
                    finish_record(record_index, initialized)
                    continue

                active.append(initialized)
                total_stream_units += len(initialized.stream_units)
                if not args.stream_progress:
                    continue
                stream_progress.total = total_stream_units
                stream_progress.refresh()

        fill_active_batch()
        while active:
            batch_states = active[: args.compaction_batch_size]
            prompts: list[str] = []
            batch_units: list[StreamUnit] = []
            for state in batch_states:
                stream_unit = state.stream_units[state.next_stream_index]
                batch_units.append(stream_unit)
                prompts.append(
                    render_compaction_prompt(
                        current_summary=state.current_summary,
                        stream_unit=stream_unit,
                        stream_position=state.next_stream_index + 1,
                        args=args,
                    )
                )

            batch_start_time = time.monotonic()
            try:
                results = generator.generate_batch(prompts)
            except Exception as exc:
                logger.exception("Failed batched autocompaction generation")
                completed_ids = set()
                for state in batch_states:
                    completed_ids.add(id(state))
                    finish_record(
                        state.record_index,
                        build_error_summary_record(
                            question_entry=state.question_entry,
                            generator=generator,
                            args=args,
                            error=f"Batch generation error: {exc}",
                        ),
                    )
                active[:] = [state for state in active if id(state) not in completed_ids]
                fill_active_batch()
                continue

            batch_seconds = time.monotonic() - batch_start_time
            if len(results) < len(batch_states):
                results = [
                    *results,
                    *[
                        GenerationResult(text="", raw_response="")
                        for _ in range(len(batch_states) - len(results))
                    ],
                ]

            completed_ids: set[int] = set()
            for state, stream_unit, prompt, result in zip(batch_states, batch_units, prompts, results):
                previous_summary = state.current_summary
                if result.text:
                    state.current_summary = result.text
                summary_token_estimate = (
                    max(1, (len(state.current_summary) + 3) // 4)
                    if state.current_summary
                    else 0
                )
                elapsed_seconds = time.monotonic() - state.start_time
                step_record: dict[str, Any] = {
                    "stream_position": state.next_stream_index + 1,
                    "path": stream_unit.document.rel_path,
                    "fragment_index": stream_unit.fragment_index,
                    "num_fragments": stream_unit.num_fragments,
                    "document_char_count": len(stream_unit.text),
                    "prompt_token_count": generator.count_prompt_tokens(prompt),
                    "summary_char_count": len(state.current_summary),
                    "summary_token_estimate": summary_token_estimate,
                    "generation_seconds": round(batch_seconds, 3),
                    "batch_generation_seconds": round(batch_seconds, 3),
                    "batch_size": len(batch_states),
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "generated_token_count": result.generated_token_count,
                    "finish_reason": result.finish_reason,
                }
                if not result.text:
                    step_record["empty_generation"] = True
                    state.current_summary = previous_summary
                    step_record["summary_char_count"] = len(state.current_summary)
                    step_record["summary_token_estimate"] = (
                        max(1, (len(state.current_summary) + 3) // 4)
                        if state.current_summary
                        else 0
                    )
                if args.include_intermediate_summaries:
                    step_record["summary"] = state.current_summary
                state.steps.append(step_record)
                state.next_stream_index += 1

                stream_progress.update(1)
                stream_progress.set_postfix(
                    qid=state.question_entry["question_id"],
                    doc=Path(stream_unit.document.rel_path).name[:24],
                    active=len(active),
                    batch=len(batch_states),
                    summary_tokens=step_record["summary_token_estimate"],
                    gen_s=f"{batch_seconds:.1f}",
                )

                if state.next_stream_index >= len(state.stream_units):
                    completed_ids.add(id(state))
                    finish_record(
                        state.record_index,
                        finalize_compaction_state(state, generator=generator, args=args),
                    )

            if completed_ids:
                active[:] = [state for state in active if id(state) not in completed_ids]
            fill_active_batch()

    return [record for record in records if record is not None]


def run_autocompaction_retrieval(args: argparse.Namespace) -> list[dict[str, Any]]:
    enforce_full_document_corpus(args)
    questions = select_questions(
        load_heldout_questions(args.questions_path, args.privileged_dir),
        args.query_id,
        args.limit,
    )
    logger.info("Loaded %d heldout questions", len(questions))

    generator: AutocompactionGenerator | None = None
    records: list[dict[str, Any]] = []
    try:
        if args.non_local_generator:
            logger.info("Using %s autocompaction generator", args.generator_provider)
        else:
            logger.info(
                "Using local autocompaction generator vLLM GPU memory utilization=%.3f",
                effective_generator_gpu_memory_utilization(args),
            )
        generator = build_autocompaction_generator(args)
        assert generator is not None
        logger.info(
            "Running autocompaction with batch_size=%d, stream_chunk_chars=%d, summary_max_tokens=%d",
            args.compaction_batch_size,
            args.stream_chunk_chars,
            args.summary_max_tokens,
        )
        records = run_batched_autocompaction(
            questions=questions,
            generator=generator,
            args=args,
        )
    finally:
        if generator is not None:
            stop_component("autocompaction generator", generator)
            generator = None
        release_cuda_memory()

    write_jsonl(records, args.retrieval_output_path)
    logger.info("Wrote %d autocompaction records to %s", len(records), args.retrieval_output_path)
    ensure_no_record_errors(records, stage="Autocompaction")
    return records


def build_answer_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        non_local_answerer=args.non_local_answerer,
        provider=args.provider,
        model_name_or_path=args.model_name_or_path,
        max_concurrent=args.max_concurrent,
        answer_reasoning_effort=args.answer_reasoning_effort,
        answer_use_cache=args.answer_use_cache,
        input_path=args.retrieval_output_path,
        output_path=args.output_path,
        docs_dir=args.docs_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_executor_backend=args.distributed_executor_backend,
        max_model_len=args.max_model_len,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=effective_answer_gpu_memory_utilization(args),
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        top_docs=args.top_docs,
        max_doc_chars=args.answer_max_doc_chars,
        min_doc_chars=args.min_doc_chars,
        n=args.n,
        privileged_dir=args.privileged_dir,
        judge=args.judge,
        judge_provider=args.judge_provider,
        judge_model_name=args.judge_model_name,
        judge_max_concurrent=args.judge_max_concurrent,
        judge_max_completion_tokens=args.judge_max_completion_tokens,
        judge_reasoning_effort=args.judge_reasoning_effort,
        judge_use_cache=args.judge_use_cache,
        disable_thinking=args.disable_thinking,
        limit=None,
        query_id=None,
        resume=args.resume,
        log_level=args.log_level,
    )


def run_autocompaction_rag(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retrieval_records = run_autocompaction_retrieval(args)
    answer_output_records: list[dict[str, Any]] = []
    if not args.skip_answer:
        if args.non_local_answerer:
            logger.info("Using %s answerer", args.provider)
        else:
            logger.info(
                "Using local answerer vLLM GPU memory utilization=%.3f",
                effective_answer_gpu_memory_utilization(args),
            )
        try:
            answer_output_records = answer_records(build_answer_args(args))
        finally:
            release_cuda_memory()
    return retrieval_records, answer_output_records


def main() -> None:
    load_env_file(REPO_ROOT / ".env")
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_autocompaction_rag(args)


if __name__ == "__main__":
    main()
