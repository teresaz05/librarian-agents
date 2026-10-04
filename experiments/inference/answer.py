from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from tqdm import tqdm

from experiments.inference.llm import (
    PROVIDER_DEFAULT,
    ReasoningEffort,
    create_async_client,
    generate_responses,
    get_provider_config,
)
from experiments.inference.retrieve import (
    DEFAULT_DOCS_DIR,
    DEFAULT_PRIVILEGED_DIR,
    shutdown_vllm_llm,
    write_jsonl,
)

if TYPE_CHECKING:
    from vllm import SamplingParams


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "outputs" / "inference" / "retrieval_dense.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "inference" / "answers.jsonl"
DEFAULT_NON_LOCAL_ANSWERER = True
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_ANSWER_MAX_TOKENS = 512
DEFAULT_ANSWER_TEMPERATURE = 0.7
DEFAULT_ANSWER_TOP_P = 0.8
DEFAULT_ANSWER_TOP_K = 20
DEFAULT_ANSWER_MIN_P = 0.0
DEFAULT_ANSWER_PRESENCE_PENALTY = 1.5
DEFAULT_ANSWER_REPETITION_PENALTY = 1.0
ANSWER_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_ANSWER_POLICY = "strict_grounded"

SYSTEM_PROMPT = """You are a grounded question-answering assistant.
Answer the question using only the retrieved documents below.

Instructions:
1. Treat the retrieved documents as the only source of truth.
2. Some documents may be irrelevant, incomplete, duplicated, or conflicting.
3. Ignore any document content that does not directly help answer the question.
4. Do not use outside knowledge or make assumptions.
5. If the answer is not explicitly supported by the documents, return exactly: Unknown
6. If documents conflict, use the answer best supported by the most direct evidence.
   If the conflict cannot be resolved, return exactly: Unknown
7. Return only the final short answer. Do not include explanation, citations, markdown,
   or extra text.
"""

BEST_SUPPORTED_SYSTEM_PROMPT = """You are a grounded question-answering assistant.
Answer the question using only the retrieved documents below.

Instructions:
1. The source corpus was assembled for an answerable question. Combine clues across documents
   when the answer is not stated verbatim in one place.
2. Some documents may be irrelevant, incomplete, duplicated, or conflicting. Prefer the answer
   supported by the most direct and mutually consistent evidence.
3. Make the best-supported inference available from the documents. Do not refuse merely because
   the evidence is distributed across multiple documents.
4. Do not use outside knowledge or invent unsupported facts.
5. Return exactly Unknown only when the documents contain no plausible evidence for an answer.
6. Return only the final short answer. Do not include explanation, citations, markdown, or extra text.
"""

ANSWER_POLICY_PROMPTS = {
    "strict_grounded": SYSTEM_PROMPT,
    "best_supported": BEST_SUPPORTED_SYSTEM_PROMPT,
}
ANSWER_PROMPT_RENDERING_VERSION = "retrieved-documents-v1"

USER_PROMPT_TEMPLATE = """QUESTION:
{question}

RETRIEVED DOCUMENTS:
{documents}

FINAL ANSWER:
"""

JUDGE_PROMPT_TEMPLATE = """You are given a question, sampled response, and gold answer.
Decide whether the sampled response is semantically equivalent to the gold answer.
Minor formatting, capitalization, punctuation, and unit-format differences are acceptable
when the answer means the same thing.

Return only valid JSON with this shape:
{{"correctness": 0 or 1}}

QUESTION:
{question}

SAMPLED RESPONSE:
{response}

GOLD ANSWER:
{gold_answer}
"""


@dataclass(frozen=True)
class PromptDocument:
    path: str
    rank: int | None
    retrieval_rank: int | None
    rerank_rank: int | None
    retrieval_score: float | None
    rerank_score: float | None
    text: str
    truncated_for_prompt: bool


@dataclass(frozen=True)
class AnswerPrompt:
    messages: list[dict[str, str]]
    prompt_documents: list[PromptDocument]
    prompt_token_count: int


class LLMAnswerAgent:
    def __init__(
        self,
        provider: str,
        model_name_or_path: str = "",
        max_concurrent: int = 8,
        max_completion_tokens: int = DEFAULT_ANSWER_MAX_TOKENS,
        reasoning_effort: ReasoningEffort = PROVIDER_DEFAULT,
        use_cache: bool = True,
        n: int = 1,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        self.provider = provider
        config = get_provider_config(provider)
        self.model_name_or_path = model_name_or_path or config.default_model
        self.client = create_async_client(provider)
        self.max_concurrent = max_concurrent
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.use_cache = use_cache
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.tensor_parallel_size = None

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        # A conservative approximation for API models; local vLLM uses exact tokens.
        return max(1, (len(render_messages_for_llm(messages)) + 3) // 4)

    def generate(
        self,
        prompts: list[AnswerPrompt],
        _sampling_params: SamplingParams | None = None,
    ) -> list[list[dict[str, Any]]]:
        prompt_texts = [render_messages_for_llm(prompt.messages) for prompt in prompts]
        response_groups = asyncio.run(
            generate_responses(
                self.client,
                prompt_texts,
                N=self.n,
                max_completion_tokens=self.max_completion_tokens,
                reasoning_effort=self.reasoning_effort,
                use_cache=self.use_cache,
                max_concurrent=self.max_concurrent,
                model_name=self.model_name_or_path,
                provider=self.provider,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        )

        results: list[list[dict[str, Any]]] = []
        for response_group in response_groups:
            results.append(
                [
                    {
                        "response": clean_response(raw_response),
                        "raw_response": raw_response,
                        "generated_token_count": None,
                        "finish_reason": None,
                    }
                    for raw_response in response_group
                ]
            )
        return results

    def stop(self) -> None:
        try:
            asyncio.run(self.client.close())
        except Exception:
            pass


class QwenAnswerAgent:
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
    ) -> None:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        try:
            from vllm.inputs import TokensPrompt
        except ImportError:
            from vllm.inputs.data import TokensPrompt

        if tensor_parallel_size is None:
            tensor_parallel_size = max(torch.cuda.device_count(), 1)

        self.tensor_parallel_size = tensor_parallel_size
        self.disable_thinking = disable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        llm_kwargs: dict[str, Any] = {
            "model": model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if distributed_executor_backend:
            llm_kwargs["distributed_executor_backend"] = distributed_executor_backend

        self.llm = LLM(**llm_kwargs)
        self._tokens_prompt = TokensPrompt
        self._coerce_vllm_tokenizer_max_token_id()

    def apply_chat_template(self, messages: list[dict[str, str]]) -> list[int]:
        try:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=not self.disable_thinking,
            )
        except TypeError:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        return normalize_token_ids(tokenized)

    def _coerce_vllm_tokenizer_max_token_id(self) -> None:
        engine = getattr(self.llm, "llm_engine", None)
        input_processor = getattr(engine, "input_processor", None)
        tokenizer = getattr(input_processor, "tokenizer", None)
        if tokenizer is not None:
            coerce_tokenizer_max_token_id(tokenizer)

    def generate(
        self,
        prompts: list[AnswerPrompt],
        sampling_params: SamplingParams,
    ) -> list[list[dict[str, Any]]]:
        token_prompts = [
            self._tokens_prompt(prompt_token_ids=self.apply_chat_template(prompt.messages))
            for prompt in prompts
        ]
        outputs = self.llm.generate(token_prompts, sampling_params, use_tqdm=False)

        results: list[list[dict[str, Any]]] = []
        for output in outputs:
            results.append(
                [
                    {
                        "response": clean_response(generation.text),
                        "raw_response": generation.text,
                        "generated_token_count": len(generation.token_ids),
                        "finish_reason": generation.finish_reason,
                    }
                    for generation in output.outputs
                ]
            )
        return results

    def stop(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is None:
            return

        try:
            shutdown_vllm_llm(llm)
        finally:
            del self.llm
            del llm
            gc.collect()
            try:
                import torch

                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                logger.debug("Could not release all answer-agent CUDA memory", exc_info=True)


def normalize_token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, dict):
        if "input_ids" in tokenized:
            tokenized = tokenized["input_ids"]
        elif "prompt_token_ids" in tokenized:
            tokenized = tokenized["prompt_token_ids"]
        else:
            raise TypeError(f"Tokenizer output dict is missing input_ids: {sorted(tokenized)}")

    if hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()

    if (
        isinstance(tokenized, (list, tuple))
        and len(tokenized) == 1
        and isinstance(tokenized[0], (list, tuple))
    ):
        tokenized = tokenized[0]

    if not isinstance(tokenized, (list, tuple)):
        raise TypeError(f"Expected token ids as a list, got {type(tokenized).__name__}")

    token_ids: list[int] = []
    for token_id in tokenized:
        if isinstance(token_id, bool):
            raise TypeError("Boolean token id is not valid")
        try:
            token_ids.append(int(token_id))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Token id is not int-convertible: {token_id!r}") from exc
    return token_ids


def coerce_tokenizer_max_token_id(tokenizer: Any) -> None:
    raw_max_token_id = getattr(tokenizer, "max_token_id", None)
    if isinstance(raw_max_token_id, int):
        return

    max_token_id = safe_int(raw_max_token_id)
    if max_token_id is None:
        vocab_getter = getattr(tokenizer, "get_vocab", None)
        if callable(vocab_getter):
            vocab_values = []
            for value in vocab_getter().values():
                parsed = safe_int(value)
                if parsed is not None:
                    vocab_values.append(parsed)
            if vocab_values:
                max_token_id = max(vocab_values)

    vocab_size = safe_int(getattr(tokenizer, "vocab_size", None))
    if vocab_size is not None:
        max_token_id = vocab_size if max_token_id is None else max(max_token_id, vocab_size)

    if max_token_id is None:
        return

    try:
        setattr(tokenizer, "max_token_id", max_token_id)
        return
    except (AttributeError, TypeError):
        pass

    try:
        setattr(tokenizer.__class__, "max_token_id", property(lambda _self, value=max_token_id: value))
    except (AttributeError, TypeError):
        logger.warning("Could not patch vLLM tokenizer max_token_id=%r", raw_max_token_id)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer heldout questions using retrieved or reranked documents."
    )
    parser.add_argument(
        "--non-local-answerer",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_NON_LOCAL_ANSWERER,
        dest="non_local_answerer",
        help="Use an API provider instead of local vLLM/Qwen.",
    )
    parser.add_argument("--provider", default="gemini")
    parser.add_argument(
        "--model-name-or-path",
        default="",
        help="Defaults to Qwen locally, or the selected provider's default for --non-local-answerer.",
    )
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument(
        "--answer-reasoning-effort",
        default="provider_default",
        help="For --non-local-answerer: use 'none', 'provider_default', or a provider-supported effort value.",
    )
    parser.add_argument(
        "--answer-use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For --non-local-answerer: cache answer responses through experiments.inference.llm.",
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Defaults to all visible CUDA devices. Use 4 for a 4x H200 tensor-parallel run.",
    )
    parser.add_argument("--distributed-executor-backend", default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-prompt-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_ANSWER_MAX_TOKENS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
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
    parser.add_argument("--top-docs", type=int, default=10)
    parser.add_argument("--max-doc-chars", type=int, default=12_000)
    parser.add_argument("--min-doc-chars", type=int, default=800)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Independent answer responses to generate for each prepared question prompt.",
    )
    parser.add_argument(
        "--privileged-dir",
        type=Path,
        default=DEFAULT_PRIVILEGED_DIR,
        help="Gold answer source for --judge; expects <question_id>/answer.txt.",
    )
    parser.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Gemini via experiments.inference.llm to judge each generated answer against the gold answer.",
    )
    parser.add_argument("--judge-provider", default="gemini")
    parser.add_argument(
        "--judge-model-name",
        default="",
        help="Defaults to the selected provider's configured model.",
    )
    parser.add_argument("--judge-max-concurrent", type=int, default=4)
    parser.add_argument("--judge-max-completion-tokens", type=int, default=64)
    parser.add_argument(
        "--judge-reasoning-effort",
        default="none",
        help="Use 'none', 'provider_default', or a provider-supported effort value.",
    )
    parser.add_argument(
        "--judge-use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache judge responses through experiments.inference.llm.",
    )
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking=False to Qwen chat templates when supported.",
    )
    parser.add_argument(
        "--answer-policy",
        choices=tuple(ANSWER_POLICY_PROMPTS),
        default=DEFAULT_ANSWER_POLICY,
        help=(
            "strict_grounded requires explicit documentary support; best_supported permits "
            "inference across documents."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help="Only process this question id. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to output and skip question ids already present there.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.top_docs < 1:
        parser.error("--top-docs must be at least 1")
    if args.max_doc_chars < 1:
        parser.error("--max-doc-chars must be at least 1")
    if args.min_doc_chars < 0:
        parser.error("--min-doc-chars must be >= 0")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.max_model_len <= args.max_new_tokens:
        parser.error("--max-model-len must be greater than --max-new-tokens")
    if args.max_prompt_tokens < 1:
        parser.error("--max-prompt-tokens must be at least 1")
    if args.tensor_parallel_size is not None and args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be at least 1")
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k == 0 or args.top_k < -1:
        parser.error("--top-k must be -1 or a positive integer")
    if not 0 <= args.min_p <= 1:
        parser.error("--min-p must be in [0, 1]")
    if not -2 <= args.presence_penalty <= 2:
        parser.error("--presence-penalty must be in [-2, 2]")
    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be > 0")
    if args.judge_max_concurrent < 1:
        parser.error("--judge-max-concurrent must be at least 1")
    if args.judge_max_completion_tokens < 1:
        parser.error("--judge-max-completion-tokens must be at least 1")
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records


def record_question_id(record: dict[str, Any]) -> str:
    value = record.get("question_id", record.get("query_id", ""))
    return str(value).strip()


def successful_sample_key(record: dict[str, Any]) -> tuple[str, int] | None:
    question_id = record_question_id(record)
    response = record.get("response")
    if (
        not question_id
        or record.get("error")
        or not isinstance(response, str)
        or not response.strip()
    ):
        return None
    try:
        sample_idx = int(record.get("sample_idx", 0))
    except (TypeError, ValueError):
        return None
    if sample_idx < 0:
        return None
    return question_id, sample_idx


def load_completed_samples(output_path: Path) -> dict[str, set[int]]:
    if not output_path.exists():
        return {}

    completed: dict[str, set[int]] = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = successful_sample_key(record)
            if key is None:
                continue
            question_id, sample_idx = key
            completed.setdefault(question_id, set()).add(sample_idx)
    return completed


def answer_input_sha256(input_path: Path) -> str:
    """Fingerprint only the retrieval content consumed by the answerer."""

    digest = hashlib.sha256()
    for record in load_jsonl(input_path):
        semantic_input = {
            "question_id": record_question_id(record),
            "question": str(record.get("question", "")),
            "documents": record.get("documents", []),
        }
        encoded = json.dumps(
            semantic_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def answer_experiment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input_path.resolve()
    answer_policy = resolve_answer_policy(args)
    schema_version = 1 if answer_policy == DEFAULT_ANSWER_POLICY else 2
    config = {
        "schema_version": schema_version,
        "input_path": str(input_path),
        "input_sha256": answer_input_sha256(input_path),
        "answer_model": resolve_answer_model(args),
        "non_local_answerer": args.non_local_answerer,
        "provider": args.provider,
        "answer_reasoning_effort": args.answer_reasoning_effort,
        "answer_use_cache": args.answer_use_cache,
        "tensor_parallel_size": args.tensor_parallel_size,
        "distributed_executor_backend": args.distributed_executor_backend,
        "max_model_len": args.max_model_len,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "disable_thinking": args.disable_thinking,
        "samples_per_query": args.n,
        "top_docs": args.top_docs,
        "max_doc_chars": args.max_doc_chars,
        "min_doc_chars": args.min_doc_chars,
        "docs_dir": str(args.docs_dir.resolve()),
        "privileged_dir": str(args.privileged_dir.resolve()),
        "limit": args.limit,
        "query_ids": args.query_id,
        "judge": args.judge,
        "judge_provider": args.judge_provider,
        "judge_model_name": args.judge_model_name,
        "judge_max_completion_tokens": args.judge_max_completion_tokens,
        "judge_reasoning_effort": args.judge_reasoning_effort,
    }
    if answer_policy != DEFAULT_ANSWER_POLICY:
        config["answer_policy"] = answer_policy
        config["answer_prompt_sha256"] = answer_prompt_sha256(answer_policy)
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": schema_version,
        "config_id": hashlib.sha256(encoded).hexdigest(),
        "config": config,
    }


def prepare_answer_manifest(args: argparse.Namespace) -> dict[str, Any]:
    expected = answer_experiment_manifest(args)
    manifest_path = args.output_path.with_suffix(args.output_path.suffix + ".config.json")
    if args.resume and (args.output_path.exists() or manifest_path.exists()):
        if not manifest_path.exists():
            raise ValueError(
                f"Resume output exists without its configuration manifest: {manifest_path}"
            )
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual.get("config_id") != expected["config_id"]:
            raise ValueError(
                "Resume configuration does not match the existing answer run. "
                f"Use a new --output-path or restore the settings in {manifest_path}."
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return expected


def prepare_resume_output(
    output_path: Path, *, expected_config_id: str | None = None
) -> dict[str, set[int]]:
    if not output_path.exists():
        return {}

    successful_records: dict[tuple[str, int], dict[str, Any]] = {}
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = successful_sample_key(record)
            if key is not None:
                if expected_config_id is not None and record.get("config_id") != expected_config_id:
                    raise ValueError(f"Sample configuration does not match run manifest in {output_path}")
                if key in successful_records:
                    raise ValueError(f"Duplicate completed sample in {output_path}: {key}")
                successful_records[key] = record

    temp_path = output_path.with_suffix(output_path.suffix + ".resume.tmp")
    write_jsonl(list(successful_records.values()), temp_path)
    temp_path.replace(output_path)

    completed: dict[str, set[int]] = {}
    for question_id, sample_idx in successful_records:
        completed.setdefault(question_id, set()).add(sample_idx)
    return completed


def load_completed_question_ids(output_path: Path) -> set[str]:
    """Return question IDs with at least one completed response."""

    return set(load_completed_samples(output_path))


def resolve_reasoning_arg(value: str) -> ReasoningEffort:
    if value == "none":
        return None
    if value == "provider_default":
        return PROVIDER_DEFAULT
    return value


def read_gold_answer(record: dict[str, Any], privileged_dir: Path) -> str | None:
    for key in ("gold_answer", "answer"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    question_id = str(record.get("question_id", "")).strip()
    if not question_id:
        return None

    root = privileged_dir.resolve()
    answer_path = (root / question_id / "answer.txt").resolve()
    try:
        answer_path.relative_to(root)
    except ValueError:
        raise ValueError(f"Gold answer path escapes privileged dir for question id: {question_id}")

    if not answer_path.exists():
        return None
    return answer_path.read_text(encoding="utf-8", errors="replace").strip()


def select_records(
    records: list[dict[str, Any]],
    query_ids: list[str] | None,
    limit: int | None,
    completed_samples: dict[str, set[int]],
    samples_per_question: int = 1,
) -> list[dict[str, Any]]:
    if query_ids:
        selected = {str(query_id) for query_id in query_ids}
        records = [record for record in records if str(record.get("question_id")) in selected]
    if completed_samples:
        expected = set(range(samples_per_question))
        records = [
            record
            for record in records
            if not expected.issubset(completed_samples.get(record_question_id(record), set()))
        ]
    if limit is not None:
        records = records[:limit]
    return records


def batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def doc_sort_key(doc: dict[str, Any]) -> tuple[int, int]:
    rank = doc.get("rank", doc.get("rerank_rank", doc.get("retrieval_rank", 10**9)))
    retrieval_rank = doc.get("retrieval_rank", 10**9)
    return safe_int(rank, 10**9), safe_int(retrieval_rank, 10**9)


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_doc_text(doc: dict[str, Any], docs_dir: Path) -> str:
    text = doc.get("text")
    if isinstance(text, str):
        return text

    rel_path = doc.get("path")
    if not rel_path:
        return ""
    path = (docs_dir / str(rel_path)).resolve()
    try:
        path.relative_to(docs_dir.resolve())
    except ValueError:
        raise ValueError(f"Retrieved document path escapes docs dir: {rel_path}")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Retrieved document text missing and file not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def render_document(doc: PromptDocument) -> str:
    metadata = [f"path: {doc.path}"]
    if doc.rerank_rank is not None:
        metadata.append(f"rerank_rank: {doc.rerank_rank}")
    if doc.rerank_score is not None:
        metadata.append(f"rerank_score: {doc.rerank_score:.6f}")
    if doc.retrieval_rank is not None:
        metadata.append(f"retrieval_rank: {doc.retrieval_rank}")
    if doc.retrieval_score is not None:
        metadata.append(f"retrieval_score: {doc.retrieval_score:.6f}")
    if doc.truncated_for_prompt:
        metadata.append("truncated_for_prompt: true")
    return "\n".join(metadata) + "\ncontent:\n" + doc.text.strip()


def render_documents(docs: list[PromptDocument]) -> str:
    return "\n\n".join(
        f"[Document {index}]\n{render_document(doc)}"
        for index, doc in enumerate(docs, start=1)
    )


def resolve_answer_policy(args: argparse.Namespace) -> str:
    policy = str(getattr(args, "answer_policy", DEFAULT_ANSWER_POLICY))
    if policy not in ANSWER_POLICY_PROMPTS:
        available = ", ".join(ANSWER_POLICY_PROMPTS)
        raise ValueError(f"Unknown answer policy {policy!r}; expected one of {available}")
    return policy


def answer_prompt_sha256(policy: str) -> str:
    try:
        prompt = ANSWER_POLICY_PROMPTS[policy]
    except KeyError:
        available = ", ".join(ANSWER_POLICY_PROMPTS)
        raise ValueError(f"Unknown answer policy {policy!r}; expected one of {available}") from None
    payload = "\0".join((prompt, USER_PROMPT_TEMPLATE, ANSWER_PROMPT_RENDERING_VERSION))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_messages(
    question: str,
    docs: list[PromptDocument],
    *,
    answer_policy: str = DEFAULT_ANSWER_POLICY,
) -> list[dict[str, str]]:
    documents_text = render_documents(docs) if docs else "<no retrieved documents>"
    return [
        {"role": "system", "content": ANSWER_POLICY_PROMPTS[answer_policy]},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                question=question.strip(),
                documents=documents_text,
            ),
        },
    ]


def render_messages_for_llm(messages: list[dict[str, str]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = str(message.get("content", "")).strip()
        rendered.append(f"{role}:\n{content}")
    return "\n\n".join(rendered).strip()


def count_prompt_tokens(agent: Any, messages: list[dict[str, str]]) -> int:
    counter = getattr(agent, "count_prompt_tokens", None)
    if callable(counter):
        return int(counter(messages))
    return len(agent.apply_chat_template(messages))


def prompt_budget(args: argparse.Namespace) -> int:
    model_budget = args.max_model_len - args.max_new_tokens - 16
    return max(1, min(args.max_prompt_tokens, model_budget))


def make_prompt_document(raw_doc: dict[str, Any], text: str, truncated_for_prompt: bool) -> PromptDocument:
    return PromptDocument(
        path=str(raw_doc.get("path", raw_doc.get("filename", ""))),
        rank=safe_int(raw_doc.get("rank")),
        retrieval_rank=safe_int(raw_doc.get("retrieval_rank")),
        rerank_rank=safe_int(raw_doc.get("rerank_rank")),
        retrieval_score=safe_float(raw_doc.get("retrieval_score")),
        rerank_score=safe_float(raw_doc.get("rerank_score")),
        text=text,
        truncated_for_prompt=truncated_for_prompt,
    )


def fit_prompt(
    agent: Any,
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> AnswerPrompt:
    question = str(record.get("question", ""))
    budget = prompt_budget(args)
    answer_policy = resolve_answer_policy(args)
    raw_documents = sorted(record.get("documents", []), key=doc_sort_key)[: args.top_docs]

    prompt_documents: list[PromptDocument] = []
    for raw_doc in raw_documents:
        full_text = load_doc_text(raw_doc, args.docs_dir)
        clipped_text = full_text[: args.max_doc_chars]
        full_prompt_doc = make_prompt_document(
            raw_doc,
            clipped_text,
            truncated_for_prompt=len(full_text) > len(clipped_text),
        )

        candidate_docs = prompt_documents + [full_prompt_doc]
        candidate_messages = build_messages(
            question,
            candidate_docs,
            answer_policy=answer_policy,
        )
        if count_prompt_tokens(agent, candidate_messages) <= budget:
            prompt_documents.append(full_prompt_doc)
            continue

        low = 0
        high = len(clipped_text)
        best_text = ""
        while low <= high:
            mid = (low + high) // 2
            test_text = clipped_text[:mid]
            test_doc = make_prompt_document(raw_doc, test_text, truncated_for_prompt=True)
            test_messages = build_messages(
                question,
                prompt_documents + [test_doc],
                answer_policy=answer_policy,
            )
            if count_prompt_tokens(agent, test_messages) <= budget:
                best_text = test_text
                low = mid + 1
            else:
                high = mid - 1

        if len(best_text.strip()) >= args.min_doc_chars:
            prompt_documents.append(make_prompt_document(raw_doc, best_text, truncated_for_prompt=True))
        break

    messages = build_messages(question, prompt_documents, answer_policy=answer_policy)
    return AnswerPrompt(
        messages=messages,
        prompt_documents=prompt_documents,
        prompt_token_count=count_prompt_tokens(agent, messages),
    )


def clean_response(text: str) -> str:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    return cleaned


def parse_correctness(content: str) -> float:
    text = content.strip()
    parsed: Any | None = None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))

    if isinstance(parsed, dict) and "correctness" in parsed:
        value = parsed["correctness"]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return 1.0 if value >= 0.5 else 0.0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "correct", "yes"}:
                return 1.0
            if normalized in {"0", "false", "incorrect", "no"}:
                return 0.0

    match = re.search(r'"?correctness"?\s*[:=]\s*"?([01])"?', text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))

    normalized_text = text.lower()
    if normalized_text in {"1", "true", "correct", "yes"}:
        return 1.0
    if normalized_text in {"0", "false", "incorrect", "no"}:
        return 0.0

    raise ValueError(f"Could not parse correctness from judge response: {content!r}")


def prompt_doc_to_record(doc: PromptDocument) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": doc.path,
        "truncated_for_prompt": doc.truncated_for_prompt,
    }
    for key in ("rank", "retrieval_rank", "rerank_rank", "retrieval_score", "rerank_score"):
        value = getattr(doc, key)
        if value is not None:
            record[key] = value
    return record


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "n": 1 if args.temperature == 0 else args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_new_tokens,
    }
    if args.top_k > 0:
        kwargs["top_k"] = args.top_k
    return SamplingParams(**kwargs)


def generate_answer_groups(
    agent: Any,
    prompts: list[AnswerPrompt],
    sampling_params: SamplingParams | None,
    *,
    greedy_repeats: int,
) -> list[list[dict[str, Any]]]:
    if greedy_repeats == 1:
        return agent.generate(prompts, sampling_params)

    expanded_prompts = [prompt for prompt in prompts for _ in range(greedy_repeats)]
    expanded_groups = agent.generate(expanded_prompts, sampling_params)
    grouped: list[list[dict[str, Any]]] = []
    for prompt_index in range(len(prompts)):
        group: list[dict[str, Any]] = []
        start = prompt_index * greedy_repeats
        for repeat_index in range(greedy_repeats):
            expanded_index = start + repeat_index
            response_group = (
                expanded_groups[expanded_index]
                if expanded_index < len(expanded_groups)
                else []
            )
            if response_group:
                group.append(response_group[0])
            else:
                group.append(
                    {
                        "response": "",
                        "error": f"Greedy answer generation returned no response for repeat {repeat_index}",
                    }
                )
        grouped.append(group)
    return grouped


def resolve_answer_model(args: argparse.Namespace) -> str:
    if args.model_name_or_path:
        return args.model_name_or_path
    if args.non_local_answerer:
        return get_provider_config(args.provider).default_model
    return DEFAULT_LOCAL_MODEL


async def judge_answer_records(
    output_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    config = get_provider_config(args.judge_provider)
    judge_model = args.judge_model_name or config.default_model
    reasoning_effort = resolve_reasoning_arg(args.judge_reasoning_effort)

    prompts: list[str] = []
    prompt_indices: list[int] = []
    judged_records = [dict(record) for record in output_records]

    for index, record in enumerate(judged_records):
        record["judge_provider"] = config.name
        record["judge_model"] = judge_model
        record["correctness"] = None
        record["judge_response"] = ""
        record["judge_error"] = None

        try:
            gold_answer = read_gold_answer(record, args.privileged_dir)
        except Exception as exc:
            record["gold_answer"] = None
            record["judge_error"] = str(exc)
            continue

        record["gold_answer"] = gold_answer
        if not gold_answer:
            record["judge_error"] = "Missing gold answer"
            continue

        if record.get("error"):
            record["judge_error"] = f"Answer generation error: {record['error']}"
            continue

        response = str(record.get("response", "")).strip()
        if not response:
            record["correctness"] = 0.0
            record["judge_error"] = "Missing sampled response"
            continue

        prompts.append(
            JUDGE_PROMPT_TEMPLATE.format(
                question=str(record.get("question", "")).strip(),
                response=response,
                gold_answer=gold_answer,
            )
        )
        prompt_indices.append(index)

    if not prompts:
        return judged_records

    try:
        client = create_async_client(config)
    except Exception as exc:
        for index in prompt_indices:
            judged_records[index]["judge_error"] = f"Judge client error: {exc}"
        return judged_records

    try:
        try:
            responses = await generate_responses(
                client,
                prompts,
                N=1,
                max_completion_tokens=args.judge_max_completion_tokens,
                reasoning_effort=reasoning_effort,
                use_cache=args.judge_use_cache,
                max_concurrent=args.judge_max_concurrent,
                model_name=judge_model,
                provider=config,
                temperature=0.0,
            )
        except Exception as exc:
            for index in prompt_indices:
                judged_records[index]["judge_error"] = f"Judge request error: {exc}"
            return judged_records
    finally:
        await client.close()

    for index, response_group in zip(prompt_indices, responses):
        content = response_group[0] if response_group else ""
        judged_records[index]["judge_response"] = content
        try:
            judged_records[index]["correctness"] = parse_correctness(content)
        except Exception as exc:
            judged_records[index]["judge_error"] = str(exc)

    return judged_records


def answer_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = prepare_answer_manifest(args)
    completed_samples = (
        prepare_resume_output(
            args.output_path,
            expected_config_id=manifest["config_id"],
        )
        if args.resume
        else {}
    )
    records = select_records(
        load_jsonl(args.input_path),
        args.query_id,
        args.limit,
        completed_samples,
        args.n,
    )
    logger.info("Loaded %d records to answer", len(records))
    if not records:
        if not args.resume:
            write_jsonl([], args.output_path)
        return []
    answer_model = resolve_answer_model(args)

    if args.non_local_answerer:
        agent = LLMAnswerAgent(
            args.provider,
            answer_model,
            args.max_concurrent,
            args.max_new_tokens,
            resolve_reasoning_arg(args.answer_reasoning_effort),
            args.answer_use_cache,
            args.n,
        )
        agent.temperature = args.temperature
        agent.top_p = args.top_p
    else:
        agent = QwenAnswerAgent(
            answer_model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            distributed_executor_backend=args.distributed_executor_backend,
            disable_thinking=args.disable_thinking,
        )
    sampling_params = None if args.non_local_answerer else build_sampling_params(args)
    greedy_repeats = args.n if not args.non_local_answerer and args.temperature == 0 else 1

    output_records: list[dict[str, Any]] = []
    try:
        with tqdm(total=len(records), desc="Answering", unit="question") as progress:
            for batch in batches(records, args.batch_size):
                prompt_records: list[
                    tuple[dict[str, Any], AnswerPrompt | None, str | None, list[int]]
                ] = []
                prompts: list[AnswerPrompt] = []

                for record in batch:
                    question_id = record_question_id(record)
                    completed_for_question = completed_samples.get(question_id, set())
                    missing_sample_indices = [
                        sample_idx
                        for sample_idx in range(args.n)
                        if sample_idx not in completed_for_question
                    ]
                    if not missing_sample_indices:
                        continue
                    try:
                        prompt = fit_prompt(agent, record, args=args)
                        prompt_records.append(
                            (record, prompt, None, missing_sample_indices)
                        )
                        prompts.append(prompt)
                    except Exception as exc:
                        prompt_records.append(
                            (record, None, str(exc), missing_sample_indices)
                        )

                generation_groups = (
                    generate_answer_groups(
                        agent,
                        prompts,
                        sampling_params,
                        greedy_repeats=greedy_repeats,
                    )
                    if prompts
                    else []
                )
                generation_iter = iter(generation_groups)

                for record, prompt, error, missing_sample_indices in prompt_records:
                    base_output = {
                        "config_id": manifest["config_id"],
                        "question_id": record_question_id(record),
                        "question": str(record.get("question", "")),
                        "input_path": str(args.input_path),
                        "answer_model": answer_model,
                        "answer_provider": getattr(agent, "provider", "local"),
                        "tensor_parallel_size": getattr(agent, "tensor_parallel_size", None),
                        "answer_temperature": args.temperature,
                        "answer_top_p": args.top_p,
                        "answer_top_k": args.top_k,
                        "answer_min_p": args.min_p,
                        "answer_presence_penalty": args.presence_penalty,
                        "answer_repetition_penalty": args.repetition_penalty,
                        "answer_thinking_enabled": not args.disable_thinking,
                    }
                    answer_policy = resolve_answer_policy(args)
                    if answer_policy != DEFAULT_ANSWER_POLICY:
                        base_output["answer_policy"] = answer_policy
                        base_output["answer_prompt_sha256"] = answer_prompt_sha256(
                            answer_policy
                        )
                    gold_answer = read_gold_answer(record, args.privileged_dir)
                    if gold_answer:
                        base_output["gold_answer"] = gold_answer
                    if error is not None or prompt is None:
                        output_records.extend(
                            {
                                **base_output,
                                "sample_idx": sample_idx,
                                "response": "",
                                "error": error,
                            }
                            for sample_idx in missing_sample_indices
                        )
                        continue

                    generation_group = next(generation_iter, [])
                    for sample_idx in missing_sample_indices:
                        if sample_idx < len(generation_group):
                            generation = generation_group[sample_idx]
                        else:
                            generation = {
                                "response": "",
                                "error": (
                                    "Answerer returned "
                                    f"{len(generation_group)} of {args.n} requested responses"
                                ),
                            }
                        output_records.append(
                            {
                                **base_output,
                                **generation,
                                "sample_idx": sample_idx,
                                "num_retrieved_documents": len(record.get("documents", [])),
                                "num_prompt_documents": len(prompt.prompt_documents),
                                "prompt_token_count": prompt.prompt_token_count,
                                "prompt_documents": [
                                    prompt_doc_to_record(doc)
                                    for doc in prompt.prompt_documents
                                ],
                            }
                        )
                progress.update(len(batch))
    finally:
        agent.stop()

    if args.judge:
        output_records = asyncio.run(judge_answer_records(output_records, args))

    if args.resume and args.output_path.exists():
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("a", encoding="utf-8") as f:
            for record in output_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        write_jsonl(output_records, args.output_path)

    logger.info("Wrote %d answer records to %s", len(output_records), args.output_path)
    failed_count = sum(successful_sample_key(record) is None for record in output_records)
    if failed_count:
        raise RuntimeError(
            f"Answer generation failed for {failed_count} samples; rerun with --resume to retry them."
        )
    return output_records


def main() -> None:
    load_env_file(REPO_ROOT / ".env")
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    answer_records(args)


if __name__ == "__main__":
    main()
