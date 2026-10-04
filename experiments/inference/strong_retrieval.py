from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tqdm import tqdm

from experiments.inference.passage_retrieval import (
    DEFAULT_CANDIDATE_DEPTH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_PER_SOURCE,
    DEFAULT_TOP_K,
    DensePassageIndex,
    PassageHit,
    PassageIndex,
    compute_bm25_scores,
    load_or_build_passage_index,
    passage_hit_to_record,
    rank_passages,
    reciprocal_rank_fusion,
    source_fingerprint,
    source_diverse_top_k,
)
from experiments.inference.retrieve import (
    DEFAULT_DOCS_DIR,
    DEFAULT_PRIVILEGED_DIR,
    DEFAULT_QUESTIONS_PATH,
    CandidateDocument,
    Qwen3EmbeddingVllm,
    build_embedding_llm_kwargs,
    ensure_no_record_errors,
    load_candidate_documents,
    load_heldout_questions,
    select_questions,
)


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "outputs" / "cache" / "passage_index"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "inference" / "browsecomp_plus_strong_v1"
MANIFEST_SCHEMA_VERSION = 1
STRATEGIES = (
    "bm25_chunked",
    "embed_chunked",
    "hybrid_rrf_chunked",
    "lateinter_chunked",
    "rerank_fixed_chunked",
)


@dataclass(frozen=True)
class QuestionWork:
    question_entry: dict[str, str]
    num_documents: int
    index: PassageIndex


@dataclass(frozen=True)
class DenseCandidateWork:
    work: QuestionWork
    candidates: tuple[PassageHit, ...]


class PinnedQwen3EmbeddingVllm(Qwen3EmbeddingVllm):
    """Qwen embedding runner with an enforceable Hugging Face revision."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None,
        instruction: str | None,
        tensor_parallel_size: int | None,
        max_model_len: int | None,
        gpu_memory_utilization: float | None,
    ) -> None:
        from vllm import LLM

        self.instruction = instruction or (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        kwargs = build_embedding_llm_kwargs(
            model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if revision is not None:
            kwargs["revision"] = revision
        self.model = LLM(**kwargs)


def _supports_logits_to_keep(model: Any) -> bool:
    try:
        return "logits_to_keep" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


class Qwen3RerankerForwardLogits:
    """Qwen3 reranker using the model-card forward-logit scoring path."""

    SYSTEM_PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
        'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        instruction: str = (
            "Given a web search query, retrieve relevant passages that answer the query"
        ),
        max_length: int = 2048,
        device: str = "cuda",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        tokenizer_kwargs: dict[str, Any] = {
            "padding_side": "left",
            "trust_remote_code": trust_remote_code,
        }
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision is not None:
            tokenizer_kwargs["revision"] = revision
            model_kwargs["revision"] = revision
        if dtype != "auto":
            try:
                model_kwargs["torch_dtype"] = getattr(torch, dtype)
            except AttributeError as exc:
                raise ValueError(f"Unsupported torch dtype: {dtype}") from exc
        else:
            model_kwargs["torch_dtype"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.to(device)
        self.model.eval()
        self.model_forward_kwargs = (
            {"logits_to_keep": 1} if _supports_logits_to_keep(self.model) else {}
        )
        self.device = device
        self.instruction = instruction
        self.max_length = max_length
        self.prefix_tokens = self.tokenizer.encode(self.SYSTEM_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.ASSISTANT_SUFFIX, add_special_tokens=False)
        self.true_token_id = self._single_token_id("yes")
        self.false_token_id = self._single_token_id("no")
        if len(self.prefix_tokens) + len(self.suffix_tokens) >= max_length:
            raise ValueError("max_length is too small for the reranker prompt wrapper")

    def _single_token_id(self, value: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(value)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Reranker label {value!r} is not a vocabulary token")
        return int(token_id)

    def _format_pair(self, query: str, document: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"

    def _prepare_inputs(self, pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
        available_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        encoded = self.tokenizer(
            [self._format_pair(query, document) for query, document in pairs],
            padding=False,
            truncation="longest_first",
            max_length=available_length,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        encoded["input_ids"] = [
            self.prefix_tokens + list(token_ids) + self.suffix_tokens
            for token_ids in encoded["input_ids"]
        ]
        padded = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in padded.items()}

    def compute_scores(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        import torch

        if not pairs:
            return []
        inputs = self._prepare_inputs(pairs)
        with torch.inference_mode():
            final_logits = self.model(**inputs, **self.model_forward_kwargs).logits[:, -1, :]
            probabilities = yes_probabilities_from_logits(
                final_logits,
                false_token_id=self.false_token_id,
                true_token_id=self.true_token_id,
            )
        return [float(score) for score in probabilities.detach().cpu().tolist()]

    def stop(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def yes_probabilities_from_logits(
    final_logits: Any,
    *,
    false_token_id: int,
    true_token_id: int,
) -> Any:
    """Return P(yes | {no, yes}) directly from a model's final forward logits."""
    import torch

    binary_logits = torch.stack(
        [final_logits[:, false_token_id], final_logits[:, true_token_id]],
        dim=1,
    )
    return torch.softmax(binary_logits, dim=1)[:, 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run passage-level retrieval baselines.")
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument(
        "--privileged-dir",
        type=Path,
        default=DEFAULT_PRIVILEGED_DIR,
        help="Used only to hydrate question text; manifest labels are never loaded.",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--chunk-tokenizer-name-or-path", default=None)
    parser.add_argument("--chunk-tokenizer-revision", default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--candidate-depth", type=int, default=DEFAULT_CANDIDATE_DEPTH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE)

    parser.add_argument("--embedding-model-name-or-path", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--embedding-model-revision", default=None)
    parser.add_argument("--embedding-instruction", default=None)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--embedding-max-model-len", type=int, default=None)
    parser.add_argument("--embedding-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--rrf-k", type=float, default=60.0)

    parser.add_argument("--lateinter-model-name-or-path", default="lightonai/Reason-ModernColBERT")
    parser.add_argument("--lateinter-model-revision", default=None)
    parser.add_argument("--lateinter-device", default=None)
    parser.add_argument("--lateinter-batch-size", type=int, default=32)
    parser.add_argument("--lateinter-query-prefix", default=None)

    parser.add_argument("--reranker-model-name-or-path", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--reranker-model-revision", default=None)
    parser.add_argument(
        "--reranker-instruction",
        default="Given a web search query, retrieve relevant passages that answer the query",
    )
    parser.add_argument("--reranker-max-length", type=int, default=2048)
    parser.add_argument("--reranker-batch-size", type=int, default=64)
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument(
        "--reranker-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-id", action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be non-negative and smaller than --chunk-size")
    if args.candidate_depth < 1 or args.top_k < 1 or args.max_per_source < 1:
        parser.error("--candidate-depth, --top-k, and --max-per-source must be at least 1")
    if args.embedding_batch_size < 1 or args.lateinter_batch_size < 1:
        parser.error("retrieval batch sizes must be at least 1")
    if args.reranker_batch_size < 1 or args.reranker_max_length < 1:
        parser.error("reranker batch size and max length must be at least 1")
    if args.bm25_k1 <= 0 or not 0 <= args.bm25_b <= 1 or args.rrf_k <= 0:
        parser.error("invalid BM25 or RRF parameters")
    if args.output_path is None:
        args.output_path = DEFAULT_OUTPUT_DIR / f"{args.strategy}.jsonl"
    if args.chunk_tokenizer_name_or_path is None:
        args.chunk_tokenizer_name_or_path = args.embedding_model_name_or_path
    if (
        args.chunk_tokenizer_revision is None
        and args.chunk_tokenizer_name_or_path == args.embedding_model_name_or_path
    ):
        args.chunk_tokenizer_revision = args.embedding_model_revision
    return args


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_fingerprints(
    docs_dir: Path,
    questions: Sequence[dict[str, str]],
) -> dict[str, str]:
    """Fingerprint only the candidate source text, never privileged metadata."""

    fingerprints: dict[str, str] = {}
    for question in questions:
        question_id = question["question_id"]
        documents = load_candidate_documents(
            docs_dir,
            None,
            question_id,
            max_doc_chars=0,
            gold_and_support_only=False,
        )
        fingerprints[question_id] = source_fingerprint(documents)
    return fingerprints


def retrieval_experiment_manifest(
    args: argparse.Namespace,
    questions: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """Bind every output-affecting static retrieval input to one run id."""

    config: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "variant": args.strategy,
        "questions_path": str(args.questions_path.resolve()),
        "selected_questions_sha256": _sha256_json(list(questions)),
        "selected_question_ids": [question["question_id"] for question in questions],
        "docs_dir": str(args.docs_dir.resolve()),
        "source_fingerprints": _source_fingerprints(args.docs_dir, questions),
        "privileged_dir": (
            None if args.privileged_dir is None else str(args.privileged_dir.resolve())
        ),
        "passage_cache_dir": str(args.cache_dir.resolve()),
        "passage": {
            "tokenizer": args.chunk_tokenizer_name_or_path,
            "tokenizer_revision": args.chunk_tokenizer_revision,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "trust_remote_code": args.trust_remote_code,
        },
        "selection": {
            "candidate_depth": args.candidate_depth,
            "top_k": args.top_k,
            "max_per_source": args.max_per_source,
            "include_text": not args.no_text,
            "limit": args.limit,
            "query_ids": args.query_id,
        },
    }

    if args.strategy in {"embed_chunked", "hybrid_rrf_chunked", "rerank_fixed_chunked"}:
        config["embedding"] = {
            "model": args.embedding_model_name_or_path,
            "model_revision": args.embedding_model_revision,
            "instruction": args.embedding_instruction,
            "dim": args.embedding_dim,
            "batch_size": args.embedding_batch_size,
            "tensor_parallel_size": args.embedding_tensor_parallel_size,
            "max_model_len": args.embedding_max_model_len,
            "gpu_memory_utilization": args.embedding_gpu_memory_utilization,
            "normalize": args.normalize,
        }
    if args.strategy in {"bm25_chunked", "hybrid_rrf_chunked"}:
        config["bm25"] = {"k1": args.bm25_k1, "b": args.bm25_b}
    if args.strategy == "hybrid_rrf_chunked":
        config["rrf"] = {"k": args.rrf_k, "weights": [1.0, 1.0]}
    if args.strategy == "lateinter_chunked":
        config["late_interaction"] = {
            "model": args.lateinter_model_name_or_path,
            "model_revision": args.lateinter_model_revision,
            "device": args.lateinter_device,
            "batch_size": args.lateinter_batch_size,
            "query_prefix": args.lateinter_query_prefix,
            "normalize_embeddings": True,
            "scoring_method": "colbert_maxsim",
        }
    if args.strategy == "rerank_fixed_chunked":
        config["reranker"] = {
            "model": args.reranker_model_name_or_path,
            "model_revision": args.reranker_model_revision,
            "instruction": args.reranker_instruction,
            "prompt_wrapper_sha256": _sha256_json(
                {
                    "system_prefix": Qwen3RerankerForwardLogits.SYSTEM_PREFIX,
                    "assistant_suffix": Qwen3RerankerForwardLogits.ASSISTANT_SUFFIX,
                }
            ),
            "max_length": args.reranker_max_length,
            "batch_size": args.reranker_batch_size,
            "device": args.reranker_device,
            "dtype": args.reranker_dtype,
            "trust_remote_code": args.trust_remote_code,
            "scoring_method": "forward_logits_yes_no_softmax",
        }

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config_id": _sha256_json(config),
        "config": config,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def prepare_retrieval_manifest(
    args: argparse.Namespace,
    questions: Sequence[dict[str, str]],
) -> dict[str, Any]:
    expected = retrieval_experiment_manifest(args, questions)
    manifest_path = args.output_path.with_suffix(args.output_path.suffix + ".config.json")
    if args.resume and (args.output_path.exists() or manifest_path.exists()):
        if not manifest_path.exists():
            raise ValueError(
                f"Resume output exists without its configuration manifest: {manifest_path}"
            )
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid retrieval configuration manifest: {manifest_path}") from exc
        if actual.get("config_id") != expected["config_id"]:
            raise ValueError(
                "Resume configuration does not match the existing retrieval run. "
                f"Use a new --output-path or restore the settings in {manifest_path}."
            )
    _atomic_write_text(
        manifest_path,
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
    )
    return expected


def load_completed_retrieval_records(
    output_path: Path,
    *,
    config_id: str,
    question_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load only successful records; error records intentionally remain pending."""

    if not output_path.exists():
        return {}
    expected_ids = None if question_ids is None else set(question_ids)
    completed: dict[str, dict[str, Any]] = {}
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {output_path}:{line_number}; refuse unsafe resume"
                ) from exc
            if record.get("config_id") != config_id:
                raise ValueError(f"Retrieval record configuration mismatch in {output_path}")
            question_id = str(record.get("question_id", "")).strip()
            if not question_id:
                raise ValueError(f"Retrieval record has no question_id in {output_path}")
            if expected_ids is not None and question_id not in expected_ids:
                raise ValueError(
                    f"Unexpected retrieval question_id {question_id!r} in {output_path}"
                )
            if "error" in record:
                continue
            if not isinstance(record.get("documents"), list):
                raise ValueError(
                    f"Successful retrieval record for {question_id} has no documents list"
                )
            if question_id in completed:
                raise ValueError(f"Duplicate retrieval record for {question_id} in {output_path}")
            completed[question_id] = record
    return completed


def write_retrieval_checkpoint(
    output_path: Path,
    questions: Sequence[dict[str, str]],
    records_by_question: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Atomically publish a checkpoint in the selected question order."""

    ordered = [
        records_by_question[question["question_id"]]
        for question in questions
        if question["question_id"] in records_by_question
    ]
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered)
    _atomic_write_text(output_path, payload)
    return ordered


def _load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.chunk_tokenizer_revision is not None:
        kwargs["revision"] = args.chunk_tokenizer_revision
    return AutoTokenizer.from_pretrained(args.chunk_tokenizer_name_or_path, **kwargs)


def _load_question_work(
    question_entry: dict[str, str],
    tokenizer: Any,
    args: argparse.Namespace,
) -> QuestionWork:
    # Do not pass privileged_dir: manifest labels must never enter passage metadata.
    documents = load_candidate_documents(
        args.docs_dir,
        None,
        question_entry["question_id"],
        max_doc_chars=0,
    )
    index = load_or_build_passage_index(
        documents,
        tokenizer,
        cache_dir=args.cache_dir,
        tokenizer_name=args.chunk_tokenizer_name_or_path,
        tokenizer_revision=args.chunk_tokenizer_revision,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    return QuestionWork(question_entry, len(documents), index)


def _record(
    work: QuestionWork,
    hits: Sequence[PassageHit],
    args: argparse.Namespace,
    *,
    config_id: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "config_id": config_id,
        "question_id": work.question_entry["question_id"],
        "question": work.question_entry["question"],
        "variant": args.strategy,
        "num_candidate_docs": work.num_documents,
        "num_passages": len(work.index.passages),
        "candidate_depth": args.candidate_depth,
        "top_k": args.top_k,
        "max_per_source": args.max_per_source,
        "source_fingerprint": work.index.source_fingerprint,
        "passage_config_fingerprint": work.index.config_fingerprint,
        "passage_cache_hit": work.index.cache_hit,
        "chunk_tokenizer": args.chunk_tokenizer_name_or_path,
        "chunk_tokenizer_revision": args.chunk_tokenizer_revision,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "documents": [passage_hit_to_record(hit, include_text=not args.no_text) for hit in hits],
    }
    if args.strategy in {"embed_chunked", "hybrid_rrf_chunked", "rerank_fixed_chunked"}:
        record.update(
            {
                "embedding_model": args.embedding_model_name_or_path,
                "embedding_model_revision": args.embedding_model_revision,
                "embedding_instruction": args.embedding_instruction,
                "embedding_dim": args.embedding_dim,
                "embedding_normalized": args.normalize,
            }
        )
    if args.strategy == "bm25_chunked":
        record.update(
            {
                "retrieval_model": "bm25",
                "bm25_k1": args.bm25_k1,
                "bm25_b": args.bm25_b,
            }
        )
    elif args.strategy == "hybrid_rrf_chunked":
        record.update(
            {
                "retrieval_model": "qwen_embedding_bm25_rrf",
                "bm25_k1": args.bm25_k1,
                "bm25_b": args.bm25_b,
                "rrf_k": args.rrf_k,
            }
        )
    elif args.strategy == "lateinter_chunked":
        record.update(
            {
                "retrieval_model": args.lateinter_model_name_or_path,
                "retrieval_model_revision": args.lateinter_model_revision,
                "retrieval_backend": "pylate",
                "scoring_method": "colbert_maxsim",
                "query_prefix": args.lateinter_query_prefix,
                "normalized": True,
            }
        )
    elif args.strategy == "rerank_fixed_chunked":
        record.update(
            {
                "retrieval_model": args.embedding_model_name_or_path,
                "reranker_model": args.reranker_model_name_or_path,
                "reranker_model_revision": args.reranker_model_revision,
                "reranker_scoring_method": "forward_logits_yes_no_softmax",
                "reranker_instruction": args.reranker_instruction,
                "reranker_max_length": args.reranker_max_length,
            }
        )
    else:
        record["retrieval_model"] = args.embedding_model_name_or_path
        record["retrieval_model_revision"] = args.embedding_model_revision
    return record


def _error_record(
    question_entry: dict[str, str],
    error: Exception,
    args: argparse.Namespace,
    *,
    config_id: str,
) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "variant": args.strategy,
        "error": str(error),
        "documents": [],
    }


def _embedding_model(args: argparse.Namespace) -> Qwen3EmbeddingVllm:
    return PinnedQwen3EmbeddingVllm(
        args.embedding_model_name_or_path,
        revision=args.embedding_model_revision,
        instruction=args.embedding_instruction,
        tensor_parallel_size=args.embedding_tensor_parallel_size,
        max_model_len=args.embedding_max_model_len,
        gpu_memory_utilization=args.embedding_gpu_memory_utilization,
    )


def _embed_or_hybrid_records(
    questions: Sequence[dict[str, str]],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    config_id: str,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    model = _embedding_model(args)
    records: list[dict[str, Any]] = []
    try:
        for question_entry in tqdm(questions, desc=args.strategy):
            try:
                work = _load_question_work(question_entry, tokenizer, args)
                dense_index = DensePassageIndex(
                    model,
                    work.index.passages,
                    dim=args.embedding_dim,
                    batch_size=args.embedding_batch_size,
                    normalize=args.normalize,
                )
                dense_hits = dense_index.search(
                    question_entry["question"],
                    top_k=args.candidate_depth,
                )
                if args.strategy == "hybrid_rrf_chunked":
                    bm25_hits = rank_passages(
                        work.index.passages,
                        compute_bm25_scores(
                            question_entry["question"],
                            work.index.passages,
                            k1=args.bm25_k1,
                            b=args.bm25_b,
                        ),
                        component="bm25",
                    )[: args.candidate_depth]
                    ranked = reciprocal_rank_fusion(
                        [dense_hits, bm25_hits],
                        rrf_k=args.rrf_k,
                        depth=args.candidate_depth,
                        weights=[1.0, 1.0],
                        names=["dense", "bm25"],
                    )
                else:
                    ranked = dense_hits
                hits = source_diverse_top_k(
                    ranked,
                    top_k=args.top_k,
                    max_per_source=args.max_per_source,
                )
                record = _record(work, hits, args, config_id=config_id)
            except Exception as exc:
                logger.exception(
                    "Failed %s for question %s", args.strategy, question_entry["question_id"]
                )
                record = _error_record(question_entry, exc, args, config_id=config_id)
            records.append(record)
            if on_record is not None:
                on_record(record)
    finally:
        model.stop()
    return records


def _bm25_records(
    questions: Sequence[dict[str, str]],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    config_id: str,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for question_entry in tqdm(questions, desc=args.strategy):
        try:
            work = _load_question_work(question_entry, tokenizer, args)
            ranked = rank_passages(
                work.index.passages,
                compute_bm25_scores(
                    question_entry["question"],
                    work.index.passages,
                    k1=args.bm25_k1,
                    b=args.bm25_b,
                ),
                component="bm25",
            )[: args.candidate_depth]
            hits = source_diverse_top_k(
                ranked,
                top_k=args.top_k,
                max_per_source=args.max_per_source,
            )
            record = _record(work, hits, args, config_id=config_id)
        except Exception as exc:
            logger.exception("Failed BM25 for question %s", question_entry["question_id"])
            record = _error_record(question_entry, exc, args, config_id=config_id)
        records.append(record)
        if on_record is not None:
            on_record(record)
    return records


def _lateinter_records(
    questions: Sequence[dict[str, str]],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    config_id: str,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    from experiments.inference.retrieve_colbert import ReasonModernColBERTRetriever

    model_name_or_path = resolve_hf_revision(
        args.lateinter_model_name_or_path,
        args.lateinter_model_revision,
    )
    model = ReasonModernColBERTRetriever(
        model_name_or_path,
        device=args.lateinter_device,
    )
    records: list[dict[str, Any]] = []
    try:
        for question_entry in tqdm(questions, desc=args.strategy):
            try:
                work = _load_question_work(question_entry, tokenizer, args)
                candidates = [
                    CandidateDocument(
                        question_id=passage.question_id,
                        path=args.docs_dir / passage.source_path,
                        rel_path=passage.chunk_id,
                        text=passage.text,
                        truncated=False,
                        metadata={},
                    )
                    for passage in work.index.passages
                ]
                by_chunk_id = {
                    candidate.rel_path: passage
                    for candidate, passage in zip(candidates, work.index.passages)
                }
                retrieved = model.retrieve(
                    question_entry["question"],
                    candidates,
                    top_k=args.candidate_depth,
                    batch_size=args.lateinter_batch_size,
                    normalize_embeddings=True,
                    query_prefix=args.lateinter_query_prefix,
                    show_progress_bar=False,
                )
                ranked = [
                    PassageHit(
                        passage=by_chunk_id[item.document.rel_path],
                        score=item.retrieval_score,
                        rank=rank,
                        component_scores={"colbert": item.retrieval_score},
                        component_ranks={"colbert": rank},
                    )
                    for rank, item in enumerate(retrieved, start=1)
                ]
                hits = source_diverse_top_k(
                    ranked,
                    top_k=args.top_k,
                    max_per_source=args.max_per_source,
                )
                record = _record(work, hits, args, config_id=config_id)
            except Exception as exc:
                logger.exception(
                    "Failed %s for question %s", args.strategy, question_entry["question_id"]
                )
                record = _error_record(question_entry, exc, args, config_id=config_id)
            records.append(record)
            if on_record is not None:
                on_record(record)
    finally:
        model.stop()
    return records


def resolve_hf_revision(model_name_or_path: str, revision: str | None) -> str:
    if revision is None or Path(model_name_or_path).exists():
        return model_name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_name_or_path, revision=revision)


def _rerank_hits(
    reranker: Qwen3RerankerForwardLogits,
    question: str,
    candidates: Sequence[PassageHit],
    *,
    batch_size: int,
) -> list[PassageHit]:
    scores: list[float] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        scores.extend(reranker.compute_scores([(question, hit.passage.text) for hit in batch]))
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -scores[index],
            candidates[index].rank,
            candidates[index].passage.chunk_id,
        ),
    )
    ranked: list[PassageHit] = []
    for rank, index in enumerate(order, start=1):
        dense_hit = candidates[index]
        ranked.append(
            PassageHit(
                passage=dense_hit.passage,
                score=scores[index],
                rank=rank,
                component_scores={"dense": dense_hit.score, "rerank": scores[index]},
                component_ranks={"dense": dense_hit.rank, "rerank": rank},
            )
        )
    return ranked


def _rerank_records(
    questions: Sequence[dict[str, str]],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    config_id: str,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    embedding_model = _embedding_model(args)
    work_items: list[DenseCandidateWork | dict[str, Any]] = []
    try:
        for question_entry in tqdm(questions, desc="rerank dense candidates"):
            try:
                work = _load_question_work(question_entry, tokenizer, args)
                dense_index = DensePassageIndex(
                    embedding_model,
                    work.index.passages,
                    dim=args.embedding_dim,
                    batch_size=args.embedding_batch_size,
                    normalize=args.normalize,
                )
                candidates = dense_index.search(
                    question_entry["question"],
                    top_k=args.candidate_depth,
                )
                work_items.append(DenseCandidateWork(work, tuple(candidates)))
            except Exception as exc:
                logger.exception(
                    "Failed rerank candidate retrieval for %s", question_entry["question_id"]
                )
                work_items.append(_error_record(question_entry, exc, args, config_id=config_id))
    finally:
        embedding_model.stop()
        del embedding_model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    successful = [item for item in work_items if isinstance(item, DenseCandidateWork)]
    if not successful:
        records = [item for item in work_items if isinstance(item, dict)]
        if on_record is not None:
            for record in records:
                on_record(record)
        return records

    reranker = Qwen3RerankerForwardLogits(
        args.reranker_model_name_or_path,
        revision=args.reranker_model_revision,
        instruction=args.reranker_instruction,
        max_length=args.reranker_max_length,
        device=args.reranker_device,
        dtype=args.reranker_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    records: list[dict[str, Any]] = []
    try:
        for item in tqdm(work_items, desc="forward-logit reranking"):
            if isinstance(item, dict):
                record = item
                records.append(record)
                if on_record is not None:
                    on_record(record)
                continue
            try:
                ranked = _rerank_hits(
                    reranker,
                    item.work.question_entry["question"],
                    item.candidates,
                    batch_size=args.reranker_batch_size,
                )
                hits = source_diverse_top_k(
                    ranked,
                    top_k=args.top_k,
                    max_per_source=args.max_per_source,
                )
                record = _record(item.work, hits, args, config_id=config_id)
            except Exception as exc:
                question_entry = item.work.question_entry
                logger.exception(
                    "Failed forward-logit reranking for %s", question_entry["question_id"]
                )
                record = _error_record(
                    question_entry,
                    exc,
                    args,
                    config_id=config_id,
                )
            records.append(record)
            if on_record is not None:
                on_record(record)
    finally:
        reranker.stop()
    return records


def run_strong_retrieval(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = select_questions(
        load_heldout_questions(args.questions_path, args.privileged_dir),
        args.query_id,
        args.limit,
    )
    manifest = prepare_retrieval_manifest(args, questions)
    question_ids = [question["question_id"] for question in questions]
    completed = (
        load_completed_retrieval_records(
            args.output_path,
            config_id=manifest["config_id"],
            question_ids=question_ids,
        )
        if args.resume
        else {}
    )
    pending = [question for question in questions if question["question_id"] not in completed]
    logger.info(
        "Loaded %d questions (%d complete, %d pending)",
        len(questions),
        len(completed),
        len(pending),
    )

    if not args.resume:
        write_retrieval_checkpoint(args.output_path, questions, {})
    if not pending:
        ordered = write_retrieval_checkpoint(args.output_path, questions, completed)
        ensure_no_record_errors(ordered, stage=args.strategy)
        return ordered

    records_by_question = dict(completed)

    def checkpoint(record: dict[str, Any]) -> None:
        records_by_question[str(record["question_id"])] = record
        write_retrieval_checkpoint(args.output_path, questions, records_by_question)

    tokenizer = _load_tokenizer(args)
    if args.strategy == "bm25_chunked":
        _bm25_records(
            pending,
            tokenizer,
            args,
            config_id=manifest["config_id"],
            on_record=checkpoint,
        )
    elif args.strategy in {"embed_chunked", "hybrid_rrf_chunked"}:
        _embed_or_hybrid_records(
            pending,
            tokenizer,
            args,
            config_id=manifest["config_id"],
            on_record=checkpoint,
        )
    elif args.strategy == "lateinter_chunked":
        _lateinter_records(
            pending,
            tokenizer,
            args,
            config_id=manifest["config_id"],
            on_record=checkpoint,
        )
    elif args.strategy == "rerank_fixed_chunked":
        _rerank_records(
            pending,
            tokenizer,
            args,
            config_id=manifest["config_id"],
            on_record=checkpoint,
        )
    else:
        raise ValueError(f"Unsupported retrieval strategy: {args.strategy}")

    records = write_retrieval_checkpoint(args.output_path, questions, records_by_question)
    logger.info("Wrote %d %s records to %s", len(records), args.strategy, args.output_path)
    ensure_no_record_errors(records, stage=args.strategy)
    return records


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_strong_retrieval(args)


if __name__ == "__main__":
    main()
