# coding: utf-8
from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Union

from tqdm import tqdm

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS_PATH = REPO_ROOT / "data" / "splits" / "heldout_ids.json"
DEFAULT_DOCS_DIR = REPO_ROOT / "data" / "train"
DEFAULT_PRIVILEGED_DIR = REPO_ROOT / "data" / "train_privileged"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "inference" / "retrieval_dense.jsonl"
GOLD_AND_SUPPORT_LABELS = frozenset({"gold_docs", "evidence_docs", "support_docs", "supporting_docs"})
DEFAULT_GOLD_AND_SUPPORT_ONLY = False

@dataclass(frozen=True)
class CandidateDocument:
    question_id: str
    path: Path
    rel_path: str
    text: str
    truncated: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievedDocument:
    document: CandidateDocument
    retrieval_score: float
    retrieval_rank: int


def build_embedding_llm_kwargs(
    model_name_or_path: str,
    *,
    tensor_parallel_size: int | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
) -> dict[str, Any]:
    import torch

    from vllm import LLM

    llm_signature = inspect.signature(LLM)
    llm_kwargs: dict[str, Any] = {
        "model": model_name_or_path,
        "hf_overrides": {"is_matryoshka": True},
    }
    if {"runner", "convert"}.issubset(llm_signature.parameters):
        llm_kwargs["runner"] = "pooling"
        llm_kwargs["convert"] = "embed"
    else:
        llm_kwargs["task"] = "embed"

    if tensor_parallel_size is not None:
        llm_kwargs["tensor_parallel_size"] = tensor_parallel_size
    else:
        llm_kwargs["tensor_parallel_size"] = max(torch.cuda.device_count(), 1)
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    return llm_kwargs


class Qwen3EmbeddingVllm:
    def __init__(
        self,
        model_name_or_path: str,
        instruction: str | None = None,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
    ) -> None:
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        self.instruction = instruction

        from vllm import LLM

        self.model = LLM(
            **build_embedding_llm_kwargs(
                model_name_or_path,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        )

    def get_detailed_instruct(self, task_description: str | None, query: str) -> str:
        if task_description is None:
            task_description = self.instruction
        return f"Instruct: {task_description}\nQuery:{query}"

    def encode(
        self,
        sentences: Union[list[str], str],
        is_query: bool = False,
        instruction: str | None = None,
        dim: int = -1,
    ) -> torch.Tensor:
        import torch

        if isinstance(sentences, str):
            sentences = [sentences]
        if is_query:
            sentences = [self.get_detailed_instruct(instruction, sent) for sent in sentences]
        if dim > 0:
            from vllm import PoolingParams

            output = self.model.embed(sentences, pooling_params=PoolingParams(dimensions=dim))
        else:
            output = self.model.embed(sentences)
        return torch.tensor([o.outputs.embedding for o in output])

    def stop(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return

        try:
            shutdown_vllm_llm(model)
        finally:
            del self.model
            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                logger.debug("Could not release all embedding CUDA memory", exc_info=True)


def shutdown_vllm_llm(llm: Any) -> None:
    """Stop vLLM workers before releasing the caller's final model reference."""
    engine = getattr(llm, "llm_engine", None)
    renderer = getattr(llm, "renderer", None)
    if renderer is None:
        renderer = getattr(engine, "renderer", None)
    engine_core = getattr(engine, "engine_core", None)
    model_executor = getattr(engine, "model_executor", None)
    shutdown = None
    for target in (engine_core, engine, model_executor, llm):
        candidate = getattr(target, "shutdown", None)
        if callable(candidate):
            shutdown = candidate
            break

    try:
        renderer_shutdown = getattr(renderer, "shutdown", None)
        if callable(renderer_shutdown):
            renderer_shutdown()
    finally:
        if shutdown is not None:
            shutdown()
        else:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            try:
                destroy_model_parallel()
            finally:
                destroy_distributed_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve candidate documents for heldout questions.")
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument(
        "--privileged-dir",
        type=Path,
        default=DEFAULT_PRIVILEGED_DIR,
        help=(
            "Optional manifest source for docid/url/label metadata, e.g. data/train_privileged. "
            "Required by --gold-and-support-only."
        ),
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-doc-chars", type=int, default=30_000)
    parser.add_argument(
        "--gold-and-support-only",
        action=argparse.BooleanOptionalAction,
        help=(
            "Only retrieve from manifest docs labeled as gold or supporting/evidence docs. "
            "Requires --privileged-dir."
        ),
        default=DEFAULT_GOLD_AND_SUPPORT_ONLY,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before scoring.")
    parser.add_argument("--no-text", action="store_true", help="Omit document text from the output JSONL.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help="Only process this question id. Can be supplied multiple times.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_doc_chars < 0:
        parser.error("--max-doc-chars must be >= 0")
    if args.gold_and_support_only and args.privileged_dir is None:
        parser.error("--gold-and-support-only requires --privileged-dir")
    return args


def load_heldout_questions(
    path: Path, privileged_dir: Path | None = DEFAULT_PRIVILEGED_DIR
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        raw_questions = payload.get("questions", payload.get("query_ids", []))
    else:
        raw_questions = payload
    if not isinstance(raw_questions, list):
        raise ValueError(f"Expected a list of questions in {path}")

    questions: list[dict[str, str]] = []
    for item in raw_questions:
        if isinstance(item, (str, int)):
            question_id = str(item)
            question = ""
        elif isinstance(item, dict):
            raw_id = item.get("question_id", item.get("query_id", item.get("id")))
            if raw_id is None:
                raise ValueError(f"Question entry has no id: {item!r}")
            question_id = str(raw_id)
            question = str(item.get("question", "")).strip()
        else:
            raise ValueError(f"Invalid question entry: {item!r}")
        if not question:
            if privileged_dir is None:
                raise ValueError(
                    f"Question {question_id} has no text and no privileged directory was provided"
                )
            query_path = privileged_dir / question_id / "query.txt"
            if not query_path.exists():
                raise FileNotFoundError(f"Missing question text for {question_id}: {query_path}")
            question = query_path.read_text(encoding="utf-8").strip()
        questions.append({"question_id": question_id, "question": question})
    return questions


def select_questions(
    questions: list[dict[str, str]],
    query_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, str]]:
    if query_ids:
        selected = {str(query_id) for query_id in query_ids}
        questions = [question for question in questions if question["question_id"] in selected]
    if limit is not None:
        questions = questions[:limit]
    return questions


def load_manifest_by_filename(privileged_dir: Path | None, question_id: str) -> dict[str, dict[str, Any]]:
    if privileged_dir is None:
        return {}

    manifest_path = privileged_dir / question_id / "manifest.json"
    if not manifest_path.exists():
        return {}

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    metadata: dict[str, dict[str, Any]] = {}
    for doc in manifest.get("documents", []):
        agent_file = doc.get("agent_file")
        if not agent_file:
            continue
        filename = Path(str(agent_file)).name
        metadata[filename] = {
            "label": doc.get("label"),
            "docid": doc.get("docid"),
            "url": doc.get("url"),
        }
    return metadata


def read_document_text(path: Path, max_doc_chars: int) -> tuple[str, bool]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_doc_chars > 0 and len(text) > max_doc_chars:
        return text[:max_doc_chars], True
    return text, False


def load_candidate_documents(
    docs_dir: Path,
    privileged_dir: Path | None,
    question_id: str,
    max_doc_chars: int,
    gold_and_support_only: bool = False,
) -> list[CandidateDocument]:
    question_docs_dir = docs_dir / question_id
    if not question_docs_dir.exists():
        raise FileNotFoundError(f"No document directory found for question {question_id}: {question_docs_dir}")
    if not question_docs_dir.is_dir():
        raise NotADirectoryError(f"Document path is not a directory: {question_docs_dir}")
    if gold_and_support_only and privileged_dir is None:
        raise ValueError("gold_and_support_only requires a privileged_dir with manifest.json files")
    if gold_and_support_only:
        assert privileged_dir is not None
        manifest_path = privileged_dir / question_id / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest found for gold_and_support_only question {question_id}: {manifest_path}"
            )

    manifest_metadata = load_manifest_by_filename(privileged_dir, question_id)
    documents: list[CandidateDocument] = []

    for path in sorted(question_docs_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file() or path.suffix != ".txt":
            continue
        if path.name in {"query.txt", "answer.txt"}:
            continue

        metadata = manifest_metadata.get(path.name, {})
        if gold_and_support_only and metadata.get("label") not in GOLD_AND_SUPPORT_LABELS:
            continue

        text, truncated = read_document_text(path, max_doc_chars)
        documents.append(
            CandidateDocument(
                question_id=question_id,
                path=path,
                rel_path=path.relative_to(docs_dir).as_posix(),
                text=text,
                truncated=truncated,
                metadata=metadata,
            )
        )

    return documents


def batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def maybe_normalize(embeddings: torch.Tensor, normalize: bool) -> torch.Tensor:
    if normalize:
        import torch.nn.functional as F

        return F.normalize(embeddings, p=2, dim=1)
    return embeddings


def retrieve_documents(
    model: Qwen3EmbeddingVllm,
    question: str,
    documents: list[CandidateDocument],
    *,
    top_k: int,
    dim: int,
    batch_size: int,
    normalize: bool,
) -> list[RetrievedDocument]:
    import torch

    if not documents:
        return []

    with torch.inference_mode():
        query_embedding = model.encode(question, is_query=True, dim=dim)
        query_embedding = maybe_normalize(query_embedding, normalize)

        doc_embeddings: list[torch.Tensor] = []
        for batch in batches(documents, batch_size):
            batch_embeddings = model.encode([doc.text for doc in batch], dim=dim)
            doc_embeddings.append(maybe_normalize(batch_embeddings, normalize))

        all_doc_embeddings = torch.cat(doc_embeddings, dim=0)
        scores = ((query_embedding @ all_doc_embeddings.T).squeeze(0) * 100).detach().cpu()

    k = min(top_k, len(documents))
    top_scores, top_indices = torch.topk(scores, k=k)
    retrieved: list[RetrievedDocument] = []
    for rank, (score, index) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
        retrieved.append(
            RetrievedDocument(
                document=documents[index],
                retrieval_score=float(score),
                retrieval_rank=rank,
            )
        )
    return retrieved


def retrieved_document_to_record(doc: RetrievedDocument, *, include_text: bool) -> dict[str, Any]:
    candidate = doc.document
    record: dict[str, Any] = {
        "rank": doc.retrieval_rank,
        "retrieval_rank": doc.retrieval_rank,
        "retrieval_score": doc.retrieval_score,
        "path": candidate.rel_path,
        "filename": candidate.path.name,
        "truncated": candidate.truncated,
    }
    record.update({key: value for key, value in candidate.metadata.items() if value is not None})
    if include_text:
        record["text"] = candidate.text
    return record


def build_retrieval_record(
    question_entry: dict[str, str],
    documents: list[CandidateDocument],
    retrieved_documents: list[RetrievedDocument],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "num_candidate_docs": len(documents),
        "top_k": args.top_k,
        "gold_and_support_only": bool(getattr(args, "gold_and_support_only", False)),
        "retrieval_model": args.model_name_or_path,
        "embedding_dim": args.dim,
        "normalized": bool(args.normalize),
        "documents": [
            retrieved_document_to_record(doc, include_text=not args.no_text)
            for doc in retrieved_documents
        ],
    }


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_no_record_errors(records: Iterable[dict[str, Any]], *, stage: str) -> None:
    failures = [record for record in records if record.get("error")]
    if not failures:
        return
    question_ids = [str(record.get("question_id", "?")) for record in failures]
    raise RuntimeError(
        f"{stage} failed for {len(failures)} question(s): {', '.join(question_ids)}"
    )


def run_retrieval(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = select_questions(
        load_heldout_questions(args.questions_path, args.privileged_dir),
        args.query_id,
        args.limit,
    )
    logger.info("Loaded %d heldout questions", len(questions))

    model = Qwen3EmbeddingVllm(
        args.model_name_or_path,
        instruction=args.instruction,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    records: list[dict[str, Any]] = []
    try:
        for question_entry in tqdm(questions, desc="Retrieving"):
            question_id = question_entry["question_id"]
            try:
                documents = load_candidate_documents(
                    args.docs_dir,
                    args.privileged_dir,
                    question_id,
                    args.max_doc_chars,
                    gold_and_support_only=bool(getattr(args, "gold_and_support_only", False)),
                )
                retrieved_documents = retrieve_documents(
                    model,
                    question_entry["question"],
                    documents,
                    top_k=args.top_k,
                    dim=args.dim,
                    batch_size=args.batch_size,
                    normalize=args.normalize,
                )
                record = build_retrieval_record(question_entry, documents, retrieved_documents, args=args)
            except Exception as exc:
                logger.exception("Failed retrieval for question %s", question_id)
                record = {
                    "question_id": question_id,
                    "question": question_entry["question"],
                    "gold_and_support_only": bool(getattr(args, "gold_and_support_only", False)),
                    "error": str(exc),
                    "documents": [],
                }
            records.append(record)
    finally:
        model.stop()

    write_jsonl(records, args.output_path)
    logger.info("Wrote %d records to %s", len(records), args.output_path)
    ensure_no_record_errors(records, stage="Embedding retrieval")
    return records


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_retrieval(args)


if __name__ == "__main__":
    main()
