from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from experiments.inference.retrieve import (
    DEFAULT_DOCS_DIR,
    DEFAULT_PRIVILEGED_DIR,
    DEFAULT_QUESTIONS_PATH,
    CandidateDocument,
    RetrievedDocument,
    ensure_no_record_errors,
    load_candidate_documents,
    load_heldout_questions,
    retrieved_document_to_record,
    select_questions,
    write_jsonl,
)


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME_OR_PATH = "lightonai/Reason-ModernColBERT"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "inference" / "retrieval_colbert.jsonl"
DEFAULT_GOLD_AND_SUPPORT_ONLY = False


class ReasonModernColBERTRetriever:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None,
    ) -> None:
        try:
            from pylate import models, rank
        except ImportError as exc:
            raise ImportError(
                "PyLate is required for the Reason-ModernColBERT baseline. "
                "Install it with: pip install -U pylate"
            ) from exc

        self.model = models.ColBERT(model_name_or_path=model_name_or_path, device=device)
        self.rank = rank

    def retrieve(
        self,
        question: str,
        documents: list[CandidateDocument],
        *,
        top_k: int,
        batch_size: int,
        normalize_embeddings: bool,
        query_prefix: str | None,
        show_progress_bar: bool,
    ) -> list[RetrievedDocument]:
        if not documents:
            return []

        query = f"{query_prefix}{question}" if query_prefix else question
        document_ids = list(range(len(documents)))
        document_texts = [[doc.text for doc in documents]]

        with torch.inference_mode():
            query_embeddings = self.model.encode(
                [query],
                batch_size=batch_size,
                is_query=True,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
            )
            document_embeddings = self.model.encode(
                document_texts,
                batch_size=batch_size,
                is_query=False,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
            )
            reranked = self.rank.rerank(
                documents_ids=[document_ids],
                queries_embeddings=query_embeddings,
                documents_embeddings=document_embeddings,
            )

        retrieved: list[RetrievedDocument] = []
        for rank_index, result in enumerate(single_query_results(reranked)[: min(top_k, len(documents))], start=1):
            doc_index = result_document_index(result, document_ids)
            retrieved.append(
                RetrievedDocument(
                    document=documents[doc_index],
                    retrieval_score=result_score(result),
                    retrieval_rank=rank_index,
                )
            )
        return retrieved

    def stop(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve candidate documents for heldout questions with Reason-ModernColBERT."
    )
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
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-doc-chars", type=int, default=30_000)
    parser.add_argument(
        "--gold-and-support-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GOLD_AND_SUPPORT_ONLY,
        help=(
            "Only retrieve from manifest docs labeled as gold or supporting/evidence docs. "
            "Requires --privileged-dir."
        ),
    )
    parser.add_argument(
        "--normalize-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize ColBERT token embeddings before MaxSim scoring.",
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help="Optional text prepended to each question before query encoding.",
    )
    parser.add_argument("--device", default=None, help='PyTorch device for PyLate, e.g. "cuda", "cuda:0", or "cpu".')
    parser.add_argument("--show-progress-bar", action="store_true", help="Show PyLate encoding progress bars.")
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


def single_query_results(reranked: Any) -> list[Any]:
    if reranked is None:
        return []
    if hasattr(reranked, "tolist"):
        reranked = reranked.tolist()
    if not isinstance(reranked, list):
        raise ValueError(f"Unexpected PyLate rerank output type: {type(reranked)!r}")
    if not reranked:
        return []
    first = reranked[0]
    if isinstance(first, list):
        return first
    return reranked


def result_field(result: Any, field_names: tuple[str, ...]) -> Any:
    for field_name in field_names:
        if isinstance(result, dict) and field_name in result:
            return result[field_name]
        if hasattr(result, field_name):
            return getattr(result, field_name)
    raise ValueError(f"PyLate rerank result is missing one of {field_names}: {result!r}")


def result_document_index(result: Any, document_ids: list[int]) -> int:
    result_id = result_field(result, ("id", "document_id", "doc_id", "corpus_id"))
    if hasattr(result_id, "item"):
        result_id = result_id.item()
    try:
        doc_index = int(result_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unexpected PyLate document id: {result_id!r}") from exc
    if doc_index not in document_ids:
        raise ValueError(f"PyLate returned unknown document id: {result_id!r}")
    return doc_index


def result_score(result: Any) -> float:
    score = result_field(result, ("score", "retrieval_score", "colbert_score"))
    if hasattr(score, "item"):
        score = score.item()
    return float(score)


def build_moderncolbert_retrieval_record(
    question_entry: dict[str, str],
    documents: list[CandidateDocument],
    retrieved_documents: list[RetrievedDocument],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "num_candidate_docs": len(documents),
        "top_k": args.top_k,
        "gold_and_support_only": bool(args.gold_and_support_only),
        "retrieval_model": args.model_name_or_path,
        "retrieval_backend": "pylate",
        "scoring_method": "colbert_maxsim",
        "normalized": bool(args.normalize_embeddings),
        "documents": [
            retrieved_document_to_record(doc, include_text=not args.no_text)
            for doc in retrieved_documents
        ],
    }
    if args.query_prefix:
        record["query_prefix"] = args.query_prefix
    return record


def build_error_record(
    question_entry: dict[str, str],
    error: Exception,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "gold_and_support_only": bool(args.gold_and_support_only),
        "retrieval_model": args.model_name_or_path,
        "retrieval_backend": "pylate",
        "scoring_method": "colbert_maxsim",
        "error": str(error),
        "documents": [],
    }


def run_moderncolbert_retrieval(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = select_questions(
        load_heldout_questions(args.questions_path, args.privileged_dir),
        args.query_id,
        args.limit,
    )
    logger.info("Loaded %d heldout questions", len(questions))
    if not questions:
        write_jsonl([], args.output_path)
        logger.info("Wrote 0 records to %s", args.output_path)
        return []

    model = ReasonModernColBERTRetriever(args.model_name_or_path, device=args.device)

    records: list[dict[str, Any]] = []
    try:
        for question_entry in tqdm(questions, desc="Reason-ModernColBERT retrieving"):
            question_id = question_entry["question_id"]
            try:
                documents = load_candidate_documents(
                    args.docs_dir,
                    args.privileged_dir,
                    question_id,
                    args.max_doc_chars,
                    gold_and_support_only=bool(args.gold_and_support_only),
                )
                retrieved_documents = model.retrieve(
                    question_entry["question"],
                    documents,
                    top_k=args.top_k,
                    batch_size=args.batch_size,
                    normalize_embeddings=args.normalize_embeddings,
                    query_prefix=args.query_prefix,
                    show_progress_bar=args.show_progress_bar,
                )
                record = build_moderncolbert_retrieval_record(
                    question_entry,
                    documents,
                    retrieved_documents,
                    args=args,
                )
            except Exception as exc:
                logger.exception("Failed Reason-ModernColBERT retrieval for question %s", question_id)
                record = build_error_record(question_entry, exc, args=args)
            records.append(record)
    finally:
        model.stop()

    write_jsonl(records, args.output_path)
    logger.info("Wrote %d records to %s", len(records), args.output_path)
    ensure_no_record_errors(records, stage="Reason-ModernColBERT retrieval")
    return records


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_moderncolbert_retrieval(args)


if __name__ == "__main__":
    main()
