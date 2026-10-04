from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import io
import json
import logging
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    batches,
    ensure_no_record_errors,
    load_candidate_documents,
    load_heldout_questions,
    select_questions,
    write_jsonl,
)


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVAL_OUTPUT_PATH = REPO_ROOT / "outputs" / "inference" / "graph_retrieval.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "inference" / "graph_answers.jsonl"
DEFAULT_GOLD_AND_SUPPORT_ONLY = False
DEFAULT_MAX_INDEX_DOCS = 0
FORCE_FULL_DOCUMENT_CORPUS = True
DEFAULT_NON_LOCAL_GRAPH = True
DEFAULT_NON_LOCAL_ANSWERER = True
DEFAULT_LOCAL_GRAPH_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_LOCAL_ANSWER_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_LOCAL_GRAPH_MAX_MODEL_LEN = 8192

PROMPT_SOURCE_URLS = [
    "https://github.com/microsoft/graphrag/tree/main/packages/graphrag/graphrag/prompts/index",
    "https://github.com/microsoft/graphrag/tree/main/packages/graphrag/graphrag/prompts/query",
]

# Prompt structure adapted from Microsoft GraphRAG's MIT-licensed prompt family:
# entity/relationship extraction, description summarization, community reports,
# and report-grounded query context. The tuple delimiters are intentionally kept
# because they are easy to recover from local models that do not reliably emit JSON.
GRAPH_EXTRACTION_PROMPT = """You are building a knowledge graph for retrieval-augmented question answering.

Goal:
Given one text unit, extract the salient entities and relationships needed to build a reusable corpus-level
knowledge graph for retrieval-augmented question answering.
Keep specific entities, bridge entities, dates, quantities, events, works, and facts that could support
future multi-hop questions over this corpus.

Entity types to consider:
{entity_types}

Output format:
Use "{record_delimiter}" between records and finish with "{completion_delimiter}".
For each entity:
("entity"<|>ENTITY_NAME<|>ENTITY_TYPE<|>ENTITY_DESCRIPTION)
For each relationship:
("relationship"<|>SOURCE_ENTITY_NAME<|>TARGET_ENTITY_NAME<|>RELATIONSHIP_DESCRIPTION<|>RELATIONSHIP_STRENGTH)

Rules:
- Use canonical, specific entity names.
- Descriptions must be concise and grounded in the text unit.
- Relationship strength is a number from 1 to 10.
- Do not infer facts beyond the text unit.
- If there are no useful records, output only "{completion_delimiter}".

Text unit metadata:
path: {path}
text_unit_id: {text_unit_id}

Text unit:
{text}
"""

SUMMARIZE_DESCRIPTION_PROMPT = """You are reconciling extracted knowledge graph descriptions.

Given one entity or relationship name and several source descriptions, write one concise, faithful summary.
Preserve specific names, dates, quantities, and qualifications that may matter for question answering.
Do not add facts that are not present in the descriptions.

Name:
{name}

Descriptions:
{descriptions}

Return only the merged summary.
"""

COMMUNITY_REPORT_PROMPT = """You are writing a GraphRAG community report for retrieval-augmented question answering.

Use only the provided entity and relationship tables. The report should summarize what this community contains
so it can be reused for later retrieval-augmented question answering.

Return only valid JSON with this shape:
{{
  "title": "short community title",
  "summary": "one paragraph summary grounded in the tables",
  "rating": 0-10,
  "findings": [
    {{"summary": "specific finding", "explanation": "brief explanation with entity or relationship ids"}}
  ]
}}

Maximum report words: {max_report_words}

Community tables:
{community_context}
"""


@dataclass(frozen=True)
class TextUnit:
    text_unit_id: str
    document: CandidateDocument
    unit_index: int
    text: str


@dataclass(frozen=True)
class ParsedEntity:
    name: str
    entity_type: str
    description: str


@dataclass(frozen=True)
class ParsedRelationship:
    source: str
    target: str
    description: str
    strength: float


@dataclass
class GraphEntity:
    name: str
    entity_type: str
    descriptions: list[str] = field(default_factory=list)
    text_unit_ids: set[str] = field(default_factory=set)
    document_paths: set[str] = field(default_factory=set)
    summary: str = ""


@dataclass
class GraphRelationship:
    source: str
    target: str
    descriptions: list[str] = field(default_factory=list)
    strengths: list[float] = field(default_factory=list)
    text_unit_ids: set[str] = field(default_factory=set)
    document_paths: set[str] = field(default_factory=set)
    summary: str = ""

    @property
    def weight(self) -> float:
        if not self.strengths:
            return 1.0
        return sum(self.strengths) / len(self.strengths)


@dataclass(frozen=True)
class GraphCommunity:
    community_id: int
    entity_names: list[str]
    relationship_keys: list[tuple[str, str]]
    selection_score: float


@dataclass(frozen=True)
class CommunityReport:
    community: GraphCommunity
    raw_response: str
    title: str
    summary: str
    rating: float | None
    findings: list[dict[str, str]]
    text: str
    retrieval_score: float = 0.0


@dataclass(frozen=True)
class GraphIndex:
    text_units: list[TextUnit]
    entities: dict[str, GraphEntity]
    relationships: dict[tuple[str, str], GraphRelationship]
    extraction_stats: list[dict[str, Any]]


class NonLocalGraphGenerator:
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
        self.client = create_async_client(self.config)
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.use_cache = use_cache
        self.max_concurrent = max_concurrent

    def generate_batch(self, prompts: Sequence[str], *, batch_size: int) -> list[str]:
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
        return [responses[0] if responses else "" for responses in response_groups]

    def stop(self) -> None:
        try:
            asyncio.run(self.client.close())
        except Exception:
            pass


class LocalGraphGenerator:
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
        sampling_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_completion_tokens,
        }
        if top_k > 0:
            sampling_kwargs["top_k"] = top_k
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def generate_batch(self, prompts: Sequence[str], *, batch_size: int) -> list[str]:
        prompt_records = [
            AnswerPrompt(
                messages=[{"role": "user", "content": prompt}],
                prompt_documents=[],
                prompt_token_count=0,
            )
            for prompt in prompts
        ]
        generation_groups: list[list[dict[str, Any]]] = []
        for batch in batches(prompt_records, batch_size):
            generation_groups.extend(self.agent.generate(batch, self.sampling_params))
        return [
            str(group[0].get("response") or group[0].get("raw_response") or "")
            if group
            else ""
            for group in generation_groups
        ]

    def stop(self) -> None:
        try:
            self.agent.stop()
        except Exception:
            pass


GraphGenerator = NonLocalGraphGenerator | LocalGraphGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Microsoft GraphRAG baseline over heldout candidate documents, then answer and judge."
    )
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--privileged-dir", type=Path, default=DEFAULT_PRIVILEGED_DIR)
    parser.add_argument("--retrieval-output-path", type=Path, default=DEFAULT_RETRIEVAL_OUTPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)

    parser.add_argument("--non-local-graph", action=argparse.BooleanOptionalAction, default=DEFAULT_NON_LOCAL_GRAPH)
    parser.add_argument("--graph-provider", default="gemini")
    parser.add_argument(
        "--graph-model-name",
        default="",
        help=(
            "Defaults to the selected provider's model for --non-local-graph, "
            f"or {DEFAULT_LOCAL_MODEL} for --no-non-local-graph."
        ),
    )
    parser.add_argument("--graph-max-completion-tokens", type=int, default=1536)
    parser.add_argument("--graph-reasoning-effort", default="provider_default")
    parser.add_argument("--graph-use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-max-concurrent", type=int, default=8)
    parser.add_argument("--graph-batch-size", type=int, default=8)
    parser.add_argument("--graph-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--graph-distributed-executor-backend", default=None)
    parser.add_argument("--graph-max-model-len", type=int, default=DEFAULT_LOCAL_GRAPH_MAX_MODEL_LEN)
    parser.add_argument(
        "--graph-gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "GPU memory fraction for local graph-generator vLLM. Defaults to "
            f"{DEFAULT_LOCAL_GRAPH_GPU_MEMORY_UTILIZATION}."
        ),
    )
    parser.add_argument("--graph-dtype", default="auto")
    parser.add_argument("--graph-trust-remote-code", action="store_true")
    parser.add_argument("--graph-temperature", type=float, default=0.0)
    parser.add_argument("--graph-top-p", type=float, default=1.0)
    parser.add_argument("--graph-top-k", type=int, default=-1)
    parser.add_argument("--graph-disable-thinking", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--entity-types", default="PERSON,ORGANIZATION,LOCATION,EVENT,WORK,PRODUCT,CONCEPT,DATE,NUMBER,OTHER")
    parser.add_argument("--chunk-chars", type=int, default=6000)
    parser.add_argument("--chunk-overlap", type=int, default=300)
    parser.add_argument("--max-chunks-per-doc", type=int, default=2)
    parser.add_argument("--max-text-units", type=int, default=80)
    parser.add_argument(
        "--max-index-docs",
        type=int,
        default=DEFAULT_MAX_INDEX_DOCS,
        help=(
            "Dense-prefilter to this many documents before graph extraction. "
            "The paper configuration uses full graph indexing."
        ),
    )
    parser.add_argument("--max-extract-input-chars", type=int, default=6500)
    parser.add_argument("--summarize-descriptions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-description-summaries", type=int, default=40)
    parser.add_argument("--max-community-entities", type=int, default=40)
    parser.add_argument("--max-community-reports", type=int, default=8)
    parser.add_argument("--max-report-input-chars", type=int, default=16000)
    parser.add_argument("--max-community-report-words", type=int, default=300)
    parser.add_argument("--include-raw-graph-responses", action="store_true")

    parser.add_argument("--embedding-model-name-or-path", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--embedding-instruction", default=None)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--embedding-max-model-len", type=int, default=None)
    parser.add_argument("--embedding-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max-doc-chars", type=int, default=30_000)
    parser.add_argument(
        "--gold-and-support-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GOLD_AND_SUPPORT_ONLY,
        help="Retrieval always uses all candidate documents.",
    )
    parser.add_argument("--no-text", action="store_true", help="Omit document text from retrieval JSONL.")

    parser.add_argument("--report-context-top-k", type=int, default=3)
    parser.add_argument("--source-context-top-k", type=int, default=5)

    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--non-local-answerer", action=argparse.BooleanOptionalAction, default=DEFAULT_NON_LOCAL_ANSWERER)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument(
        "--model-name-or-path",
        default="",
        help="Defaults to Qwen locally, or the selected provider's default for --non-local-answerer.",
    )
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--answer-reasoning-effort", default="provider_default")
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
    parser.add_argument("--top-docs", type=int, default=10)
    parser.add_argument("--answer-max-doc-chars", type=int, default=12_000)
    parser.add_argument("--min-doc-chars", type=int, default=800)
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

    if args.graph_max_completion_tokens < 1:
        parser.error("--graph-max-completion-tokens must be at least 1")
    if args.graph_max_concurrent < 1:
        parser.error("--graph-max-concurrent must be at least 1")
    if args.graph_batch_size < 1:
        parser.error("--graph-batch-size must be at least 1")
    if not args.non_local_graph and args.graph_max_model_len <= args.graph_max_completion_tokens:
        parser.error("--graph-max-model-len must be greater than --graph-max-completion-tokens")
    for option_name in (
        "graph_gpu_memory_utilization",
        "embedding_gpu_memory_utilization",
        "gpu_memory_utilization",
    ):
        value = getattr(args, option_name)
        if value is not None and not (0.0 < value <= 1.0):
            parser.error(f"--{option_name.replace('_', '-')} must be in (0, 1]")
    if args.chunk_chars < 100:
        parser.error("--chunk-chars must be at least 100")
    if args.chunk_overlap < 0:
        parser.error("--chunk-overlap must be >= 0")
    if args.chunk_overlap >= args.chunk_chars:
        parser.error("--chunk-overlap must be smaller than --chunk-chars")
    if args.max_chunks_per_doc < 0:
        parser.error("--max-chunks-per-doc must be >= 0")
    if args.max_text_units < 0:
        parser.error("--max-text-units must be >= 0")
    if args.max_index_docs < 0:
        parser.error("--max-index-docs must be >= 0")
    if args.max_extract_input_chars < 100:
        parser.error("--max-extract-input-chars must be at least 100")
    if args.max_description_summaries < 0:
        parser.error("--max-description-summaries must be >= 0")
    if args.max_community_entities < 1:
        parser.error("--max-community-entities must be at least 1")
    if args.max_community_reports < 0:
        parser.error("--max-community-reports must be >= 0")
    if args.max_report_input_chars < 100:
        parser.error("--max-report-input-chars must be at least 100")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be at least 1")
    if args.max_doc_chars < 0:
        parser.error("--max-doc-chars must be >= 0")
    if FORCE_FULL_DOCUMENT_CORPUS:
        args.gold_and_support_only = False
        args.max_index_docs = 0
    elif args.gold_and_support_only and args.privileged_dir is None:
        parser.error("--gold-and-support-only requires --privileged-dir")
    if args.report_context_top_k < 0:
        parser.error("--report-context-top-k must be >= 0")
    if args.source_context_top_k < 1:
        parser.error("--source-context-top-k must be at least 1")
    if args.no_text and not args.skip_answer and args.report_context_top_k > 0:
        parser.error(
            "--no-text cannot be used with graph report context and answering; "
            "pass --skip-answer or --report-context-top-k 0"
        )
    return args


def enforce_full_document_corpus(args: argparse.Namespace) -> None:
    if FORCE_FULL_DOCUMENT_CORPUS:
        args.gold_and_support_only = False
        args.max_index_docs = 0


def effective_graph_gpu_memory_utilization(args: argparse.Namespace) -> float:
    if args.non_local_graph:
        raise ValueError("graph GPU memory utilization only applies to local graph generators")
    if args.graph_gpu_memory_utilization is not None:
        return args.graph_gpu_memory_utilization
    return DEFAULT_LOCAL_GRAPH_GPU_MEMORY_UTILIZATION


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


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:json|text)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def split_document_text(text: str, *, chunk_chars: int, overlap: int, max_chunks: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= chunk_chars:
        return [stripped]

    chunks: list[str] = []
    start = 0
    while start < len(stripped):
        end = min(len(stripped), start + chunk_chars)
        if end < len(stripped):
            newline = stripped.rfind("\n", start + chunk_chars // 2, end)
            sentence = stripped.rfind(". ", start + chunk_chars // 2, end)
            boundary = max(newline, sentence)
            if boundary > start:
                end = boundary + (1 if boundary == newline else 2)

        chunk = stripped[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if max_chunks and len(chunks) >= max_chunks:
            break
        if end >= len(stripped):
            break
        start = max(0, end - overlap)
    return chunks


def make_text_units(documents: list[CandidateDocument], args: argparse.Namespace) -> list[TextUnit]:
    text_units: list[TextUnit] = []
    for document in documents:
        chunks = split_document_text(
            document.text,
            chunk_chars=args.chunk_chars,
            overlap=args.chunk_overlap,
            max_chunks=args.max_chunks_per_doc,
        )
        for unit_index, chunk in enumerate(chunks):
            text_unit_id = f"T{len(text_units) + 1}"
            text_units.append(
                TextUnit(
                    text_unit_id=text_unit_id,
                    document=document,
                    unit_index=unit_index,
                    text=chunk,
                )
            )
            if args.max_text_units and len(text_units) >= args.max_text_units:
                return text_units
    return text_units


def canonical_entity_name(name: str) -> str:
    cleaned = compact_whitespace(name.strip().strip("\"'`"))
    return cleaned.upper()


def normalize_entity_type(entity_type: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_ -]", "", entity_type.strip().upper())
    return compact_whitespace(cleaned.replace(" ", "_")) or "OTHER"


def unique_items(items: Iterable[str], *, max_items: int | None = None) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        cleaned = compact_whitespace(str(item))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        output.append(cleaned)
        seen.add(key)
        if max_items is not None and len(output) >= max_items:
            break
    return output


def parse_strength(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        return 1.0
    return max(1.0, min(10.0, float(match.group(0))))


def clean_tuple_field(value: str) -> str:
    return value.strip().strip("\"'").strip()


def parse_graph_tuple(part: str) -> tuple[str, list[str]] | None:
    cleaned = part.strip()
    cleaned = re.sub(r"^\s*[-*]?\s*\d*\.?\s*", "", cleaned)
    start = cleaned.find("(")
    end = cleaned.rfind(")")
    if start < 0 or end <= start:
        return None
    inner = cleaned[start + 1 : end]
    fields = [clean_tuple_field(field) for field in inner.split("<|>")]
    if not fields:
        return None
    record_type = fields[0].lower()
    return record_type, fields[1:]


def parse_extraction_response(raw_response: str) -> tuple[list[ParsedEntity], list[ParsedRelationship]]:
    text = strip_code_fence(raw_response)
    text = text.split("<|COMPLETE|>", 1)[0]
    parts = re.split(r"##|\n(?=\s*\()", text)
    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []

    for part in parts:
        parsed = parse_graph_tuple(part)
        if parsed is None:
            continue
        record_type, fields = parsed
        if record_type == "entity" and len(fields) >= 3:
            name = canonical_entity_name(fields[0])
            description = compact_whitespace(fields[2])
            if name and description:
                entities.append(
                    ParsedEntity(
                        name=name,
                        entity_type=normalize_entity_type(fields[1]),
                        description=description,
                    )
                )
        elif record_type == "relationship" and len(fields) >= 4:
            source = canonical_entity_name(fields[0])
            target = canonical_entity_name(fields[1])
            description = compact_whitespace(fields[2])
            if source and target and source != target and description:
                relationships.append(
                    ParsedRelationship(
                        source=source,
                        target=target,
                        description=description,
                        strength=parse_strength(fields[3]),
                    )
                )
    return entities, relationships


def build_extraction_prompt(text_unit: TextUnit, args: argparse.Namespace) -> str:
    return GRAPH_EXTRACTION_PROMPT.format(
        entity_types=args.entity_types,
        record_delimiter="##",
        completion_delimiter="<|COMPLETE|>",
        path=text_unit.document.rel_path,
        text_unit_id=text_unit.text_unit_id,
        text=truncate_text(text_unit.text, args.max_extract_input_chars),
    ).strip()


def ensure_entity(entities: dict[str, GraphEntity], name: str, entity_type: str = "OTHER") -> GraphEntity:
    entity = entities.get(name)
    if entity is None:
        entity = GraphEntity(name=name, entity_type=entity_type)
        entities[name] = entity
    elif entity.entity_type == "OTHER" and entity_type != "OTHER":
        entity.entity_type = entity_type
    return entity


def relationship_key(source: str, target: str) -> tuple[str, str]:
    return source, target


def build_graph_index(
    *,
    text_units: list[TextUnit],
    generator: GraphGenerator,
    args: argparse.Namespace,
) -> GraphIndex:
    prompts = [build_extraction_prompt(text_unit, args) for text_unit in text_units]
    raw_responses = generator.generate_batch(prompts, batch_size=args.graph_batch_size)
    entities: dict[str, GraphEntity] = {}
    relationships: dict[tuple[str, str], GraphRelationship] = {}
    extraction_stats: list[dict[str, Any]] = []

    for text_unit, raw_response in zip(text_units, raw_responses):
        parsed_entities, parsed_relationships = parse_extraction_response(raw_response)
        for parsed_entity in parsed_entities:
            entity = ensure_entity(entities, parsed_entity.name, parsed_entity.entity_type)
            entity.descriptions.append(parsed_entity.description)
            entity.text_unit_ids.add(text_unit.text_unit_id)
            entity.document_paths.add(text_unit.document.rel_path)

        for parsed_relationship in parsed_relationships:
            ensure_entity(entities, parsed_relationship.source)
            ensure_entity(entities, parsed_relationship.target)
            key = relationship_key(parsed_relationship.source, parsed_relationship.target)
            relationship = relationships.get(key)
            if relationship is None:
                relationship = GraphRelationship(
                    source=parsed_relationship.source,
                    target=parsed_relationship.target,
                )
                relationships[key] = relationship
            relationship.descriptions.append(parsed_relationship.description)
            relationship.strengths.append(parsed_relationship.strength)
            relationship.text_unit_ids.add(text_unit.text_unit_id)
            relationship.document_paths.add(text_unit.document.rel_path)

        stat = {
            "text_unit_id": text_unit.text_unit_id,
            "path": text_unit.document.rel_path,
            "num_entities": len(parsed_entities),
            "num_relationships": len(parsed_relationships),
        }
        if args.include_raw_graph_responses:
            stat["raw_response"] = raw_response
        extraction_stats.append(stat)

    return GraphIndex(
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        extraction_stats=extraction_stats,
    )


def merged_description(descriptions: list[str], *, max_items: int = 4, max_chars: int = 1000) -> str:
    merged = "; ".join(unique_items(descriptions, max_items=max_items))
    return truncate_text(merged, max_chars)


def build_description_summary_prompt(name: str, descriptions: list[str]) -> str:
    rendered_descriptions = "\n".join(f"- {description}" for description in unique_items(descriptions, max_items=8))
    return SUMMARIZE_DESCRIPTION_PROMPT.format(name=name, descriptions=rendered_descriptions).strip()


def summarize_graph_descriptions(
    graph: GraphIndex,
    generator: GraphGenerator,
    args: argparse.Namespace,
) -> None:
    if not args.summarize_descriptions or args.max_description_summaries == 0:
        for entity in graph.entities.values():
            entity.summary = merged_description(entity.descriptions)
        for relationship in graph.relationships.values():
            relationship.summary = merged_description(relationship.descriptions)
        return

    candidates: list[tuple[str, GraphEntity | GraphRelationship, list[str]]] = []
    for entity in graph.entities.values():
        candidates.append((entity.name, entity, entity.descriptions))
    for relationship in graph.relationships.values():
        name = f"{relationship.source} -> {relationship.target}"
        candidates.append((name, relationship, relationship.descriptions))

    candidates.sort(key=lambda item: (len(item[2]), sum(len(desc) for desc in item[2])), reverse=True)
    summarize_candidates = [
        item for item in candidates
        if len(unique_items(item[2])) > 1
    ][: args.max_description_summaries]
    prompt_by_object_id = {
        id(obj): build_description_summary_prompt(name, descriptions)
        for name, obj, descriptions in summarize_candidates
    }
    summaries = generator.generate_batch(list(prompt_by_object_id.values()), batch_size=args.graph_batch_size)
    summary_by_object_id = dict(zip(prompt_by_object_id.keys(), summaries))

    for _name, obj, descriptions in candidates:
        summary = summary_by_object_id.get(id(obj))
        if summary:
            obj.summary = compact_whitespace(strip_code_fence(summary))
        else:
            obj.summary = merged_description(descriptions)


def detect_communities(graph: GraphIndex, args: argparse.Namespace) -> list[GraphCommunity]:
    adjacency: dict[str, set[str]] = {name: set() for name in graph.entities}
    for source, target in graph.relationships:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    components: list[list[str]] = []
    seen: set[str] = set()
    for entity_name in sorted(adjacency):
        if entity_name in seen:
            continue
        queue: deque[str] = deque([entity_name])
        seen.add(entity_name)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    split_components: list[list[str]] = []
    for component in components:
        if len(component) <= args.max_community_entities:
            split_components.append(component)
            continue
        ranked = sorted(component, key=lambda name: (-len(adjacency.get(name, ())), name))
        for start in range(0, len(ranked), args.max_community_entities):
            split_components.append(ranked[start : start + args.max_community_entities])

    communities: list[GraphCommunity] = []
    for community_id, entity_names in enumerate(split_components, start=1):
        entity_set = set(entity_names)
        rel_keys = [
            key for key in graph.relationships
            if key[0] in entity_set and key[1] in entity_set
        ]
        relationship_weight = sum(graph.relationships[key].weight for key in rel_keys)
        score = relationship_weight
        score += math.log1p(len(entity_names)) * 0.5
        score += math.log1p(len(rel_keys)) * 0.5
        communities.append(
            GraphCommunity(
                community_id=community_id,
                entity_names=sorted(entity_names),
                relationship_keys=rel_keys,
                selection_score=score,
            )
        )

    communities.sort(
        key=lambda community: (
            community.selection_score,
            len(community.relationship_keys),
            len(community.entity_names),
        ),
        reverse=True,
    )
    if args.max_community_reports:
        return communities[: args.max_community_reports]
    return communities


def csv_table(headers: list[str], rows: list[list[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue().strip()


def community_source_counts(
    community: GraphCommunity,
    graph: GraphIndex,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entity_name in community.entity_names:
        counts.update(graph.entities[entity_name].document_paths)
    for relationship_key_value in community.relationship_keys:
        counts.update(graph.relationships[relationship_key_value].document_paths)
    return counts


def build_community_context(
    community: GraphCommunity,
    graph: GraphIndex,
    *,
    max_chars: int,
) -> str:
    entity_rows = []
    for entity_index, entity_name in enumerate(community.entity_names, start=1):
        entity = graph.entities[entity_name]
        entity_rows.append(
            [
                f"E{entity_index}",
                entity.name,
                entity.entity_type,
                entity.summary or merged_description(entity.descriptions),
                "; ".join(sorted(entity.document_paths)[:6]),
            ]
        )

    relationship_rows = []
    entity_id_by_name = {name: f"E{index}" for index, name in enumerate(community.entity_names, start=1)}
    for relationship_index, key in enumerate(community.relationship_keys, start=1):
        relationship = graph.relationships[key]
        relationship_rows.append(
            [
                f"R{relationship_index}",
                entity_id_by_name.get(relationship.source, relationship.source),
                entity_id_by_name.get(relationship.target, relationship.target),
                f"{relationship.weight:.2f}",
                relationship.summary or merged_description(relationship.descriptions),
                "; ".join(sorted(relationship.document_paths)[:6]),
            ]
        )

    context = (
        "Entities\n"
        + csv_table(["id", "name", "type", "description", "source_paths"], entity_rows)
        + "\n\nRelationships\n"
        + csv_table(["id", "source", "target", "weight", "description", "source_paths"], relationship_rows)
    )
    return truncate_text(context, max_chars)


def parse_jsonish(text: str) -> Any:
    stripped = strip_code_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError("response did not contain valid JSON")


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_findings(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    findings: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            summary = compact_whitespace(str(item.get("summary", "")))
            explanation = compact_whitespace(str(item.get("explanation", "")))
        else:
            summary = compact_whitespace(str(item))
            explanation = ""
        if summary or explanation:
            findings.append({"summary": summary, "explanation": explanation})
    return findings


def render_report_text(
    *,
    community_id: int,
    title: str,
    summary: str,
    rating: float | None,
    findings: list[dict[str, str]],
    source_counts: Counter[str],
) -> str:
    lines = [f"Graph community {community_id}: {title or 'Untitled community report'}"]
    if rating is not None:
        lines.append(f"rating: {rating:.2f}")
    if summary:
        lines.extend(["", "summary:", summary])
    if findings:
        lines.extend(["", "findings:"])
        for finding in findings:
            item = finding.get("summary", "")
            explanation = finding.get("explanation", "")
            if explanation:
                item = f"{item} - {explanation}" if item else explanation
            if item:
                lines.append(f"- {item}")
    if source_counts:
        lines.extend(["", "source_paths:"])
        for path, count in source_counts.most_common(12):
            lines.append(f"- {path} (graph_mentions={count})")
    return "\n".join(lines).strip()


def build_community_report(
    community: GraphCommunity,
    raw_response: str,
    graph: GraphIndex,
) -> CommunityReport:
    title = ""
    summary = ""
    rating: float | None = None
    findings: list[dict[str, str]] = []
    try:
        parsed = parse_jsonish(raw_response)
        if isinstance(parsed, dict):
            title = compact_whitespace(str(parsed.get("title", "")))
            summary = compact_whitespace(str(parsed.get("summary", "")))
            rating = safe_float(parsed.get("rating"))
            findings = normalize_findings(parsed.get("findings"))
    except Exception:
        summary = compact_whitespace(strip_code_fence(raw_response))

    if not title:
        first_entities = ", ".join(community.entity_names[:4])
        title = f"Community around {first_entities}" if first_entities else "Graph community"
    source_counts = community_source_counts(community, graph)
    text = render_report_text(
        community_id=community.community_id,
        title=title,
        summary=summary,
        rating=rating,
        findings=findings,
        source_counts=source_counts,
    )
    return CommunityReport(
        community=community,
        raw_response=raw_response,
        title=title,
        summary=summary,
        rating=rating,
        findings=findings,
        text=text,
    )


def generate_community_reports(
    *,
    graph: GraphIndex,
    communities: list[GraphCommunity],
    generator: GraphGenerator,
    args: argparse.Namespace,
) -> list[CommunityReport]:
    prompts = []
    for community in communities:
        context = build_community_context(
            community,
            graph,
            max_chars=args.max_report_input_chars,
        )
        prompts.append(
            COMMUNITY_REPORT_PROMPT.format(
                max_report_words=args.max_community_report_words,
                community_context=context,
            ).strip()
        )
    raw_reports = generator.generate_batch(prompts, batch_size=args.graph_batch_size)
    return [
        build_community_report(community, raw_response, graph)
        for community, raw_response in zip(communities, raw_reports)
    ]


def rank_reports(
    reports: list[CommunityReport],
) -> list[CommunityReport]:
    rescored = [
        CommunityReport(
            community=report.community,
            raw_response=report.raw_response,
            title=report.title,
            summary=report.summary,
            rating=report.rating,
            findings=report.findings,
            text=report.text,
            retrieval_score=report.community.selection_score,
        )
        for report in reports
    ]
    return sorted(
        rescored,
        key=lambda report: (
            report.community.selection_score,
            len(report.community.relationship_keys),
            len(report.community.entity_names),
        ),
        reverse=True,
    )


def choose_index_documents(
    documents: list[CandidateDocument],
    args: argparse.Namespace,
) -> list[CandidateDocument]:
    if FORCE_FULL_DOCUMENT_CORPUS:
        return documents
    if not args.max_index_docs or len(documents) <= args.max_index_docs:
        return documents
    return documents[: args.max_index_docs]


def rank_source_documents(
    *,
    documents: list[CandidateDocument],
    ranked_reports: list[CommunityReport],
    graph: GraphIndex,
) -> list[tuple[CandidateDocument, float, dict[str, Any]]]:
    by_path = {document.rel_path: document for document in documents}
    source_order = {document.rel_path: index for index, document in enumerate(documents, start=1)}

    graph_scores: defaultdict[str, float] = defaultdict(float)
    graph_mentions: defaultdict[str, int] = defaultdict(int)
    source_communities: defaultdict[str, list[int]] = defaultdict(list)

    for report_rank, report in enumerate(ranked_reports, start=1):
        report_counts = community_source_counts(report.community, graph)
        report_weight = max(report.retrieval_score, 0.0) / math.sqrt(report_rank)
        for path, count in report_counts.items():
            if path not in by_path:
                continue
            graph_scores[path] += report_weight * math.log1p(count)
            graph_mentions[path] += count
            if report.community.community_id not in source_communities[path]:
                source_communities[path].append(report.community.community_id)

    ranked: list[tuple[CandidateDocument, float, dict[str, Any]]] = []
    for document in documents:
        graph_component = graph_scores.get(document.rel_path, 0.0)
        metadata = {
            "graph_score": graph_component,
            "graph_mentions": graph_mentions.get(document.rel_path, 0),
            "graph_community_ids": source_communities.get(document.rel_path, []),
            "corpus_order": source_order[document.rel_path],
        }
        ranked.append((document, graph_component, metadata))

    ranked.sort(
        key=lambda item: (
            item[1],
            item[2].get("graph_mentions", 0),
            -item[2].get("corpus_order", 10**9),
        ),
        reverse=True,
    )
    return ranked


def community_report_to_record(
    report: CommunityReport,
    *,
    rank: int,
    include_text: bool,
    include_raw: bool,
    question_id: str,
    graph: GraphIndex,
) -> dict[str, Any]:
    filename = f"community_{report.community.community_id}.report.txt"
    source_counts = community_source_counts(report.community, graph)
    record: dict[str, Any] = {
        "rank": rank,
        "retrieval_rank": rank,
        "retrieval_score": report.retrieval_score,
        "path": f"{question_id}/_graph/{filename}",
        "filename": filename,
        "truncated": False,
        "graph_kind": "community_report",
        "community_id": report.community.community_id,
        "community_selection_score": report.community.selection_score,
        "community_entity_count": len(report.community.entity_names),
        "community_relationship_count": len(report.community.relationship_keys),
        "community_title": report.title,
        "community_rating": report.rating,
        "community_source_paths": [path for path, _count in source_counts.most_common(12)],
    }
    if include_text:
        record["text"] = report.text
    if include_raw:
        record["raw_graph_response"] = report.raw_response
    return record


def source_document_to_record(
    document: CandidateDocument,
    *,
    rank: int,
    score: float,
    graph_metadata: dict[str, Any],
    include_text: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rank": rank,
        "retrieval_rank": rank,
        "retrieval_score": score,
        "path": document.rel_path,
        "filename": document.path.name,
        "truncated": document.truncated,
        "graph_kind": "source_document",
    }
    record.update({key: value for key, value in document.metadata.items() if value is not None})
    record.update({key: value for key, value in graph_metadata.items() if value is not None})
    if include_text:
        record["text"] = document.text
    return record


def graph_report_summary(report: CommunityReport, graph: GraphIndex, *, include_raw: bool) -> dict[str, Any]:
    source_counts = community_source_counts(report.community, graph)
    summary: dict[str, Any] = {
        "community_id": report.community.community_id,
        "retrieval_score": report.retrieval_score,
        "selection_score": report.community.selection_score,
        "title": report.title,
        "rating": report.rating,
        "num_entities": len(report.community.entity_names),
        "num_relationships": len(report.community.relationship_keys),
        "source_paths": [path for path, _count in source_counts.most_common(12)],
    }
    if include_raw:
        summary["raw_response"] = report.raw_response
    return summary


def build_fallback_record(
    *,
    question_entry: dict[str, str],
    documents: list[CandidateDocument],
    args: argparse.Namespace,
    reason: str,
) -> dict[str, Any]:
    final_docs = documents[: args.source_context_top_k]
    return {
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "num_candidate_docs": len(documents),
        "top_k": len(final_docs),
        "graph_rag": True,
        "graph_fallback": True,
        "graph_fallback_reason": reason,
        "graph_extraction_scope": "corpus_global",
        "graph_report_scope": "corpus_global",
        "graph_ranking_scope": "question_blind",
        "graph_construction_uses_question": False,
        "gold_and_support_only": bool(args.gold_and_support_only),
        "retrieval_model": None,
        "embedding_dim": None,
        "normalized": False,
        "documents": [
            source_document_to_record(
                document,
                rank=rank,
                score=0.0,
                graph_metadata={
                    "graph_score": 0.0,
                    "graph_mentions": 0,
                    "graph_community_ids": [],
                    "corpus_order": rank,
                },
                include_text=not args.no_text,
            )
            for rank, document in enumerate(final_docs, start=1)
        ],
    }


def run_graph_retrieval_for_question(
    *,
    question_entry: dict[str, str],
    documents: list[CandidateDocument],
    generator: GraphGenerator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    enforce_full_document_corpus(args)
    if not documents:
        return build_fallback_record(
            question_entry=question_entry,
            documents=documents,
            args=args,
            reason="no_candidate_documents",
        )

    index_documents = choose_index_documents(documents, args)
    text_units = make_text_units(index_documents, args)
    if not text_units:
        return build_fallback_record(
            question_entry=question_entry,
            documents=documents,
            args=args,
            reason="no_text_units",
        )

    graph = build_graph_index(
        text_units=text_units,
        generator=generator,
        args=args,
    )
    summarize_graph_descriptions(graph, generator, args)
    if not graph.entities:
        return build_fallback_record(
            question_entry=question_entry,
            documents=documents,
            args=args,
            reason="no_extracted_entities",
        )

    communities = detect_communities(graph, args)
    reports = generate_community_reports(
        graph=graph,
        communities=communities,
        generator=generator,
        args=args,
    )
    ranked_reports = rank_reports(reports)
    ranked_sources = rank_source_documents(
        documents=documents,
        ranked_reports=ranked_reports,
        graph=graph,
    )

    final_documents: list[dict[str, Any]] = []
    rank = 1
    for report in ranked_reports[: args.report_context_top_k]:
        final_documents.append(
            community_report_to_record(
                report,
                rank=rank,
                include_text=not args.no_text,
                include_raw=args.include_raw_graph_responses,
                question_id=question_entry["question_id"],
                graph=graph,
            )
        )
        rank += 1

    for document, score, metadata in ranked_sources[: args.source_context_top_k]:
        final_documents.append(
            source_document_to_record(
                document,
                rank=rank,
                score=score,
                graph_metadata=metadata,
                include_text=not args.no_text,
            )
        )
        rank += 1

    return {
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "num_candidate_docs": len(documents),
        "num_index_docs": len(index_documents),
        "top_k": len(final_documents),
        "graph_rag": True,
        "graph_prompt_source_urls": PROMPT_SOURCE_URLS,
        "graph_extraction_scope": "corpus_global",
        "graph_report_scope": "corpus_global",
        "graph_ranking_scope": "question_blind",
        "graph_construction_uses_question": False,
        "graph_non_local": bool(args.non_local_graph),
        "graph_provider": generator.provider_name,
        "graph_model": generator.model_name,
        "graph_gpu_memory_utilization": (
            None if args.non_local_graph else effective_graph_gpu_memory_utilization(args)
        ),
        "graph_num_text_units": len(text_units),
        "graph_num_entities": len(graph.entities),
        "graph_num_relationships": len(graph.relationships),
        "graph_num_communities": len(communities),
        "graph_num_reports": len(ranked_reports),
        "graph_extraction_stats": graph.extraction_stats,
        "graph_community_reports": [
            graph_report_summary(report, graph, include_raw=args.include_raw_graph_responses)
            for report in ranked_reports
        ],
        "gold_and_support_only": bool(args.gold_and_support_only),
        "retrieval_model": None,
        "embedding_dim": None,
        "normalized": False,
        "documents": final_documents,
    }


def resolve_graph_model(args: argparse.Namespace) -> str:
    if args.graph_model_name:
        return args.graph_model_name
    if args.non_local_graph:
        return get_provider_config(args.graph_provider).default_model
    return DEFAULT_LOCAL_MODEL


def build_graph_generator(args: argparse.Namespace) -> GraphGenerator:
    graph_model = resolve_graph_model(args)
    if args.non_local_graph:
        return NonLocalGraphGenerator(
            args.graph_provider,
            graph_model,
            max_completion_tokens=args.graph_max_completion_tokens,
            reasoning_effort=resolve_reasoning_arg(args.graph_reasoning_effort),
            use_cache=args.graph_use_cache,
            max_concurrent=args.graph_max_concurrent,
        )

    return LocalGraphGenerator(
        graph_model,
        tensor_parallel_size=args.graph_tensor_parallel_size,
        max_model_len=args.graph_max_model_len,
        gpu_memory_utilization=effective_graph_gpu_memory_utilization(args),
        dtype=args.graph_dtype,
        trust_remote_code=args.graph_trust_remote_code,
        distributed_executor_backend=args.graph_distributed_executor_backend,
        disable_thinking=args.graph_disable_thinking,
        max_completion_tokens=args.graph_max_completion_tokens,
        temperature=args.graph_temperature,
        top_p=args.graph_top_p,
        top_k=args.graph_top_k,
    )


def stop_component(name: str, component: Any) -> None:
    try:
        component.stop()
    except Exception:
        logger.exception("Failed to stop %s cleanly", name)


def run_graph_retrieval(args: argparse.Namespace) -> list[dict[str, Any]]:
    enforce_full_document_corpus(args)
    questions = select_questions(
        load_heldout_questions(args.questions_path, args.privileged_dir),
        args.query_id,
        args.limit,
    )
    logger.info("Loaded %d heldout questions", len(questions))

    generator: GraphGenerator | None = None
    records: list[dict[str, Any]] = []
    try:
        if args.non_local_graph:
            logger.info("Using %s graph generator", args.graph_provider)
        else:
            logger.info(
                "Using local graph-generator vLLM GPU memory utilization=%.3f",
                effective_graph_gpu_memory_utilization(args),
            )
        generator = build_graph_generator(args)
        assert generator is not None

        for question_entry in tqdm(questions, desc="GraphRAG retrieval"):
            question_id = question_entry["question_id"]
            try:
                documents = load_candidate_documents(
                    args.docs_dir,
                    args.privileged_dir,
                    question_id,
                    args.max_doc_chars,
                    gold_and_support_only=False,
                )
                record = run_graph_retrieval_for_question(
                    question_entry=question_entry,
                    documents=documents,
                    generator=generator,
                    args=args,
                )
            except Exception as exc:
                assert generator is not None
                logger.exception("Failed GraphRAG retrieval for question %s", question_id)
                record = {
                    "question_id": question_id,
                    "question": question_entry["question"],
                    "graph_rag": True,
                    "graph_extraction_scope": "corpus_global",
                    "graph_report_scope": "corpus_global",
                    "graph_ranking_scope": "question_blind",
                    "graph_construction_uses_question": False,
                    "graph_non_local": bool(args.non_local_graph),
                    "graph_provider": generator.provider_name,
                    "graph_model": generator.model_name,
                    "graph_gpu_memory_utilization": (
                        None if args.non_local_graph else effective_graph_gpu_memory_utilization(args)
                    ),
                    "gold_and_support_only": bool(args.gold_and_support_only),
                    "error": str(exc),
                    "documents": [],
                }
            records.append(record)
    finally:
        if generator is not None:
            stop_component("graph generator", generator)
            generator = None
        release_cuda_memory()

    write_jsonl(records, args.retrieval_output_path)
    logger.info("Wrote %d GraphRAG retrieval records to %s", len(records), args.retrieval_output_path)
    ensure_no_record_errors(records, stage="GraphRAG retrieval")
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


def run_graph_rag(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retrieval_records = run_graph_retrieval(args)
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
    run_graph_rag(args)


if __name__ == "__main__":
    main()
