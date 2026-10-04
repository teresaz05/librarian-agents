from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
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
    generate_response,
    get_provider_config,
)
from experiments.inference.retrieve import (
    DEFAULT_DOCS_DIR,
    DEFAULT_PRIVILEGED_DIR,
    DEFAULT_QUESTIONS_PATH,
    CandidateDocument,
    Qwen3EmbeddingVllm,
    batches,
    ensure_no_record_errors,
    load_candidate_documents,
    load_heldout_questions,
    maybe_normalize,
    select_questions,
    write_jsonl,
)


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENT_PROMPT_PATH = Path(__file__).parent / "prompts" / "agentic_rag.md"
DEFAULT_RETRIEVAL_OUTPUT_PATH = REPO_ROOT / "outputs" / "inference" / "agentic_retrieval.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "inference" / "agentic_answers.jsonl"
DEFAULT_GOLD_AND_SUPPORT_ONLY=False
FORCE_FULL_DOCUMENT_CORPUS = True
DEFAULT_NON_LOCAL_AGENT = True
DEFAULT_NON_LOCAL_ANSWERER = True
DEFAULT_LOCAL_RETRIEVAL_GPU_MEMORY_UTILIZATION = 0.9
MIN_LOCAL_RETRIEVAL_GPU_MEMORY_UTILIZATION = 0.05
DEFAULT_LOCAL_ANSWER_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_LOCAL_AGENT_MAX_MODEL_LEN = 8192
DEFAULT_PLANNER_SEARCH_BUDGET = 1
LOW_YIELD_NEW_DOC_THRESHOLD = 1
PLANNER_SEARCH_HISTORY_LIMIT = 2
PLANNER_SNIPPET_CHAR_LIMIT = 350

@dataclass(frozen=True)
class SearchHit:
    document: CandidateDocument
    score: float
    rank: int
    query: str
    iteration: int


@dataclass
class AggregatedDocument:
    document: CandidateDocument
    best_score: float
    best_rank: int
    best_query: str
    first_seen_iteration: int
    hit_count: int = 0
    seen_queries: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentAction:
    action: str
    thought: str
    query: str | None
    document_paths: list[str]
    raw_response: str
    parse_error: str | None = None


class QuestionRetrievalIndex:
    def __init__(
        self,
        model: Qwen3EmbeddingVllm,
        documents: list[CandidateDocument],
        *,
        dim: int,
        batch_size: int,
        normalize: bool,
    ) -> None:
        self.model = model
        self.documents = documents
        self.dim = dim
        self.normalize = normalize
        self.doc_embeddings = self._embed_documents(batch_size)

    def _embed_documents(self, batch_size: int) -> torch.Tensor | None:
        if not self.documents:
            return None

        doc_embeddings: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in batches(self.documents, batch_size):
                embeddings = self.model.encode([doc.text for doc in batch], dim=self.dim)
                doc_embeddings.append(maybe_normalize(embeddings, self.normalize))
        return torch.cat(doc_embeddings, dim=0)

    def search(self, query: str, *, top_k: int, iteration: int) -> list[SearchHit]:
        if self.doc_embeddings is None or not self.documents:
            return []

        with torch.inference_mode():
            query_embedding = self.model.encode(query, is_query=True, dim=self.dim)
            query_embedding = maybe_normalize(query_embedding, self.normalize)
            scores = ((query_embedding @ self.doc_embeddings.T).squeeze(0) * 100).detach().cpu()

        k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(scores, k=k)
        return [
            SearchHit(
                document=self.documents[index],
                score=float(score),
                rank=rank,
                query=query,
                iteration=iteration,
            )
            for rank, (score, index) in enumerate(
                zip(top_scores.tolist(), top_indices.tolist()),
                start=1,
            )
        ]


class AgenticQueryPlanner:
    def __init__(
        self,
        provider: str,
        model_name: str,
        *,
        max_completion_tokens: int,
        reasoning_effort: ReasoningEffort,
        use_cache: bool,
    ) -> None:
        self.config = get_provider_config(provider)
        self.model_name = model_name or self.config.default_model
        self.client = create_async_client(self.config)
        self.provider_name = self.config.name
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.use_cache = use_cache

    def propose(self, prompt: str) -> str:
        responses = asyncio.run(
            generate_response(
                self.client,
                prompt,
                N=1,
                max_completion_tokens=self.max_completion_tokens,
                reasoning_effort=self.reasoning_effort,
                use_cache=self.use_cache,
                model_name=self.model_name,
                provider=self.config,
            )
        )
        return responses[0] if responses else ""

    def stop(self) -> None:
        try:
            asyncio.run(self.client.close())
        except Exception:
            pass


class LocalAgenticQueryPlanner:
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

    def propose(self, prompt: str) -> str:
        prompt_record = AnswerPrompt(
            messages=[{"role": "user", "content": prompt}],
            prompt_documents=[],
            prompt_token_count=0,
        )
        generation_groups = self.agent.generate([prompt_record], self.sampling_params)
        if not generation_groups or not generation_groups[0]:
            return ""
        generation = generation_groups[0][0]
        return str(generation.get("response") or generation.get("raw_response") or "")

    def stop(self) -> None:
        try:
            self.agent.stop()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run agentic query expansion RAG, then answer and judge heldout questions."
    )
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--privileged-dir", type=Path, default=DEFAULT_PRIVILEGED_DIR)
    parser.add_argument("--retrieval-output-path", type=Path, default=DEFAULT_RETRIEVAL_OUTPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)

    parser.add_argument("--agent-prompt-path", type=Path, default=DEFAULT_AGENT_PROMPT_PATH)
    parser.add_argument(
        "--non-local-agent",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_NON_LOCAL_AGENT,
        help=(
            "Use experiments.inference.llm for the query-planning agent. "
            "Pass --no-non-local-agent to run the agent with local vLLM/Qwen."
        ),
    )
    parser.add_argument("--agent-provider", default="gemini")
    parser.add_argument(
        "--agent-model-name",
        default="",
        help=(
            "Defaults to the selected provider's model for --non-local-agent, "
            f"or {DEFAULT_LOCAL_MODEL} for --no-non-local-agent."
        ),
    )
    parser.add_argument("--agent-max-completion-tokens", type=int, default=1024)
    parser.add_argument(
        "--agent-reasoning-effort",
        default="provider_default",
        help="For --non-local-agent: use 'none', 'provider_default', or a provider-supported effort value.",
    )
    parser.add_argument(
        "--agent-use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For --non-local-agent: cache query-planner responses through experiments.inference.llm.",
    )
    parser.add_argument("--agent-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--agent-distributed-executor-backend", default=None)
    parser.add_argument("--agent-max-model-len", type=int, default=DEFAULT_LOCAL_AGENT_MAX_MODEL_LEN)
    parser.add_argument(
        "--agent-gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "GPU memory fraction for local planner vLLM. Defaults to an automatic split "
            "with the embedding vLLM when --no-non-local-agent is used."
        ),
    )
    parser.add_argument("--agent-dtype", default="auto")
    parser.add_argument("--agent-trust-remote-code", action="store_true")
    parser.add_argument("--agent-temperature", type=float, default=0.0)
    parser.add_argument("--agent-top-p", type=float, default=1.0)
    parser.add_argument("--agent-top-k", type=int, default=-1)
    parser.add_argument("--agent-disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--agent-observation-docs", type=int, default=5)
    parser.add_argument("--agent-context-docs", type=int, default=12)
    parser.add_argument("--agent-max-snippet-chars", type=int, default=900)
    parser.add_argument("--agent-extended-relevance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-enforce-top-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--planner-search-budget",
        type=int,
        default=DEFAULT_PLANNER_SEARCH_BUDGET,
        help=(
            "Maximum number of agent-planned searches after the optional initial search. "
            "With --initial-search, the dense retrieval hit budget is roughly "
            "(1 + planner_search_budget) * search_top_k before choosing final_top_k documents."
        ),
    )

    parser.add_argument("--embedding-model-name-or-path", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--embedding-instruction", default=None)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--embedding-max-model-len", type=int, default=None)
    parser.add_argument(
        "--embedding-gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "GPU memory fraction for embedding vLLM. Defaults to an automatic split "
            "with the local planner vLLM when --no-non-local-agent is used."
        ),
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max-doc-chars", type=int, default=30_000)
    parser.add_argument(
        "--gold-and-support-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GOLD_AND_SUPPORT_ONLY,
        help="Retrieval always uses all candidate documents.",
    )
    parser.add_argument("--no-text", action="store_true", help="Omit document text from retrieval JSONL.")

    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument(
        "--non-local-answerer",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_NON_LOCAL_ANSWERER,
    )
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

    if args.agent_max_completion_tokens < 1:
        parser.error("--agent-max-completion-tokens must be at least 1")
    if args.agent_tensor_parallel_size is not None and args.agent_tensor_parallel_size < 1:
        parser.error("--agent-tensor-parallel-size must be at least 1")
    if not args.non_local_agent and args.agent_max_model_len <= args.agent_max_completion_tokens:
        parser.error("--agent-max-model-len must be greater than --agent-max-completion-tokens")
    for option_name in (
        "agent_gpu_memory_utilization",
        "embedding_gpu_memory_utilization",
        "gpu_memory_utilization",
    ):
        value = getattr(args, option_name)
        if value is not None and not (0.0 < value <= 1.0):
            parser.error(f"--{option_name.replace('_', '-')} must be in (0, 1]")
    if not args.non_local_agent:
        agent_gpu_memory, embedding_gpu_memory = effective_local_retrieval_gpu_memory_utilizations(args)
        if agent_gpu_memory + embedding_gpu_memory > 1.0:
            parser.error(
                "local planner and embedding vLLM GPU memory utilizations must sum to <= 1.0; "
                f"got agent={agent_gpu_memory:.3f}, embedding={embedding_gpu_memory:.3f}. "
                "Set --agent-gpu-memory-utilization and --embedding-gpu-memory-utilization explicitly."
            )
    if args.max_iterations < 0:
        parser.error("--max-iterations must be >= 0")
    if args.planner_search_budget < 0:
        parser.error("--planner-search-budget must be >= 0")
    if args.search_top_k < 1:
        parser.error("--search-top-k must be at least 1")
    if args.final_top_k < 1:
        parser.error("--final-top-k must be at least 1")
    if args.agent_observation_docs < 1:
        parser.error("--agent-observation-docs must be at least 1")
    if args.agent_context_docs < 1:
        parser.error("--agent-context-docs must be at least 1")
    if args.agent_max_snippet_chars < 0:
        parser.error("--agent-max-snippet-chars must be >= 0")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be at least 1")
    if args.max_doc_chars < 0:
        parser.error("--max-doc-chars must be >= 0")
    if FORCE_FULL_DOCUMENT_CORPUS:
        args.gold_and_support_only = False
    elif args.gold_and_support_only and args.privileged_dir is None:
        parser.error("--gold-and-support-only requires --privileged-dir")
    return args


def enforce_full_document_corpus(args: argparse.Namespace) -> None:
    if FORCE_FULL_DOCUMENT_CORPUS:
        args.gold_and_support_only = False


def effective_local_retrieval_gpu_memory_utilizations(args: argparse.Namespace) -> tuple[float, float]:
    agent_gpu_memory = args.agent_gpu_memory_utilization
    embedding_gpu_memory = args.embedding_gpu_memory_utilization

    if agent_gpu_memory is None and embedding_gpu_memory is None:
        split_gpu_memory = DEFAULT_LOCAL_RETRIEVAL_GPU_MEMORY_UTILIZATION / 2.0
        return split_gpu_memory, split_gpu_memory

    if agent_gpu_memory is None:
        if embedding_gpu_memory is None:
            raise ValueError("embedding GPU memory utilization is required to infer planner budget")
        agent_gpu_memory = max(
            MIN_LOCAL_RETRIEVAL_GPU_MEMORY_UTILIZATION,
            DEFAULT_LOCAL_RETRIEVAL_GPU_MEMORY_UTILIZATION - embedding_gpu_memory,
        )
    if embedding_gpu_memory is None:
        embedding_gpu_memory = max(
            MIN_LOCAL_RETRIEVAL_GPU_MEMORY_UTILIZATION,
            DEFAULT_LOCAL_RETRIEVAL_GPU_MEMORY_UTILIZATION - agent_gpu_memory,
        )
    return agent_gpu_memory, embedding_gpu_memory


def effective_agent_gpu_memory_utilization(args: argparse.Namespace) -> float:
    if args.non_local_agent:
        raise ValueError("agent GPU memory utilization only applies to local planners")
    agent_gpu_memory, _ = effective_local_retrieval_gpu_memory_utilizations(args)
    return agent_gpu_memory


def effective_embedding_gpu_memory_utilization(args: argparse.Namespace) -> float | None:
    if args.non_local_agent:
        return args.embedding_gpu_memory_utilization
    _, embedding_gpu_memory = effective_local_retrieval_gpu_memory_utilizations(args)
    return embedding_gpu_memory


def effective_answer_gpu_memory_utilization(args: argparse.Namespace) -> float:
    if args.gpu_memory_utilization is not None:
        return args.gpu_memory_utilization
    return DEFAULT_LOCAL_ANSWER_GPU_MEMORY_UTILIZATION


def format_gpu_memory_utilization(value: float | None) -> str:
    return "vLLM default" if value is None else f"{value:.3f}"


def release_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        logger.debug("Could not run torch.cuda.ipc_collect()", exc_info=True)


def render_agent_template(
    path: Path,
    *,
    extended_relevance: bool,
    enforce_top_k: bool,
    top_k: int,
    with_init_docs: bool,
) -> str:
    template = path.read_text(encoding="utf-8")
    context = {
        "extended_relevance": extended_relevance,
        "enforce_top_k": enforce_top_k,
        "top_k": top_k,
        "with_init_docs": with_init_docs,
    }

    try:
        from jinja2 import Template

        return Template(template).render(**context).strip()
    except Exception:
        return render_tiny_jinja(template, context).strip()


def render_tiny_jinja(template: str, context: dict[str, Any]) -> str:
    pattern = re.compile(r"{%-?\s*if\s+(\w+)\s*-?%}(.*?){%-?\s*endif\s*-?%}", flags=re.DOTALL)
    rendered = template
    while True:
        match = pattern.search(rendered)
        if match is None:
            break
        key, body = match.group(1), match.group(2)
        rendered = rendered[: match.start()] + (body if context.get(key) else "") + rendered[match.end() :]

    def replace_var(match: re.Match[str]) -> str:
        return str(context.get(match.group(1), ""))

    rendered = re.sub(r"{{\s*(\w+)\s*}}", replace_var, rendered)
    rendered = re.sub(r"{#.*?#}", "", rendered, flags=re.DOTALL)
    return rendered


def render_messages_for_planner(system_prompt: str, user_prompt: str) -> str:
    return f"SYSTEM:\n{system_prompt.strip()}\n\nUSER:\n{user_prompt.strip()}".strip()


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_title(text: str) -> str:
    for line in text.splitlines()[:12]:
        stripped = line.strip()
        if stripped.lower().startswith("title:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    text = compact_whitespace(text)
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def effective_agent_observation_docs(args: argparse.Namespace) -> int:
    return max(1, min(args.agent_observation_docs, args.search_top_k, args.final_top_k))


def effective_agent_context_docs(args: argparse.Namespace) -> int:
    return max(1, min(args.agent_context_docs, max(args.search_top_k, args.final_top_k)))


def effective_agent_snippet_chars(args: argparse.Namespace) -> int:
    if args.agent_max_snippet_chars <= 0:
        return 0
    return min(args.agent_max_snippet_chars, PLANNER_SNIPPET_CHAR_LIMIT)


def effective_search_history(
    search_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return search_steps[-PLANNER_SEARCH_HISTORY_LIMIT:]


def hit_to_observation(hit: SearchHit, *, max_snippet_chars: int) -> dict[str, Any]:
    doc = hit.document
    observation: dict[str, Any] = {
        "rank": hit.rank,
        "score": hit.score,
        "path": doc.rel_path,
        "filename": doc.path.name,
    }
    if doc.metadata.get("label") is not None:
        observation["label"] = doc.metadata["label"]
    title = extract_title(doc.text)
    if title:
        observation["title"] = title
    snippet = truncate_text(doc.text, max_snippet_chars)
    if snippet:
        observation["snippet"] = snippet
    return observation


def aggregated_to_observation(doc: AggregatedDocument, *, max_snippet_chars: int) -> dict[str, Any]:
    candidate = doc.document
    observation: dict[str, Any] = {
        "path": candidate.rel_path,
        "filename": candidate.path.name,
        "best_score": doc.best_score,
        "best_search_rank": doc.best_rank,
        "first_seen_iteration": doc.first_seen_iteration,
        "hit_count": doc.hit_count,
        "best_query": doc.best_query,
    }
    title = extract_title(candidate.text)
    if title:
        observation["title"] = title
    snippet = truncate_text(candidate.text, max_snippet_chars)
    if snippet:
        observation["snippet"] = snippet
    return observation


def build_planner_prompt(
    system_prompt: str,
    *,
    question: str,
    search_steps: list[dict[str, Any]],
    aggregate: dict[str, AggregatedDocument],
    iteration: int,
    agent_search_count: int,
    args: argparse.Namespace,
) -> str:
    snippet_chars = effective_agent_snippet_chars(args)
    ranked_candidates = rank_aggregated_documents(aggregate)[: effective_agent_context_docs(args)]
    searches_remaining = max(0, args.planner_search_budget - agent_search_count)
    user_payload = {
        "question": question,
        "iteration": iteration,
        "max_iterations": args.max_iterations,
        "search_top_k": args.search_top_k,
        "final_top_k": args.final_top_k,
        "previous_searches": effective_search_history(search_steps),
        "agent_searches_used": agent_search_count,
        "planner_search_budget": args.planner_search_budget,
        "agent_searches_remaining": searches_remaining,
        "best_candidate_documents": [
            aggregated_to_observation(doc, max_snippet_chars=snippet_chars)
            for doc in ranked_candidates
        ],
    }
    user_prompt = f"""
Use the retrieval history below to decide the next tool call.

Available tools:
1. search: retrieve documents for one new query.
2. final_results: stop and choose the final document paths.

Return only valid JSON, with one of these shapes:
{{"thought": "brief reason", "action": "search", "query": "new dense-retrieval query"}}
{{"thought": "brief reason", "action": "final_results", "document_paths": ["path/a.txt", "path/b.txt"]}}

Do not answer the question. Search queries should be specific, diverse, and useful for finding missing evidence.
If agent_searches_remaining is 0, return final_results.

PAYLOAD:
{json.dumps(user_payload, ensure_ascii=False, indent=2)}
"""
    return render_messages_for_planner(system_prompt, user_prompt)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_jsonish(text: str) -> Any:
    stripped = strip_code_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError("response did not contain valid JSON")


def normalize_document_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    paths: list[str] = []
    for item in value:
        if isinstance(item, str):
            candidate = item.strip()
        elif isinstance(item, dict):
            candidate = str(
                item.get("path")
                or item.get("document_path")
                or item.get("filename")
                or item.get("docid")
                or ""
            ).strip()
        else:
            candidate = ""
        if candidate:
            paths.append(candidate)
    return paths


def parse_agent_action(raw_response: str) -> AgentAction:
    try:
        parsed = parse_jsonish(raw_response)
        if not isinstance(parsed, dict):
            raise ValueError("top-level JSON is not an object")
        action = str(parsed.get("action", "")).strip().lower()
        if action in {"final", "finish", "final_result"}:
            action = "final_results"
        if action not in {"search", "final_results"}:
            raise ValueError(f"unsupported action: {action!r}")
        query = parsed.get("query") or parsed.get("search_query")
        document_paths = normalize_document_paths(
            parsed.get("document_paths")
            or parsed.get("documents")
            or parsed.get("results")
            or []
        )
        return AgentAction(
            action=action,
            thought=str(parsed.get("thought", parsed.get("reason", ""))).strip(),
            query=str(query).strip() if query is not None else None,
            document_paths=document_paths,
            raw_response=raw_response,
        )
    except Exception as exc:
        return AgentAction(
            action="final_results",
            thought="Could not parse planner response; falling back to ranked retrieved documents.",
            query=None,
            document_paths=[],
            raw_response=raw_response,
            parse_error=str(exc),
        )


def normalized_query(query: str) -> str:
    return compact_whitespace(query).lower()


def add_hits_to_aggregate(aggregate: dict[str, AggregatedDocument], hits: list[SearchHit]) -> int:
    num_new_documents = 0
    for hit in hits:
        key = hit.document.rel_path
        existing = aggregate.get(key)
        if existing is None:
            num_new_documents += 1
            aggregate[key] = AggregatedDocument(
                document=hit.document,
                best_score=hit.score,
                best_rank=hit.rank,
                best_query=hit.query,
                first_seen_iteration=hit.iteration,
                hit_count=1,
                seen_queries=[hit.query],
            )
            continue

        existing.hit_count += 1
        if hit.query not in existing.seen_queries:
            existing.seen_queries.append(hit.query)
        if hit.score > existing.best_score:
            existing.best_score = hit.score
            existing.best_rank = hit.rank
            existing.best_query = hit.query
    return num_new_documents


def rank_aggregated_documents(aggregate: dict[str, AggregatedDocument]) -> list[AggregatedDocument]:
    return sorted(
        aggregate.values(),
        key=lambda doc: (
            doc.best_score,
            doc.hit_count,
            -doc.best_rank,
            -doc.first_seen_iteration,
        ),
        reverse=True,
    )


def resolve_selected_documents(
    aggregate: dict[str, AggregatedDocument],
    selected_paths: list[str],
    *,
    final_top_k: int,
) -> list[AggregatedDocument]:
    by_path = {doc.document.rel_path: doc for doc in aggregate.values()}
    by_filename = {doc.document.path.name: doc for doc in aggregate.values()}
    selected: list[AggregatedDocument] = []
    seen: set[str] = set()

    for raw_path in selected_paths:
        key = raw_path.strip()
        doc = by_path.get(key) or by_filename.get(Path(key).name)
        if doc is None or doc.document.rel_path in seen:
            continue
        selected.append(doc)
        seen.add(doc.document.rel_path)
        if len(selected) >= final_top_k:
            return selected

    for doc in rank_aggregated_documents(aggregate):
        if doc.document.rel_path in seen:
            continue
        selected.append(doc)
        seen.add(doc.document.rel_path)
        if len(selected) >= final_top_k:
            break
    return selected


def aggregated_document_to_record(
    doc: AggregatedDocument,
    *,
    rank: int,
    include_text: bool,
) -> dict[str, Any]:
    candidate = doc.document
    record: dict[str, Any] = {
        "rank": rank,
        "retrieval_rank": rank,
        "retrieval_score": doc.best_score,
        "agentic_best_search_rank": doc.best_rank,
        "agentic_first_seen_iteration": doc.first_seen_iteration,
        "agentic_hit_count": doc.hit_count,
        "agentic_best_query": doc.best_query,
        "agentic_seen_queries": doc.seen_queries,
        "path": candidate.rel_path,
        "filename": candidate.path.name,
        "truncated": candidate.truncated,
    }
    record.update({key: value for key, value in candidate.metadata.items() if value is not None})
    if include_text:
        record["text"] = candidate.text
    return record


def search_step_record(
    *,
    iteration: int,
    query: str,
    hits: list[SearchHit],
    num_new_documents: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    snippet_chars = effective_agent_snippet_chars(args)
    return {
        "iteration": iteration,
        "query": query,
        "num_hits": len(hits),
        "num_new_documents": num_new_documents,
        "documents": [
            hit_to_observation(hit, max_snippet_chars=snippet_chars)
            for hit in hits[: effective_agent_observation_docs(args)]
        ],
    }


QueryPlanner = AgenticQueryPlanner | LocalAgenticQueryPlanner


def resolve_agent_model(args: argparse.Namespace) -> str:
    if args.agent_model_name:
        return args.agent_model_name
    if args.non_local_agent:
        return get_provider_config(args.agent_provider).default_model
    return DEFAULT_LOCAL_MODEL


def build_agentic_query_planner(args: argparse.Namespace) -> QueryPlanner:
    agent_model = resolve_agent_model(args)
    if args.non_local_agent:
        return AgenticQueryPlanner(
            args.agent_provider,
            agent_model,
            max_completion_tokens=args.agent_max_completion_tokens,
            reasoning_effort=resolve_reasoning_arg(args.agent_reasoning_effort),
            use_cache=args.agent_use_cache,
        )

    return LocalAgenticQueryPlanner(
        agent_model,
        tensor_parallel_size=args.agent_tensor_parallel_size,
        max_model_len=args.agent_max_model_len,
        gpu_memory_utilization=effective_agent_gpu_memory_utilization(args),
        dtype=args.agent_dtype,
        trust_remote_code=args.agent_trust_remote_code,
        distributed_executor_backend=args.agent_distributed_executor_backend,
        disable_thinking=args.agent_disable_thinking,
        max_completion_tokens=args.agent_max_completion_tokens,
        temperature=args.agent_temperature,
        top_p=args.agent_top_p,
        top_k=args.agent_top_k,
    )


def stop_component(name: str, component: Any) -> None:
    try:
        component.stop()
    except Exception:
        logger.exception("Failed to stop %s cleanly", name)


def run_agentic_retrieval_for_question(
    *,
    question_entry: dict[str, str],
    documents: list[CandidateDocument],
    embedding_model: Qwen3EmbeddingVllm,
    planner: QueryPlanner,
    system_prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    enforce_full_document_corpus(args)
    question = question_entry["question"]
    index = QuestionRetrievalIndex(
        embedding_model,
        documents,
        dim=args.embedding_dim,
        batch_size=args.embedding_batch_size,
        normalize=args.normalize,
    )
    aggregate: dict[str, AggregatedDocument] = {}
    search_steps: list[dict[str, Any]] = []
    planner_steps: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    selected_paths: list[str] = []
    stop_reason = "max_iterations"
    agent_search_count = 0

    if args.initial_search:
        hits = index.search(question, top_k=args.search_top_k, iteration=0)
        num_new_documents = add_hits_to_aggregate(aggregate, hits)
        search_steps.append(
            search_step_record(
                iteration=0,
                query=question,
                hits=hits,
                num_new_documents=num_new_documents,
                args=args,
            )
        )
        seen_queries.add(normalized_query(question))

    for iteration in range(1, args.max_iterations + 1):
        prompt = build_planner_prompt(
            system_prompt,
            question=question,
            search_steps=search_steps,
            aggregate=aggregate,
            iteration=iteration,
            agent_search_count=agent_search_count,
            args=args,
        )
        raw_response = planner.propose(prompt)
        action = parse_agent_action(raw_response)
        planner_record: dict[str, Any] = {
            "iteration": iteration,
            "action": action.action,
            "thought": action.thought,
            "raw_response": action.raw_response,
        }
        if action.parse_error:
            planner_record["parse_error"] = action.parse_error

        if action.action == "final_results":
            selected_paths = action.document_paths
            planner_record["document_paths"] = selected_paths
            planner_steps.append(planner_record)
            stop_reason = "final_results"
            break

        query = (action.query or "").strip()
        planner_record["query"] = query
        if agent_search_count >= args.planner_search_budget:
            planner_record["search_budget_exhausted"] = True
            planner_steps.append(planner_record)
            stop_reason = "search_budget_exhausted"
            break

        normalized = normalized_query(query)
        if not query:
            planner_record["parse_error"] = "Planner requested search without a query"
            planner_steps.append(planner_record)
            stop_reason = "empty_query"
            break
        if normalized in seen_queries:
            planner_record["duplicate_query"] = True
            planner_steps.append(planner_record)
            stop_reason = "duplicate_query"
            break

        hits = index.search(query, top_k=args.search_top_k, iteration=iteration)
        num_new_documents = add_hits_to_aggregate(aggregate, hits)
        agent_search_count += 1
        planner_record["num_new_documents"] = num_new_documents
        seen_queries.add(normalized)
        search_steps.append(
            search_step_record(
                iteration=iteration,
                query=query,
                hits=hits,
                num_new_documents=num_new_documents,
                args=args,
            )
        )
        planner_steps.append(planner_record)
        if (
            LOW_YIELD_NEW_DOC_THRESHOLD > 0
            and num_new_documents < LOW_YIELD_NEW_DOC_THRESHOLD
        ):
            planner_record["low_yield_search"] = True

    final_documents = resolve_selected_documents(
        aggregate,
        selected_paths,
        final_top_k=args.final_top_k,
    )
    queries = [step["query"] for step in search_steps]
    return {
        "question_id": question_entry["question_id"],
        "question": question,
        "num_candidate_docs": len(documents),
        "top_k": args.final_top_k,
        "retrieval_top_k": args.search_top_k,
        "agentic": True,
        "agent_non_local": bool(args.non_local_agent),
        "agent_provider": planner.provider_name,
        "agent_model": planner.model_name,
        "agent_gpu_memory_utilization": (
            None if args.non_local_agent else effective_agent_gpu_memory_utilization(args)
        ),
        "agent_max_iterations": args.max_iterations,
        "agent_planner_search_budget": args.planner_search_budget,
        "agent_initial_search": bool(args.initial_search),
        "agent_retrieval_hit_budget": (
            (1 if args.initial_search else 0) + args.planner_search_budget
        )
        * args.search_top_k,
        "agent_num_searches": len(search_steps),
        "agent_num_planned_searches": agent_search_count,
        "agent_num_candidate_documents_seen": len(aggregate),
        "agent_low_yield_new_doc_threshold": LOW_YIELD_NEW_DOC_THRESHOLD,
        "agent_num_iterations": len(planner_steps),
        "agent_stop_reason": stop_reason,
        "agent_queries": queries,
        "agent_planner_steps": planner_steps,
        "agent_search_steps": search_steps,
        "gold_and_support_only": bool(args.gold_and_support_only),
        "retrieval_model": args.embedding_model_name_or_path,
        "embedding_gpu_memory_utilization": effective_embedding_gpu_memory_utilization(args),
        "embedding_dim": args.embedding_dim,
        "normalized": bool(args.normalize),
        "documents": [
            aggregated_document_to_record(doc, rank=rank, include_text=not args.no_text)
            for rank, doc in enumerate(final_documents, start=1)
        ],
    }


def run_agentic_retrieval(args: argparse.Namespace) -> list[dict[str, Any]]:
    enforce_full_document_corpus(args)
    questions = select_questions(
        load_heldout_questions(args.questions_path, args.privileged_dir),
        args.query_id,
        args.limit,
    )
    logger.info("Loaded %d heldout questions", len(questions))

    system_prompt = render_agent_template(
        args.agent_prompt_path,
        extended_relevance=args.agent_extended_relevance,
        enforce_top_k=args.agent_enforce_top_k,
        top_k=args.final_top_k,
        with_init_docs=args.initial_search,
    )
    planner: QueryPlanner | None = None
    embedding_model: Qwen3EmbeddingVllm | None = None
    records: list[dict[str, Any]] = []
    try:
        if args.non_local_agent:
            logger.info(
                "Using %s planner and embedding vLLM GPU memory utilization=%s",
                args.agent_provider,
                format_gpu_memory_utilization(effective_embedding_gpu_memory_utilization(args)),
            )
        else:
            agent_gpu_memory, embedding_gpu_memory = effective_local_retrieval_gpu_memory_utilizations(args)
            logger.info(
                "Using local retrieval vLLM GPU memory utilizations: planner=%.3f, embedding=%.3f",
                agent_gpu_memory,
                embedding_gpu_memory,
            )
        planner = build_agentic_query_planner(args)
        embedding_model = Qwen3EmbeddingVllm(
            args.embedding_model_name_or_path,
            instruction=args.embedding_instruction,
            tensor_parallel_size=args.embedding_tensor_parallel_size,
            max_model_len=args.embedding_max_model_len,
            gpu_memory_utilization=effective_embedding_gpu_memory_utilization(args),
        )
        assert planner is not None
        assert embedding_model is not None

        for question_entry in tqdm(questions, desc="Agentic retrieval"):
            question_id = question_entry["question_id"]
            try:
                documents = load_candidate_documents(
                    args.docs_dir,
                    args.privileged_dir,
                    question_id,
                    args.max_doc_chars,
                    gold_and_support_only=False,
                )
                record = run_agentic_retrieval_for_question(
                    question_entry=question_entry,
                    documents=documents,
                    embedding_model=embedding_model,
                    planner=planner,
                    system_prompt=system_prompt,
                    args=args,
                )
            except Exception as exc:
                assert planner is not None
                logger.exception("Failed agentic retrieval for question %s", question_id)
                record = {
                    "question_id": question_id,
                    "question": question_entry["question"],
                    "agentic": True,
                    "agent_non_local": bool(args.non_local_agent),
                    "agent_provider": planner.provider_name,
                    "agent_model": planner.model_name,
                    "agent_gpu_memory_utilization": (
                        None if args.non_local_agent else effective_agent_gpu_memory_utilization(args)
                    ),
                    "agent_planner_search_budget": args.planner_search_budget,
                    "agent_initial_search": bool(args.initial_search),
                    "agent_retrieval_hit_budget": (
                        (1 if args.initial_search else 0) + args.planner_search_budget
                    )
                    * args.search_top_k,
                    "gold_and_support_only": bool(args.gold_and_support_only),
                    "error": str(exc),
                    "retrieval_model": args.embedding_model_name_or_path,
                    "embedding_gpu_memory_utilization": effective_embedding_gpu_memory_utilization(args),
                    "documents": [],
                }
            records.append(record)
    finally:
        if planner is not None:
            stop_component("agent planner", planner)
            planner = None
        if embedding_model is not None:
            stop_component("embedding model", embedding_model)
            embedding_model = None
        release_cuda_memory()

    write_jsonl(records, args.retrieval_output_path)
    logger.info("Wrote %d agentic retrieval records to %s", len(records), args.retrieval_output_path)
    ensure_no_record_errors(records, stage="Agentic retrieval")
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


def run_agentic_rag(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retrieval_records = run_agentic_retrieval(args)
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
    run_agentic_rag(args)


if __name__ == "__main__":
    main()
