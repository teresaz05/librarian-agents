from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    import torch


PASSAGE_CACHE_VERSION = 1
DEFAULT_CHUNK_SIZE = 768
DEFAULT_CHUNK_OVERLAP = 96
DEFAULT_CANDIDATE_DEPTH = 100
DEFAULT_TOP_K = 12
DEFAULT_MAX_PER_SOURCE = 2
TOKEN_RE = re.compile(r"(?u)\b\w+\b")


@dataclass(frozen=True)
class Passage:
    question_id: str
    chunk_id: str
    source_path: str
    filename: str
    text: str
    token_start: int
    token_end: int
    token_count: int
    source_sha256: str


@dataclass(frozen=True)
class PassageHit:
    passage: Passage
    score: float
    rank: int
    component_scores: Mapping[str, float] = field(default_factory=dict)
    component_ranks: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PassageIndex:
    passages: tuple[Passage, ...]
    source_fingerprint: str
    config_fingerprint: str
    cache_hit: bool


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_fingerprint(documents: Sequence[Any]) -> str:
    """Fingerprint full source contents without privileged document metadata."""
    entries = []
    for document in sorted(documents, key=lambda item: str(item.rel_path)):
        text = str(document.text)
        entries.append(
            {
                "path": str(document.rel_path),
                "sha256": _text_sha256(text),
                "utf8_bytes": len(text.encode("utf-8")),
            }
        )
    return _canonical_sha256(entries)


def passage_config_fingerprint(
    *,
    tokenizer_name: str,
    tokenizer_revision: str | None,
    chunk_size: int,
    overlap: int,
) -> str:
    _validate_chunk_config(chunk_size, overlap)
    return _canonical_sha256(
        {
            "cache_version": PASSAGE_CACHE_VERSION,
            "tokenizer": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
    )


def _validate_chunk_config(chunk_size: int, overlap: int) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def _flat_int_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _offset_list(value: Any, expected_length: int) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
        value = value[0]
    try:
        offsets = [(int(start), int(end)) for start, end in value]
    except (TypeError, ValueError):
        return None
    if len(offsets) != expected_length or any(end < start for start, end in offsets):
        return None
    return offsets


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        token_ids = _flat_int_list(encoded["input_ids"])
        offsets = _offset_list(encoded.get("offset_mapping"), len(token_ids))
        return token_ids, offsets
    except (KeyError, NotImplementedError, TypeError, ValueError):
        token_ids = _flat_int_list(tokenizer.encode(text, add_special_tokens=False))
        return token_ids, None


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    try:
        return str(tokenizer.decode(list(token_ids), **kwargs))
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        return str(tokenizer.decode(list(token_ids), **kwargs))


def _chunk_text(
    text: str,
    token_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]] | None,
    start: int,
    end: int,
    tokenizer: Any,
) -> str:
    if offsets is None:
        return _decode_tokens(tokenizer, token_ids[start:end])

    nonempty_offsets = [(left, right) for left, right in offsets[start:end] if right > left]
    if not nonempty_offsets:
        return _decode_tokens(tokenizer, token_ids[start:end])
    char_start = 0 if start == 0 else nonempty_offsets[0][0]
    char_end = len(text) if end == len(token_ids) else nonempty_offsets[-1][1]
    return text[char_start:char_end]


def chunk_document(
    document: Any,
    tokenizer: Any,
    *,
    tokenizer_name: str,
    tokenizer_revision: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Passage]:
    """Split one complete source document into stable, overlapping token passages."""
    _validate_chunk_config(chunk_size, overlap)
    text = str(document.text)
    if not text:
        return []

    source_path = str(document.rel_path)
    question_id = str(document.question_id)
    filename = Path(source_path).name
    document_sha256 = _text_sha256(text)
    config_sha256 = passage_config_fingerprint(
        tokenizer_name=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    token_ids, offsets = _tokenize_with_offsets(tokenizer, text)
    if not token_ids:
        chunk_id = _canonical_sha256(
            {
                "source_path": source_path,
                "source_sha256": document_sha256,
                "config_fingerprint": config_sha256,
                "token_start": 0,
                "token_end": 0,
            }
        )
        return [
            Passage(
                question_id=question_id,
                chunk_id=chunk_id,
                source_path=source_path,
                filename=filename,
                text=text,
                token_start=0,
                token_end=0,
                token_count=0,
                source_sha256=document_sha256,
            )
        ]

    stride = chunk_size - overlap
    passages: list[Passage] = []
    for start in range(0, len(token_ids), stride):
        end = min(start + chunk_size, len(token_ids))
        chunk_id = _canonical_sha256(
            {
                "source_path": source_path,
                "source_sha256": document_sha256,
                "config_fingerprint": config_sha256,
                "token_start": start,
                "token_end": end,
            }
        )
        passages.append(
            Passage(
                question_id=question_id,
                chunk_id=chunk_id,
                source_path=source_path,
                filename=filename,
                text=_chunk_text(text, token_ids, offsets, start, end, tokenizer),
                token_start=start,
                token_end=end,
                token_count=end - start,
                source_sha256=document_sha256,
            )
        )
        if end == len(token_ids):
            break
    return passages


def _cache_file(
    cache_dir: Path,
    documents: Sequence[Any],
    source_sha256: str,
    config_sha256: str,
) -> Path:
    question_ids = sorted({str(document.question_id) for document in documents})
    question_key = _canonical_sha256(question_ids)[:16]
    return cache_dir / question_key / f"{source_sha256}.{config_sha256}.json"


def _load_cached_index(
    path: Path,
    *,
    source_sha256: str,
    config_sha256: str,
) -> PassageIndex | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("cache_version") != PASSAGE_CACHE_VERSION:
            return None
        if payload.get("source_fingerprint") != source_sha256:
            return None
        if payload.get("config_fingerprint") != config_sha256:
            return None
        passages = tuple(Passage(**item) for item in payload["passages"])
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return PassageIndex(passages, source_sha256, config_sha256, True)


def _write_cached_index(path: Path, index: PassageIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": PASSAGE_CACHE_VERSION,
        "source_fingerprint": index.source_fingerprint,
        "config_fingerprint": index.config_fingerprint,
        "passages": [asdict(passage) for passage in index.passages],
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def load_or_build_passage_index(
    documents: Sequence[Any],
    tokenizer: Any,
    *,
    cache_dir: Path | None,
    tokenizer_name: str,
    tokenizer_revision: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> PassageIndex:
    """Load a content-addressed passage index or build it from full source documents."""
    if tokenizer_revision is None:
        tokenizer_revision = getattr(tokenizer, "_commit_hash", None)
    if tokenizer_revision is None:
        tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    ordered_documents = sorted(documents, key=lambda item: str(item.rel_path))
    source_sha256 = source_fingerprint(ordered_documents)
    config_sha256 = passage_config_fingerprint(
        tokenizer_name=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    cache_path = None
    if cache_dir is not None:
        cache_path = _cache_file(cache_dir, ordered_documents, source_sha256, config_sha256)
        cached = _load_cached_index(
            cache_path,
            source_sha256=source_sha256,
            config_sha256=config_sha256,
        )
        if cached is not None:
            return cached

    passages = tuple(
        passage
        for document in ordered_documents
        for passage in chunk_document(
            document,
            tokenizer,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    )
    index = PassageIndex(passages, source_sha256, config_sha256, False)
    if cache_path is not None:
        _write_cached_index(cache_path, index)
    return index


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as torch_functional

    return torch_functional.normalize(tensor, p=2, dim=1)


class DensePassageIndex:
    """Embed passages once, then serve any number of dense queries against them."""

    def __init__(
        self,
        model: Any,
        passages: Sequence[Passage],
        *,
        dim: int = 1024,
        batch_size: int = 128,
        normalize: bool = True,
    ) -> None:
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model = model
        self.passages = tuple(passages)
        self.dim = dim
        self.batch_size = batch_size
        self.normalize = normalize
        embeddings: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(self.passages), batch_size):
                batch = self.passages[start : start + batch_size]
                values = model.encode([passage.text for passage in batch], dim=dim)
                embeddings.append(_normalize_rows(values) if normalize else values)
        self._embeddings = torch.cat(embeddings, dim=0) if embeddings else None

    def scores(self, query: str) -> list[float]:
        import torch

        if self._embeddings is None:
            return []
        with torch.inference_mode():
            query_embedding = self.model.encode(query, is_query=True, dim=self.dim)
            if self.normalize:
                query_embedding = _normalize_rows(query_embedding)
            scores = (query_embedding @ self._embeddings.T).squeeze(0).detach().cpu()
        return [float(score) for score in scores.tolist()]

    def search(self, query: str, top_k: int = DEFAULT_CANDIDATE_DEPTH) -> list[PassageHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        return rank_passages(self.passages, self.scores(query), component="dense")[:top_k]


def compute_dense_scores(
    model: Any,
    query: str,
    passages: Sequence[Passage],
    *,
    dim: int = 1024,
    batch_size: int = 128,
    normalize: bool = True,
) -> list[float]:
    return DensePassageIndex(
        model,
        passages,
        dim=dim,
        batch_size=batch_size,
        normalize=normalize,
    ).scores(query)


def tokenize_for_bm25(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def compute_bm25_scores(
    query: str,
    passages: Sequence[Passage],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if k1 <= 0:
        raise ValueError("k1 must be positive")
    if not 0 <= b <= 1:
        raise ValueError("b must be between 0 and 1")
    if not passages:
        return []

    tokenized_passages = [tokenize_for_bm25(passage.text) for passage in passages]
    query_terms = Counter(tokenize_for_bm25(query))
    if not query_terms:
        return [0.0] * len(passages)

    term_frequencies = [Counter(tokens) for tokens in tokenized_passages]
    document_frequencies: Counter[str] = Counter()
    for counts in term_frequencies:
        document_frequencies.update(counts.keys())
    lengths = [len(tokens) for tokens in tokenized_passages]
    average_length = sum(lengths) / len(lengths)

    scores: list[float] = []
    for counts, length in zip(term_frequencies, lengths):
        length_norm = k1 * (1 - b + b * length / average_length) if average_length else k1
        score = 0.0
        for term, query_frequency in query_terms.items():
            term_frequency = counts.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = document_frequencies[term]
            idf = math.log(
                1
                + (len(passages) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            score += (
                query_frequency
                * idf
                * term_frequency
                * (k1 + 1)
                / (term_frequency + length_norm)
            )
        scores.append(float(score))
    return scores


def _passage_tie_key(passage: Passage) -> tuple[str, int, int, str]:
    return passage.source_path, passage.token_start, passage.token_end, passage.chunk_id


def rank_passages(
    passages: Sequence[Passage],
    scores: Sequence[float],
    *,
    component: str = "retrieval",
) -> list[PassageHit]:
    if len(passages) != len(scores):
        raise ValueError("passages and scores must have equal lengths")
    if not component:
        raise ValueError("component must not be empty")
    order = sorted(
        range(len(passages)),
        key=lambda index: (-float(scores[index]), _passage_tie_key(passages[index])),
    )
    return [
        PassageHit(
            passage=passages[index],
            score=float(scores[index]),
            rank=rank,
            component_scores={component: float(scores[index])},
            component_ranks={component: rank},
        )
        for rank, index in enumerate(order, start=1)
    ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[PassageHit]],
    *,
    rrf_k: float = 60.0,
    depth: int = DEFAULT_CANDIDATE_DEPTH,
    weights: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
) -> list[PassageHit]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    if names is not None and len(names) != len(rankings):
        raise ValueError("names must match rankings")

    passages_by_id: dict[str, Passage] = {}
    fused_scores: defaultdict[str, float] = defaultdict(float)
    component_scores: defaultdict[str, dict[str, float]] = defaultdict(dict)
    component_ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for ranking_index, (ranking, weight) in enumerate(zip(rankings, weights)):
        component_name = names[ranking_index] if names is not None else f"ranking_{ranking_index}"
        seen: set[str] = set()
        for position, hit in enumerate(ranking[:depth], start=1):
            chunk_id = hit.passage.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            passages_by_id[chunk_id] = hit.passage
            fused_scores[chunk_id] += float(weight) / (rrf_k + position)
            component_scores[chunk_id][component_name] = float(hit.score)
            component_ranks[chunk_id][component_name] = position

    ordered_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            _passage_tie_key(passages_by_id[chunk_id]),
        ),
    )
    return [
        PassageHit(
            passage=passages_by_id[chunk_id],
            score=fused_scores[chunk_id],
            rank=rank,
            component_scores=component_scores[chunk_id],
            component_ranks=component_ranks[chunk_id],
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]


def source_diverse_top_k(
    hits: Iterable[PassageHit],
    *,
    top_k: int = DEFAULT_TOP_K,
    max_per_source: int = DEFAULT_MAX_PER_SOURCE,
) -> list[PassageHit]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if max_per_source < 1:
        raise ValueError("max_per_source must be at least 1")
    counts: Counter[str] = Counter()
    selected: list[PassageHit] = []
    for hit in hits:
        source_path = hit.passage.source_path
        if counts[source_path] >= max_per_source:
            continue
        counts[source_path] += 1
        selected.append(
            PassageHit(
                passage=hit.passage,
                score=hit.score,
                rank=len(selected) + 1,
                component_scores=dict(hit.component_scores),
                component_ranks=dict(hit.component_ranks),
            )
        )
        if len(selected) == top_k:
            break
    return selected


def passage_hit_to_record(hit: PassageHit, *, include_text: bool = True) -> dict[str, Any]:
    passage = hit.passage
    record: dict[str, Any] = {
        "rank": hit.rank,
        "retrieval_rank": hit.rank,
        "retrieval_score": hit.score,
        "path": passage.source_path,
        "source_path": passage.source_path,
        "filename": passage.filename,
        "chunk_id": passage.chunk_id,
        "token_start": passage.token_start,
        "token_end": passage.token_end,
        "token_count": passage.token_count,
        "source_sha256": passage.source_sha256,
        "truncated": False,
    }
    for component, score in sorted(hit.component_scores.items()):
        record[f"{component}_score"] = score
    for component, rank in sorted(hit.component_ranks.items()):
        record[f"{component}_rank"] = rank
    if include_text:
        record["text"] = passage.text
    return record
