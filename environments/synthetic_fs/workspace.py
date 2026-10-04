from __future__ import annotations

import asyncio
import difflib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from tinker_cookbook.tool_use import simple_tool_result, tool
from tinker_cookbook.tool_use.types import ToolResult
from tinker_cookbook.utils import trace

from environments.synthetic_fs.clients import call_chat_model_messages
from environments.synthetic_fs.common import (
    BAD_SYNTHETIC_TEXT_VALUES,
    READ_FILE_MAX_CHARS,
    READ_FILE_MAX_LINES,
    READ_MANY_MAX_FILES,
    SYNTHETIC_PATH_SLUG_MAX_BYTES,
    AnswererWorkspaceMode,
    JudgeBackend,
    NodeKind,
)


def _format_line_window_payload(
    *,
    path: str,
    start_line: int,
    chunk_lines: list[str],
    total_lines: int,
    max_chars: int = READ_FILE_MAX_CHARS,
) -> dict[str, Any]:
    content = "\n".join(chunk_lines)
    truncated_by_chars = False
    if len(content) > max_chars:
        content = content[:max_chars].rstrip()
        truncated_by_chars = True
    end_line = start_line + len(chunk_lines) - 1 if chunk_lines else start_line - 1
    return {
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
        "has_more": end_line < total_lines or truncated_by_chars,
        "truncated_by_chars": truncated_by_chars,
        "content": content,
    }


def _read_text_line_window(
    *,
    path: str,
    text: str,
    start_line: int,
    num_lines: int,
    max_chars: int = READ_FILE_MAX_CHARS,
) -> dict[str, Any]:
    all_lines = text.splitlines()
    start_index = start_line - 1
    chunk_lines = all_lines[start_index : start_index + num_lines]
    return _format_line_window_payload(
        path=path,
        start_line=start_line,
        chunk_lines=chunk_lines,
        total_lines=len(all_lines),
        max_chars=max_chars,
    )


def _read_file_line_window(
    *,
    path: str,
    file_path: Path,
    start_line: int,
    num_lines: int,
    max_chars: int = READ_FILE_MAX_CHARS,
) -> dict[str, Any]:
    chunk_lines: list[str] = []
    start_index = start_line - 1
    stop_index = start_index + num_lines
    total_lines = 0
    with file_path.open(encoding="utf-8") as handle:
        for total_lines, line in enumerate(handle, start=1):
            line_index = total_lines - 1
            if start_index <= line_index < stop_index:
                chunk_lines.append(line.rstrip("\r\n"))
    return _format_line_window_payload(
        path=path,
        start_line=start_line,
        chunk_lines=chunk_lines,
        total_lines=total_lines,
        max_chars=max_chars,
    )


class SyntheticFilesystemDatum(TypedDict):
    question_id: str
    question: str
    gold_answer: str
    agent_query_dir: str
    privileged_query_dir: str
    num_docs: int
    dataset_type: str
    files: list[dict[str, Any]]
    construction_budget_active_files: NotRequired[int]
    construction_budget_reference_builder_turns: NotRequired[int]
    construction_budget_reference_builder_tokens: NotRequired[int]
    amortization_questions: NotRequired[list[dict[str, Any]]]
    num_questions: NotRequired[int]
    domain: NotRequired[str]
    cluster_id: NotRequired[str]


@dataclass
class SyntheticFileRecord:
    node_id: str
    kind: NodeKind
    title: str
    slug: str
    content: str
    created_round: int
    parent_id: str | None = None
    direct_raw_doc_paths: list[str] = field(default_factory=list)
    direct_synthetic_ids: list[str] = field(default_factory=list)
    active: bool = True
    deleted_round: int | None = None
    delete_reason: str | None = None

    def to_snapshot(self) -> dict[str, Any]:
        """Return the JSON-safe representation used by reusable filesystem snapshots."""
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "title": self.title,
            "slug": self.slug,
            "content": self.content,
            "created_round": self.created_round,
            "parent_id": self.parent_id,
            "direct_raw_doc_paths": list(self.direct_raw_doc_paths),
            "direct_synthetic_ids": list(self.direct_synthetic_ids),
            "active": self.active,
            "deleted_round": self.deleted_round,
            "delete_reason": self.delete_reason,
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> SyntheticFileRecord:
        required = {
            "node_id",
            "kind",
            "title",
            "slug",
            "content",
            "created_round",
            "parent_id",
            "direct_raw_doc_paths",
            "direct_synthetic_ids",
            "active",
            "deleted_round",
            "delete_reason",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Snapshot file record is missing fields: {missing}")
        kind = str(payload["kind"])
        if kind not in {"cluster", "merge_summary"}:
            raise ValueError(f"Invalid snapshot node kind: {kind!r}")
        raw_paths = payload["direct_raw_doc_paths"]
        synthetic_ids = payload["direct_synthetic_ids"]
        if not isinstance(raw_paths, list) or not all(isinstance(path, str) for path in raw_paths):
            raise ValueError("Snapshot direct_raw_doc_paths must be a list of strings")
        if not isinstance(synthetic_ids, list) or not all(
            isinstance(node_id, str) for node_id in synthetic_ids
        ):
            raise ValueError("Snapshot direct_synthetic_ids must be a list of strings")
        return cls(
            node_id=str(payload["node_id"]),
            kind=kind,  # type: ignore[arg-type]
            title=str(payload["title"]),
            slug=str(payload["slug"]),
            content=str(payload["content"]),
            created_round=int(payload["created_round"]),
            parent_id=(str(payload["parent_id"]) if payload["parent_id"] is not None else None),
            direct_raw_doc_paths=list(raw_paths),
            direct_synthetic_ids=list(synthetic_ids),
            active=bool(payload["active"]),
            deleted_round=(
                int(payload["deleted_round"])
                if payload["deleted_round"] is not None
                else None
            ),
            delete_reason=(
                str(payload["delete_reason"])
                if payload["delete_reason"] is not None
                else None
            ),
        )


@dataclass
class SyntheticFilesystemState:
    raw_doc_root: Path
    visible_raw_doc_paths: list[str]
    question_id: str
    raw_read_max_chars: int = READ_FILE_MAX_CHARS
    files_by_id: dict[str, SyntheticFileRecord] = field(default_factory=dict)
    next_cluster_idx: int = 1
    next_merge_idx: int = 1
    operation_round: int = 0

    SNAPSHOT_SCHEMA_VERSION = 1

    @staticmethod
    def _validate_snapshot_raw_path(path: str) -> str:
        clean = path.replace("\\", "/").strip().lstrip("/")
        parts = PurePosixPath(clean).parts
        if not clean or clean != path or ".." in parts or PurePosixPath(clean).is_absolute():
            raise ValueError(f"Invalid raw-document path in snapshot: {path!r}")
        return clean

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize this filesystem without copying the source document corpus.

        The caller supplies ``raw_doc_root`` again when loading. Keeping source text
        out of the snapshot makes one constructed filesystem reusable across many
        questions while keeping source documents separate from result artifacts.
        """
        return {
            "schema_version": self.SNAPSHOT_SCHEMA_VERSION,
            "question_id": self.question_id,
            "visible_raw_doc_paths": list(self.visible_raw_doc_paths),
            "next_cluster_idx": self.next_cluster_idx,
            "next_merge_idx": self.next_merge_idx,
            "operation_round": self.operation_round,
            "files": [
                self.files_by_id[node_id].to_snapshot()
                for node_id in sorted(self.files_by_id)
            ],
        }

    @classmethod
    def from_snapshot(
        cls,
        payload: dict[str, Any],
        *,
        raw_doc_root: Path,
    ) -> SyntheticFilesystemState:
        if payload.get("schema_version") != cls.SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported synthetic filesystem snapshot schema version: "
                f"{payload.get('schema_version')!r}"
            )
        raw_paths = payload.get("visible_raw_doc_paths")
        files = payload.get("files")
        if not isinstance(raw_paths, list) or not all(isinstance(path, str) for path in raw_paths):
            raise ValueError("Snapshot visible_raw_doc_paths must be a list of strings")
        if not isinstance(files, list) or not all(isinstance(record, dict) for record in files):
            raise ValueError("Snapshot files must be a list of objects")
        normalized_raw_paths = [cls._validate_snapshot_raw_path(path) for path in raw_paths]
        if len(normalized_raw_paths) != len(set(normalized_raw_paths)):
            raise ValueError("Snapshot visible_raw_doc_paths contains duplicates")
        state = cls(
            raw_doc_root=raw_doc_root.resolve(),
            visible_raw_doc_paths=normalized_raw_paths,
            question_id=str(payload.get("question_id", "")),
            next_cluster_idx=int(payload.get("next_cluster_idx", 1)),
            next_merge_idx=int(payload.get("next_merge_idx", 1)),
            operation_round=int(payload.get("operation_round", 0)),
        )
        if not state.question_id:
            raise ValueError("Snapshot question_id must be non-empty")
        if min(state.next_cluster_idx, state.next_merge_idx) < 1 or state.operation_round < 0:
            raise ValueError("Snapshot counters must be non-negative and start at one")
        for record_payload in files:
            record = SyntheticFileRecord.from_snapshot(record_payload)
            if not record.node_id or record.node_id in state.files_by_id:
                raise ValueError(f"Duplicate or empty snapshot node id: {record.node_id!r}")
            record.direct_raw_doc_paths = [
                cls._validate_snapshot_raw_path(path.removeprefix("raw_docs/"))
                for path in record.direct_raw_doc_paths
            ]
            record.direct_raw_doc_paths = [f"raw_docs/{path}" for path in record.direct_raw_doc_paths]
            state.files_by_id[record.node_id] = record
        for record in state.files_by_id.values():
            if record.parent_id is not None and record.parent_id not in state.files_by_id:
                raise ValueError(f"Snapshot node {record.node_id} has an unknown parent")
            unknown_sources = set(record.direct_synthetic_ids) - set(state.files_by_id)
            if unknown_sources:
                raise ValueError(
                    f"Snapshot node {record.node_id} has unknown synthetic sources: "
                    f"{sorted(unknown_sources)}"
                )
            for raw_path in record.direct_raw_doc_paths:
                if raw_path.removeprefix("raw_docs/") not in normalized_raw_paths:
                    raise ValueError(
                        f"Snapshot node {record.node_id} references a hidden raw document: {raw_path}"
                    )
        # Reject malformed parent graphs at load time.
        for node_id in state.files_by_id:
            state.active_path(node_id)
        return state

    def save_snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_snapshot(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_snapshot(cls, path: Path, *, raw_doc_root: Path) -> SyntheticFilesystemState:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid synthetic filesystem snapshot JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise TypeError("Synthetic filesystem snapshot root must be an object")
        return cls.from_snapshot(payload, raw_doc_root=raw_doc_root)

    def _slugify(self, text: str) -> str:
        chars: list[str] = []
        for ch in text.strip().lower():
            if ch.isalnum():
                chars.append(ch)
            else:
                chars.append("_")
        slug = "".join(chars).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return self._truncate_path_slug(slug or "untitled")

    def _truncate_path_slug(self, slug: str) -> str:
        if len(slug.encode("utf-8")) <= SYNTHETIC_PATH_SLUG_MAX_BYTES:
            return slug
        chars: list[str] = []
        used_bytes = 0
        for ch in slug:
            ch_bytes = len(ch.encode("utf-8"))
            if used_bytes + ch_bytes > SYNTHETIC_PATH_SLUG_MAX_BYTES:
                break
            chars.append(ch)
            used_bytes += ch_bytes
        return "".join(chars).rstrip("_") or "untitled"

    def _cluster_node_id(self) -> str:
        node_id = f"cluster_{self.next_cluster_idx:04d}"
        self.next_cluster_idx += 1
        return node_id

    def _merge_node_id(self) -> str:
        node_id = f"merge_{self.next_merge_idx:04d}"
        self.next_merge_idx += 1
        return node_id

    def _record_sort_key(self, rec: SyntheticFileRecord) -> tuple[int, str, str]:
        return (0 if rec.kind == "merge_summary" else 1, rec.title.lower(), rec.node_id)

    def _children(self, parent_id: str | None, *, active_only: bool = True) -> list[SyntheticFileRecord]:
        records = [
            rec
            for rec in self.files_by_id.values()
            if rec.parent_id == parent_id and (rec.active or not active_only)
        ]
        return sorted(records, key=self._record_sort_key)

    def _dir_name(self, rec: SyntheticFileRecord) -> str:
        return f"{rec.node_id}__{self._truncate_path_slug(rec.slug)}"

    def _file_name(self, rec: SyntheticFileRecord) -> str:
        return f"{rec.node_id}__{self._truncate_path_slug(rec.slug)}.txt"

    def _active_path_parts(self, node_id: str) -> list[str]:
        rec = self.files_by_id[node_id]
        parts: list[str] = []
        seen = {node_id}
        parent_chain = [node_id]
        parent_id = rec.parent_id
        while parent_id is not None:
            if parent_id in seen:
                parent_chain.append(parent_id)
                raise ValueError(
                    "Cycle detected in synthetic filesystem parent chain: "
                    + " -> ".join(parent_chain)
                )
            seen.add(parent_id)
            parent_chain.append(parent_id)
            parent = self.files_by_id.get(parent_id)
            if parent is None:
                raise ValueError(
                    f"Synthetic filesystem parent {parent_id!r} referenced by "
                    f"{parent_chain[-2]!r} does not exist"
                )
            parts.append(self._dir_name(parent))
            parent_id = parent.parent_id
        parts.reverse()
        return parts

    def active_path(self, node_id: str) -> str:
        rec = self.files_by_id[node_id]
        parts = ["synthetic_fs", *self._active_path_parts(node_id)]
        if rec.kind == "merge_summary":
            return "/".join(parts + [self._dir_name(rec), "summary.txt"])
        return "/".join(parts + [self._file_name(rec)])

    def history_path(self, node_id: str) -> str:
        rec = self.files_by_id[node_id]
        if rec.kind == "merge_summary":
            return f"history/{self._dir_name(rec)}/summary.txt"
        return f"history/{self._file_name(rec)}"

    def workspace_relative_path(self, node_id: str) -> str:
        active = self.active_path(node_id)
        if not active.startswith("synthetic_fs/"):
            raise ValueError(f"Unexpected active synthetic path: {active}")
        return active[len("synthetic_fs/") :]

    def active_records(self) -> list[SyntheticFileRecord]:
        return sorted([rec for rec in self.files_by_id.values() if rec.active], key=lambda rec: self.active_path(rec.node_id))

    def _active_path_index(self) -> dict[str, str]:
        return {self.active_path(rec.node_id): rec.node_id for rec in self.active_records()}

    def _history_path_index(self) -> dict[str, str]:
        return {self.history_path(rec.node_id): rec.node_id for rec in self.files_by_id.values()}

    def raw_doc_rel_paths(self) -> list[str]:
        return [f"raw_docs/{rel}" for rel in self.visible_raw_doc_paths]

    def _raw_handle_index(self) -> dict[str, str]:
        return {
            f"raw_{idx:04d}": path
            for idx, path in enumerate(self.raw_doc_rel_paths(), start=1)
        }

    def _raw_listing_metadata(self) -> dict[str, dict[str, str]]:
        return {
            path: {"handle": handle}
            for handle, path in self._raw_handle_index().items()
        }

    def _clean_tool_path(self, path: str) -> str:
        clean = str(path).strip()
        clean = clean.strip("\"'`")
        clean = clean.replace("\\", "/")
        while clean.startswith("./"):
            clean = clean[2:]
        clean = clean.lstrip("/")
        return clean

    def _path_match_key(self, path: str) -> str:
        return re.sub(r"[^0-9a-z]+", "", self._clean_tool_path(path).lower())

    def _unique_close_path_match(self, path: str, candidates: Sequence[str]) -> str | None:
        clean_key = self._path_match_key(PurePosixPath(self._clean_tool_path(path)).name)
        if len(clean_key) < 16:
            return None
        scored: list[tuple[float, str]] = []
        for candidate in candidates:
            candidate_key = self._path_match_key(PurePosixPath(candidate).name)
            if len(candidate_key) < 16:
                continue
            score = difflib.SequenceMatcher(None, clean_key, candidate_key).ratio()
            if score >= 0.92:
                scored.append((score, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.03:
            return None
        return scored[0][1]

    def _unique_path_match(self, path: str, candidates: Sequence[str]) -> str | None:
        clean = self._clean_tool_path(path)
        if clean in candidates:
            return clean
        lower_matches = [candidate for candidate in candidates if candidate.lower() == clean.lower()]
        if len(lower_matches) == 1:
            return lower_matches[0]
        suffix_matches = [candidate for candidate in candidates if candidate.endswith(clean)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        lower_suffix_matches = [candidate for candidate in candidates if candidate.lower().endswith(clean.lower())]
        if len(lower_suffix_matches) == 1:
            return lower_suffix_matches[0]
        active_suffix_matches = [candidate for candidate in suffix_matches if candidate.startswith("synthetic_fs/")]
        if len(active_suffix_matches) == 1:
            return active_suffix_matches[0]
        basename = PurePosixPath(clean).name
        basename_matches = [candidate for candidate in candidates if PurePosixPath(candidate).name == basename]
        if len(basename_matches) == 1:
            return basename_matches[0]
        lower_basename_matches = [
            candidate
            for candidate in candidates
            if PurePosixPath(candidate).name.lower() == basename.lower()
        ]
        if len(lower_basename_matches) == 1:
            return lower_basename_matches[0]
        active_basename_matches = [candidate for candidate in basename_matches if candidate.startswith("synthetic_fs/")]
        if len(active_basename_matches) == 1:
            return active_basename_matches[0]
        clean_match_key = self._path_match_key(clean)
        normalized_matches = [
            candidate for candidate in candidates if self._path_match_key(candidate) == clean_match_key
        ]
        if len(normalized_matches) == 1:
            return normalized_matches[0]
        clean_basename_key = self._path_match_key(basename)
        normalized_basename_matches = [
            candidate
            for candidate in candidates
            if self._path_match_key(PurePosixPath(candidate).name) == clean_basename_key
        ]
        if len(normalized_basename_matches) == 1:
            return normalized_basename_matches[0]
        close_match = self._unique_close_path_match(clean, candidates)
        if close_match is not None:
            return close_match
        return None

    def _node_id_from_path_text(self, path: str) -> str | None:
        import re

        match = re.search(r"(?<![A-Za-z0-9_])(?:cluster|merge)_\d{4}(?!\d)", path)
        return match.group(0) if match else None

    def _raw_handle_from_path_text(self, path: str) -> str | None:
        clean = self._clean_tool_path(path)
        if clean.startswith("raw_docs/"):
            clean = clean[len("raw_docs/") :]
        if re.fullmatch(r"raw_\d{4}\.txt", clean):
            clean = clean[:-4]
        return self._raw_handle_index().get(clean)

    def _looks_like_raw_handle_path(self, path: str) -> bool:
        clean = self._clean_tool_path(path)
        if clean.startswith("raw_docs/"):
            clean = clean[len("raw_docs/") :]
        return bool(re.fullmatch(r"raw_\d{4}(?:\.txt)?", clean))

    def path_suggestions(self, path: str, *, limit: int = 8) -> list[str]:
        clean = self._clean_tool_path(path).lower()
        candidates = [
            *self.raw_doc_rel_paths(),
            *self._active_path_index().keys(),
            *self._history_path_index().keys(),
        ]
        if not clean:
            return candidates[:limit]
        scored: list[tuple[int, str]] = []
        clean_base = PurePosixPath(clean).name
        clean_key = self._path_match_key(clean)
        clean_base_key = self._path_match_key(clean_base)
        for candidate in candidates:
            candidate_lower = candidate.lower()
            candidate_base = PurePosixPath(candidate_lower).name
            candidate_key = self._path_match_key(candidate)
            candidate_base_key = self._path_match_key(candidate_base)
            score = 0
            if candidate_lower == clean:
                score += 100
            if candidate_key == clean_key:
                score += 95
            if candidate_lower.endswith(clean):
                score += 70
            if clean_base and candidate_base == clean_base:
                score += 60
            if clean_base_key and candidate_base_key == clean_base_key:
                score += 55
            if clean in candidate_lower:
                score += 30
            if clean_base and clean_base in candidate_base:
                score += 20
            if clean_base_key and clean_base_key in candidate_base_key:
                score += 15
            if clean_base_key and len(clean_base_key) >= 16:
                close_score = difflib.SequenceMatcher(None, clean_base_key, candidate_base_key).ratio()
                if close_score >= 0.92:
                    score += int(close_score * 40)
            node_id = self._node_id_from_path_text(clean)
            if node_id and node_id in candidate_lower:
                score += 80
            if score > 0:
                scored.append((score, candidate))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [candidate for _, candidate in scored[:limit]]

    def resolve_raw_doc(self, path: str) -> Path:
        path = self._clean_tool_path(path)
        handle_match = self._raw_handle_from_path_text(path)
        if handle_match is not None:
            path = handle_match
        elif self._looks_like_raw_handle_path(path):
            raise ValueError(f"Raw document handle not visible in this run: {path}")
        if not path.startswith("raw_docs/"):
            matched = self._unique_path_match(path, self.raw_doc_rel_paths())
            if matched is None:
                raise ValueError(f"Not a raw document path: {path}")
            path = matched
        rel = path[len("raw_docs/") :]
        rel_match = self._unique_path_match(rel, self.visible_raw_doc_paths)
        if rel_match is not None:
            rel = rel_match
            path = f"raw_docs/{rel}"
        if rel not in self.visible_raw_doc_paths:
            raise ValueError(f"Raw document not visible in this run: {path}")
        raw_doc_root = self.raw_doc_root.resolve()
        candidate = (raw_doc_root / rel).resolve()
        if candidate != raw_doc_root and raw_doc_root not in candidate.parents:
            raise ValueError(f"Path escapes raw document root: {path}")
        if not candidate.exists() or not candidate.is_file():
            raise ValueError(f"Raw document not found: {path}")
        return candidate

    def resolve_synthetic_node(self, path: str) -> SyntheticFileRecord:
        path = self._clean_tool_path(path)
        embedded_node_id = self._node_id_from_path_text(path)
        if embedded_node_id is not None and not path.startswith(("synthetic_fs/", "history/")):
            rec = self.files_by_id.get(embedded_node_id)
            if rec is not None and rec.active:
                return rec
        if path.startswith("synthetic_fs/"):
            path_index = self._active_path_index()
        elif path.startswith("history/"):
            path_index = self._history_path_index()
        else:
            all_paths = [*self._active_path_index().keys(), *self._history_path_index().keys()]
            matched = self._unique_path_match(path, all_paths)
            if matched is None:
                raise ValueError(f"Path is not in synthetic filesystem or history: {path}")
            path = matched
            path_index = self._active_path_index() if path.startswith("synthetic_fs/") else self._history_path_index()
        node_id = path_index.get(path)
        if node_id is None:
            embedded_node_id = self._node_id_from_path_text(path)
            if embedded_node_id is not None:
                rec = self.files_by_id.get(embedded_node_id)
                if rec is not None and (path.startswith("history/") or rec.active):
                    return rec
        if node_id is None:
            all_paths = [*self._active_path_index().keys(), *self._history_path_index().keys()]
            matched = self._unique_path_match(path, all_paths)
            if matched is not None:
                node_id = self._active_path_index().get(matched) or self._history_path_index().get(matched)
        if node_id is None:
            raise ValueError(f"Synthetic file not found: {path}")
        return self.files_by_id[node_id]

    def _render_header_block(self, rec: SyntheticFileRecord) -> list[str]:
        return [
            f"TITLE: {rec.title}",
            f"KIND: {rec.kind}",
            f"DIRECT_SOURCE_RAW_DOCS: {json.dumps(rec.direct_raw_doc_paths, ensure_ascii=False)}",
            f"DIRECT_SOURCE_SYNTHETIC_FILES: {json.dumps(rec.direct_synthetic_ids, ensure_ascii=False)}",
        ]

    def _render_final_workspace_header_block(self, rec: SyntheticFileRecord) -> list[str]:
        return [
            f"TITLE: {rec.title}",
            f"KIND: {rec.kind}",
            f"DIRECT_SOURCE_RAW_DOCS: {json.dumps(rec.direct_raw_doc_paths, ensure_ascii=False)}",
            f"DIRECT_SOURCE_SYNTHETIC_FILES: {json.dumps(rec.direct_synthetic_ids, ensure_ascii=False)}",
        ]

    def render_synthetic_file(self, rec: SyntheticFileRecord) -> str:
        header = self._render_header_block(rec)
        body_label = "MERGE_SUMMARY" if rec.kind == "merge_summary" else "CLUSTER_TEXT"
        lines = header + ["", f"{body_label}:", rec.content.strip()]
        return "\n".join(lines).strip() + "\n"

    def render_final_workspace_file(self, rec: SyntheticFileRecord) -> str:
        header = self._render_final_workspace_header_block(rec)
        body_label = "MERGE_SUMMARY" if rec.kind == "merge_summary" else "CLUSTER_TEXT"
        lines = header + ["", f"{body_label}:", rec.content.strip()]
        return "\n".join(lines).strip() + "\n"

    def read_lines(self, path: str, start_line: int = 1, num_lines: int = READ_FILE_MAX_LINES) -> dict[str, Any]:
        path = self._clean_tool_path(path)
        if start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {start_line}")
        if num_lines < 1:
            raise ValueError(f"num_lines must be >= 1, got {num_lines}")
        num_lines = min(num_lines, READ_FILE_MAX_LINES)
        raw_handle_match = self._raw_handle_from_path_text(path)
        if raw_handle_match is not None:
            path = raw_handle_match
        raw_path_match = path if path.startswith("raw_docs/") else self._unique_path_match(path, self.raw_doc_rel_paths())
        if raw_path_match is not None or self._looks_like_raw_handle_path(path):
            if raw_path_match is not None:
                path = raw_path_match
            raw_doc_path = self.resolve_raw_doc(path)
            path = f"raw_docs/{raw_doc_path.relative_to(self.raw_doc_root.resolve())}"
            return _read_file_line_window(
                path=path,
                file_path=raw_doc_path,
                start_line=start_line,
                num_lines=num_lines,
                max_chars=self.raw_read_max_chars,
            )
        else:
            rec = self.resolve_synthetic_node(path)
            path = self.history_path(rec.node_id) if path.startswith("history/") or not rec.active else self.active_path(rec.node_id)
            text = self.render_synthetic_file(rec)
        return _read_text_line_window(
            path=path,
            text=text,
            start_line=start_line,
            num_lines=num_lines,
        )

    def _tree_lines(self, parent_id: str | None = None, indent: int = 0) -> list[str]:
        lines: list[str] = []
        prefix = "  " * indent
        for rec in self._children(parent_id, active_only=True):
            active_path = self.active_path(rec.node_id)
            if rec.kind == "merge_summary":
                lines.append(f"{prefix}- [MERGE] {rec.title} :: {active_path}")
                lines.extend(self._tree_lines(rec.node_id, indent + 1))
            else:
                lines.append(f"{prefix}- [CLUSTER] {rec.title} :: {active_path}")
        return lines

    def active_tree_string(self) -> str:
        lines = self._tree_lines()
        return "\n".join(lines) if lines else "(synthetic filesystem is currently empty)"

    def _transitive_raw_docs_for_node(self, node_id: str, memo: dict[str, set[str]] | None = None) -> set[str]:
        if memo is None:
            memo = {}
        if node_id in memo:
            return memo[node_id]
        rec = self.files_by_id[node_id]
        result = set(rec.direct_raw_doc_paths)
        for source_id in rec.direct_synthetic_ids:
            if source_id in self.files_by_id:
                result.update(self._transitive_raw_docs_for_node(source_id, memo))
        memo[node_id] = result
        return result

    def _virtual_list_from_paths(
        self,
        path: str,
        all_paths: Sequence[str],
        metadata_by_path: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        metadata_by_path = metadata_by_path or {}
        clean = path.strip().strip("/") or "."
        if clean in {"", "."}:
            clean = "."
        if clean == ".":
            return {
                "path": ".",
                "type": "dir",
                "entries": [
                    {"path": "raw_docs", "type": "dir"},
                    {"path": "synthetic_fs", "type": "dir"},
                    {"path": "history", "type": "dir"},
                ],
            }

        if clean in all_paths:
            payload = metadata_by_path.get(clean, {})
            return {
                "path": clean,
                "type": "file",
                **payload,
                "entries": [],
            }

        prefix = clean + "/"
        entries_by_path: dict[str, str] = {}
        has_descendants = False
        for item in all_paths:
            if not item.startswith(prefix):
                continue
            has_descendants = True
            remainder = item[len(prefix) :]
            if not remainder:
                continue
            child_name = remainder.split("/", 1)[0]
            child_path = str(PurePosixPath(clean) / child_name)
            child_type = "file" if "/" not in remainder else "dir"
            prev = entries_by_path.get(child_path)
            if prev == "dir":
                continue
            if prev == "file" and child_type == "dir":
                entries_by_path[child_path] = "dir"
                continue
            entries_by_path.setdefault(child_path, child_type)

        if not has_descendants:
            if clean in {"raw_docs", "synthetic_fs", "history"}:
                return {
                    "path": clean,
                    "type": "dir",
                    "entries": [],
                }
            raise ValueError(f"Path not found: {path}")

        entries = []
        for child_path, child_type in sorted(
            entries_by_path.items(), key=lambda item: (item[1] == "file", item[0].lower())
        ):
            entries.append(
                {
                    "path": child_path,
                    "type": child_type,
                    **metadata_by_path.get(child_path, {}),
                }
            )
        return {
            "path": clean,
            "type": "dir",
            "entries": entries,
        }

    def _record_listing_metadata(
        self,
        records: Sequence[SyntheticFileRecord],
        path_fn: Any,
    ) -> dict[str, dict[str, str]]:
        metadata: dict[str, dict[str, str]] = {}
        for rec in records:
            path = path_fn(rec.node_id)
            item = {
                "handle": rec.node_id,
                "node_id": rec.node_id,
                "title": rec.title,
                "kind": rec.kind,
            }
            metadata[path] = item
            if rec.kind == "merge_summary" and path.endswith("/summary.txt"):
                metadata[str(PurePosixPath(path).parent)] = item
        return metadata

    def list_entries(self, path: str = ".") -> dict[str, Any]:
        clean = self._clean_tool_path(path).strip("/") or "."
        if clean in {"", "."}:
            clean = "."

        if clean == ".":
            return self._virtual_list_from_paths(".", [])

        raw_handle_match = self._raw_handle_index().get(clean)
        if raw_handle_match is not None:
            clean = raw_handle_match
        if clean == "raw_docs" or clean.startswith("raw_docs/"):
            return self._virtual_list_from_paths(clean, self.raw_doc_rel_paths(), self._raw_listing_metadata())

        if clean == "synthetic_fs" or clean.startswith("synthetic_fs/"):
            active_records = self.active_records()
            active_paths = [self.active_path(rec.node_id) for rec in active_records]
            metadata = self._record_listing_metadata(active_records, self.active_path)
            return self._virtual_list_from_paths(clean, active_paths, metadata)

        if clean == "history" or clean.startswith("history/"):
            history_records = sorted(
                self.files_by_id.values(), key=lambda rec: (rec.created_round, rec.node_id)
            )
            history_paths = [self.history_path(rec.node_id) for rec in history_records]
            metadata = self._record_listing_metadata(history_records, self.history_path)
            return self._virtual_list_from_paths(clean, history_paths, metadata)

        raise ValueError(f"Path not found: {path}")

    def list_payload(self, view: Literal["active", "all"] = "active") -> dict[str, Any]:
        synthetic_entries: list[dict[str, Any]] = []
        if view == "active":
            records: Iterable[SyntheticFileRecord] = self.active_records()
        elif view == "all":
            records = sorted(self.files_by_id.values(), key=lambda rec: (rec.created_round, rec.node_id))
        else:
            raise ValueError(f"Unknown view: {view}")

        for rec in records:
            synthetic_entries.append(
                {
                    "node_id": rec.node_id,
                    "title": rec.title,
                    "kind": rec.kind,
                    "active": rec.active,
                    "active_path": self.active_path(rec.node_id) if rec.active else "",
                    "history_path": self.history_path(rec.node_id),
                    "created_round": rec.created_round,
                }
            )

        return {
            "question_id": self.question_id,
            "view": view,
            "raw_docs": self.raw_doc_rel_paths(),
            "synthetic_files": synthetic_entries,
            "synthetic_tree": self.active_tree_string(),
        }

    def _validate_title(self, title: str) -> str:
        cleaned = title.strip()
        if not cleaned:
            raise ValueError("title must be non-empty")
        return cleaned

    def _validate_synthetic_text(self, text: str, field_name: str) -> str:
        cleaned = str(text or "").strip()
        if cleaned.lower() in BAD_SYNTHETIC_TEXT_VALUES:
            raise ValueError(f"{field_name} must be non-empty and not a placeholder")
        return cleaned

    def _resolve_parent_id(self, parent_path: str | None) -> str | None:
        if not parent_path:
            return None
        parent = self.resolve_synthetic_node(parent_path)
        if not parent.active:
            raise ValueError(f"Parent synthetic file is not active: {parent_path}")
        if parent.kind != "merge_summary":
            raise ValueError(f"Parent path must point to an active merge summary: {parent_path}")
        return parent.node_id

    def _validate_merge_parent(self, child_ids: Sequence[str], parent_id: str | None) -> None:
        """Reject reparenting that would introduce or extend a parent cycle."""
        if parent_id is None:
            return

        child_id_set = set(child_ids)
        seen: set[str] = set()
        parent_chain: list[str] = []
        current_id: str | None = parent_id
        while current_id is not None:
            parent_chain.append(current_id)
            if current_id in child_id_set:
                raise ValueError(
                    "parent_path must not point to a child being merged or any of "
                    f"its descendants (parent chain: {' -> '.join(parent_chain)})"
                )
            if current_id in seen:
                raise ValueError(
                    "Cycle detected in existing synthetic filesystem parent chain: "
                    + " -> ".join(parent_chain)
                )
            seen.add(current_id)
            current = self.files_by_id.get(current_id)
            if current is None:
                raise ValueError(
                    f"Synthetic filesystem parent {current_id!r} does not exist"
                )
            current_id = current.parent_id

    def _partition_input_paths(self, input_paths: Sequence[str]) -> tuple[list[str], list[str]]:
        if not input_paths:
            raise ValueError("input_paths must be non-empty")
        raw_doc_paths: list[str] = []
        synthetic_ids: list[str] = []
        seen_paths: set[str] = set()
        invalid_paths: list[str] = []
        for path in input_paths:
            path = self._clean_tool_path(path)
            raw_handle_match = self._raw_handle_index().get(path)
            if raw_handle_match is not None:
                path = raw_handle_match
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                if path.startswith("raw_docs/") or self._unique_path_match(path, self.raw_doc_rel_paths()):
                    resolved = self.resolve_raw_doc(path)
                    rel = str(resolved.relative_to(self.raw_doc_root.resolve()))
                    raw_doc_paths.append(f"raw_docs/{rel}")
                else:
                    rec = self.resolve_synthetic_node(path)
                    synthetic_ids.append(rec.node_id)
            except Exception as exc:
                invalid_paths.append(f"{path}: {exc}")
        if not raw_doc_paths and not synthetic_ids:
            details = "; ".join(invalid_paths[:3])
            raise ValueError(f"input_paths did not resolve to any visible files: {details}")
        return raw_doc_paths, synthetic_ids

    def create_cluster(
        self,
        *,
        title: str,
        input_paths: Sequence[str],
        cluster_text: str,
        parent_path: str | None = None,
    ) -> dict[str, Any]:
        title = self._validate_title(title)
        cluster_text = self._validate_synthetic_text(cluster_text, "cluster_text")
        raw_doc_paths, synthetic_ids = self._partition_input_paths(input_paths)
        self.operation_round += 1
        rec = SyntheticFileRecord(
            node_id=self._cluster_node_id(),
            kind="cluster",
            title=title,
            slug=self._slugify(title),
            content=cluster_text,
            created_round=self.operation_round,
            parent_id=self._resolve_parent_id(parent_path),
            direct_raw_doc_paths=raw_doc_paths,
            direct_synthetic_ids=synthetic_ids,
        )
        self.files_by_id[rec.node_id] = rec
        return {
            "created": {
                "handle": rec.node_id,
                "node_id": rec.node_id,
                "title": rec.title,
                "active_path": self.active_path(rec.node_id),
                "history_path": self.history_path(rec.node_id),
            },
        }

    def merge_clusters(
        self,
        *,
        title: str,
        child_paths: Sequence[str],
        merge_summary_text: str,
        parent_path: str | None = None,
    ) -> dict[str, Any]:
        title = self._validate_title(title)
        merge_summary_text = self._validate_synthetic_text(
            merge_summary_text, "merge_summary_text"
        )
        if not child_paths:
            raise ValueError("child_paths must be non-empty")
        child_ids: list[str] = []
        seen: set[str] = set()
        for path in child_paths:
            rec = self.resolve_synthetic_node(path)
            if not rec.active:
                raise ValueError(f"Cannot merge inactive synthetic file: {path}")
            if rec.node_id in seen:
                continue
            seen.add(rec.node_id)
            child_ids.append(rec.node_id)
        parent_id = self._resolve_parent_id(parent_path)
        self._validate_merge_parent(child_ids, parent_id)
        self.operation_round += 1
        merge_rec = SyntheticFileRecord(
            node_id=self._merge_node_id(),
            kind="merge_summary",
            title=title,
            slug=self._slugify(title),
            content=merge_summary_text,
            created_round=self.operation_round,
            parent_id=parent_id,
            direct_raw_doc_paths=[],
            direct_synthetic_ids=child_ids.copy(),
        )
        self.files_by_id[merge_rec.node_id] = merge_rec
        for child_id in child_ids:
            self.files_by_id[child_id].parent_id = merge_rec.node_id
        return {
            "created": {
                "handle": merge_rec.node_id,
                "node_id": merge_rec.node_id,
                "title": merge_rec.title,
                "active_path": self.active_path(merge_rec.node_id),
                "history_path": self.history_path(merge_rec.node_id),
            },
            "moved_children": [self.active_path(child_id) for child_id in child_ids],
        }

    def delete(self, *, target_paths: Sequence[str], reason: str = "") -> dict[str, Any]:
        if not target_paths:
            raise ValueError("target_paths must be non-empty")
        deleted: list[dict[str, str]] = []
        seen: set[str] = set()
        for path in target_paths:
            rec = self.resolve_synthetic_node(path)
            if not rec.active:
                raise ValueError(f"Synthetic file is already inactive: {path}")
            if rec.node_id in seen:
                continue
            seen.add(rec.node_id)
            old_path = self.active_path(rec.node_id)
            if rec.kind == "merge_summary":
                children = self._children(rec.node_id, active_only=True)
                for child in children:
                    child.parent_id = rec.parent_id
            rec.active = False
            rec.deleted_round = self.operation_round + 1
            rec.delete_reason = reason.strip() or None
            deleted.append({
                "handle": rec.node_id,
                "node_id": rec.node_id,
                "title": rec.title,
                "old_active_path": old_path,
                "history_path": self.history_path(rec.node_id),
            })
        self.operation_round += 1
        return {
            "deleted": deleted,
        }

    def transitive_raw_docs_for_active_files(self) -> list[str]:
        raw_doc_paths: set[str] = set()
        memo: dict[str, set[str]] = {}
        for rec in self.active_records():
            raw_doc_paths.update(self._transitive_raw_docs_for_node(rec.node_id, memo))
        return sorted(raw_doc_paths)

    def raw_doc_fallback_order(self) -> list[str]:
        provenance_docs = self.transitive_raw_docs_for_active_files()
        remaining = [path for path in self.raw_doc_rel_paths() if path not in set(provenance_docs)]
        return provenance_docs + remaining

    def generate_readme(self, *, include_raw_docs: bool = False) -> str:
        lines = [
            "# Final Synthetic Filesystem",
            "",
            "This workspace contains a synthetic filesystem built before answering.",
            "README.md is for navigation and workspace conventions only; it does not contain evidence for the answer.",
            "",
            "How to use it:",
            "1. The top-level synthetic entries are the highest-value entry points and are intended to be read first.",
            "2. If needed, browse into synthetic subdirectories and read their summary.txt files first.",
            "3. Do not guess: if the synthetic files read so far are insufficient to uniquely identify the answer, keep exploring the synthetic filesystem before answering.",
        ]
        if include_raw_docs:
            lines.append(
                "4. If the synthetic filesystem still lacks a decisive detail, use raw_docs/ only through provenance links exposed by synthetic files."
            )
        lines.extend(
            [
            "",
            "Conventions:",
            "- Root-level synthetic entries are intended to be the first places to inspect and are directly loaded into context at the start of answering.",
            "- Merge-summary directories contain a summary.txt and then more specific child files underneath.",
            "- Synthetic files include provenance fields that point to relevant raw docs and prior synthetic files.",
            "- Use list_files to discover the available files and directories.",
            "- Do not answer from README.md alone; inspect synthetic files for evidence.",
            "",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def materialize_final_workspace(
        self, root_dir: Path, *, mode: AnswererWorkspaceMode = "synthetic_only"
    ) -> Path:
        root_dir.mkdir(parents=True, exist_ok=True)
        include_raw_docs = mode == "synthetic_plus_raw"
        (root_dir / "README.md").write_text(
            self.generate_readme(include_raw_docs=include_raw_docs), encoding="utf-8"
        )

        if include_raw_docs:
            raw_docs_dir = root_dir / "raw_docs"
            raw_docs_dir.mkdir(parents=True, exist_ok=True)
            for relative_path in self.raw_doc_fallback_order():
                source = self.resolve_raw_doc(relative_path)
                target = raw_docs_dir / relative_path.removeprefix("raw_docs/")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        for rec in self.active_records():
            rel_path = self.workspace_relative_path(rec.node_id)
            target = root_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.render_final_workspace_file(rec), encoding="utf-8")
        return root_dir

    def root_active_records(self) -> list[SyntheticFileRecord]:
        return self._children(None, active_only=True)

    def workspace_path_to_node_id(self) -> dict[str, str]:
        return {self.workspace_relative_path(rec.node_id): rec.node_id for rec in self.active_records()}

    def answerer_bootstrap_entries(self) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for rec in self.root_active_records():
            entries.append(
                {
                    "path": self.workspace_relative_path(rec.node_id),
                    "content": self.render_final_workspace_file(rec),
                }
            )
        return entries


class ReadOnlyAnswererWorkspaceTools:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir.resolve()

    def _resolve(self, relative_path: str) -> Path:
        clean = relative_path.strip() or "."
        candidate = (self.root_dir / clean).resolve()
        if candidate != self.root_dir and self.root_dir not in candidate.parents:
            raise ValueError(f"Path escapes workspace root: {relative_path}")
        if not candidate.exists():
            raise ValueError(f"Path not found: {relative_path}")
        return candidate

    def list_files(self, path: str = ".") -> dict[str, Any]:
        target = self._resolve(path)
        if target.is_file():
            return {
                "path": path,
                "type": "file",
                "entries": [],
            }
        entries: list[dict[str, Any]] = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            rel = str(child.relative_to(self.root_dir))
            entries.append(
                {
                    "path": rel,
                    "type": "dir" if child.is_dir() else "file",
                }
            )
        return {
            "path": "." if target == self.root_dir else str(target.relative_to(self.root_dir)),
            "type": "dir",
            "entries": entries,
        }

    def read_file(self, path: str, start_line: int = 1, num_lines: int = READ_FILE_MAX_LINES) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise ValueError(f"Path is not a file: {path}")
        if start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {start_line}")
        if num_lines < 1:
            raise ValueError(f"num_lines must be >= 1, got {num_lines}")
        num_lines = min(num_lines, READ_FILE_MAX_LINES)
        return _read_file_line_window(
            path=path,
            file_path=target,
            start_line=start_line,
            num_lines=num_lines,
        )


@dataclass(frozen=True)
class FrozenSyntheticFileExecutor:
    """External frozen-model executor that writes synthetic files from planner-selected sources."""

    backend: JudgeBackend = "gemini"
    model: str = "gemini-3.1-flash-lite-preview"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GEMINI_API_KEY"
    max_source_chars: int = 24000
    max_output_tokens: int = 768
    temperature: float = 0.0
    top_k: int | None = None
    deterministic: bool = False

    @staticmethod
    def _extractive_text(title: str, source_blocks: str, *, kind: str) -> str:
        if not source_blocks.strip():
            raise RuntimeError(f"Executor received no source text for {kind}")
        return f"{title}\n\n{source_blocks}".strip()

    def _source_blocks(
        self,
        state: SyntheticFilesystemState,
        paths: Sequence[str],
    ) -> str:
        blocks: list[str] = []
        remaining_chars = max(1000, self.max_source_chars)
        for path in paths:
            if remaining_chars <= 0:
                break
            payload = state.read_lines(
                path=path,
                start_line=1,
                num_lines=READ_FILE_MAX_LINES,
            )
            content = str(payload.get("content", "")).strip()
            if len(content) > remaining_chars:
                content = content[:remaining_chars].rstrip()
            blocks.append(
                "\n".join(
                    [
                        f"[SOURCE] {payload.get('path', path)}",
                        f"LINES: {payload.get('start_line', 1)}-{payload.get('end_line', 0)} of {payload.get('total_lines', 0)}",
                        content,
                    ]
                ).strip()
            )
            remaining_chars -= len(content)
        return "\n\n".join(blocks)

    def generate_cluster_text(
        self,
        *,
        state: SyntheticFilesystemState,
        title: str,
        input_paths: Sequence[str],
        generation_instructions: str = "",
    ) -> str:
        source_blocks = self._source_blocks(state, input_paths)
        system_prompt = (
            "You are a frozen executor model that writes synthetic filesystem memory banks. "
            "Use only the provided source excerpts. Do not answer any particular downstream question. "
            "Write a concise, high-density cluster that preserves concrete facts useful for many future questions. "
            "Include names, aliases, dates, places, relationships, titles, and uncertainties when present. "
            "Return only the cluster text, with no JSON, no markdown fence, and no explanation."
        )
        user_prompt = (
            f"Cluster title: {title}\n"
            f"Planner instructions: {generation_instructions.strip() or 'Create one coherent reusable memory bank from these sources.'}\n\n"
            "Source excerpts:\n"
            f"{source_blocks}"
        )
        if self.deterministic:
            return self._extractive_text(title, source_blocks, kind="cluster")
        text = call_chat_model_messages(
            backend=self.backend,
            model=self.model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_json=False,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            enable_thinking=False,
        ).strip()
        if text.lower() in BAD_SYNTHETIC_TEXT_VALUES:
            raise RuntimeError("Executor returned empty cluster text")
        return text

    def generate_merge_summary_text(
        self,
        *,
        state: SyntheticFilesystemState,
        title: str,
        child_paths: Sequence[str],
        generation_instructions: str = "",
    ) -> str:
        source_blocks = self._source_blocks(state, child_paths)
        system_prompt = (
            "You are a frozen executor model that writes parent summaries for a synthetic filesystem. "
            "Use only the provided child synthetic files. Build a navigation-oriented summary that explains "
            "the shared theme, important distinctions among children, and what future questions this parent helps route. "
            "Return only the merge summary text, with no JSON, no markdown fence, and no explanation."
        )
        user_prompt = (
            f"Parent summary title: {title}\n"
            f"Planner instructions: {generation_instructions.strip() or 'Summarize and organize these child synthetic files.'}\n\n"
            "Child synthetic files:\n"
            f"{source_blocks}"
        )
        if self.deterministic:
            return self._extractive_text(title, source_blocks, kind="merge summary")
        text = call_chat_model_messages(
            backend=self.backend,
            model=self.model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_json=False,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            enable_thinking=False,
        ).strip()
        if text.lower() in BAD_SYNTHETIC_TEXT_VALUES:
            raise RuntimeError("Executor returned empty merge summary text")
        return text


class BuilderFilesystemTools:
    def __init__(self, state: SyntheticFilesystemState, executor: FrozenSyntheticFileExecutor | None = None):
        self.state = state
        self.executor = executor

    async def _generate_cluster_text_with_executor(
        self,
        *,
        title: str,
        input_paths: Sequence[str],
        generation_instructions: str,
    ) -> str:
        assert self.executor is not None
        async with trace.scope_span("remote/builder_executor_cluster_call"):
            return await asyncio.to_thread(
                self.executor.generate_cluster_text,
                state=self.state,
                title=title,
                input_paths=input_paths,
                generation_instructions=generation_instructions,
            )

    async def _generate_merge_summary_text_with_executor(
        self,
        *,
        title: str,
        child_paths: Sequence[str],
        generation_instructions: str,
    ) -> str:
        assert self.executor is not None
        async with trace.scope_span("remote/builder_executor_merge_call"):
            return await asyncio.to_thread(
                self.executor.generate_merge_summary_text,
                state=self.state,
                title=title,
                child_paths=child_paths,
                generation_instructions=generation_instructions,
            )

    def _error_code(self, tool_name: str, exc: Exception, *, directory_listing: dict[str, Any] | None = None) -> str:
        if directory_listing is not None:
            return "directory_path"
        message = str(exc).lower()
        if "start_line" in message or "num_lines" in message:
            return "bad_line_range"
        if "input_paths did not resolve" in message or "child_paths" in message:
            return "bad_input_paths"
        if "not active" in message or "inactive" in message:
            return "inactive_path"
        if "path not found" in message or "not found" in message:
            return "path_not_found"
        if "not a raw document path" in message or "not in synthetic filesystem" in message:
            return "wrong_namespace"
        if "not visible" in message:
            return "not_visible"
        if "must be non-empty" in message:
            return "empty_required_field"
        return f"{tool_name}_error"

    def _error_payload(
        self,
        *,
        tool_name: str,
        exc: Exception,
        path: str | None = None,
        paths: Sequence[str] | None = None,
        directory_listing: dict[str, Any] | None = None,
        hint: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        suggestion_paths = list(paths or ([] if path is None else [path]))
        suggestions = [
            suggestion
            for suggestion_path in suggestion_paths
            for suggestion in self.state.path_suggestions(suggestion_path, limit=3)
        ][:10]
        error_code = self._error_code(tool_name, exc, directory_listing=directory_listing)
        if error_code == "path_not_found" and not suggestions:
            suggestions = [self.state.active_path(rec.node_id) for rec in self.state.active_records()[:10]]
        if not suggestions and error_code == "not_visible":
            suggestions = self.state.raw_doc_rel_paths()[:10]
        payload: dict[str, Any] = {
            "ok": False,
            "tool": tool_name,
            "error_code": error_code,
            "error": str(exc),
            "suggestions": suggestions,
            "hint": hint,
        }
        if path is not None:
            payload["path"] = path
        if paths is not None:
            payload["paths"] = list(paths)
        if directory_listing is not None:
            payload["directory_listing"] = directory_listing
        if error_code == "not_visible":
            payload["available_raw_docs"] = self.state.raw_doc_rel_paths()[:25]
            payload["available_raw_doc_handles"] = [
                {"handle": handle, "path": path}
                for handle, path in list(self.state._raw_handle_index().items())[:25]
            ]
        if error_code == "path_not_found":
            payload["available_synthetic_files"] = [
                {
                    "handle": rec.node_id,
                    "path": self.state.active_path(rec.node_id),
                    "title": rec.title,
                    "kind": rec.kind,
                }
                for rec in self.state.active_records()[:25]
            ]
        if extra:
            payload.update(extra)
        return payload

    @tool
    async def list_files(
        self,
        path: Annotated[str, "Path under ., raw_docs/, synthetic_fs/, or history/"] = ".",
    ) -> ToolResult:
        try:
            payload = self.state.list_entries(path=path)
        except Exception as exc:
            payload = self._error_payload(
                tool_name="list_files",
                exc=exc,
                path=path,
                hint="Use one of the suggested paths, or list_files('.') to restart navigation.",
            )
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def read_file(
        self,
        path: Annotated[str, "File path or handle returned by list_files/create_cluster under raw_docs/, synthetic_fs/, or history/"],
        start_line: Annotated[int, "1-based line number to start reading from"] = 1,
        num_lines: Annotated[int, "How many lines to read in this chunk (max 100); prefer focused reads on long files"] = READ_FILE_MAX_LINES,
    ) -> ToolResult:
        try:
            payload = self.state.read_lines(path=path, start_line=start_line, num_lines=num_lines)
        except Exception as exc:
            directory_listing: dict[str, Any] | None = None
            try:
                candidate_listing = self.state.list_entries(path=path)
                if candidate_listing.get("type") == "dir":
                    directory_listing = candidate_listing
            except Exception:
                directory_listing = None
            payload = self._error_payload(
                tool_name="read_file",
                exc=exc,
                path=path,
                directory_listing=directory_listing,
                hint=(
                    "read_file expects a file path. If this is a directory, use list_files on it; "
                    "otherwise retry with one of the suggested file paths."
                ),
                extra={"start_line": start_line, "num_lines": num_lines},
            )
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def read_many(
        self,
        reads: Annotated[list[dict[str, Any]], "List of read specs: {path, optional start_line, optional num_lines}. At most 4 reads per call."],
    ) -> ToolResult:
        payload_reads: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for spec in list(reads or [])[:READ_MANY_MAX_FILES]:
            path = str(spec.get("path", ""))
            start_line = int(spec.get("start_line", 1))
            num_lines = int(spec.get("num_lines", READ_FILE_MAX_LINES))
            try:
                payload_reads.append(
                    self.state.read_lines(path=path, start_line=start_line, num_lines=num_lines)
                )
            except Exception as exc:
                errors.append(
                    self._error_payload(
                        tool_name="read_many",
                        exc=exc,
                        path=path,
                        hint="Each read_many item must use an exact file path or handle returned by list_files or a prior successful tool result.",
                        extra={"start_line": start_line, "num_lines": num_lines},
                    )
                )
        omitted = max(0, len(reads or []) - READ_MANY_MAX_FILES)
        payload: dict[str, Any] = {
            "ok": bool(payload_reads),
            "reads": payload_reads,
            "errors": errors,
            "omitted_specs": omitted,
        }
        if not payload_reads and errors:
            payload["error"] = "No read_many items succeeded"
            payload["error_code"] = "all_reads_failed"
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def create_cluster(
        self,
        title: Annotated[str, "Short descriptive title for the new synthetic memory bank"],
        input_paths: Annotated[list[str], "List of raw_docs/ or synthetic_fs/ or history/ paths/handles used to build one coherent synthetic memory bank"],
        cluster_text: Annotated[str, "Optional builder-written cluster text. Leave empty when frozen executor mode is enabled."] = "",
        generation_instructions: Annotated[str, "Short scope/focus instructions for the frozen executor when cluster_text is empty"] = "",
        parent_path: Annotated[str, "Optional active merge-summary path under synthetic_fs/ where the cluster should be placed"] = "",
    ) -> ToolResult:
        try:
            if self.executor is not None:
                cluster_text = await self._generate_cluster_text_with_executor(
                    title=title,
                    input_paths=input_paths,
                    generation_instructions=generation_instructions,
                )
            payload = self.state.create_cluster(
                title=title,
                input_paths=input_paths,
                cluster_text=cluster_text,
                parent_path=parent_path or None,
            )
            payload["executor_generated"] = self.executor is not None
            payload["created_count"] = 1.0
        except Exception as exc:
            payload = self._error_payload(
                tool_name="create_cluster",
                exc=exc,
                paths=input_paths,
                hint="Retry create_cluster with at least one valid raw_docs/, synthetic_fs/, or history/ input path. In executor mode, provide title plus short generation_instructions instead of long cluster_text.",
                extra={"title": title, "input_paths": input_paths},
            )
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def create_clusters(
        self,
        clusters: Annotated[list[dict[str, Any]], "Batch of cluster specs. Each spec needs title, input_paths, and optional generation_instructions/parent_path/cluster_text."],
    ) -> ToolResult:
        created: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for idx, spec in enumerate(clusters or [], start=1):
            title = str(spec.get("title", "")).strip()
            input_paths = list(spec.get("input_paths") or [])
            parent_path = str(spec.get("parent_path", "") or "")
            generation_instructions = str(spec.get("generation_instructions", "") or "")
            cluster_text = str(spec.get("cluster_text", "") or "")
            try:
                if self.executor is not None:
                    cluster_text = await self._generate_cluster_text_with_executor(
                        title=title,
                        input_paths=input_paths,
                        generation_instructions=generation_instructions,
                    )
                payload = self.state.create_cluster(
                    title=title,
                    input_paths=input_paths,
                    cluster_text=cluster_text,
                    parent_path=parent_path or None,
                )
                created_payload = dict(payload["created"])
                created_payload["batch_index"] = idx
                created.append(created_payload)
            except Exception as exc:
                errors.append(
                    self._error_payload(
                        tool_name="create_clusters",
                        exc=exc,
                        paths=input_paths,
                        hint="Each create_clusters item needs a non-empty title and valid source paths. In executor mode, cluster_text is generated by the executor.",
                        extra={"batch_index": idx, "title": title, "input_paths": input_paths},
                    )
                )
        payload: dict[str, Any] = {
            "ok": bool(created),
            "created": created,
            "created_count": float(len(created)),
            "errors": errors,
            "executor_generated": self.executor is not None,
        }
        if not created and errors:
            payload["error"] = "No create_clusters items succeeded"
            payload["error_code"] = "all_creates_failed"
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def merge_clusters(
        self,
        title: Annotated[str, "Title for the new parent summary / directory"],
        child_paths: Annotated[list[str], "Active synthetic_fs/ file paths or handles that should be grouped under the new parent"],
        merge_summary_text: Annotated[str, "Optional builder-written parent summary. Leave empty when frozen executor mode is enabled."] = "",
        generation_instructions: Annotated[str, "Short scope/focus instructions for the frozen executor when merge_summary_text is empty"] = "",
        parent_path: Annotated[str, "Optional active merge-summary path under synthetic_fs/ where the new merge node should be placed"] = "",
    ) -> ToolResult:
        try:
            if self.executor is not None:
                merge_summary_text = await self._generate_merge_summary_text_with_executor(
                    title=title,
                    child_paths=child_paths,
                    generation_instructions=generation_instructions,
                )
            payload = self.state.merge_clusters(
                title=title,
                child_paths=child_paths,
                merge_summary_text=merge_summary_text,
                parent_path=parent_path or None,
            )
            payload["executor_generated"] = self.executor is not None
            payload["created_count"] = 1.0
        except Exception as exc:
            payload = self._error_payload(
                tool_name="merge_clusters",
                exc=exc,
                paths=child_paths,
                hint="Retry merge_clusters with active synthetic_fs/ child file paths. In executor mode, provide title plus short generation_instructions instead of long merge_summary_text.",
                extra={"title": title, "child_paths": child_paths},
            )
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def merge_many(
        self,
        merges: Annotated[list[dict[str, Any]], "Batch of merge specs. Each spec needs title, child_paths, and optional generation_instructions/parent_path/merge_summary_text."],
    ) -> ToolResult:
        created: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        moved_children: list[str] = []
        for idx, spec in enumerate(merges or [], start=1):
            title = str(spec.get("title", "")).strip()
            child_paths = list(spec.get("child_paths") or [])
            parent_path = str(spec.get("parent_path", "") or "")
            generation_instructions = str(spec.get("generation_instructions", "") or "")
            merge_summary_text = str(spec.get("merge_summary_text", "") or "")
            try:
                if self.executor is not None:
                    merge_summary_text = await self._generate_merge_summary_text_with_executor(
                        title=title,
                        child_paths=child_paths,
                        generation_instructions=generation_instructions,
                    )
                payload = self.state.merge_clusters(
                    title=title,
                    child_paths=child_paths,
                    merge_summary_text=merge_summary_text,
                    parent_path=parent_path or None,
                )
                created_payload = dict(payload["created"])
                created_payload["batch_index"] = idx
                created.append(created_payload)
                moved_children.extend(list(payload.get("moved_children", [])))
            except Exception as exc:
                errors.append(
                    self._error_payload(
                        tool_name="merge_many",
                        exc=exc,
                        paths=child_paths,
                        hint="Each merge_many item needs a non-empty title and active synthetic_fs/ child paths.",
                        extra={"batch_index": idx, "title": title, "child_paths": child_paths},
                    )
                )
        payload = {
            "ok": bool(created),
            "created": created,
            "created_count": float(len(created)),
            "moved_children": moved_children,
            "errors": errors,
            "executor_generated": self.executor is not None,
        }
        if not created and errors:
            payload["error"] = "No merge_many items succeeded"
            payload["error_code"] = "all_merges_failed"
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def delete(
        self,
        target_paths: Annotated[list[str], "Active synthetic_fs/ file paths or handles to remove from the final answerer-facing synthetic filesystem. Never pass raw_docs/ paths or history/ paths here."],
        reason: Annotated[str, "Optional short reason for deleting these files"] = "",
    ) -> ToolResult:
        try:
            payload = self.state.delete(target_paths=target_paths, reason=reason)
        except Exception as exc:
            payload = self._error_payload(
                tool_name="delete",
                exc=exc,
                paths=target_paths,
                hint="Retry delete only with active synthetic_fs/ paths, never raw_docs/ or history/.",
                extra={"target_paths": target_paths},
            )
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
