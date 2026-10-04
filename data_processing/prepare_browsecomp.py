from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "browsecomp_plus_source.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "browsecomp_plus"
DEFAULT_HELDOUT_IDS = REPO_ROOT / "data" / "splits" / "heldout_ids.json"
DEFAULT_EXCLUDED_IDS = REPO_ROOT / "data" / "splits" / "excluded_ids.jsonl"
DEFAULT_SOURCE_METADATA = REPO_ROOT / "data" / "splits" / "browsecomp_plus_source.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the BrowseComp+ training and evaluation split."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--heldout-ids", type=Path, default=DEFAULT_HELDOUT_IDS)
    parser.add_argument("--excluded-ids", type=Path, default=DEFAULT_EXCLUDED_IDS)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--noise-seed", type=int, default=20260523)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(source: Path, heldout_ids: Path, metadata_path: Path) -> int:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_record = metadata["validated_source"]
    expected_bytes = int(source_record["bytes"])
    if source.stat().st_size != expected_bytes:
        raise ValueError(f"Unexpected source size: {source.stat().st_size} != {expected_bytes}")
    expected_source_hash = str(source_record["sha256"])
    if sha256_file(source) != expected_source_hash:
        raise ValueError("BrowseComp+ source SHA-256 does not match the release manifest")
    expected_heldout_hash = str(metadata["heldout_validation"]["manifest_sha256"])
    if sha256_file(heldout_ids) != expected_heldout_hash:
        raise ValueError("Held-out manifest SHA-256 does not match the release manifest")
    return int(source_record["rows"])


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def load_qids(path: Path) -> set[str]:
    if path.suffix == ".jsonl":
        qids = {
            str(row.get("question_id") or row.get("query_id") or "").strip()
            for row in iter_jsonl(path)
        }
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("query_ids", []) if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise ValueError(f"Expected a JSON list or query_ids object: {path}")
        qids = {str(value).strip() for value in values}
    qids.discard("")
    if not qids:
        raise ValueError(f"No question IDs found: {path}")
    return qids


def safe_slug(text: str, max_length: int = 120) -> str:
    slug = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return (slug.strip("_") or "doc")[:max_length]


def url_slug(url: str, fallback: str) -> str:
    parsed = urlparse(url.strip())
    parts: list[str] = []
    if parsed.netloc:
        parts.append(parsed.netloc)
    path_parts = [part for part in parsed.path.split("/") if part]
    if path_parts:
        parts.extend(path_parts[-3:])
    if parsed.query:
        parts.append(parsed.query[:40])
    return safe_slug("__".join(parts) if parts else fallback)


def normalize_documents(row: dict[str, Any]) -> list[dict[str, Any]]:
    documents_by_id: dict[str, dict[str, Any]] = {}
    document_ids: list[str] = []
    for source_key, role in (
        ("gold_docs", "gold"),
        ("evidence_docs", "evidence"),
        ("negative_docs", "negative"),
    ):
        for raw_document in row.get(source_key) or []:
            if not isinstance(raw_document, dict):
                continue
            doc_id = str(raw_document.get("docid") or raw_document.get("doc_id") or "").strip()
            text = str(raw_document.get("text") or "").strip()
            if not doc_id or not text:
                continue
            if doc_id not in documents_by_id:
                documents_by_id[doc_id] = {
                    "doc_id": doc_id,
                    "text": text,
                    "url": str(raw_document.get("url") or ""),
                    "is_gold": False,
                    "is_evidence": False,
                    "is_negative": False,
                    "source_roles": [],
                }
                document_ids.append(doc_id)
            document = documents_by_id[doc_id]
            document[f"is_{role}"] = True
            if role not in document["source_roles"]:
                document["source_roles"].append(role)
    return [documents_by_id[doc_id] for doc_id in document_ids]


def ordered_documents(
    documents: list[dict[str, Any]], *, question_id: str, noise_seed: int
) -> list[dict[str, Any]]:
    support = [
        document
        for document in documents
        if bool(document["is_gold"]) or bool(document["is_evidence"])
    ]
    noise = [
        document
        for document in documents
        if bool(document["is_negative"])
        and not bool(document["is_gold"])
        and not bool(document["is_evidence"])
    ]
    seed_bytes = hashlib.sha256(
        f"{noise_seed}:{question_id}:noise_order".encode("utf-8")
    ).digest()
    random.Random(int.from_bytes(seed_bytes[:8], byteorder="big")).shuffle(noise)
    return support + noise


def render_document(document: dict[str, Any]) -> str:
    return (
        f"DOC_ID: {document['doc_id']}\n"
        f"URL: {document['url']}\n\n"
        f"{document['text'].strip()}\n"
    )


def support_count(documents: list[dict[str, Any]]) -> int:
    return sum(bool(document["is_gold"]) or bool(document["is_evidence"]) for document in documents)


def noise_count(documents: list[dict[str, Any]]) -> int:
    return sum(
        bool(document["is_negative"])
        and not bool(document["is_gold"])
        and not bool(document["is_evidence"])
        for document in documents
    )


def prepare_output_root(root: Path, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root exists: {root}; pass --overwrite to replace it")
        shutil.rmtree(root)
    root.mkdir(parents=True)


def write_row(
    *, output_dir: Path, row: dict[str, Any], documents: list[dict[str, Any]]
) -> dict[str, Any]:
    question_id = str(row["query_id"]).strip()
    agent_dir = output_dir / "agent_data" / safe_slug(question_id, max_length=180)
    privileged_dir = output_dir / "privileged_data" / safe_slug(question_id, max_length=180)
    agent_dir.mkdir(parents=True)
    privileged_dir.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for ordinal, document in enumerate(documents, start=1):
        fallback = str(document["doc_id"] or f"doc_{ordinal:03d}")
        stem = f"doc_{ordinal:03d}__{url_slug(str(document['url']), fallback)}"
        filename = f"{stem}.txt"
        if filename in used_names:
            filename = f"{stem}__{ordinal}.txt"
        used_names.add(filename)
        (agent_dir / filename).write_text(render_document(document), encoding="utf-8")
        files.append(
            {
                "relative_path": filename,
                "doc_id": document["doc_id"],
                "url": document["url"],
                "is_gold": document["is_gold"],
                "is_evidence": document["is_evidence"],
                "is_negative": document["is_negative"],
                "source_roles": document["source_roles"],
            }
        )
    (privileged_dir / "query.txt").write_text(
        str(row.get("query") or "").strip() + "\n", encoding="utf-8"
    )
    (privileged_dir / "answer.txt").write_text(
        str(row.get("answer") or "").strip() + "\n", encoding="utf-8"
    )
    support_docs = support_count(documents)
    noise_docs = noise_count(documents)
    (privileged_dir / "manifest.json").write_text(
        json.dumps(
            {
                "question_id": question_id,
                "query_file": "query.txt",
                "answer_file": "answer.txt",
                "agent_dir": str(agent_dir),
                "documents": files,
                "num_support_docs": support_docs,
                "num_noise_docs": noise_docs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "question_id": question_id,
        "agent_query_dir": str(agent_dir),
        "privileged_query_dir": str(privileged_dir),
        "num_docs": len(files),
        "num_support_docs": support_docs,
        "num_noise_docs": noise_docs,
        "dataset_type": "official_browsecomp_plus_noisy_docs_all",
        "files": files,
    }


def write_split(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    split_name: str,
    source: Path,
    noise_seed: int,
) -> None:
    output_dir.mkdir(parents=True)
    index_rows = [
        write_row(
            output_dir=output_dir,
            row=row,
            documents=ordered_documents(
                normalize_documents(row),
                question_id=str(row["query_id"]),
                noise_seed=noise_seed,
            ),
        )
        for row in rows
    ]
    with (output_dir / "index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_type": "official_browsecomp_plus_noisy_docs_all",
                "split_name": split_name,
                "source_jsonl": str(source),
                "index_jsonl": str(output_dir / "index.jsonl"),
                "num_examples": len(index_rows),
                "noise_seed": noise_seed,
                "document_order": "evidence_then_seeded_negatives",
                "role_labels_visible": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    heldout_path = args.heldout_ids.expanduser().resolve()
    expected_source_rows = verify_inputs(
        source, heldout_path, args.source_metadata.expanduser().resolve()
    )
    heldout_ids = load_qids(heldout_path)
    excluded_ids = load_qids(args.excluded_ids.expanduser().resolve())
    if heldout_ids & excluded_ids:
        raise ValueError("The held-out and excluded ID manifests overlap")
    source_rows = list(iter_jsonl(source))
    if len(source_rows) != expected_source_rows:
        raise ValueError(
            f"Expected {expected_source_rows} source rows, found {len(source_rows)}"
        )
    rows_by_id = {str(row.get("query_id") or "").strip(): row for row in source_rows}
    if len(rows_by_id) != len(source_rows) or "" in rows_by_id:
        raise ValueError("The source contains duplicate or empty query IDs")
    missing = heldout_ids - set(rows_by_id)
    if missing:
        raise ValueError(f"Held-out IDs are absent from source: {sorted(missing)}")
    prepare_output_root(output_root, args.overwrite)
    train_rows = [
        row
        for row in source_rows
        if str(row["query_id"]) not in heldout_ids and str(row["query_id"]) not in excluded_ids
    ]
    heldout_rows = [row for row in source_rows if str(row["query_id"]) in heldout_ids]
    write_split(
        output_dir=output_root / "train",
        rows=train_rows,
        split_name="train",
        source=source,
        noise_seed=args.noise_seed,
    )
    write_split(
        output_dir=output_root / "heldout50",
        rows=heldout_rows,
        split_name="heldout50",
        source=source,
        noise_seed=args.noise_seed,
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "train_rows": len(train_rows),
                "heldout_rows": len(heldout_rows),
                "excluded_rows": len(excluded_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
