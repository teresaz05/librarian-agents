"""Prepare one question-blind BRIGHT construction task per domain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from data_processing.prepare_training_index import write_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"Empty BRIGHT index: {path}")
    return rows


def _resolve(path: object, *, relative_to: Path) -> Path:
    value = Path(str(path))
    return value.resolve() if value.is_absolute() else (relative_to / value).resolve()


def prepare(shared_root: Path, domain: str, output_root: Path) -> dict[str, Any]:
    source_index = shared_root.resolve() / "by_domain" / domain / "index.jsonl"
    rows = _load_rows(source_index)
    question_ids = [str(row.get("question_id", "")) for row in rows]
    if len(question_ids) != 50 or not all(question_ids) or len(set(question_ids)) != 50:
        raise ValueError(f"{domain} must contain exactly 50 distinct evaluation questions")

    source_dirs = [
        _resolve(row["agent_query_dir"], relative_to=source_index.parent) for row in rows
    ]
    canonical_dir = source_dirs[0]
    files = sorted(path.relative_to(canonical_dir).as_posix() for path in canonical_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"BRIGHT domain corpus is empty: {canonical_dir}")
    hashes = {relative: _sha256(canonical_dir / relative) for relative in files}
    for candidate in source_dirs[1:]:
        candidate_files = sorted(
            path.relative_to(candidate).as_posix() for path in candidate.rglob("*") if path.is_file()
        )
        if candidate_files != files or any(
            _sha256(candidate / relative) != hashes[relative] for relative in files
        ):
            raise ValueError(f"{domain} questions do not share one fixed corpus")

    output_root = output_root.resolve()
    private_dir = output_root / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    (private_dir / "query.txt").write_text(
        "Organize this domain corpus into a reusable research filesystem for many "
        "unknown future questions.\n",
        encoding="utf-8",
    )
    (private_dir / "answer.txt").write_text("Question-blind construction task.\n", encoding="utf-8")

    construction_id = f"bright_{domain}_shared_filesystem"
    row = {
        "question_id": construction_id,
        "agent_query_dir": str(canonical_dir),
        "privileged_query_dir": str(private_dir),
        "num_docs": len(files),
        "dataset_type": "bright_question_blind_shared_construction",
        "files": [{"relative_path": relative} for relative in files],
    }
    source = {
        "benchmark": "BRIGHT answer generation",
        "protocol": "one_question_blind_filesystem_per_domain",
        "domain": domain,
        "question_count": 50,
    }
    evaluation_index = write_index(output_root / "evaluation_index", [row], source=source)
    seed_row = {**row, "question_id": f"{construction_id}_seed"}
    seed_index = write_index(output_root / "seed_index", [seed_row], source=source)

    manifest = {
        "schema_version": 1,
        **source,
        "construction_id": construction_id,
        "document_count": len(files),
        "document_hashes": hashes,
        "evaluation_question_ids": question_ids,
        "evaluation_index": evaluation_index.relative_to(output_root).as_posix(),
        "seed_index": seed_index.relative_to(output_root).as_posix(),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare(args.shared_root, args.domain, args.output_root)
    print(json.dumps({key: manifest[key] for key in ("domain", "document_count")}, indent=2))


if __name__ == "__main__":
    main()
