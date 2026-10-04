"""Prepare the BRIGHT QA evaluation set."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

from data_processing.prepare_bright import (
    BRIGHT_DATASET,
    SelectedExample,
    collect_required_doc_ids,
    load_domain_documents,
    slugify,
    unique_doc_ids,
    validate_required_documents,
    write_dataset,
)
from data_processing.prepare_training_index import build_row, write_index


DEFAULT_REVISION = "0dc152a5d69327d457c0390c692baf8e5808424b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument(
        "--extension-per-domain",
        type=int,
        default=0,
        help="Add deterministic questions beyond those in the supplied manifest.",
    )
    parser.add_argument("--max-documents-per-domain", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_example(domain: str, row: dict[str, Any]) -> SelectedExample:
    query_id = str(row["id"])
    gold_ids = unique_doc_ids(row.get("gold_ids", []))
    if not gold_ids:
        raise ValueError(f"BRIGHT {domain}/{query_id} has no gold documents.")
    return SelectedExample(
        domain=domain,
        query_id=query_id,
        question_id=f"{slugify(domain)}__{slugify(query_id)}",
        query=str(row["query"]),
        gold_answer=str(row.get("gold_answer", "")),
        gold_ids=gold_ids,
    )


def reference_ids(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions", [])
    by_domain: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(str(row["bright_query_id"]))
    if not by_domain:
        raise ValueError(f"Reference manifest has no questions: {path}")
    return dict(by_domain)


def choose_extension_ids(
    *,
    domain: str,
    candidates: list[str],
    count: int,
    seed: int,
) -> list[str]:
    if count < 0:
        raise ValueError("extension count cannot be negative")
    if count == 0:
        return []
    candidates = sorted(candidates)
    if len(candidates) < count:
        raise ValueError(f"{domain} has only {len(candidates)} remaining queries; need {count}.")
    rng = random.Random(f"bright50:{seed}:{domain}")
    return sorted(rng.sample(candidates, count))


def write_domain_indexes(output_root: Path, selected: dict[str, list[SelectedExample]]) -> None:
    for domain, examples in selected.items():
        rows = [
            build_row(
                output_root / "train" / example.question_id,
                output_root / "train_privileged" / example.question_id,
            )
            for example in examples
        ]
        for row, example in zip(rows, examples, strict=True):
            row["dataset_type"] = "bright50_shared_domain_corpus"
            row["bright_domain"] = domain
            row["bright_query_id"] = example.query_id
        out_dir = output_root / "by_domain" / domain
        write_index(
            out_dir,
            rows,
            source={
                "benchmark": "BRIGHT QA",
                "selection": "paper_50_per_domain",
                "domain": domain,
            },
        )
        question_path = output_root / "questions_by_domain" / f"{domain}.json"
        question_path.parent.mkdir(parents=True, exist_ok=True)
        question_path.write_text(
            json.dumps({"query_ids": [row["question_id"] for row in rows]}, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    if args.extension_per_domain < 0 or args.max_documents_per_domain < 1:
        raise SystemExit("Extension count cannot be negative; document count must be positive.")
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output_root}; use --overwrite to replace it.")
        shutil.rmtree(output_root)

    reference = reference_ids(args.reference_manifest.resolve())
    examples = load_dataset(BRIGHT_DATASET, "examples", revision=args.revision)
    documents = load_dataset(BRIGHT_DATASET, "documents", revision=args.revision)
    if not isinstance(examples, DatasetDict) or not isinstance(documents, DatasetDict):
        raise TypeError("Expected BRIGHT DatasetDict splits.")

    selected: dict[str, list[SelectedExample]] = {}
    extension_manifest: dict[str, list[str]] = {}
    for domain, existing_ids in sorted(reference.items()):
        rows_by_id = {str(row["id"]): dict(row) for row in examples[domain]}
        missing = sorted(set(existing_ids) - set(rows_by_id))
        if missing:
            raise ValueError(f"Reference IDs missing from {domain}: {missing[:5]}")
        extension_ids = choose_extension_ids(
            domain=domain,
            candidates=[query_id for query_id in rows_by_id if query_id not in set(existing_ids)],
            count=args.extension_per_domain,
            seed=args.seed,
        )
        selected[domain] = [selected_example(domain, rows_by_id[qid]) for qid in existing_ids + extension_ids]
        extension_manifest[domain] = extension_ids

    docs_by_domain = {
        domain: load_domain_documents(
            documents[domain],
            required_doc_ids=collect_required_doc_ids(domain_examples),
            max_documents_per_domain=args.max_documents_per_domain,
            gold_docs_only=False,
            domain=domain,
            seed=args.seed,
        )
        for domain, domain_examples in selected.items()
    }
    validate_required_documents(selected, docs_by_domain)
    write_dataset(
        selected_by_domain=selected,
        docs_by_domain=docs_by_domain,
        output_root=output_root,
        queries_per_domain=len(next(iter(selected.values()))),
        max_documents_per_domain=args.max_documents_per_domain,
        gold_docs_only=False,
        seed=args.seed,
    )
    write_domain_indexes(output_root, selected)
    heldout = {
        "questions": [
            {"question_id": example.question_id, "question": example.query}
            for domain_examples in selected.values()
            for example in domain_examples
        ]
    }
    (output_root / "heldout_ids.json").write_text(
        json.dumps(heldout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "benchmark": "BRIGHT QA",
        "selection": "paper_50_per_domain",
        "reference_manifest": args.reference_manifest.name,
        "dataset": BRIGHT_DATASET,
        "revision": args.revision,
        "seed": args.seed,
        "existing_per_domain": {
            domain: len(ids) for domain, ids in sorted(reference.items())
        },
        "extension_per_domain": args.extension_per_domain,
        "questions_per_domain": {
            domain: len(rows) for domain, rows in sorted(selected.items())
        },
        "documents_per_domain": {domain: len(docs) for domain, docs in docs_by_domain.items()},
        "reference_query_ids": reference,
        "extension_query_ids": extension_manifest,
    }
    (output_root / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
