"""Create and score the BrowseComp hierarchy ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


VARIANTS = ("original", "flattened", "shuffled")


def summarize_score_ledger(
    path: Path,
    *,
    expected_corpora: int = 49,
    expected_questions: int = 435,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_924,
) -> dict[str, Any]:
    rows = _read_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        variant = str(row["variant"])
        if variant not in grouped:
            raise ValueError(f"Unexpected hierarchy variant: {variant}")
        key = str(row["corpus_id"]), str(row["question_id"]), variant
        if key in seen:
            raise ValueError(f"Duplicate hierarchy score: {key}")
        seen.add(key)
        grouped[variant].append(row)
    expected_keys = {
        (str(row["corpus_id"]), str(row["question_id"])) for row in grouped["original"]
    }
    if len(expected_keys) != expected_questions:
        raise ValueError(f"Hierarchy ledger must contain {expected_questions} questions")
    if len({corpus_id for corpus_id, _ in expected_keys}) != expected_corpora:
        raise ValueError(f"Hierarchy ledger must contain {expected_corpora} corpora")
    result: dict[str, Any] = {
        "standard_error_unit": "corpus_cluster_bootstrap",
        "variants": {},
    }
    scores: dict[str, dict[str, tuple[float, int]]] = {name: {} for name in VARIANTS}
    token_costs: dict[str, dict[str, tuple[float, int]]] = {name: {} for name in VARIANTS}
    for variant in VARIANTS:
        variant_rows = grouped[variant]
        keys = {(str(row["corpus_id"]), str(row["question_id"])) for row in variant_rows}
        if keys != expected_keys:
            raise ValueError(f"Hierarchy variant {variant} does not cover the same questions")
        result["variants"][variant] = {
            "questions": len(variant_rows),
            "question_micro_accuracy": sum(float(row["correctness"]) for row in variant_rows)
            / len(variant_rows),
            "mean_answerer_tokens_read": sum(
                float(row["answerer_total_tokens_read"]) for row in variant_rows
            )
            / len(variant_rows),
        }
        for corpus_id in sorted({str(row["corpus_id"]) for row in variant_rows}):
            corpus_rows = [row for row in variant_rows if str(row["corpus_id"]) == corpus_id]
            scores[variant][corpus_id] = (
                sum(float(row["correctness"]) for row in corpus_rows),
                len(corpus_rows),
            )
            token_costs[variant][corpus_id] = (
                sum(float(row["answerer_total_tokens_read"]) for row in corpus_rows),
                len(corpus_rows),
            )
    score_differences, _ = _paired_bootstrap(
        scores, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    token_differences, token_ratios = _paired_bootstrap(
        token_costs, resamples=bootstrap_resamples, seed=bootstrap_seed + 1
    )
    score_means = _cluster_bootstrap_means(
        scores, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    token_means = _cluster_bootstrap_means(
        token_costs, resamples=bootstrap_resamples, seed=bootstrap_seed + 1
    )
    for variant in VARIANTS:
        result["variants"][variant][
            "question_micro_accuracy_cluster_bootstrap_se"
        ] = statistics.stdev(score_means[variant])
        result["variants"][variant][
            "mean_answerer_tokens_read_cluster_bootstrap_se"
        ] = statistics.stdev(token_means[variant])
    result["paired_bootstrap"] = {
        "unit": "paired_corpus",
        "resamples": bootstrap_resamples,
        "seed": bootstrap_seed,
        "variant_minus_original": {
            variant: {
                "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
                "bootstrap_se": statistics.stdev(values),
            }
            for variant, values in score_differences.items()
        },
    }
    result["paired_read_cost_bootstrap"] = {
        "unit": "paired_corpus",
        "resamples": bootstrap_resamples,
        "seed": bootstrap_seed + 1,
        "variant_minus_original_tokens": {
            variant: {
                "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
                "bootstrap_se": statistics.stdev(values),
            }
            for variant, values in token_differences.items()
        },
        "variant_over_original_tokens": {
            variant: {
                "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
                "bootstrap_se": statistics.stdev(values),
            }
            for variant, values in token_ratios.items()
        },
    }
    return result


def write_compact_score_ledger(
    *,
    results: Path,
    output: Path,
    expected_rows: int = 1_305,
    expected_corpora: int = 49,
    expected_questions: int = 435,
) -> None:
    rows = _read_jsonl(results)
    if len(rows) != expected_rows:
        raise ValueError(f"Hierarchy evaluation must contain {expected_rows} rows")
    if any(
        bool(row.get("answerer_error"))
        or bool(row.get("judge_error"))
        or bool(row.get("error"))
        or not bool(row.get("construction_complete", True))
        for row in rows
    ):
        raise ValueError("Hierarchy evaluation contains an incomplete or errored row")
    fields = (
        "answerer_total_tokens_read",
        "corpus_id",
        "correctness",
        "question_id",
        "variant",
    )
    compact = [{field: row[field] for field in fields} for row in rows]
    summarize_score_ledger(
        results,
        expected_corpora=expected_corpora,
        expected_questions=expected_questions,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in compact),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _clone(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def _files(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    files = snapshot.get("files")
    if not isinstance(files, list) or not all(isinstance(record, dict) for record in files):
        raise ValueError("Snapshot files must be a list of objects")
    return files


def _active_parent_edges(snapshot: dict[str, Any]) -> list[tuple[str, str | None]]:
    return sorted(
        (str(record["node_id"]), record.get("parent_id"))
        for record in _files(snapshot)
        if bool(record.get("active"))
    )


def flatten_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    result = _clone(snapshot)
    for record in _files(result):
        if bool(record.get("active")):
            record["parent_id"] = None
    return result


def shuffle_snapshot(snapshot: dict[str, Any], *, seed: int) -> dict[str, Any]:
    result = _clone(snapshot)
    active = [record for record in _files(result) if bool(record.get("active"))]
    node_ids = sorted(str(record["node_id"]) for record in active)
    shuffled = list(node_ids)
    random.Random(seed).shuffle(shuffled)
    permutation = dict(zip(node_ids, shuffled, strict=True))
    parents = {str(record["node_id"]): record.get("parent_id") for record in active}
    reassigned: dict[str, str | None] = {}
    for child, parent in parents.items():
        reassigned[permutation[child]] = permutation[str(parent)] if parent is not None else None
    for record in active:
        record["parent_id"] = reassigned[str(record["node_id"])]
    return result


def _payload_fingerprint(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "parent_id"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _depth_profile(snapshot: dict[str, Any]) -> Counter[int]:
    active = {
        str(record["node_id"]): record
        for record in _files(snapshot)
        if bool(record.get("active"))
    }
    profile: Counter[int] = Counter()
    for node_id in active:
        depth = 0
        cursor: str | None = node_id
        seen: set[str] = set()
        while cursor is not None:
            if cursor in seen:
                raise ValueError("Parent cycle in hierarchy")
            seen.add(cursor)
            record = active.get(cursor)
            if record is None:
                raise ValueError("Active node references a missing parent")
            cursor = record.get("parent_id")
            if cursor is not None:
                depth += 1
        profile[depth] += 1
    return profile


def validate_transforms(
    original: dict[str, Any], flattened: dict[str, Any], shuffled: dict[str, Any]
) -> dict[str, Any]:
    node_ids = sorted(str(record["node_id"]) for record in _files(original))
    payloads = Counter(_payload_fingerprint(record) for record in _files(original))
    for label, candidate in (("flattened", flattened), ("shuffled", shuffled)):
        if sorted(str(record["node_id"]) for record in _files(candidate)) != node_ids:
            raise ValueError(f"{label} changed the snapshot node set")
        if Counter(_payload_fingerprint(record) for record in _files(candidate)) != payloads:
            raise ValueError(f"{label} changed file contents or metadata")
    if any(parent is not None for _, parent in _active_parent_edges(flattened)):
        raise ValueError("Flattened snapshot retains a parent edge")
    if _depth_profile(shuffled) != _depth_profile(original):
        raise ValueError("Shuffled snapshot changed the hierarchy depth profile")
    return {
        "payload_multiset_preserved": True,
        "node_ids_preserved": True,
        "flattened_has_no_parent_edges": True,
        "shuffled_depth_profile_preserved": True,
    }


def create_hierarchy_variants(*, snapshot: Path, output_dir: Path, seed: int) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    original = _read_json(snapshot)
    variants = {
        "original": original,
        "flattened": flatten_snapshot(original),
        "shuffled": shuffle_snapshot(original, seed=seed),
    }
    invariants = validate_transforms(
        variants["original"], variants["flattened"], variants["shuffled"]
    )
    output_dir.mkdir(parents=True)
    for name, payload in variants.items():
        (output_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha256": _sha256(snapshot),
                "shuffle_seed": seed,
                "variants": {
                    name: {"sha256": _sha256(output_dir / f"{name}.json")}
                    for name in VARIANTS
                },
                "invariants": invariants,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def create_hierarchy_tree(
    *, snapshots_root: Path, index: Path, output_dir: Path, seed: int, skip_missing: bool
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    rows = _read_jsonl(index)
    output_dir.mkdir(parents=True)
    included: list[str] = []
    excluded: list[str] = []
    for row in rows:
        corpus_id = str(row["question_id"])
        snapshot = snapshots_root / corpus_id / "trajectory_00.json"
        if not snapshot.is_file():
            if skip_missing:
                excluded.append(corpus_id)
                continue
            raise FileNotFoundError(snapshot)
        corpus_seed = seed + int(hashlib.sha256(corpus_id.encode()).hexdigest()[:8], 16)
        create_hierarchy_variants(
            snapshot=snapshot,
            output_dir=output_dir / corpus_id,
            seed=corpus_seed,
        )
        included.append(corpus_id)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shuffle_seed": seed,
                "included_corpus_ids": included,
                "missing_snapshot_corpus_ids": excluded,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_hierarchy_tree(
    *, snapshots_root: Path, index: Path, output_dir: Path, seed: int
) -> None:
    rows = _read_jsonl(index)
    included = [
        str(row["question_id"])
        for row in rows
        if (snapshots_root / str(row["question_id"]) / "trajectory_00.json").is_file()
    ]
    excluded = [str(row["question_id"]) for row in rows if str(row["question_id"]) not in included]
    tree_manifest = _read_json(output_dir / "manifest.json")
    if tree_manifest.get("shuffle_seed") != seed:
        raise ValueError("Hierarchy variants use a different shuffle seed")
    if tree_manifest.get("included_corpus_ids") != included:
        raise ValueError("Hierarchy variants do not match the current source snapshots")
    if tree_manifest.get("missing_snapshot_corpus_ids") != excluded:
        raise ValueError("Hierarchy variant missing-snapshot set changed")
    unexpected = {
        path.name for path in output_dir.iterdir() if path.is_dir()
    } - set(included)
    if unexpected:
        raise ValueError(f"Hierarchy variants contain unexpected corpora: {sorted(unexpected)}")
    for corpus_id in included:
        source = snapshots_root / corpus_id / "trajectory_00.json"
        manifest = _read_json(output_dir / corpus_id / "manifest.json")
        expected_seed = seed + int(hashlib.sha256(corpus_id.encode()).hexdigest()[:8], 16)
        if manifest.get("shuffle_seed") != expected_seed:
            raise ValueError(f"Shuffle seed changed for {corpus_id}")
        if manifest.get("source_sha256") != _sha256(source):
            raise ValueError(f"Source snapshot changed for {corpus_id}")
        for variant in VARIANTS:
            variant_path = output_dir / corpus_id / f"{variant}.json"
            if manifest.get("variants", {}).get(variant, {}).get("sha256") != _sha256(
                variant_path
            ):
                raise ValueError(f"Hierarchy variant changed for {corpus_id}/{variant}")


def create_completed_snapshot_index(
    *, snapshots_root: Path, index: Path, output: Path, manifest: Path
) -> None:
    rows = _read_jsonl(index)
    selected = [
        row
        for row in rows
        if (snapshots_root / str(row["question_id"]) / "trajectory_00.json").is_file()
    ]
    if not selected:
        raise ValueError("No completed filesystem snapshots are available")
    selected_ids = {str(row["question_id"]) for row in selected}
    excluded = [str(row["question_id"]) for row in rows if str(row["question_id"]) not in selected_ids]
    _write_new(output, "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    _write_new(
        manifest,
        json.dumps(
            {
                "schema_version": 1,
                "included_corpora": len(selected),
                "included_questions": sum(
                    len(row.get("amortization_questions") or []) for row in selected
                ),
                "excluded_corpus_ids": excluded,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_bootstrap(
    values: dict[str, dict[str, tuple[float, int]]], *, resamples: int, seed: int
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    corpus_ids = sorted(values["original"])
    if resamples < 1 or not corpus_ids:
        raise ValueError("Bootstrap requires corpora and positive resamples")
    for variant in VARIANTS:
        if set(values[variant]) != set(corpus_ids):
            raise ValueError("Variants do not cover identical corpora")
    rng = random.Random(seed)
    differences = {variant: [] for variant in VARIANTS[1:]}
    ratios = {variant: [] for variant in VARIANTS[1:]}
    for _ in range(resamples):
        sample = [rng.choice(corpus_ids) for _ in corpus_ids]
        denominator = sum(values["original"][corpus][1] for corpus in sample)
        means = {
            variant: sum(values[variant][corpus][0] for corpus in sample) / denominator
            for variant in VARIANTS
        }
        for variant in VARIANTS[1:]:
            differences[variant].append(means[variant] - means["original"])
            ratios[variant].append(means[variant] / means["original"])
    return differences, ratios


def _cluster_bootstrap_means(
    values: dict[str, dict[str, tuple[float, int]]], *, resamples: int, seed: int
) -> dict[str, list[float]]:
    corpus_ids = sorted(values["original"])
    if resamples < 1 or not corpus_ids:
        raise ValueError("Bootstrap requires corpora and positive resamples")
    for variant in VARIANTS:
        if set(values[variant]) != set(corpus_ids):
            raise ValueError("Variants do not cover identical corpora")
    rng = random.Random(seed)
    means = {variant: [] for variant in VARIANTS}
    for _ in range(resamples):
        sample = [rng.choice(corpus_ids) for _ in corpus_ids]
        for variant in VARIANTS:
            denominator = sum(values[variant][corpus][1] for corpus in sample)
            means[variant].append(
                sum(values[variant][corpus][0] for corpus in sample) / denominator
            )
    return means


def report_hierarchy(
    *,
    results: Path,
    index: Path,
    variants_root: Path,
    output: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    expected_corpora: int | None = None,
    expected_questions: int | None = None,
) -> None:
    index_rows = _read_jsonl(index)
    question_counts = {
        str(row["question_id"]): len(row.get("amortization_questions") or [])
        for row in index_rows
    }
    question_to_corpus = {
        str(question["question_id"]): str(row["question_id"])
        for row in index_rows
        for question in row.get("amortization_questions") or []
    }
    if expected_corpora is not None and len(question_counts) != expected_corpora:
        raise ValueError(f"Hierarchy evaluation requires exactly {expected_corpora} corpora")
    if expected_questions is not None and sum(question_counts.values()) != expected_questions:
        raise ValueError(f"Hierarchy evaluation requires exactly {expected_questions} questions")
    expected = {
        (variant, str(question["question_id"]))
        for row in index_rows
        for question in row.get("amortization_questions") or []
        for variant in VARIANTS
    }
    if not expected:
        raise ValueError("The evaluation index contains no amortization questions")
    rows = _read_jsonl(results)
    for row in rows:
        question_id = str(row["question_id"])
        if str(row["corpus_id"]) != question_to_corpus.get(question_id):
            raise ValueError(f"Question-to-corpus mismatch: {question_id}")
    actual = {(str(row["variant"]), str(row["question_id"])) for row in rows}
    if len(actual) != len(rows) or actual != expected:
        raise ValueError("Results do not cover the complete evaluation manifest")

    scores = {variant: {} for variant in VARIANTS}
    token_costs = {variant: {} for variant in VARIANTS}
    summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if any(
            bool(row.get("answerer_error")) or bool(row.get("judge_error"))
            for row in selected
        ):
            raise ValueError("Hierarchy evaluation contains answerer or judge errors")
        for corpus_id, count in question_counts.items():
            corpus_rows = [row for row in selected if row["corpus_id"] == corpus_id]
            if len(corpus_rows) != count:
                raise ValueError(f"{variant}/{corpus_id} has the wrong denominator")
            scores[variant][corpus_id] = (
                sum(float(row["correctness"]) for row in corpus_rows),
                count,
            )
            token_costs[variant][corpus_id] = (
                sum(float(row.get("answerer_total_tokens_read", 0)) for row in corpus_rows),
                count,
            )
        summaries[variant] = {
            "questions": len(selected),
            "question_micro_accuracy": (
                sum(float(row["correctness"]) for row in selected) / len(selected)
            ),
            "mean_answerer_tokens_read": (
                sum(float(row.get("answerer_total_tokens_read", 0)) for row in selected)
                / len(selected)
            ),
        }

    score_differences, _ = _paired_bootstrap(
        scores, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    token_differences, token_ratios = _paired_bootstrap(
        token_costs, resamples=bootstrap_resamples, seed=bootstrap_seed + 1
    )
    score_means = _cluster_bootstrap_means(
        scores, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    token_means = _cluster_bootstrap_means(
        token_costs, resamples=bootstrap_resamples, seed=bootstrap_seed + 1
    )
    for variant in VARIANTS:
        summaries[variant][
            "question_micro_accuracy_cluster_bootstrap_se"
        ] = statistics.stdev(score_means[variant])
        summaries[variant][
            "mean_answerer_tokens_read_cluster_bootstrap_se"
        ] = statistics.stdev(token_means[variant])
    invariant_counts: Counter[str] = Counter()
    for corpus_id in question_counts:
        manifest = _read_json(variants_root / corpus_id / "manifest.json")
        if not all(manifest["invariants"].values()):
            raise ValueError(f"Transform invariant failed for {corpus_id}")
        invariant_counts.update(manifest["invariants"])
    tree_manifest = _read_json(variants_root / "manifest.json")
    payload = {
        "schema_version": 1,
        "report_type": "browsecomp_hierarchy_ablation",
        "standard_error_unit": "corpus_cluster_bootstrap",
        "transformation_seed": tree_manifest["shuffle_seed"],
        "failure_policy": "answerer or judge errors invalidate the run",
        "variants": summaries,
        "paired_bootstrap": {
            "unit": "paired_corpus",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "variant_minus_original": {
                variant: {
                    "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
                    "bootstrap_se": statistics.stdev(values),
                }
                for variant, values in score_differences.items()
            },
        },
        "paired_read_cost_bootstrap": {
            "unit": "paired_corpus",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed + 1,
            "variant_minus_original_tokens": {
                variant: {
                    "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
                    "bootstrap_se": statistics.stdev(values),
                }
                for variant, values in token_differences.items()
            },
            "variant_over_original_tokens": {
                variant: {
                    "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
                    "bootstrap_se": statistics.stdev(values),
                }
                for variant, values in token_ratios.items()
            },
        },
        "transformation_checks": {
            "corpora": len(question_counts),
            "all_invariants_verified": True,
            "payload_definition": (
                "all file fields except parent_id, including text, metadata, and provenance"
            ),
            "corpora_by_invariant": dict(sorted(invariant_counts.items())),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tree = subparsers.add_parser("hierarchy-tree")
    tree.add_argument("--snapshots-root", type=Path, required=True)
    tree.add_argument("--index", type=Path, required=True)
    tree.add_argument("--output-dir", type=Path, required=True)
    tree.add_argument("--seed", type=int, default=19_105)
    tree.add_argument("--skip-missing", action="store_true")

    validate_tree = subparsers.add_parser("validate-hierarchy-tree")
    validate_tree.add_argument("--snapshots-root", type=Path, required=True)
    validate_tree.add_argument("--index", type=Path, required=True)
    validate_tree.add_argument("--output-dir", type=Path, required=True)
    validate_tree.add_argument("--seed", type=int, default=20_260_924)

    completed = subparsers.add_parser("completed-snapshot-index")
    completed.add_argument("--snapshots-root", type=Path, required=True)
    completed.add_argument("--index", type=Path, required=True)
    completed.add_argument("--output", type=Path, required=True)
    completed.add_argument("--manifest", type=Path, required=True)

    report = subparsers.add_parser("hierarchy-report")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--index", type=Path, required=True)
    report.add_argument("--variants-root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--bootstrap-resamples", type=int, default=10_000)
    report.add_argument("--bootstrap-seed", type=int, default=19_105)
    report.add_argument("--expected-corpora", type=int, default=49)
    report.add_argument("--expected-questions", type=int, default=435)

    compact = subparsers.add_parser("compact-ledger")
    compact.add_argument("--results", type=Path, required=True)
    compact.add_argument("--output", type=Path, required=True)
    compact.add_argument("--expected-rows", type=int, default=1_305)
    compact.add_argument("--expected-corpora", type=int, default=49)
    compact.add_argument("--expected-questions", type=int, default=435)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "hierarchy-tree":
        create_hierarchy_tree(
            snapshots_root=args.snapshots_root,
            index=args.index,
            output_dir=args.output_dir,
            seed=args.seed,
            skip_missing=args.skip_missing,
        )
    elif args.command == "completed-snapshot-index":
        create_completed_snapshot_index(
            snapshots_root=args.snapshots_root,
            index=args.index,
            output=args.output,
            manifest=args.manifest,
        )
    elif args.command == "validate-hierarchy-tree":
        validate_hierarchy_tree(
            snapshots_root=args.snapshots_root,
            index=args.index,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.command == "hierarchy-report":
        report_hierarchy(
            results=args.results,
            index=args.index,
            variants_root=args.variants_root,
            output=args.output,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            expected_corpora=args.expected_corpora,
            expected_questions=args.expected_questions,
        )
    elif args.command == "compact-ledger":
        write_compact_score_ledger(
            results=args.results,
            output=args.output,
            expected_rows=args.expected_rows,
            expected_corpora=args.expected_corpora,
            expected_questions=args.expected_questions,
        )


if __name__ == "__main__":
    main()
