"""Score matched BrowseComp+ filesystem evaluations."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"Empty evaluation output: {path}")
    return rows


def _manifest_questions(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    rows = _read(path)
    for row in rows:
        corpus_id = str(row["question_id"])
        for question in row.get("amortization_questions") or []:
            question_id = str(question["question_id"])
            if question_id in expected:
                raise ValueError(f"Duplicate question in manifest: {question_id}")
            expected[question_id] = corpus_id
    if len(rows) != 50 or len(expected) != 443:
        raise ValueError("BrowseComp+ manifest must contain 50 corpora and 443 questions")
    return expected


def _validated(path: Path, expected: dict[str, str]) -> dict[str, dict[str, Any]]:
    rows = _read(path)
    by_question: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in by_question:
            raise ValueError(f"Duplicate question in {path}: {question_id}")
        if row.get("error") or row.get("answerer_error") or row.get("judge_error"):
            raise ValueError(f"Evaluation error in {path}: {question_id}")
        if row.get("construction_complete") is False and float(row["correctness"]) != 0.0:
            raise ValueError(f"Missing construction must score zero in {path}: {question_id}")
        if question_id not in expected:
            raise ValueError(f"Question outside the frozen manifest: {question_id}")
        if str(row["corpus_id"]) != expected[question_id]:
            raise ValueError(f"Question-to-corpus mismatch in {path}: {question_id}")
        by_question[question_id] = row
    if set(by_question) != set(expected):
        raise ValueError(f"Evaluation output does not cover the frozen 443-question manifest: {path}")
    return by_question


def _percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    index = probability * (len(values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize(
    librarian_path: Path,
    base_path: Path,
    manifest_path: Path,
    *,
    resamples: int = 10_000,
    seed: int = 19_105,
) -> dict[str, Any]:
    expected = _manifest_questions(manifest_path)
    librarian = _validated(librarian_path, expected)
    base = _validated(base_path, expected)
    if set(librarian) != set(base):
        raise ValueError("Librarian and base outputs must cover identical questions")

    question_ids = sorted(librarian)
    librarian_correct = sum(float(librarian[q]["correctness"]) for q in question_ids)
    base_correct = sum(float(base[q]["correctness"]) for q in question_ids)
    corpus_ids = sorted({str(librarian[q]["corpus_id"]) for q in question_ids})
    grouped = {
        corpus_id: [q for q in question_ids if str(librarian[q]["corpus_id"]) == corpus_id]
        for corpus_id in corpus_ids
    }

    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(resamples):
        sample = [rng.choice(corpus_ids) for _ in corpus_ids]
        sampled_questions = [question for corpus in sample for question in grouped[corpus]]
        difference = sum(
            float(librarian[q]["correctness"]) - float(base[q]["correctness"])
            for q in sampled_questions
        ) / len(sampled_questions)
        differences.append(difference)

    return {
        "questions": len(question_ids),
        "corpora": len(corpus_ids),
        "librarian_accuracy": librarian_correct / len(question_ids),
        "base_accuracy": base_correct / len(question_ids),
        "accuracy_difference": (librarian_correct - base_correct) / len(question_ids),
        "accuracy_difference_ci95": [
            _percentile(differences, 0.025),
            _percentile(differences, 0.975),
        ],
        "bootstrap_unit": "corpus",
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def summarize_ledger(
    ledger_path: Path,
    *,
    resamples: int = 10_000,
    seed: int = 19_105,
) -> dict[str, Any]:
    rows = _read(ledger_path)
    by_arm: dict[str, dict[str, dict[str, Any]]] = {"base": {}, "rl": {}}
    for row in rows:
        arm = "rl" if row.get("arm") == "ft" else str(row.get("arm"))
        if arm not in by_arm:
            raise ValueError(f"Unexpected ledger arm: {arm}")
        corpus_id = str(row["corpus_id"])
        if corpus_id in by_arm[arm]:
            raise ValueError(f"Duplicate ledger row: {arm}/{corpus_id}")
        if row.get("status") not in {"complete", "max_token_capped_failure"}:
            raise ValueError(f"Unexpected construction status: {arm}/{corpus_id}")
        by_arm[arm][corpus_id] = row

    corpus_ids = sorted(by_arm["base"])
    if len(corpus_ids) != 50 or set(corpus_ids) != set(by_arm["rl"]):
        raise ValueError("Ledger must contain the same 50 corpora for both arms")

    summaries: dict[str, Any] = {}
    for arm in ("base", "rl"):
        arm_rows = [by_arm[arm][corpus_id] for corpus_id in corpus_ids]
        questions = sum(int(row["expected_questions"]) for row in arm_rows)
        correct = sum(float(row["amortization_correct_sum"]) for row in arm_rows)
        tokens = sum(int(row["builder_tokens"]) for row in arm_rows)
        if questions != 443:
            raise ValueError(f"{arm} ledger must cover exactly 443 questions")
        summaries[arm] = {
            "corpora": 50,
            "questions": questions,
            "correct_sum": correct,
            "question_micro_accuracy": correct / questions,
            "builder_tokens_total": tokens,
            "builder_tokens_per_corpus": tokens / 50,
            "correct_answers_per_million_builder_tokens": correct / tokens * 1_000_000,
            "capped_failure_corpora": sum(
                row["status"] == "max_token_capped_failure" for row in arm_rows
            ),
        }

    rng = random.Random(seed)
    accuracy_differences: list[float] = []
    efficiency_ratios: list[float] = []
    accuracy_samples: dict[str, list[float]] = {"base": [], "rl": []}
    for _ in range(resamples):
        sample = [rng.choice(corpus_ids) for _ in corpus_ids]
        sampled: dict[str, tuple[float, int, int]] = {}
        for arm in ("base", "rl"):
            arm_rows = [by_arm[arm][corpus_id] for corpus_id in sample]
            sampled[arm] = (
                sum(float(row["amortization_correct_sum"]) for row in arm_rows),
                sum(int(row["expected_questions"]) for row in arm_rows),
                sum(int(row["builder_tokens"]) for row in arm_rows),
            )
        base_correct, base_questions, base_tokens = sampled["base"]
        rl_correct, rl_questions, rl_tokens = sampled["rl"]
        accuracy_samples["base"].append(base_correct / base_questions)
        accuracy_samples["rl"].append(rl_correct / rl_questions)
        accuracy_differences.append(rl_correct / rl_questions - base_correct / base_questions)
        efficiency_ratios.append((rl_correct / rl_tokens) / (base_correct / base_tokens))

    for arm in ("base", "rl"):
        summaries[arm]["question_micro_accuracy_cluster_bootstrap_se"] = statistics.stdev(
            accuracy_samples[arm]
        )

    return {
        "schema_version": 1,
        "report_type": "browsecomp_equal_construction_cap_base_vs_rl",
        "standard_error_unit": "corpus_cluster_bootstrap",
        "scores": summaries,
        "paired_rl_minus_base_accuracy": {
            "estimate": summaries["rl"]["question_micro_accuracy"]
            - summaries["base"]["question_micro_accuracy"],
            "ci95": [
                _percentile(accuracy_differences, 0.025),
                _percentile(accuracy_differences, 0.975),
            ],
            "bootstrap_se": statistics.stdev(accuracy_differences),
            "unit": "paired_corpus",
            "resamples": resamples,
            "seed": seed,
        },
        "rl_over_base_correct_per_builder_token": {
            "estimate": summaries["rl"]["correct_answers_per_million_builder_tokens"]
            / summaries["base"]["correct_answers_per_million_builder_tokens"],
            "ci95": [
                _percentile(efficiency_ratios, 0.025),
                _percentile(efficiency_ratios, 0.975),
            ],
            "bootstrap_se": statistics.stdev(efficiency_ratios),
        },
    }
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--librarian", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/browsecomp_amortized50.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ledger:
        if args.librarian or args.base:
            parser.error("--ledger cannot be combined with --librarian or --base")
        result = summarize_ledger(args.ledger)
    else:
        if not args.librarian or not args.base:
            parser.error("provide --ledger or both --librarian and --base")
        result = summarize(args.librarian, args.base, args.manifest)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
