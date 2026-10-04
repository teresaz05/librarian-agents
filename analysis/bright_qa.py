"""Score the 150-question BRIGHT answer-generation evaluation."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DOMAINS = ("aops", "biology", "stackoverflow")
QUESTIONS_PER_DOMAIN = 50
RESPONSES_PER_QUESTION = 4


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _slugify(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return re.sub(r"_+", "_", slug) or "unknown"


def _validate(grouped: dict[tuple[str, str], list[int]]) -> None:
    expected_questions = len(DOMAINS) * QUESTIONS_PER_DOMAIN
    if len(grouped) != expected_questions:
        raise ValueError(f"Expected {expected_questions} questions, found {len(grouped)}")
    for domain in DOMAINS:
        count = sum(key[0] == domain for key in grouped)
        if count != QUESTIONS_PER_DOMAIN:
            raise ValueError(f"Expected 50 {domain} questions, found {count}")
    incomplete = [key for key, values in grouped.items() if len(values) != RESPONSES_PER_QUESTION]
    if incomplete:
        raise ValueError(f"Every question must have four responses; invalid keys: {incomplete[:3]}")


def validate_manifest(
    grouped: dict[tuple[str, str], list[int]], manifest_path: Path
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        (
            str(row["domain"]),
            f"{_slugify(row['domain'])}__{_slugify(row['bright_query_id'])}",
        )
        for row in payload.get("questions", [])
    }
    actual = set(grouped)
    if actual != expected:
        missing = sorted(expected - actual)[:3]
        extra = sorted(actual - expected)[:3]
        raise ValueError(f"Scores do not match the BRIGHT manifest; missing={missing}, extra={extra}")


def load_evaluation_outputs(root: Path) -> dict[tuple[str, str], list[int]]:
    samples: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    for domain in DOMAINS:
        for row in _read_jsonl(root / f"{domain}.jsonl"):
            row_domain = str(row.get("domain", domain))
            if row_domain != domain:
                raise ValueError(f"Unexpected domain {row_domain!r} in {domain}.jsonl")
            if row.get("error") or row.get("judge_error"):
                raise ValueError(f"Evaluation error for {domain}/{row.get('question_id')}")
            question_id = str(row["question_id"])
            sample_idx = int(row["sample_idx"])
            key = domain, question_id
            if sample_idx in samples[key]:
                raise ValueError(f"Duplicate sample {sample_idx} for {domain}/{question_id}")
            samples[key][sample_idx] = int(float(row.get("correctness", 0.0) or 0.0) > 0)
    grouped = {
        key: [by_sample[index] for index in range(RESPONSES_PER_QUESTION)]
        for key, by_sample in samples.items()
        if set(by_sample) == set(range(RESPONSES_PER_QUESTION))
    }
    if len(grouped) != len(samples):
        raise ValueError("At least one question is missing one or more response samples")
    _validate(grouped)
    return grouped


def load_method_outputs(root: Path) -> dict[str, dict[tuple[str, str], list[int]]]:
    methods: dict[str, dict[tuple[str, str], list[int]]] = {}
    paths = sorted(root.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"No answer files found in {root}")
    for path in paths:
        samples: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
        for row in _read_jsonl(path):
            if row.get("error") or row.get("judge_error"):
                raise ValueError(f"Evaluation error in {path.name}/{row.get('question_id')}")
            question_id = str(row["question_id"])
            domain = question_id.split("__", 1)[0]
            if domain not in DOMAINS:
                raise ValueError(f"Cannot infer BRIGHT domain from {question_id!r}")
            sample_idx = int(row["sample_idx"])
            key = domain, question_id
            if sample_idx in samples[key]:
                raise ValueError(f"Duplicate sample {sample_idx} for {question_id}")
            samples[key][sample_idx] = int(float(row.get("correctness", 0.0) or 0.0) > 0)
        grouped = {
            key: [by_sample[index] for index in range(RESPONSES_PER_QUESTION)]
            for key, by_sample in samples.items()
            if set(by_sample) == set(range(RESPONSES_PER_QUESTION))
        }
        if len(grouped) != len(samples):
            raise ValueError(f"At least one question in {path.name} is missing responses")
        _validate(grouped)
        methods[path.stem] = grouped
    return methods


def load_score_ledger(path: Path) -> dict[str, dict[tuple[str, str], list[int]]]:
    methods: dict[str, dict[tuple[str, str], list[int]]] = defaultdict(dict)
    for row in _read_jsonl(path):
        method = str(row["method"])
        key = str(row["domain"]), str(row["question_id"])
        if key in methods[method]:
            raise ValueError(f"Duplicate score row for {method}/{key[0]}/{key[1]}")
        scores = row.get("scores")
        if not isinstance(scores, list) or len(scores) != RESPONSES_PER_QUESTION:
            raise ValueError(f"Expected four scores for {method}/{key[0]}/{key[1]}")
        parsed = [int(score) for score in scores]
        if any(score not in {0, 1} for score in parsed):
            raise ValueError(f"Scores must be binary for {method}/{key[0]}/{key[1]}")
        methods[method][key] = parsed
    if not methods:
        raise ValueError(f"Empty score ledger: {path}")
    for grouped in methods.values():
        _validate(grouped)
    return dict(methods)


def summarize_methods(
    methods: dict[str, dict[tuple[str, str], list[int]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "aggregate": {},
        "per_domain": {},
        "checks": {"all_cells_complete": True},
        "standard_error_unit": "question",
    }
    for method, grouped in sorted(methods.items()):
        result["aggregate"][method] = summarize(grouped)
        result["per_domain"][method] = {}
        for domain in DOMAINS:
            domain_scores = {key: value for key, value in grouped.items() if key[0] == domain}
            if len(domain_scores) != QUESTIONS_PER_DOMAIN:
                raise ValueError(f"Expected 50 {domain} questions, found {len(domain_scores)}")
            result["per_domain"][method][domain] = _summarize(domain_scores)
    return result


def _mean_and_se(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("Standard error requires at least two questions")
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values))


def _pass_at_k(scores: list[int], k: int) -> float:
    samples = len(scores)
    correct = sum(scores)
    if not 1 <= k <= samples:
        raise ValueError(f"pass@k requires 1 <= k <= {samples}")
    failures = samples - correct
    all_failed = math.comb(failures, k) / math.comb(samples, k) if failures >= k else 0.0
    return 1.0 - all_failed


def _summarize(grouped: dict[tuple[str, str], list[int]]) -> dict[str, int | float]:
    question_accuracy = [sum(scores) / len(scores) for scores in grouped.values()]
    pass_at_3 = [_pass_at_k(scores, 3) for scores in grouped.values()]
    pass_at_1_mean, pass_at_1_se = _mean_and_se(question_accuracy)
    pass_at_3_mean, pass_at_3_se = _mean_and_se(pass_at_3)
    return {
        "questions": len(grouped),
        "responses": len(grouped) * RESPONSES_PER_QUESTION,
        "correct_responses": sum(sum(scores) for scores in grouped.values()),
        "pass_at_1": pass_at_1_mean,
        "pass_at_1_se": pass_at_1_se,
        "pass_at_3": pass_at_3_mean,
        "pass_at_3_se": pass_at_3_se,
    }


def summarize(grouped: dict[tuple[str, str], list[int]]) -> dict[str, int | float]:
    _validate(grouped)
    return _summarize(grouped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--outputs-root", type=Path)
    inputs.add_argument("--scores", type=Path)
    inputs.add_argument("--answers-dir", type=Path)
    parser.add_argument("--method", default="librarian")
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/bright_qa150.json"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scores:
        methods = load_score_ledger(args.scores)
        for grouped in methods.values():
            validate_manifest(grouped, args.manifest)
        result = summarize_methods(methods)
    elif args.answers_dir:
        methods = load_method_outputs(args.answers_dir)
        for grouped in methods.values():
            validate_manifest(grouped, args.manifest)
        result = summarize_methods(methods)
    else:
        grouped = load_evaluation_outputs(args.outputs_root)
        validate_manifest(grouped, args.manifest)
        result = {"method": args.method, **summarize(grouped)}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
