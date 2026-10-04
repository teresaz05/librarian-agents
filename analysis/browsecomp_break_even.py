"""Compute BrowseComp+ build-once amortization from the per-corpus ledger."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _components(rows: list[dict[str, Any]]) -> dict[str, float]:
    questions = sum(int(row["expected_questions"]) for row in rows)
    correct = sum(float(row["amortization_correct_sum"]) for row in rows)
    return {
        "construction_tokens_per_corpus": sum(int(row["builder_tokens"]) for row in rows)
        / len(rows),
        "answer_tokens_per_question": sum(
            float(row["answerer_total_tokens_read"]) * int(row["expected_questions"])
            for row in rows
        )
        / questions,
        "accuracy": correct / questions,
    }


def _expected_correct_tokens(component: dict[str, float], questions: int) -> float:
    tokens = (
        component["construction_tokens_per_corpus"] / questions
        + component["answer_tokens_per_question"]
    )
    return tokens / component["accuracy"]


def _crossover(base: dict[str, float], rl: dict[str, float]) -> int | None:
    fixed = (
        rl["construction_tokens_per_corpus"] / rl["accuracy"]
        - base["construction_tokens_per_corpus"] / base["accuracy"]
    )
    slope = (
        rl["answer_tokens_per_question"] / rl["accuracy"]
        - base["answer_tokens_per_question"] / base["accuracy"]
    )
    if slope == 0:
        return None
    crossing = -fixed / slope
    return max(1, math.ceil(crossing)) if crossing > 0 else None


def summarize(
    path: Path,
    *,
    query_counts: tuple[int, ...] = (1, 2, 5, 10, 20, 25, 50, 100),
    resamples: int = 10_000,
    seed: int = 19_105,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_arm = {
        "base": {str(row["corpus_id"]): row for row in rows if row["arm"] == "base"},
        "rl": {str(row["corpus_id"]): row for row in rows if row["arm"] == "ft"},
    }
    corpus_ids = sorted(by_arm["base"])
    if len(corpus_ids) != 50 or set(corpus_ids) != set(by_arm["rl"]):
        raise ValueError("Ledger must contain matched base and RL rows for 50 corpora")
    components = {
        arm: _components([by_arm[arm][corpus_id] for corpus_id in corpus_ids])
        for arm in ("base", "rl")
    }
    recorded_tokens = {
        str(count): {
            arm: components[arm]["construction_tokens_per_corpus"] / count
            + components[arm]["answer_tokens_per_question"]
            for arm in ("base", "rl")
        }
        for count in query_counts
    }
    expected_correct_tokens = {
        str(count): {
            arm: _expected_correct_tokens(components[arm], count)
            for arm in ("base", "rl")
        }
        for count in query_counts
    }
    rng = random.Random(seed)
    ratios_at_10: list[float] = []
    crossovers: list[int] = []
    for _ in range(resamples):
        sample = [rng.choice(corpus_ids) for _ in corpus_ids]
        sampled = {
            arm: _components([by_arm[arm][corpus_id] for corpus_id in sample])
            for arm in ("base", "rl")
        }
        ratios_at_10.append(
            _expected_correct_tokens(sampled["rl"], 10)
            / _expected_correct_tokens(sampled["base"], 10)
        )
        crossing = _crossover(sampled["base"], sampled["rl"])
        if crossing is not None:
            crossovers.append(crossing)
    return {
        "schema_version": 1,
        "components": components,
        "recorded_tokens_per_query": recorded_tokens,
        "tokens_per_expected_correct": expected_correct_tokens,
        "rl_over_base_at_10_queries": {
            "estimate": expected_correct_tokens["10"]["rl"]
            / expected_correct_tokens["10"]["base"],
            "ci95": [_percentile(ratios_at_10, 0.025), _percentile(ratios_at_10, 0.975)],
        },
        "crossover_queries_per_corpus": {
            "estimate": _crossover(components["base"], components["rl"]),
            "ci95": [_percentile(crossovers, 0.025), _percentile(crossovers, 0.975)],
        },
        "bootstrap": {"unit": "paired_corpus", "resamples": resamples, "seed": seed},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("results/browsecomp_per_corpus.jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.ledger)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
