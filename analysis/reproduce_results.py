"""Reproduce reported metrics from the bundled score ledgers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from analysis.bright_qa import load_score_ledger, summarize_methods, validate_manifest
from analysis.browsecomp import summarize_ledger
from analysis.browsecomp_break_even import summarize as summarize_break_even
from analysis.hierarchy_ablation import summarize_score_ledger
from analysis.training_curve import read_points


ROOT = Path(__file__).resolve().parents[1]


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"Expected {expected}, found {actual}")


def check_configs() -> None:
    browsecomp = _json("configs/paper_browsecomp.json")
    assert browsecomp["dataset"]["training_tasks"] == 680
    assert browsecomp["training"]["batch_size"] == 8
    assert browsecomp["builder"]["model"] == "Qwen/Qwen3.5-4B"
    assert browsecomp["executor"]["model"] == "Qwen/Qwen3.5-35B-A3B"
    assert browsecomp["reward"]["answerability_delta_scale"] == 1.0
    assert browsecomp["reward"]["terminal_correctness"] is True
    bright = _json("configs/paper_bright.json")
    assert bright["domains"] == ["aops", "biology", "stackoverflow"]
    assert bright["questions_per_domain"] == 50
    assert bright["samples_per_question"] == 4
    assert bright["builder_checkpoint_step"] == 89


def reproduce_browsecomp() -> dict[str, Any]:
    actual = summarize_ledger(ROOT / "results/browsecomp_per_corpus.jsonl")
    expected = _json("results/browsecomp.json")
    for arm in ("base", "rl"):
        assert actual["scores"][arm]["questions"] == expected["scores"][arm]["questions"]
        _close(
            actual["scores"][arm]["question_micro_accuracy"],
            expected["scores"][arm]["question_micro_accuracy"],
        )
        _close(
            actual["scores"][arm]["question_micro_accuracy_cluster_bootstrap_se"],
            expected["scores"][arm]["question_micro_accuracy_cluster_bootstrap_se"],
        )
        assert (
            actual["scores"][arm]["capped_failure_corpora"]
            == expected["scores"][arm]["capped_failure_corpora"]
        )
    _close(
        actual["paired_rl_minus_base_accuracy"]["estimate"],
        expected["paired_rl_minus_base_accuracy"]["estimate"],
    )
    _close(
        actual["paired_rl_minus_base_accuracy"]["bootstrap_se"],
        expected["paired_rl_minus_base_accuracy"]["bootstrap_se"],
    )
    break_even = summarize_break_even(ROOT / "results/browsecomp_per_corpus.jsonl")
    _close(break_even["rl_over_base_at_10_queries"]["estimate"], 0.6619572088466861)
    _close(break_even["rl_over_base_at_10_queries"]["ci95"][0], 0.5040523396054453)
    _close(break_even["rl_over_base_at_10_queries"]["ci95"][1], 0.8363246499804015)
    assert break_even["crossover_queries_per_corpus"] == {
        "estimate": 31,
        "ci95": [18.0, 54.0],
    }
    _close(break_even["recorded_tokens_per_query"]["20"]["rl"], 16_762.891546275393)
    return actual


def reproduce_bright() -> dict[str, Any]:
    methods = load_score_ledger(ROOT / "results/bright_per_question.jsonl")
    manifest = ROOT / "data/splits/bright_qa150.json"
    for grouped in methods.values():
        validate_manifest(grouped, manifest)
    actual = summarize_methods(methods)
    expected = _json("results/bright.json")
    assert set(actual["aggregate"]) == set(expected["aggregate"])
    for method, summary in actual["aggregate"].items():
        reference = expected["aggregate"][method]
        assert summary["correct_responses"] == reference["correct_responses"]
        for field in ("pass_at_1", "pass_at_1_se", "pass_at_3", "pass_at_3_se"):
            _close(summary[field], reference[field])
        for domain, domain_summary in actual["per_domain"][method].items():
            domain_reference = expected["per_domain"][method][domain]
            for field in ("pass_at_1", "pass_at_1_se", "pass_at_3", "pass_at_3_se"):
                _close(domain_summary[field], domain_reference[field])
    return actual


def reproduce_hierarchy_ablation() -> dict[str, Any]:
    actual = summarize_score_ledger(ROOT / "results/hierarchy_per_question.jsonl")
    expected = _json("results/hierarchy_ablation.json")
    assert expected["transformation_seed"] == 20_260_924
    assert expected["failure_policy"] == "answerer or judge errors invalidate the run"
    assert expected["transformation_checks"]["all_invariants_verified"] is True
    assert expected["transformation_checks"]["corpora"] == 49
    assert set(expected["transformation_checks"]["corpora_by_invariant"].values()) == {49}
    for variant, summary in actual["variants"].items():
        reference = expected["variants"][variant]
        assert summary["questions"] == reference["questions"] == 435
        for field in (
            "question_micro_accuracy",
            "question_micro_accuracy_cluster_bootstrap_se",
            "mean_answerer_tokens_read",
            "mean_answerer_tokens_read_cluster_bootstrap_se",
        ):
            _close(summary[field], reference[field])
    for section, comparison in (
        ("paired_bootstrap", "variant_minus_original"),
        ("paired_read_cost_bootstrap", "variant_minus_original_tokens"),
        ("paired_read_cost_bootstrap", "variant_over_original_tokens"),
    ):
        for variant, summary in actual[section][comparison].items():
            for found, reference in zip(
                summary["ci95"], expected[section][comparison][variant]["ci95"], strict=True
            ):
                _close(found, reference)
            _close(
                summary["bootstrap_se"],
                expected[section][comparison][variant]["bootstrap_se"],
            )
    return actual


def check_reported_aggregates() -> None:
    table = _json("results/browsecomp_cross_answerer.json")["table"]
    reported = _json("results/reported_aggregates.json")
    assert max(value for key, value in table["Qwen3.5-4B"].items() if key != "librarian") == 0.44
    assert reported["table_1"]["rows"]["librarian"] == {
        "accuracy": 0.77,
        "construction_tokens": 178_926,
    }
    assert reported["table_4"]["rows"]["macro_rl"] == [32.9, 8813, 9049, 53.4, 45.4, 8.0]
    assert reported["table_5"]["rows"]["rl"] == [26.9, 9040, 9309, 42.2, 34.4, 7.8]
    assert len(reported["table_6"]["domains"]) == 9
    assert all(
        trained > base
        for trained, base in zip(
            reported["table_6"]["librarian"], reported["table_6"]["base"], strict=True
        )
    )
    points = read_points(ROOT / "results/training_reward.csv")
    assert len(points) == 54
    assert points[0] == (0, 0.278125)
    assert points[-1] == (53, 0.834375)


def main() -> None:
    check_configs()
    browsecomp = reproduce_browsecomp()
    bright = reproduce_bright()
    hierarchy = reproduce_hierarchy_ablation()
    check_reported_aggregates()
    print(
        "BrowseComp+: "
        f"{browsecomp['scores']['rl']['question_micro_accuracy']:.2%} vs "
        f"{browsecomp['scores']['base']['question_micro_accuracy']:.2%}"
    )
    print(
        "BRIGHT: "
        f"{bright['aggregate']['librarian']['pass_at_1']:.2%} pass@1, "
        f"{bright['aggregate']['librarian']['pass_at_3']:.2%} pass@3"
    )
    print(
        "Hierarchy ablation: "
        + ", ".join(
            f"{variant} {summary['question_micro_accuracy']:.2%} / "
            f"{summary['mean_answerer_tokens_read']:,.2f} navigation tokens"
            for variant, summary in hierarchy["variants"].items()
        )
    )
    print("Reported aggregates and 54-point training curve: consistency checks passed")


if __name__ == "__main__":
    main()
