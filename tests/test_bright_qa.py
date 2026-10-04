from __future__ import annotations

import json

import pytest

from analysis import bright_qa


def _complete_scores() -> dict[tuple[str, str], list[int]]:
    return {
        (domain, f"{domain}-{index}"): [1, 0, 0, 0]
        for domain in bright_qa.DOMAINS
        for index in range(bright_qa.QUESTIONS_PER_DOMAIN)
    }


def test_summarize_uses_response_and_question_denominators() -> None:
    result = bright_qa.summarize(_complete_scores())

    assert result == {
        "questions": 150,
        "responses": 600,
        "correct_responses": 150,
        "pass_at_1": 0.25,
        "pass_at_1_se": 0.0,
        "pass_at_3": 0.75,
        "pass_at_3_se": 0.0,
    }


def test_load_evaluation_outputs_rejects_errors(tmp_path) -> None:
    for domain in bright_qa.DOMAINS:
        rows = []
        for question in range(bright_qa.QUESTIONS_PER_DOMAIN):
            for sample in range(bright_qa.RESPONSES_PER_QUESTION):
                rows.append(
                    {
                        "domain": domain,
                        "question_id": f"{domain}-{question}",
                        "sample_idx": sample,
                        "correctness": 1.0,
                        "error": "",
                        "judge_error": False,
                    }
                )
        if domain == "aops":
            rows[0]["judge_error"] = True
        (tmp_path / f"{domain}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="Evaluation error"):
        bright_qa.load_evaluation_outputs(tmp_path)


def test_load_method_outputs_uses_complete_fixed_denominator(tmp_path) -> None:
    rows = []
    for domain in bright_qa.DOMAINS:
        for question in range(bright_qa.QUESTIONS_PER_DOMAIN):
            for sample in range(bright_qa.RESPONSES_PER_QUESTION):
                rows.append(
                    {
                        "question_id": f"{domain}__q{question}",
                        "sample_idx": sample,
                        "correctness": float(sample == 0),
                        "error": "",
                        "judge_error": False,
                    }
                )
    (tmp_path / "bm25.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    methods = bright_qa.load_method_outputs(tmp_path)

    assert set(methods) == {"bm25"}
    assert bright_qa.summarize(methods["bm25"])["pass_at_1"] == 0.25


def test_bundled_scores_reproduce_paper_table() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    methods = bright_qa.load_score_ledger(root / "results/bright_per_question.jsonl")
    for grouped in methods.values():
        bright_qa.validate_manifest(grouped, root / "data/splits/bright_qa150.json")
    report = bright_qa.summarize_methods(methods)

    assert report["aggregate"]["librarian"]["pass_at_1"] == 312 / 600
    assert report["aggregate"]["librarian"]["pass_at_3"] == 0.6683333333333333
    assert report["aggregate"]["embed"]["pass_at_3"] == 0.28
