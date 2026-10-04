from __future__ import annotations

import json

from analysis.browsecomp import summarize, summarize_ledger
from analysis.browsecomp_break_even import summarize as summarize_break_even


def test_summarize_matched_outputs(tmp_path) -> None:
    librarian = tmp_path / "librarian.jsonl"
    base = tmp_path / "base.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    manifest_rows: list[dict] = []
    librarian_rows: list[dict] = []
    base_rows: list[dict] = []
    question_index = 0
    for corpus in range(50):
        question_count = 9 if corpus < 43 else 8
        questions = []
        for _ in range(question_count):
            question_id = f"q{question_index}"
            questions.append({"question_id": question_id})
            common = {
                "question_id": question_id,
                "corpus_id": f"c{corpus}",
                "error": "",
                "answerer_error": False,
                "judge_error": False,
            }
            librarian_rows.append({**common, "correctness": 1})
            base_rows.append({**common, "correctness": 0})
            question_index += 1
        manifest_rows.append(
            {
                "question_id": f"c{corpus}",
                "amortization_questions": questions,
            }
        )
    assert question_index == 443
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    librarian.write_text(
        "".join(json.dumps(row) + "\n" for row in librarian_rows), encoding="utf-8"
    )
    base.write_text(
        "".join(json.dumps(row) + "\n" for row in base_rows), encoding="utf-8"
    )

    report = summarize(librarian, base, manifest, resamples=100, seed=7)

    assert report["questions"] == 443
    assert report["corpora"] == 50
    assert report["librarian_accuracy"] == 1.0
    assert report["base_accuracy"] == 0.0
    assert report["accuracy_difference"] == 1.0


def test_bundled_ledger_reproduces_report() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    report = summarize_ledger(root / "results/browsecomp_per_corpus.jsonl")

    assert report["scores"]["base"]["correct_sum"] == 107
    assert report["scores"]["rl"]["correct_sum"] == 282
    assert report["scores"]["base"]["capped_failure_corpora"] == 14
    assert report["scores"]["rl"]["capped_failure_corpora"] == 0


def test_bundled_ledger_reproduces_break_even() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    report = summarize_break_even(root / "results/browsecomp_per_corpus.jsonl")

    assert report["rl_over_base_at_10_queries"]["estimate"] == 0.6619572088466861
    assert report["rl_over_base_at_10_queries"]["ci95"] == [
        0.5040523396054453,
        0.8363246499804015,
    ]
    assert report["crossover_queries_per_corpus"] == {
        "estimate": 31,
        "ci95": [18.0, 54.0],
    }
