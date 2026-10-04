from __future__ import annotations

import json

from data_processing.prepare_bright_frozen_construction import prepare


def test_prepare_bright_construction_hides_evaluation_queries(tmp_path) -> None:
    shared_root = tmp_path / "bright"
    corpus = shared_root / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "document.txt").write_text("domain evidence", encoding="utf-8")

    index_dir = shared_root / "by_domain" / "aops"
    index_dir.mkdir(parents=True)
    rows = [
        {
            "question_id": f"aops__q{index}",
            "agent_query_dir": "../../corpus",
            "question": f"held-out question {index}",
            "gold_answer": f"answer {index}",
        }
        for index in range(50)
    ]
    (index_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    output = tmp_path / "construction"
    manifest = prepare(shared_root, "aops", output)

    assert manifest["question_count"] == 50
    assert manifest["document_count"] == 1
    prompt = (output / "private" / "query.txt").read_text(encoding="utf-8")
    assert "held-out question" not in prompt
    assert "answer 0" not in prompt
    assert (output / "evaluation_index" / "index.jsonl").is_file()
    assert (output / "seed_index" / "index.jsonl").is_file()
