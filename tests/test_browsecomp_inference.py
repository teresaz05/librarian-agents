from __future__ import annotations

import argparse
import asyncio
import json

from experiments.inference.browsecomp_hierarchy_ablation import run


def test_missing_snapshot_is_written_as_zero_credit(tmp_path) -> None:
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "question_id": "corpus-1",
                "amortization_questions": [
                    {"question_id": "q-1", "question": "Question?", "gold_answer": "Answer"}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "results.jsonl"
    args = argparse.Namespace(
        index=index,
        variants_root=tmp_path / "variants",
        output=output,
        answerer_model="answerer",
        answerer_base_url="https://answerer.invalid",
        answerer_api_key_env="",
        judge_model="judge",
        judge_base_url="https://judge.invalid",
        judge_api_key_env="",
        answerer_max_turns=32,
        answerer_max_output_tokens=512,
        answerer_final_answer_max_tokens=128,
        concurrency=1,
        variants=["original"],
        resume=False,
        retry_errors=False,
        missing_snapshot_score_zero=True,
    )

    asyncio.run(run(args))

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["construction_complete"] is False
    assert record["correctness"] == 0.0
    assert record["answerer_error"] is False
    assert record["judge_error"] is False
