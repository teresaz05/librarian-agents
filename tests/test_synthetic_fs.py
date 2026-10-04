from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("tinker_cookbook")

from environments.synthetic_fs import load_index, normalize_answer
from environments.synthetic_fs.clients import _openai_compatible_chat_completions_url
from environments.synthetic_fs.common import STOP_MESSAGE
from environments.synthetic_fs.dataset import SyntheticFilesystemDatasetBuilder
from environments.synthetic_fs.reward import SyntheticFilesystemReward
from environments.synthetic_fs.workspace import (
    FrozenSyntheticFileExecutor,
    SyntheticFilesystemState,
)


def _state_with_clusters(
    tmp_path: Path, titles: list[str]
) -> tuple[SyntheticFilesystemState, list[str]]:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_paths = [f"doc_{index}.txt" for index in range(len(titles))]
    for path in raw_paths:
        (raw_root / path).write_text(f"Evidence from {path}.", encoding="utf-8")
    state = SyntheticFilesystemState(raw_root, raw_paths, "q1")
    handles = [
        state.create_cluster(
            title=title,
            input_paths=[f"raw_docs/{path}"],
            cluster_text=f"Summary for {title}.",
        )["created"]["handle"]
        for title, path in zip(titles, raw_paths, strict=True)
    ]
    return state, handles


def test_normalization_and_endpoint_resolution() -> None:
    assert normalize_answer("The, Answer!") == "answer"
    assert _openai_compatible_chat_completions_url("http://localhost:8000/v1") == (
        "http://localhost:8000/v1/chat/completions"
    )
    assert _openai_compatible_chat_completions_url(
        "http://localhost:8000/v1/chat/completions"
    ) == "http://localhost:8000/v1/chat/completions"


def test_merge_rejects_its_own_child_as_parent_atomically(tmp_path: Path) -> None:
    state, children = _state_with_clusters(tmp_path, ["First", "Second"])
    summary = state.merge_clusters(
        title="Summary",
        child_paths=children,
        merge_summary_text="Combined summary.",
    )["created"]["handle"]
    state_before = deepcopy(state)

    with pytest.raises(ValueError, match="child being merged"):
        state.merge_clusters(
            title="Invalid merge",
            child_paths=[summary],
            merge_summary_text="This must not be created.",
            parent_path=summary,
        )

    assert state == state_before


def test_merge_rejects_descendant_parent_atomically(tmp_path: Path) -> None:
    state, children = _state_with_clusters(tmp_path, ["First", "Second", "Third"])
    inner = state.merge_clusters(
        title="Inner",
        child_paths=children[:2],
        merge_summary_text="Inner summary.",
    )["created"]["handle"]
    outer = state.merge_clusters(
        title="Outer",
        child_paths=[inner, children[2]],
        merge_summary_text="Outer summary.",
    )["created"]["handle"]
    state_before = deepcopy(state)

    with pytest.raises(ValueError, match="child being merged"):
        state.merge_clusters(
            title="Invalid ancestor merge",
            child_paths=[outer],
            merge_summary_text="This must not be created.",
            parent_path=inner,
        )

    assert state == state_before


def test_active_path_detects_preexisting_parent_cycle(tmp_path: Path) -> None:
    state, children = _state_with_clusters(tmp_path, ["First", "Second"])
    state.files_by_id[children[0]].parent_id = children[1]
    state.files_by_id[children[1]].parent_id = children[0]

    with pytest.raises(ValueError, match="Cycle detected in synthetic filesystem"):
        state.active_path(children[0])


def test_load_index_resolves_paths_relative_to_index(tmp_path: Path) -> None:
    visible = tmp_path / "data" / "train" / "1"
    private = tmp_path / "data" / "private" / "1"
    index_dir = tmp_path / "data" / "indexes" / "train"
    visible.mkdir(parents=True)
    private.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    (visible / "doc.txt").write_text("doc")
    (private / "query.txt").write_text("question")
    (private / "answer.txt").write_text("answer")
    row = {
        "question_id": "1",
        "agent_query_dir": "../../train/1",
        "privileged_query_dir": "../../private/1",
        "num_docs": 1,
    }
    index = index_dir / "index.jsonl"
    index.write_text(json.dumps(row) + "\n")

    loaded = load_index(index)
    assert loaded[0]["question"] == "question"
    assert loaded[0]["gold_answer"] == "answer"
    assert Path(loaded[0]["agent_query_dir"]) == visible.resolve()


def test_snapshot_round_trip_preserves_materialized_filesystem(tmp_path: Path) -> None:
    state, children = _state_with_clusters(tmp_path, ["First", "Second"])
    state.merge_clusters(
        title="Combined",
        child_paths=children,
        merge_summary_text="A combined summary.",
    )
    snapshot_path = tmp_path / "snapshot.json"
    state.save_snapshot(snapshot_path)
    restored = SyntheticFilesystemState.load_snapshot(snapshot_path, raw_doc_root=state.raw_doc_root)

    assert restored.to_snapshot() == state.to_snapshot()
    original_root = state.materialize_final_workspace(tmp_path / "original")
    restored_root = restored.materialize_final_workspace(tmp_path / "restored")
    assert {
        path.relative_to(original_root): path.read_text(encoding="utf-8")
        for path in original_root.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(restored_root): path.read_text(encoding="utf-8")
        for path in restored_root.rglob("*")
        if path.is_file()
    }


def test_maturity_reports_direct_raw_provenance_coverage(tmp_path: Path) -> None:
    state, _ = _state_with_clusters(tmp_path, ["First", "Second"])
    reward = SyntheticFilesystemReward(
        state=state,
        gold_answer="",
        question="",
        filesystem_direct_coverage_weight=0.4,
    )

    metrics = reward._filesystem_maturity_scores(
        active_records=state.active_records(),
        num_active_files=float(len(state.active_records())),
        num_active_merges=0.0,
        stop_called=0.0,
    )

    assert metrics["filesystem_direct_covered_raw_docs"] == 2.0
    assert metrics["filesystem_direct_coverage_score"] == 1.0
    assert reward._metrics_template(stop_called=0.0, **metrics)[
        "filesystem_direct_coverage_score"
    ] == 1.0


def test_deterministic_executor_preserves_source_text() -> None:
    text = FrozenSyntheticFileExecutor._extractive_text(
        "Cluster", "[SOURCE] raw_docs/doc.txt\nConcrete evidence.", kind="cluster"
    )

    assert text.startswith("Cluster\n\n[SOURCE]")
    assert "Concrete evidence." in text


def test_raw_documents_are_materialized_only_in_fallback_mode(tmp_path: Path) -> None:
    state, _ = _state_with_clusters(tmp_path, ["First"])
    synthetic_only = state.materialize_final_workspace(tmp_path / "synthetic-only")
    raw_fallback = state.materialize_final_workspace(
        tmp_path / "raw-fallback", mode="synthetic_plus_raw"
    )

    assert not (synthetic_only / "raw_docs").exists()
    assert (raw_fallback / "raw_docs" / "doc_0.txt").read_text(encoding="utf-8") == (
        "Evidence from doc_0.txt."
    )


def test_terminal_answerer_can_be_disabled_without_calling_answerer(tmp_path: Path) -> None:
    state, _ = _state_with_clusters(tmp_path, ["Evidence"])
    reward = SyntheticFilesystemReward(
        state=state,
        gold_answer="answer",
        question="Question?",
        terminal_answerer_enabled=False,
        filesystem_maturity_scale=0.5,
        termination_penalty=0.0,
        empty_synthetic_penalty=0.0,
    )
    reward._run_scored_answerer_repeats = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("terminal answerer must be disabled")
    )

    value, metrics = asyncio.run(reward([{"role": "assistant", "content": STOP_MESSAGE}]))

    reward._run_scored_answerer_repeats.assert_not_awaited()
    assert value == metrics["filesystem_maturity_reward"]
    assert metrics["terminal_answer_score"] == 0.0
    assert metrics["answerer_evaluated"] == 0.0
    assert metrics["terminal_answerer_enabled"] == 0.0


def test_terminal_answerer_enabled_preserves_answer_utility(tmp_path: Path) -> None:
    state, _ = _state_with_clusters(tmp_path, ["Evidence"])
    reward = SyntheticFilesystemReward(
        state=state,
        gold_answer="answer",
        question="Question?",
        terminal_answerer_enabled=True,
        terminal_answerer_repeats=1,
        filesystem_maturity_scale=0.0,
        termination_penalty=0.0,
        empty_synthetic_penalty=0.0,
    )
    answerer_result = reward._empty_answerer_result()
    answerer_result.update(
        {
            "answer_text": "Answer: answer",
            "steps": 1.0,
            "read_paths": ["clusters/evidence.txt"],
            "synthetic_chars_read": 32,
        }
    )
    reward._run_answerer = AsyncMock(return_value=answerer_result)  # type: ignore[method-assign]

    value, metrics = asyncio.run(reward([{"role": "assistant", "content": STOP_MESSAGE}]))

    reward._run_answerer.assert_awaited_once()
    assert value == 1.0
    assert metrics["terminal_answer_score"] == 1.0
    assert metrics["answerer_evaluated"] == 1.0
    assert metrics["terminal_answerer_enabled"] == 1.0


def test_binary_judge_rejects_out_of_range_response(tmp_path: Path) -> None:
    state, _ = _state_with_clusters(tmp_path, ["Evidence"])
    reward = SyntheticFilesystemReward(state=state, gold_answer="answer", question="Question?")
    reward._call_chat_model = lambda **_: '{"correct": 5}'  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="Malformed judge response"):
        asyncio.run(reward._llm_judge_score("candidate"))


def test_binary_judge_rejects_malformed_json(tmp_path: Path) -> None:
    state, _ = _state_with_clusters(tmp_path, ["Evidence"])
    reward = SyntheticFilesystemReward(state=state, gold_answer="answer", question="Question?")
    reward._call_chat_model = lambda **_: "not json"  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="Malformed judge response"):
        asyncio.run(reward._llm_judge_score("candidate"))


def test_dataset_builder_propagates_terminal_answerer_switch(tmp_path: Path) -> None:
    visible = tmp_path / "visible"
    private = tmp_path / "private"
    visible.mkdir()
    private.mkdir()
    (visible / "doc.txt").write_text("Evidence", encoding="utf-8")
    (private / "query.txt").write_text("Question?", encoding="utf-8")
    (private / "answer.txt").write_text("answer", encoding="utf-8")
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "agent_query_dir": str(visible),
                "privileged_query_dir": str(private),
                "num_docs": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    builder = SyntheticFilesystemDatasetBuilder(
        index_jsonl=str(index),
        model_name_for_tokenizer="Qwen/Qwen3.5-4B",
        batch_size=1,
        group_size=1,
        terminal_answerer_enabled=False,
    )

    train_dataset, eval_dataset = asyncio.run(builder())

    assert eval_dataset is None
    assert train_dataset.env_group_builders[0].terminal_answerer_enabled is False
