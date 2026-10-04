from __future__ import annotations

import json

import pytest

from analysis import hierarchy_ablation as controls


def _snapshot() -> dict:
    def record(node_id: str, parent_id: str | None, content: str) -> dict:
        return {
            "node_id": node_id,
            "kind": "merge_summary" if node_id.startswith("merge") else "cluster",
            "title": node_id,
            "slug": node_id,
            "content": content,
            "created_round": 1,
            "parent_id": parent_id,
            "direct_raw_doc_paths": ["raw_docs/a.txt"],
            "direct_synthetic_ids": [],
            "active": True,
            "deleted_round": None,
            "delete_reason": None,
        }

    return {
        "schema_version": 1,
        "question_id": "corpus-1",
        "visible_raw_doc_paths": ["a.txt"],
        "next_cluster_idx": 4,
        "next_merge_idx": 3,
        "operation_round": 5,
        "files": [
            record("merge_0001", None, "root one"),
            record("cluster_0001", "merge_0001", "leaf one"),
            record("merge_0002", None, "root two"),
            record("cluster_0002", "merge_0002", "leaf two"),
            record("cluster_0003", "merge_0002", "leaf three"),
        ],
    }


def test_hierarchy_controls_preserve_payloads_and_isolate_structure() -> None:
    original = _snapshot()
    flattened = controls.flatten_snapshot(original)
    shuffled = controls.shuffle_snapshot(original, seed=7)

    invariants = controls.validate_transforms(original, flattened, shuffled)

    assert invariants["payload_multiset_preserved"] is True
    assert all(parent is None for _, parent in controls._active_parent_edges(flattened))
    assert controls._depth_profile(shuffled) == controls._depth_profile(original)
    assert controls._active_parent_edges(shuffled) != controls._active_parent_edges(original)
    assert original == _snapshot()


def test_create_hierarchy_variants_writes_manifest(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_snapshot()), encoding="utf-8")
    output = tmp_path / "out"

    controls.create_hierarchy_variants(snapshot=source, output_dir=output, seed=11)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["variants"]) == set(controls.VARIANTS)
    assert manifest["invariants"]["shuffled_depth_profile_preserved"] is True
    assert (output / "original.json").is_file()
    assert (output / "flattened.json").is_file()
    assert (output / "shuffled.json").is_file()


def test_hierarchy_read_cost_bootstrap_preserves_paired_direction() -> None:
    by_variant = {
        "original": {"c1": (20.0, 2), "c2": (40.0, 2)},
        "flattened": {"c1": (40.0, 2), "c2": (80.0, 2)},
        "shuffled": {"c1": (20.0, 2), "c2": (40.0, 2)},
    }

    differences, ratios = controls._paired_bootstrap(
        by_variant,
        resamples=100,
        seed=7,
    )

    assert set(differences) == {"flattened", "shuffled"}
    assert ratios["flattened"] == [2.0] * 100
    assert all(value > 0 for value in differences["flattened"])
    assert ratios["shuffled"] == [1.0] * 100


def test_completed_snapshot_index_uses_snapshot_presence_only(tmp_path) -> None:
    index = tmp_path / "index.jsonl"
    rows = [
        {
            "question_id": "corpus-1",
            "amortization_questions": [{"question_id": "q1"}, {"question_id": "q2"}],
        },
        {
            "question_id": "corpus-2",
            "amortization_questions": [{"question_id": "q3"}],
        },
    ]
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    snapshot = tmp_path / "snapshots" / "corpus-1" / "trajectory_00.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps(_snapshot()), encoding="utf-8")
    output = tmp_path / "subset.jsonl"
    manifest = tmp_path / "subset.manifest.json"

    controls.create_completed_snapshot_index(
        snapshots_root=tmp_path / "snapshots",
        index=index,
        output=output,
        manifest=manifest,
    )

    selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert [row["question_id"] for row in selected] == ["corpus-1"]
    assert metadata["included_corpora"] == 1
    assert metadata["included_questions"] == 2
    assert metadata["excluded_corpus_ids"] == ["corpus-2"]


def test_compact_ledger_rejects_error_rows(tmp_path) -> None:
    source = tmp_path / "execution.jsonl"
    rows = [
        {
            "variant": variant,
            "corpus_id": "corpus-1",
            "question_id": "q1",
            "correctness": 1.0,
            "exact_match": 0.0,
            "answerer_total_tokens_read": 10.0,
            "construction_complete": True,
            "answerer_error": variant == "shuffled",
            "judge_error": False,
            "error": "",
        }
        for variant in controls.VARIANTS
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete or errored"):
        controls.write_compact_score_ledger(
            results=source,
            output=tmp_path / "compact.jsonl",
            expected_rows=3,
            expected_corpora=1,
            expected_questions=1,
        )

    rows[-1]["answerer_error"] = False
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "compact.jsonl"
    controls.write_compact_score_ledger(
        results=source,
        output=output,
        expected_rows=3,
        expected_corpora=1,
        expected_questions=1,
    )
    compact = [json.loads(line) for line in output.read_text().splitlines()]
    assert set(compact[0]) == {
        "answerer_total_tokens_read",
        "corpus_id",
        "correctness",
        "question_id",
        "variant",
    }
