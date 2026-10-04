#!/usr/bin/env python3
"""Evaluate original, flattened, and shuffled BrowseComp filesystem snapshots."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from functools import cache
from pathlib import Path
from typing import Any

from environments.synthetic_fs.reward import SyntheticFilesystemReward
from environments.synthetic_fs.workspace import SyntheticFilesystemState


VARIANTS = ("original", "flattened", "shuffled")
PROTOCOL = "browsecomp_hierarchy_content_preserving_ablation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["variant"]), str(row["question_id"])


@cache
def _implementation_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in (
        "experiments/inference/browsecomp_hierarchy_ablation.py",
        "environments/synthetic_fs/common.py",
        "environments/synthetic_fs/reward.py",
        "environments/synthetic_fs/prompts.py",
        "environments/synthetic_fs/workspace.py",
    ):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid evaluation manifest: {path}")
    corpus_ids = [str(row["question_id"]) for row in rows]
    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("Evaluation manifest contains duplicate corpus IDs")
    return rows


def _input_sha256(
    datum: dict[str, Any],
    question: dict[str, Any],
    variant: str,
    snapshot_sha256: str,
    args: argparse.Namespace,
) -> str:
    payload = {
        "protocol": PROTOCOL,
        "variant": variant,
        "corpus_id": str(datum["question_id"]),
        "question_id": str(question["question_id"]),
        "question": str(question["question"]),
        "gold_answer": str(question["gold_answer"]),
        "snapshot_sha256": snapshot_sha256,
        "answerer_model": args.answerer_model,
        "answerer_base_url": args.answerer_base_url,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "answerer_max_turns": args.answerer_max_turns,
        "answerer_max_output_tokens": args.answerer_max_output_tokens,
        "answerer_final_answer_max_tokens": args.answerer_final_answer_max_tokens,
        "answerer_workspace_mode": "synthetic_only",
        "implementation_sha256": _implementation_sha256(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--variants-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answerer-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--answerer-base-url", default="https://generativelanguage.googleapis.com/v1beta")
    parser.add_argument("--answerer-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--judge-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--judge-base-url", default="https://generativelanguage.googleapis.com/v1beta")
    parser.add_argument("--judge-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--answerer-max-turns", type=int, default=32)
    parser.add_argument("--answerer-max-output-tokens", type=int, default=512)
    parser.add_argument("--answerer-final-answer-max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--missing-snapshot-score-zero", action="store_true")
    return parser.parse_args()


async def _evaluate_one(
    *,
    datum: dict[str, Any],
    question: dict[str, Any],
    variant: str,
    snapshot: Path,
    snapshot_sha256: str,
    retry_index: int,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        started = time.monotonic()
        common = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "variant": variant,
            "corpus_id": str(datum["question_id"]),
            "question_id": str(question["question_id"]),
            "snapshot_sha256": snapshot_sha256,
            "retry_index": retry_index,
            "answerer_model": args.answerer_model,
            "judge_model": args.judge_model,
            "answerer_max_turns": args.answerer_max_turns,
            "answerer_max_output_tokens": args.answerer_max_output_tokens,
            "answerer_final_answer_max_tokens": args.answerer_final_answer_max_tokens,
            "answerer_workspace_mode": "synthetic_only",
            "construction_complete": True,
            "evaluation_input_sha256": _input_sha256(
                datum, question, variant, snapshot_sha256, args
            ),
        }
        try:
            state = SyntheticFilesystemState.load_snapshot(
                snapshot,
                raw_doc_root=Path(str(datum.get("agent_query_dir", "."))),
            )
            reward = SyntheticFilesystemReward(
                state=state,
                gold_answer=str(question["gold_answer"]),
                question=str(question["question"]),
                answerer_backend="gemini",
                answerer_model=args.answerer_model,
                answerer_base_url=args.answerer_base_url,
                answerer_api_key_env=args.answerer_api_key_env,
                reward_mode="hybrid",
                judge_backend="gemini",
                judge_model=args.judge_model,
                judge_base_url=args.judge_base_url,
                judge_api_key_env=args.judge_api_key_env,
                answerer_workspace_mode="synthetic_only",
                answerer_max_turns=args.answerer_max_turns,
                answerer_max_output_tokens=args.answerer_max_output_tokens,
                answerer_final_answer_max_tokens=args.answerer_final_answer_max_tokens,
                answerer_temperature=0.0,
                terminal_answerer_repeats=1,
            )
            answer, usage, score, _ = await reward._run_scored_answerer_repeats(repeats=1)
            return {
                **common,
                "correctness": float(score.get("correct", 0.0)),
                "exact_match": float(score.get("exact_match", 0.0)),
                "judge_used": bool(score.get("judge_used", 0.0)),
                "answerer_error": bool(score.get("answerer_error", 0.0)),
                "judge_error": bool(score.get("judge_error", 0.0)),
                "answerer_total_tokens_read": float(usage.get("answerer_total_tokens_read", 0.0)),
                "synthetic_read_count": float(usage.get("synthetic_read_count", 0.0)),
                "answer": str(answer.get("answer_text", "")),
                "read_paths": [str(path) for path in answer.get("read_paths", [])],
                "elapsed_seconds": time.monotonic() - started,
                "error": "",
            }
        except Exception as exc:
            return {
                **common,
                "correctness": 0.0,
                "exact_match": 0.0,
                "judge_used": False,
                "answerer_error": True,
                "judge_error": False,
                "answerer_total_tokens_read": 0.0,
                "synthetic_read_count": 0.0,
                "answer": "",
                "read_paths": [],
                "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }


async def run(args: argparse.Namespace) -> None:
    data = _load_manifest(args.index)
    expected = {
        (variant, str(question["question_id"]))
        for datum in data
        for question in datum.get("amortization_questions", [])
        for variant in args.variants
    }
    if not expected:
        raise ValueError("The evaluation index contains no amortization questions")
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                existing[_key(row)] = row
    if set(existing) - expected:
        raise ValueError("Resume output contains records outside this frozen manifest")
    previous = dict(existing)
    if args.retry_errors:
        existing = {
            key: row
            for key, row in existing.items()
            if not bool(row.get("answerer_error"))
            and not bool(row.get("judge_error"))
            and not str(row.get("error", ""))
        }

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    for datum in data:
        corpus_id = str(datum["question_id"])
        for variant in args.variants:
            snapshot = args.variants_root / corpus_id / f"{variant}.json"
            if not snapshot.is_file():
                if not args.missing_snapshot_score_zero:
                    raise FileNotFoundError(f"Missing hierarchy variant: {snapshot}")
                fingerprint = "missing_snapshot"
                for question in datum.get("amortization_questions", []):
                    key = (variant, str(question["question_id"]))
                    expected_input = _input_sha256(
                        datum, question, variant, fingerprint, args
                    )
                    if key in existing:
                        if existing[key].get("evaluation_input_sha256") != expected_input:
                            raise ValueError(f"Resume configuration mismatch for {key}")
                        continue
                    records_row = {
                        "schema_version": 1,
                        "protocol": PROTOCOL,
                        "variant": variant,
                        "corpus_id": corpus_id,
                        "question_id": str(question["question_id"]),
                        "snapshot_sha256": fingerprint,
                        "retry_index": 0,
                        "answerer_model": args.answerer_model,
                        "judge_model": args.judge_model,
                        "answerer_max_turns": args.answerer_max_turns,
                        "answerer_max_output_tokens": args.answerer_max_output_tokens,
                        "answerer_final_answer_max_tokens": (
                            args.answerer_final_answer_max_tokens
                        ),
                        "answerer_workspace_mode": "synthetic_only",
                        "evaluation_input_sha256": expected_input,
                        "construction_complete": False,
                        "correctness": 0.0,
                        "exact_match": 0.0,
                        "judge_used": False,
                        "answerer_error": False,
                        "judge_error": False,
                        "answerer_total_tokens_read": 0.0,
                        "synthetic_read_count": 0.0,
                        "answer": "",
                        "read_paths": [],
                        "elapsed_seconds": 0.0,
                        "error": "",
                    }
                    existing[key] = records_row
                continue
            fingerprint = _sha256(snapshot)
            for question in datum.get("amortization_questions", []):
                key = (variant, str(question["question_id"]))
                if key in existing:
                    expected_resume = {
                        "protocol": PROTOCOL,
                        "variant": variant,
                        "corpus_id": corpus_id,
                        "snapshot_sha256": fingerprint,
                        "answerer_model": args.answerer_model,
                        "judge_model": args.judge_model,
                        "answerer_max_turns": args.answerer_max_turns,
                        "answerer_max_output_tokens": args.answerer_max_output_tokens,
                        "answerer_final_answer_max_tokens": args.answerer_final_answer_max_tokens,
                        "answerer_workspace_mode": "synthetic_only",
                        "construction_complete": True,
                        "evaluation_input_sha256": _input_sha256(
                            datum, question, variant, fingerprint, args
                        ),
                    }
                    mismatched = [
                        field
                        for field, value in expected_resume.items()
                        if existing[key].get(field) != value
                    ]
                    if mismatched:
                        raise ValueError(f"Resume configuration mismatch for {key}: {mismatched}")
                if key not in existing:
                    prior = previous.get(key)
                    retry_index = (
                        int(prior.get("retry_index", 0)) + 1 if prior is not None else 0
                    )
                    tasks.append(
                        _evaluate_one(
                            datum=datum,
                            question=question,
                            variant=variant,
                            snapshot=snapshot,
                            snapshot_sha256=fingerprint,
                            retry_index=retry_index,
                            args=args,
                            semaphore=semaphore,
                        )
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = dict(existing)
    for future in asyncio.as_completed(tasks):
        row = await future
        records[_key(row)] = row
        ordered = sorted(records.values(), key=_key)
        args.output.write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in ordered),
            encoding="utf-8",
        )
    if set(records) != expected:
        raise ValueError(f"Incomplete hierarchy evaluation: {len(records)}/{len(expected)}")
    ordered = sorted(records.values(), key=_key)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in ordered),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
