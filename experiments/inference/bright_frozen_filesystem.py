"""Answer BRIGHT questions from one already-frozen, question-blind filesystem."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from functools import cache
from pathlib import Path
from typing import Any

from environments.synthetic_fs.dataset import load_index
from environments.synthetic_fs.reward import SyntheticFilesystemReward
from environments.synthetic_fs.workspace import SyntheticFilesystemState


PROTOCOL = "query_blind_shared_domain_corpus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--answerer-model", required=True)
    parser.add_argument("--answerer-base-url", required=True)
    parser.add_argument("--answerer-api-key-env", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-api-key-env", required=True)
    parser.add_argument("--samples-per-question", type=int, default=4)
    parser.add_argument("--answerer-temperature", type=float, default=0.7)
    parser.add_argument("--answerer-top-p", type=float, default=1.0)
    parser.add_argument("--answerer-max-turns", type=int, default=32)
    parser.add_argument("--answerer-max-output-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["question_id"]), int(record["sample_idx"])


@cache
def _implementation_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in (
        "experiments/inference/bright_frozen_filesystem.py",
        "environments/synthetic_fs/common.py",
        "environments/synthetic_fs/reward.py",
        "environments/synthetic_fs/prompts.py",
        "environments/synthetic_fs/workspace.py",
    ):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _input_sha256(
    datum: dict[str, Any], args: argparse.Namespace, snapshot_sha256: str
) -> str:
    payload = {
        "protocol": PROTOCOL,
        "domain": args.domain,
        "question_id": str(datum["question_id"]),
        "question": str(datum["question"]),
        "gold_answer": str(datum["gold_answer"]),
        "snapshot_sha256": snapshot_sha256,
        "answerer_model": args.answerer_model,
        "answerer_base_url": args.answerer_base_url,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "answerer_temperature": args.answerer_temperature,
        "answerer_top_p": args.answerer_top_p,
        "answerer_max_turns": args.answerer_max_turns,
        "answerer_max_output_tokens": args.answerer_max_output_tokens,
        "answerer_workspace_mode": "synthetic_only",
        "implementation_sha256": _implementation_sha256(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


async def answer_one(
    datum: dict[str, Any],
    sample_idx: int,
    args: argparse.Namespace,
    snapshot_sha256: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        started = time.monotonic()
        state = SyntheticFilesystemState.load_snapshot(
            args.snapshot,
            raw_doc_root=Path(str(datum["agent_query_dir"])),
        )
        reward = SyntheticFilesystemReward(
            state=state,
            gold_answer=str(datum["gold_answer"]),
            question=str(datum["question"]),
            answerer_backend="gemini",
            answerer_model=args.answerer_model,
            answerer_base_url=args.answerer_base_url,
            answerer_api_key_env=args.answerer_api_key_env,
            judge_backend="gemini",
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            judge_api_key_env=args.judge_api_key_env,
            answerer_workspace_mode="synthetic_only",
            answerer_max_turns=args.answerer_max_turns,
            answerer_max_output_tokens=args.answerer_max_output_tokens,
            answerer_final_answer_max_tokens=args.answerer_max_output_tokens,
            answerer_temperature=args.answerer_temperature,
            answerer_top_p=args.answerer_top_p,
            terminal_answerer_repeats=1,
        )
        try:
            answer = await reward.run_answerer()
            usage = reward._answerer_usage_metrics(answer)
            judged = await reward._score_answerer_output(
                answerer_output=str(answer.get("answer_text", "")),
                synthetic_read_count=usage["synthetic_read_count"],
                answerer_evaluated=1.0,
                terminal_step=1.0,
            )
            return {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "method": "librarian_frozen_snapshot",
                "domain": args.domain,
                "question_id": str(datum["question_id"]),
                "sample_idx": sample_idx,
                "snapshot": str(args.snapshot.resolve()),
                "snapshot_sha256": snapshot_sha256,
                "correctness": float(judged.get("correct", 0.0) or 0.0),
                "judge_error": bool(judged.get("judge_error", 0.0)),
                "answer": str(answer.get("answer_text", "")),
                "answerer_steps": int(answer.get("steps", 0) or 0),
                "read_paths": [str(path) for path in answer.get("read_paths", [])],
                "synthetic_chars_read": float(answer.get("synthetic_chars_read", 0) or 0),
                "raw_chars_read": float(answer.get("raw_chars_read", 0) or 0),
                "readme_chars_read": float(answer.get("readme_chars_read", 0) or 0),
                "answerer_total_tokens_read": usage["answerer_total_tokens_read"],
                "elapsed_seconds": time.monotonic() - started,
                "answer_model": args.answerer_model,
                "judge_model": args.judge_model,
                "answerer_temperature": args.answerer_temperature,
                "answerer_top_p": args.answerer_top_p,
                "answerer_max_turns": args.answerer_max_turns,
                "answerer_max_output_tokens": args.answerer_max_output_tokens,
                "answerer_workspace_mode": "synthetic_only",
                "evaluation_input_sha256": _input_sha256(datum, args, snapshot_sha256),
                "error": "",
            }
        except Exception as exc:
            return {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "method": "librarian_frozen_snapshot",
                "domain": args.domain,
                "question_id": str(datum["question_id"]),
                "sample_idx": sample_idx,
                "snapshot": str(args.snapshot.resolve()),
                "snapshot_sha256": snapshot_sha256,
                "correctness": 0.0,
                "judge_error": False,
                "answer": "",
                "answerer_steps": 0,
                "read_paths": [],
                "synthetic_chars_read": 0.0,
                "raw_chars_read": 0.0,
                "readme_chars_read": 0.0,
                "answerer_total_tokens_read": 0.0,
                "elapsed_seconds": time.monotonic() - started,
                "answer_model": args.answerer_model,
                "judge_model": args.judge_model,
                "answerer_temperature": args.answerer_temperature,
                "answerer_top_p": args.answerer_top_p,
                "answerer_max_turns": args.answerer_max_turns,
                "answerer_max_output_tokens": args.answerer_max_output_tokens,
                "answerer_workspace_mode": "synthetic_only",
                "evaluation_input_sha256": _input_sha256(datum, args, snapshot_sha256),
                "error": f"{type(exc).__name__}: {exc}",
            }


async def run(args: argparse.Namespace) -> None:
    index_path = args.shared_root / "by_domain" / args.domain / "index.jsonl"
    data = list(load_index(index_path))
    if len(data) != 50:
        raise ValueError(f"Expected 50 held-out questions in {index_path}, found {len(data)}")
    if args.samples_per_question != 4:
        raise ValueError("Final BRIGHT frozen evaluation uses exactly four answer samples per question")
    snapshot_sha256 = _sha256(args.snapshot)
    existing: dict[tuple[str, int], dict[str, Any]] = {}
    if args.resume and args.output_path.exists():
        for line in args.output_path.read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                existing[_result_key(record)] = record
    expected = {(str(datum["question_id"]), sample) for datum in data for sample in range(4)}
    by_question = {str(datum["question_id"]): datum for datum in data}
    invalid = set(existing) - expected
    if invalid:
        raise ValueError(f"Output contains records outside this exact manifest: {sorted(invalid)[:3]}")
    resume_fields = {
        "protocol": PROTOCOL,
        "method": "librarian_frozen_snapshot",
        "domain": args.domain,
        "snapshot_sha256": snapshot_sha256,
        "answer_model": args.answerer_model,
        "judge_model": args.judge_model,
        "answerer_temperature": args.answerer_temperature,
        "answerer_top_p": args.answerer_top_p,
        "answerer_max_turns": args.answerer_max_turns,
        "answerer_max_output_tokens": args.answerer_max_output_tokens,
        "answerer_workspace_mode": "synthetic_only",
    }
    for key, record in existing.items():
        expected_fields = {
            **resume_fields,
            "evaluation_input_sha256": _input_sha256(
                by_question[key[0]], args, snapshot_sha256
            ),
        }
        mismatched = [
            field for field, value in expected_fields.items() if record.get(field) != value
        ]
        if mismatched:
            raise ValueError(f"Resume configuration mismatch for {key}: {mismatched}")
    semaphore = asyncio.Semaphore(args.concurrency)
    pending = [
        answer_one(datum, sample, args, snapshot_sha256, semaphore)
        for datum in data
        for sample in range(4)
        if (str(datum["question_id"]), sample) not in existing
    ]
    records = list(existing.values()) + await asyncio.gather(*pending)
    records.sort(key=lambda record: _result_key(record))
    if {_result_key(record) for record in records} != expected:
        raise ValueError("Frozen answer output does not cover the exact 50x4 manifest")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if not args.snapshot.is_file():
        raise ValueError(f"Snapshot does not exist: {args.snapshot}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
