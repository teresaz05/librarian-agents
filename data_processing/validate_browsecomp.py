"""Validate a prepared BrowseComp+ split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id", "")).strip()
            if not question_id:
                raise ValueError(f"Missing question_id at {path}:{line_number}")
            if question_id in ids:
                raise ValueError(f"Duplicate question_id in {path}: {question_id}")
            ids.add(question_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    train = read_ids(args.root / "train" / "index.jsonl")
    heldout = read_ids(args.root / "heldout50" / "index.jsonl")
    if len(train) != 680:
        raise ValueError(f"Expected 680 training tasks, found {len(train)}")
    if len(heldout) != 50:
        raise ValueError(f"Expected 50 held-out tasks, found {len(heldout)}")
    if train & heldout:
        raise ValueError("Training and held-out IDs overlap")
    print("BrowseComp+ split: 680 train, 50 held out, no overlap")


if __name__ == "__main__":
    main()
