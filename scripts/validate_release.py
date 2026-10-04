#!/usr/bin/env python3
"""Check the repository for credentials, machine paths, and release clutter."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "outputs"}
FORBIDDEN_NAMES = {".DS_Store", "answer.txt"}
FORBIDDEN_TEXT = (
    "/Users/",
    "/home/",
    "/sailhome/",
    "/checkpoint/",
    "tinker://",
    "decrypt_browsecomp",
    "sc.stanford.edu",
    "scdt.stanford.edu",
)
SECRET_PATTERNS = (
    re.compile(r"\bgh[opurs]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
)


def release_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / value for value in completed.stdout.splitlines() if value]


def main() -> None:
    errors: list[str] = []
    for path in release_files():
        relative = path.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if path.name in FORBIDDEN_NAMES:
            errors.append(f"remove {relative}")
            continue
        if not path.is_file() or path.stat().st_size > 6_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if relative != Path("scripts/validate_release.py"):
            for marker in FORBIDDEN_TEXT:
                if marker in text:
                    errors.append(f"machine path {marker!r} in {relative}")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    errors.append(f"possible credential in {relative}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Release tree validation passed.")


if __name__ == "__main__":
    main()
