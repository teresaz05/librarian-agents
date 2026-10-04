"""Outcome-blind construction coverage contracts."""

from __future__ import annotations

from collections.abc import Iterable


def coverage_leaf_creation_required(
    *,
    enabled: bool,
    visible_raw_doc_paths: Iterable[str],
    active_direct_raw_doc_paths: Iterable[str],
    consecutive_browse_only_turns: int,
    max_browse_only_turns: int,
) -> bool:
    """Return whether the next construction action must create a source leaf.

    This is a generic construction contract: it sees only filesystem paths and
    whether active synthetic leaves carry direct raw-source provenance. It does
    not inspect questions, answers, criteria, or construction-file content.
    """
    if not enabled:
        return False
    if max_browse_only_turns < 1:
        raise ValueError("max_browse_only_turns must be positive")

    visible = set(visible_raw_doc_paths)
    covered = set(active_direct_raw_doc_paths)
    return bool(visible - covered) and consecutive_browse_only_turns >= max_browse_only_turns
