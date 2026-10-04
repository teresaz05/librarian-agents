from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).parents[1] / "environments/synthetic_fs/coverage.py"
_SPEC = importlib.util.spec_from_file_location("coverage_contract", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_COVERAGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COVERAGE)
coverage_leaf_creation_required = _COVERAGE.coverage_leaf_creation_required


def test_leaf_creation_contract_only_activates_for_incomplete_coverage() -> None:
    args = {
        "visible_raw_doc_paths": ["raw_docs/a", "raw_docs/b"],
        "active_direct_raw_doc_paths": ["raw_docs/a"],
        "max_browse_only_turns": 2,
    }

    assert not coverage_leaf_creation_required(
        enabled=False, consecutive_browse_only_turns=2, **args
    )
    assert not coverage_leaf_creation_required(
        enabled=True, consecutive_browse_only_turns=1, **args
    )
    assert coverage_leaf_creation_required(
        enabled=True, consecutive_browse_only_turns=2, **args
    )
    assert not coverage_leaf_creation_required(
        enabled=True,
        visible_raw_doc_paths=["raw_docs/a"],
        active_direct_raw_doc_paths=["raw_docs/a"],
        consecutive_browse_only_turns=2,
        max_browse_only_turns=2,
    )


def test_leaf_creation_contract_rejects_nonpositive_browse_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        coverage_leaf_creation_required(
            enabled=True,
            visible_raw_doc_paths=["raw_docs/a"],
            active_direct_raw_doc_paths=[],
            consecutive_browse_only_turns=0,
            max_browse_only_turns=0,
        )
