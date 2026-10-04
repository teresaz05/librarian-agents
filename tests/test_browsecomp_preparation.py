import hashlib
import json

import pytest

from data_processing.prepare_browsecomp import verify_inputs


def test_verify_inputs_checks_source_and_heldout_hashes(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"query_id":"q1"}\n', encoding="utf-8")
    heldout = tmp_path / "heldout.json"
    heldout.write_text('["q1"]\n', encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "validated_source": {
                    "bytes": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "rows": 1,
                },
                "heldout_validation": {
                    "manifest_sha256": hashlib.sha256(heldout.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )

    assert verify_inputs(source, heldout, metadata) == 1

    heldout.write_text('["q2"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Held-out manifest SHA-256"):
        verify_inputs(source, heldout, metadata)
