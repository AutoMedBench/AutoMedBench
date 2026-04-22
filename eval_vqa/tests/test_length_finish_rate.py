"""BUG-045: length_finish_rate metric computed from trace.jsonl."""

from __future__ import annotations

import json
import os

from inference_verifier import compute_length_finish_rate


def _write_trace(path: str, events: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for ev in events:
            handle.write(json.dumps(ev) + "\n")


def test_no_trace_file(tmp_path):
    r = compute_length_finish_rate(str(tmp_path))
    assert r["api_calls"] == 0
    assert r["length_finish_rate"] == 0.0


def test_mixed_finish_reasons(tmp_path):
    trace = str(tmp_path / "trace.jsonl")
    _write_trace(trace, [
        {"type": "api_call", "finish_reason": "stop"},
        {"type": "api_call", "finish_reason": "length"},
        {"type": "api_call", "finish_reason": "stop"},
        {"type": "api_call", "finish_reason": "tool_calls"},
        {"type": "other", "finish_reason": "length"},  # non-api_call ignored
    ])
    r = compute_length_finish_rate(str(tmp_path))
    assert r["api_calls"] == 4
    assert r["length_finish_count"] == 1
    assert r["length_finish_rate"] == 0.25


def test_all_length(tmp_path):
    trace = str(tmp_path / "trace.jsonl")
    _write_trace(trace, [
        {"type": "api_call", "finish_reason": "length"},
        {"type": "api_call", "finish_reason": "length"},
    ])
    r = compute_length_finish_rate(str(tmp_path))
    assert r["length_finish_rate"] == 1.0
