"""Unit tests for tool_usage summary + step-score blending."""

from __future__ import annotations

import json

from tool_usage import apply_tool_usage, load_tool_calls, summarize


def test_summarize_and_adjust(tmp_path):
    log = tmp_path / "tool_calls.jsonl"
    records = [
        {"tool": "inspect_image", "result_summary": {"status": "ok"}},
        {"tool": "submit_answer", "result_summary": {"status": "ok"}},
        {"tool": "submit_answer", "result_summary": {"status": "ok"}},
        {"tool": "public_medical_search", "result_summary": {"status": "ok"}},
    ]
    log.write_text("\n".join(json.dumps(r) for r in records))
    loaded = load_tool_calls(str(tmp_path))
    assert len(loaded) == 4

    summary = summarize(loaded, expected_samples=2)
    assert summary["submit_answer_ok"] == 2
    assert summary["submit_coverage"] == 1.0
    assert summary["inspect_used"] is True

    adjusted = apply_tool_usage({"s3": 0.5, "s4": 0.4}, summary, aux_weight=0.2)
    # s3: 0.8 * 0.5 + 0.2 * 1.0 = 0.6
    assert adjusted["s3"] == 0.6
    # s4: 0.8 * 0.4 + 0.2 * 1.0 = 0.52
    assert adjusted["s4"] == 0.52


def test_summarize_empty():
    summary = summarize([], expected_samples=10)
    assert summary["submit_coverage"] == 0.0
    assert summary["inspect_used"] is False
