"""Smoke tests for medbench_vqa Python helpers."""

from __future__ import annotations

import json
import os

import pytest

from medbench_vqa import inspect_image, public_medical_search, submit_answer


def test_submit_answer_mcq(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    out = submit_answer(
        question_id="q1",
        predicted_label="b",
        predicted_answer="ischemia",
        raw_model_output="The answer is (B).",
        model_name="unit-test",
        runtime_s=0.1,
    )
    assert out["status"] == "ok"
    payload = json.loads(open(out["path"]).read())
    assert payload["predicted_label"] == "B"
    assert payload["question_id"] == "q1"

    log_path = tmp_path / "tool_calls.jsonl"
    assert log_path.exists()
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert any(r.get("tool") == "submit_answer" for r in records)


def test_submit_answer_open_ended(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    out = submit_answer(
        question_id="p1",
        predicted_answer="no tumor visible",
        raw_model_output="no tumor visible",
        model_name="unit-test",
        runtime_s=0.2,
    )
    assert out["status"] == "ok"
    payload = json.loads(open(out["path"]).read())
    assert payload["predicted_label"] == ""
    assert payload["predicted_answer"] == "no tumor visible"


def test_submit_answer_rejects_bad_label(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        submit_answer(
            question_id="q2",
            predicted_label="BB",
            predicted_answer="x",
            raw_model_output="x",
            model_name="unit-test",
            runtime_s=0.0,
        )


def test_submit_answer_allows_honest_skip(tmp_path, monkeypatch):
    """Both predicted_answer and raw_model_output empty = honest skip after VLM error."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    out = submit_answer(
        question_id="q3",
        predicted_label="",
        predicted_answer="",
        raw_model_output="",
        model_name="unit-test",
        runtime_s=0.0,
    )
    assert out["status"] == "ok"


def test_submit_answer_rejects_half_empty(tmp_path, monkeypatch):
    """Answer claimed without backing raw model output is a forged shape."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        submit_answer(
            question_id="q4",
            predicted_label="",
            predicted_answer="yes",
            raw_model_output="",
            model_name="unit-test",
            runtime_s=0.0,
        )


def test_inspect_image_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    result = inspect_image(str(tmp_path / "nope.png"))
    assert result["status"] == "partial_error"
    assert result["image_count"] == 1
    assert result["per_image"][0]["exists"] is False


def test_medical_search_rejects_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    out = public_medical_search("benchmark answer key")
    assert out["status"] == "rejected"
    assert out["results"] == []
