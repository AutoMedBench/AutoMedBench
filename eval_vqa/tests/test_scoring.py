"""End-to-end unit tests for the VQA scoring pipeline."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from answer_judge import AnswerJudge
from run_eval import run_eval


def _stage(public: Path, private: Path, predictions: Path, samples: list[dict]) -> list[str]:
    qids: list[str] = []
    for sample in samples:
        qid = sample["question_id"]
        qids.append(qid)
        qdir = public / qid
        qdir.mkdir(parents=True, exist_ok=True)
        (qdir / "question.json").write_text(json.dumps({
            "question_id": qid,
            "question_type": sample.get("question_type", "mcq"),
            "medical_task": sample.get("medical_task", "unknown"),
            "body_system": sample.get("body_system", "unknown"),
            "split": "test",
        }))
        adir = private / qid
        adir.mkdir(parents=True, exist_ok=True)
        (adir / "answer.json").write_text(json.dumps({
            "question_id": qid,
            "answer_label": sample.get("answer_label"),
            "answer_text": sample.get("answer_text"),
        }))
    # pretend format_checker submission artefact exists
    (predictions / "submission.json").write_text(json.dumps({
        "task": "test", "predictions": []
    }))
    return qids


def test_mcq_scoring_end_to_end(tmp_path):
    public = tmp_path / "public"
    private = tmp_path / "private"
    preds = tmp_path / "preds"
    preds.mkdir(parents=True)

    qids = _stage(public, private, preds, [
        {"question_id": "q1", "answer_label": "B", "answer_text": "Option B"},
        {"question_id": "q2", "answer_label": "A", "answer_text": "Option A"},
    ])
    for qid, label in zip(qids, ["B", "C"]):  # 1 correct, 1 wrong
        d = preds / qid
        d.mkdir()
        (d / "answer.json").write_text(json.dumps({
            "question_id": qid,
            "predicted_label": label,
            "predicted_answer": "x",
            "raw_model_output": "x",
            "model_name": "test",
            "runtime_s": 0.1,
        }))

    report = run_eval(
        gt_dir=str(private),
        agent_dir=str(preds),
        public_dir=str(public),
        question_ids=qids,
        tier="lite",
        workspace_dir=str(preds),
        answer_mode="multiple_choice",
    )
    assert report["metrics"]["accuracy"] == 0.5
    assert report["metrics"]["completion_rate"] == 1.0
    assert "tool_usage" in report
    assert report["step_scores"]["s4"] is not None


def test_open_ended_scoring(tmp_path):
    public = tmp_path / "public"
    private = tmp_path / "private"
    preds = tmp_path / "preds"
    preds.mkdir(parents=True)

    qids = _stage(public, private, preds, [
        {"question_id": "p1", "answer_text": "yes"},
        {"question_id": "p2", "answer_text": "no tumor"},
    ])
    answers = ["yes", "tumor present"]
    for qid, ans in zip(qids, answers):
        d = preds / qid
        d.mkdir()
        (d / "answer.json").write_text(json.dumps({
            "question_id": qid,
            "predicted_label": "",
            "predicted_answer": ans,
            "raw_model_output": ans,
            "model_name": "test",
            "runtime_s": 0.1,
        }))

    report = run_eval(
        gt_dir=str(private),
        agent_dir=str(preds),
        public_dir=str(public),
        question_ids=qids,
        tier="lite",
        workspace_dir=str(preds),
        answer_mode="open_ended",
    )
    # p1: gold "yes" is yes/no -> yes_no_accuracy=1.0
    # p2: gold "no tumor" -> 0.5*EM + 0.5*F1
    #     EM=0, F1: overlap={tumor} on pred={tumor, present} gold={no, tumor}
    #     -> precision=1/2, recall=1/2, F1=0.5 -> sample=0.25
    # mean = (1.0 + 0.25) / 2 = 0.625
    assert report["metrics"]["accuracy"] == 0.625
    assert report["metrics"]["yes_no_accuracy"] == 1.0
    assert report["metrics"]["yes_no_count"] == 1


def test_open_ended_scoring_with_answer_judge(tmp_path):
    public = tmp_path / "public"
    private = tmp_path / "private"
    preds = tmp_path / "preds"
    preds.mkdir(parents=True)

    qids = _stage(public, private, preds, [
        {"question_id": "p1", "answer_text": "neoplasm"},
        {"question_id": "p2", "answer_text": "gastrointestinal"},
    ])
    # Predictions are clinically equivalent but tokenwise disjoint from gold.
    answers = ["tumor", "GI tract"]
    for qid, ans in zip(qids, answers):
        d = preds / qid
        d.mkdir()
        (d / "answer.json").write_text(json.dumps({
            "question_id": qid,
            "predicted_label": "",
            "predicted_answer": ans,
            "raw_model_output": ans,
            "model_name": "test",
            "runtime_s": 0.1,
        }))

    def _fake_backend(system: str, user: str):
        # Both pairs are synonyms — award full credit.
        return {"score": 1, "rationale": "synonym"}, 10, 5

    judge = AnswerJudge(
        model="test-synonym-judge",
        backend=_fake_backend,
        cache_path=str(tmp_path / "judge_cache.jsonl"),
        judge_backend_name="fake",
    )

    report = run_eval(
        gt_dir=str(private),
        agent_dir=str(preds),
        public_dir=str(public),
        question_ids=qids,
        tier="lite",
        workspace_dir=str(preds),
        answer_mode="open_ended",
        answer_judge=judge,
    )

    # Judge promoted to primary accuracy.
    assert report["metrics"]["accuracy"] == 1.0
    assert report["metrics"]["accuracy_judge"] == 1.0
    # Heuristic accuracy stays low because token F1 doesn't recognise synonyms.
    assert report["metrics"]["accuracy_heuristic"] < 0.5
    assert report["metrics"]["judge_enabled"] is True
    assert report["metrics"]["judge_samples"] == 2
    assert report["metrics"]["judge_model"] == "test-synonym-judge"
    for qid in qids:
        row = report["per_question"][qid]
        assert row["judge_score"] == 1.0
        assert "judge_rationale" in row


def test_answer_judge_fallback_is_counted(tmp_path, capsys):
    """When the backend raises, the judge falls back to heuristic; the scorer
    must surface that count so operators can distinguish LLM-graded from
    heuristic scores."""
    public = tmp_path / "public"
    private = tmp_path / "private"
    preds = tmp_path / "preds"
    preds.mkdir(parents=True)

    qids = _stage(public, private, preds, [
        {"question_id": "p1", "answer_text": "neoplasm"},
        {"question_id": "p2", "answer_text": "gastrointestinal"},
    ])
    for qid, ans in zip(qids, ["neoplasm", "GI tract"]):
        d = preds / qid
        d.mkdir()
        (d / "answer.json").write_text(json.dumps({
            "question_id": qid,
            "predicted_label": "",
            "predicted_answer": ans,
            "raw_model_output": ans,
            "model_name": "test",
            "runtime_s": 0.1,
        }))

    def _broken_backend(system: str, user: str):
        raise RuntimeError("simulated backend outage")

    judge = AnswerJudge(
        model="test-fallback",
        backend=_broken_backend,
        cache_path=str(tmp_path / "judge_cache.jsonl"),
        judge_backend_name="fake",
        heuristic_fallback=True,
    )

    report = run_eval(
        gt_dir=str(private),
        agent_dir=str(preds),
        public_dir=str(public),
        question_ids=qids,
        tier="lite",
        workspace_dir=str(preds),
        answer_mode="open_ended",
        answer_judge=judge,
    )

    metrics = report["metrics"]
    # p1 pred exactly matches gold → judge shortcut (score 1.0, not a fallback).
    # p2 pred differs → backend raises → fallback counted.
    assert metrics["judge_samples"] == 2
    assert metrics["judge_fallback_count"] == 1
    captured = capsys.readouterr()
    assert "fell back to heuristic" in captured.err
