#!/usr/bin/env python3
"""Deterministic exact-match scorer for staged VQA outputs."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Any

from answer_utils import VALID_LABELS, normalize_label
from answer_metrics import exact_match, token_f1, yes_no_accuracy
from answer_normalizer import is_yes_no_answer
from answer_judge import AnswerJudge, judge_agreement_rate
from format_checker import detect_placeholder


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _update_breakdown(store: dict[str, dict[str, float]], key: str, correct: bool, valid: bool) -> None:
    row = store.setdefault(key or "unknown", {"total": 0, "correct": 0, "valid": 0})
    row["total"] += 1
    row["correct"] += int(correct)
    row["valid"] += int(valid)


def _finalize_breakdown(store: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key, row in sorted(store.items()):
        total = int(row["total"])
        correct = int(row["correct"])
        valid = int(row["valid"])
        result[key] = {
            "total": total,
            "correct": correct,
            "valid": valid,
            "accuracy": _safe_ratio(correct, total),
            "valid_output_rate": _safe_ratio(valid, total),
        }
    return result


def load_ground_truth_index(gt_dir: str) -> dict[str, dict[str, Any]]:
    ground_truth: dict[str, dict[str, Any]] = {}
    csv_path = os.path.join(gt_dir, "ground_truth.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                question_id = row["question_id"]
                ground_truth[question_id] = row
    return ground_truth


def score_all(
    pred_dir: str,
    gt_dir: str,
    public_dir: str,
    question_ids: list[str],
    answer_mode: str = "multiple_choice",
    answer_judge: "AnswerJudge | None" = None,
) -> dict[str, Any]:
    per_question: dict[str, dict[str, Any]] = {}
    gt_index = load_ground_truth_index(gt_dir)
    open_ended = answer_mode == "open_ended"
    judge_active = open_ended and answer_judge is not None
    judge_score_sum = 0.0
    judge_samples = 0
    judge_fallback_count = 0
    judge_model_name = answer_judge.model if answer_judge is not None else ""
    judge_backend_name = answer_judge.judge_backend_name if answer_judge is not None else ""

    counts = {
        "expected": len(question_ids),
        "prediction_files": 0,
        "valid_outputs": 0,
        "parsed_predictions": 0,
        "correct_predictions": 0,
        "placeholder_predictions": 0,
    }
    em_sum = 0.0
    f1_sum = 0.0
    open_score_sum = 0.0
    yes_no_sum = 0.0
    yes_no_count = 0
    medical_task = defaultdict(dict)
    body_system = defaultdict(dict)
    question_type = defaultdict(dict)

    for question_id in question_ids:
        question = _load_json(os.path.join(public_dir, question_id, "question.json"))
        gt = gt_index.get(question_id) or _load_json(os.path.join(gt_dir, question_id, "answer.json"))
        pred_path = os.path.join(pred_dir, question_id, "answer.json")

        predicted_label = None
        predicted_answer_text = ""
        raw_output_text = ""
        prediction_exists = os.path.isfile(pred_path)
        valid_output = False
        is_placeholder = False
        placeholder_reason = ""
        pred = None
        if prediction_exists:
            counts["prediction_files"] += 1
            try:
                pred = _load_json(pred_path)
                predicted_label = normalize_label(pred.get("predicted_label"))
                predicted_answer_text = str(pred.get("predicted_answer") or "").strip()
                raw_output_text = str(pred.get("raw_model_output") or "")
                if open_ended:
                    valid_output = bool(predicted_answer_text)
                else:
                    valid_output = predicted_label in VALID_LABELS
                is_placeholder, placeholder_reason = detect_placeholder(
                    pred.get("raw_model_output"), pred.get("predicted_answer")
                )
            except Exception:
                pred = None
                valid_output = False

        if is_placeholder:
            counts["placeholder_predictions"] += 1
            valid_output = False

        parsed = valid_output
        sample_score = 0.0
        is_yes_no = False
        sample_yes_no_score = 0.0
        if open_ended:
            gold_text = str(gt.get("answer_text") or gt.get("answer") or "").strip()
            # answer_metrics returns 0-100; normalise to 0-1 here.
            em = (exact_match(predicted_answer_text, gold_text) / 100.0) if parsed else 0.0
            f1 = (token_f1(predicted_answer_text, gold_text) / 100.0) if parsed else 0.0
            em_sum += em
            f1_sum += f1
            is_yes_no = is_yes_no_answer(gold_text)
            if is_yes_no:
                # For yes/no questions, use strict yes/no accuracy so
                # "yes, the cyst wall ..." does not get inflated F1.
                sample_yes_no_score = (
                    yes_no_accuracy(predicted_answer_text, gold_text) / 100.0
                ) if parsed else 0.0
                yes_no_sum += sample_yes_no_score
                yes_no_count += 1
                sample_score = sample_yes_no_score
            else:
                # PathVQA / VQA-RAD convention: combine EM and token F1.
                sample_score = 0.5 * em + 0.5 * f1
            open_score_sum += sample_score
            correct = sample_score >= 1.0
        else:
            correct = bool(parsed and predicted_label == gt.get("answer_label"))

        judge_score_value: float | None = None
        judge_rationale_value: str = ""
        judge_cached_value: bool = False
        if judge_active and parsed:
            gold_for_judge = str(gt.get("answer_text") or gt.get("answer") or "").strip()
            question_text = str(question.get("question") or question.get("stem") or "")
            verdict = answer_judge.judge_one(
                qid=question_id,
                question=question_text,
                gold=gold_for_judge,
                pred=predicted_answer_text,
                raw=raw_output_text,
            )
            judge_score_value = float(verdict.score)
            judge_rationale_value = verdict.rationale
            judge_cached_value = verdict.cached
            judge_score_sum += judge_score_value
            judge_samples += 1
            if verdict.judge_backend == "heuristic_fallback":
                judge_fallback_count += 1
        counts["valid_outputs"] += int(valid_output)
        counts["parsed_predictions"] += int(parsed)
        counts["correct_predictions"] += int(correct)

        row = {
            "question_id": question_id,
            "ground_truth_label": gt.get("answer_label"),
            "ground_truth_answer": gt.get("answer_text") or gt.get("answer"),
            "predicted_label": predicted_label,
            "predicted_answer": predicted_answer_text,
            "prediction_exists": prediction_exists,
            "valid_output": valid_output,
            "parsed": parsed,
            "correct": correct,
            "sample_score": round(sample_score, 4) if open_ended else None,
            "is_placeholder": is_placeholder,
            "placeholder_reason": placeholder_reason,
            "raw_output_len": len(raw_output_text),
            "medical_task": question.get("medical_task"),
            "body_system": question.get("body_system"),
            "question_type": question.get("question_type"),
            "split": question.get("split"),
        }
        if judge_score_value is not None:
            row["judge_score"] = judge_score_value
            row["judge_rationale"] = judge_rationale_value
            row["judge_cached"] = judge_cached_value
        per_question[question_id] = row

        _update_breakdown(medical_task, row["medical_task"], correct, valid_output)
        _update_breakdown(body_system, row["body_system"], correct, valid_output)
        _update_breakdown(question_type, row["question_type"], correct, valid_output)

    total = counts["expected"]
    em_rate = round(em_sum / total, 4) if open_ended and total > 0 else 0.0
    f1_rate = round(f1_sum / total, 4) if open_ended and total > 0 else 0.0
    open_accuracy = round(open_score_sum / total, 4) if open_ended and total > 0 else 0.0
    yes_no_acc = round(yes_no_sum / yes_no_count, 4) if yes_no_count > 0 else 0.0
    heuristic_accuracy = open_accuracy if open_ended else _safe_ratio(counts["correct_predictions"], total)
    accuracy_judge = (
        round(judge_score_sum / total, 4) if judge_active and total > 0 else 0.0
    )
    # When the judge is active on open-ended, promote judge score to primary accuracy;
    # keep the heuristic accuracy available under `accuracy_heuristic` for diagnostics.
    if judge_active:
        accuracy = accuracy_judge
    else:
        accuracy = heuristic_accuracy
    agreement = judge_agreement_rate(per_question) if judge_active else 0.0
    return {
        "counts": counts,
        "answer_mode": answer_mode,
        "accuracy": accuracy,
        "accuracy_heuristic": heuristic_accuracy,
        "accuracy_judge": accuracy_judge,
        "judge_enabled": judge_active,
        "judge_samples": judge_samples,
        "judge_fallback_count": judge_fallback_count,
        "judge_model": judge_model_name,
        "judge_backend": judge_backend_name,
        "judge_agreement_rate": agreement,
        "exact_match": em_rate,
        "token_f1": f1_rate,
        "yes_no_accuracy": yes_no_acc,
        "yes_no_count": yes_no_count,
        "completion_rate": _safe_ratio(counts["prediction_files"], total),
        "parse_rate": _safe_ratio(counts["parsed_predictions"], total),
        "valid_output_rate": _safe_ratio(counts["valid_outputs"], total),
        "placeholder_rate": _safe_ratio(counts["placeholder_predictions"], total),
        "per_question": per_question,
        "breakdown": {
            "medical_task": _finalize_breakdown(medical_task),
            "body_system": _finalize_breakdown(body_system),
            "question_type": _finalize_breakdown(question_type),
        },
    }
