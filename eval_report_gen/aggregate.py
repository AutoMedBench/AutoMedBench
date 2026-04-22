#!/usr/bin/env python3
"""Aggregate scoring for report-generation benchmark."""

from __future__ import annotations


STEP_WEIGHTS = {
    "s1": 0.25,
    "s2": 0.15,
    "s3": 0.35,
    "s4": 0.15,
    "s5": 0.10,
}


def compute_s4(completion_rate: float, output_format_valid: float) -> float:
    return round(0.50 * completion_rate + 0.50 * output_format_valid, 4)


def compute_s5(has_valid_results: bool, output_format_valid: bool) -> float:
    return round(0.50 * float(has_valid_results) + 0.50 * float(output_format_valid), 4)


def compute_clinical_score(
    metrics: dict,
    weights: dict | None = None,
) -> float:
    weights = weights or {"observation_f1": 0.7, "report_similarity": 0.3}
    if not weights:
        return 0.0
    score = sum(float(weights[key]) * float(metrics.get(key, 0.0)) for key in weights)
    return round(score, 4)


def compute_workflow_score(step_scores: dict, weights: dict | None = None) -> tuple[float, list[str]]:
    weights = weights or STEP_WEIGHTS
    total_weight = sum(weights.values())
    score = sum(weights[key] * (step_scores.get(key) or 0.0) for key in weights) / total_weight
    active = sorted([key for key, value in step_scores.items() if value is not None])
    return round(score, 4), active


def compute_overall_score(workflow: float, clinical: float) -> float:
    return round(0.50 * workflow + 0.50 * clinical, 4)


def assign_medal(clinical_score: float, thresholds: dict | None = None) -> dict:
    thresholds = thresholds or {"good": 0.80, "okay": 0.55}
    if clinical_score >= thresholds["good"]:
        return {"tier": 2, "name": "good"}
    if clinical_score >= thresholds["okay"]:
        return {"tier": 1, "name": "okay"}
    return {"tier": 0, "name": "fail"}


def assign_rating(medal_tier: int, format_valid: bool) -> str:
    if not format_valid:
        return "F"
    if medal_tier >= 2:
        return "A"
    if medal_tier >= 1:
        return "B"
    return "C"


def is_resolved(rating: str) -> bool:
    return rating in {"A", "B"}


def build_report(
    format_result: dict,
    score_result: dict,
    clinical_weights: dict | None = None,
    rating_thresholds: dict | None = None,
    step_weights: dict | None = None,
) -> dict:
    completion_rate = format_result.get("completion_rate", 0.0)
    output_valid = 1.0 if format_result.get("output_format_valid") else 0.0

    observation_f1 = score_result.get("mean_observation_f1", 0.0)
    report_similarity = score_result.get("mean_report_similarity", 0.0)
    exact_match = score_result.get("mean_label_exact_match", 0.0)
    clinical_components = dict(score_result.get("clinical_components", {}))
    clinical_backend = score_result.get("clinical_score_backend", "lightweight")

    all_cases_done = completion_rate >= 1.0 and bool(format_result.get("output_format_valid"))
    if not all_cases_done:
        observation_f1 = 0.0
        report_similarity = 0.0
        exact_match = 0.0
        clinical_components = {key: 0.0 for key in clinical_components}
        format_result = dict(format_result, output_format_valid=False, submission_format_valid=False)
        output_valid = 0.0

    s4 = compute_s4(completion_rate, output_valid)
    s5 = compute_s5(bool(format_result.get("any_valid_results")), bool(output_valid))
    step_scores = {"s1": None, "s2": None, "s3": None, "s4": s4, "s5": s5}

    clinical_score = compute_clinical_score(
        metrics=clinical_components,
        weights=clinical_weights,
    )
    medal = assign_medal(clinical_score, thresholds=rating_thresholds)
    workflow_score, active_steps = compute_workflow_score(step_scores, weights=step_weights)
    overall_score = compute_overall_score(workflow_score, clinical_score)
    rating = assign_rating(medal["tier"], format_valid=bool(format_result["output_format_valid"]))
    resolved = is_resolved(rating)

    progress_rate = round(
        (float(completion_rate >= 0.9) + float(format_result["output_format_valid"])) / 2,
        4,
    )

    return {
        "step_scores": step_scores,
        "metrics": {
            "observation_f1": observation_f1,
            "report_similarity": report_similarity,
            "label_exact_match": exact_match,
            "micro_precision": score_result.get("micro_precision", 0.0),
            "micro_recall": score_result.get("micro_recall", 0.0),
            "micro_f1": score_result.get("micro_f1", 0.0),
            "clinical_score_backend": clinical_backend,
            "clinical_components": clinical_components,
            "BLEU": score_result.get("BLEU", 0.0),
            "BLEU_1": score_result.get("BLEU_1", 0.0),
            "BLEU_2": score_result.get("BLEU_2", 0.0),
            "BLEU_3": score_result.get("BLEU_3", 0.0),
            "BLEU_4": score_result.get("BLEU_4", 0.0),
            "METEOR": score_result.get("METEOR", 0.0),
            "ROUGE_L": score_result.get("ROUGE_L", 0.0),
            "F1RadGraph": score_result.get("F1RadGraph", 0.0),
            "micro_average_precision": score_result.get("micro_average_precision", 0.0),
            "micro_average_recall": score_result.get("micro_average_recall", 0.0),
            "micro_average_f1": score_result.get("micro_average_f1", 0.0),
            "medal_tier": medal["tier"],
            "medal_name": medal["name"],
        },
        "format": {
            "submission_format_valid": bool(format_result["submission_format_valid"]),
            "output_format_valid": bool(format_result["output_format_valid"]),
            "completion_rate": completion_rate,
        },
        "aggregate": {
            "rating": rating,
            "resolved": resolved,
            "overall_score": overall_score,
            "agentic_score": workflow_score,
            "clinical_score": clinical_score,
            "progress_rate": progress_rate,
            "active_steps": active_steps,
        },
    }
