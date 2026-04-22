#!/usr/bin/env python3
"""Aggregate workflow and task scoring for VQA."""

from __future__ import annotations

STEP_WEIGHTS = {
    "s1": 0.25,
    "s2": 0.15,
    "s3": 0.35,
    "s4": 0.15,
    "s5": 0.10,
}


def compute_s4(completion_rate: float, parse_rate: float) -> float:
    return round(0.50 * completion_rate + 0.50 * parse_rate, 4)


def compute_s4_with_guards(
    completion_rate: float,
    parse_rate: float,
    placeholder_rate: float = 0.0,
    model_call_detected: bool = True,
    accuracy: float = 0.0,
) -> tuple[float, list[str]]:
    base = 0.50 * completion_rate + 0.50 * parse_rate
    penalties: list[str] = []
    if placeholder_rate > 0.05:
        base = min(base, 0.2)
        penalties.append(f"placeholder_rate={placeholder_rate:.2f} > 0.05 → cap 0.2")
    if not model_call_detected:
        base = min(base, 0.3)
        penalties.append("model_call_detected=False → cap 0.3")
    # P1-B: real-but-broken — model ran but answer quality is near zero.
    if (
        completion_rate >= 0.99
        and placeholder_rate <= 0.05
        and model_call_detected
        and accuracy < 0.05
    ):
        base = min(base, 0.5)
        penalties.append(
            f"real_but_broken: accuracy={accuracy:.3f}<0.05 despite completion={completion_rate:.2f} → cap 0.5"
        )
    return round(base, 4), penalties


def infer_inference_mode(
    placeholder_rate: float,
    placeholder_reasons: dict[str, int] | None,
    completion_rate: float,
    valid_output_rate: float,
    accuracy: float = 0.0,
    model_call_detected: bool = True,
    smoke_forward_passed: bool | None = None,
    postprocess_valid: bool | None = None,
) -> str:
    reasons = placeholder_reasons or {}
    # BUG-044: S2 passed but S3 artefacts absent and no answers → stalled in
    # S3 debug loop, distinct from plain "empty" (no tool calls) runs.
    if (
        completion_rate <= 0.0
        and smoke_forward_passed is True
        and postprocess_valid is False
    ):
        return "s3_stalled"
    if completion_rate <= 0.0:
        return "empty"
    if placeholder_rate >= 0.5:
        empties = reasons.get("raw_empty_or_generic", 0) + reasons.get("raw_empty_answer_generic", 0)
        if empties >= sum(reasons.values()) / 2:
            return "empty"
        if any(k.startswith("raw_prefix_heuristic") for k in reasons):
            return "heuristic"
        return "placeholder"
    if placeholder_rate > 0.05:
        return "degraded"
    # P1-B: distinguish real-but-broken from healthy real inference.
    if (
        completion_rate >= 0.99
        and model_call_detected
        and accuracy < 0.05
    ):
        return "real_but_broken"
    return "real"


def compute_s5(has_valid_results: bool, submission_format_valid: bool) -> float:
    return round(0.50 * float(has_valid_results) + 0.50 * float(submission_format_valid), 4)


def compute_workflow_score(step_scores: dict[str, float | None], weights: dict[str, float] | None = None) -> tuple[float, list[str]]:
    weights = weights or STEP_WEIGHTS
    # Renormalize over steps that have a concrete score — a None step (e.g.
    # S1/S3 when the workflow judge is disabled) must not silently contribute
    # 0 to the numerator while still occupying the denominator, which would
    # cap even a perfect agent at sum(active_weights)/sum(weights).
    active_steps = [step for step in weights if step_scores.get(step) is not None]
    active_weight_sum = sum(weights[step] for step in active_steps)
    if active_weight_sum <= 0:
        return 0.0, active_steps
    numerator = sum(weights[step] * float(step_scores[step]) for step in active_steps)
    score = numerator / active_weight_sum
    return round(score, 4), active_steps


def compute_overall_score(workflow_score: float, task_score: float) -> float:
    return round(0.50 * workflow_score + 0.50 * task_score, 4)


def assign_rating(medal_tier: int, completion_rate: float, submission_format_valid: bool, valid_outputs: int) -> str:
    # Hard F only when submission is outright unusable: bad schema or zero valid outputs,
    # or coverage is severely low (< 0.5 means we saw less than half the split).
    if not submission_format_valid or valid_outputs <= 0 or completion_rate < 0.5:
        return "F"
    if medal_tier >= 2:
        return "A"
    if medal_tier >= 1:
        return "B"
    return "C"


def is_resolved(rating: str) -> bool:
    return rating in ("A", "B")


def compute_s2_binary(
    smoke_forward_passed: bool | None,
    model_call_detected: bool,
    env_ready: bool,
) -> tuple[float, dict[str, bool]]:
    """P3: S2 as three binary sub-criteria — env ready, model loaded (mCall),
    smoke forward artefact valid. Equal weight mean."""
    components = {
        "env_ready": bool(env_ready),
        "model_loaded": bool(model_call_detected),
        "smoke_forward_passed": bool(smoke_forward_passed),
    }
    score = sum(components.values()) / 3.0
    return round(score, 4), components


def build_report(
    format_result: dict,
    score_result: dict,
    medal_result: dict,
    step_weights: dict[str, float] | None = None,
    step_scores: dict[str, float | None] | None = None,
    model_call_detected: bool = True,
    smoke_forward_passed: bool | None = None,
    model_call_evidence: list[str] | None = None,
    postprocess_valid: bool | None = None,
    postprocess_info: dict | None = None,
) -> dict:
    completion_rate = float(score_result.get("completion_rate", 0.0))
    parse_rate = float(score_result.get("parse_rate", 0.0))
    task_score = float(score_result.get("accuracy", 0.0))
    counts = score_result.get("counts", {})
    placeholder_rate = float(score_result.get("placeholder_rate", format_result.get("placeholder_rate", 0.0)))
    placeholder_reasons = format_result.get("placeholder_reasons", {})
    valid_output_rate = float(score_result.get("valid_output_rate", 0.0))

    s4_score, s4_penalties = compute_s4_with_guards(
        completion_rate=completion_rate,
        parse_rate=parse_rate,
        placeholder_rate=placeholder_rate,
        model_call_detected=model_call_detected,
        accuracy=task_score,
    )
    inference_mode = infer_inference_mode(
        placeholder_rate=placeholder_rate,
        placeholder_reasons=placeholder_reasons,
        completion_rate=completion_rate,
        valid_output_rate=valid_output_rate,
        accuracy=task_score,
        model_call_detected=model_call_detected,
        smoke_forward_passed=smoke_forward_passed,
        postprocess_valid=postprocess_valid,
    )

    # P3: S2 binary rubric. env_ready proxy = any valid output produced
    # (if agent never got env up, they wouldn't have written any answer).
    env_ready = counts.get("valid_outputs", 0) > 0 or completion_rate > 0
    s2_score, s2_components = compute_s2_binary(
        smoke_forward_passed=smoke_forward_passed,
        model_call_detected=model_call_detected,
        env_ready=env_ready,
    )

    derived_steps = {
        "s1": None,
        "s2": s2_score,
        "s3": None,
        "s4": s4_score,
        "s5": compute_s5(
            has_valid_results=counts.get("valid_outputs", 0) > 0,
            submission_format_valid=format_result.get("submission_format_valid", False),
        ),
    }
    if step_scores:
        for key, value in step_scores.items():
            if key == "s2":
                # P3: binary S2 is authoritative — judge does not override.
                continue
            derived_steps[key] = value

    # P1-A: postprocess artefact hard cap on S3.
    s3_penalties: list[str] = []
    if postprocess_valid is False:
        cap = 0.5
        cur = derived_steps.get("s3")
        if cur is None or float(cur) > cap:
            derived_steps["s3"] = cap
        s3_penalties.append("postprocess_valid=False → S3 cap 0.5")

    # BUG-3: rating-F-level failure modes (fake/empty outputs) must also
    # deflate S1 / S3 since judge scores may otherwise inflate workflow.
    fake_mode = inference_mode in ("heuristic", "placeholder", "empty")
    if fake_mode:
        for k in ("s1", "s3"):
            cur = derived_steps.get(k)
            capped = 0.2
            if cur is None or float(cur) > capped:
                derived_steps[k] = capped

    workflow_score, active_steps = compute_workflow_score(derived_steps, step_weights)
    rating = assign_rating(
        medal_tier=medal_result["tier"],
        completion_rate=completion_rate,
        submission_format_valid=format_result.get("submission_format_valid", False),
        valid_outputs=counts.get("valid_outputs", 0),
    )
    overall_score = compute_overall_score(workflow_score, task_score)

    return {
        "step_scores": derived_steps,
        "metrics": {
            "accuracy": task_score,
            "accuracy_heuristic": score_result.get("accuracy_heuristic", score_result.get("accuracy", 0.0)),
            "accuracy_judge": score_result.get("accuracy_judge", 0.0),
            "judge_enabled": bool(score_result.get("judge_enabled", False)),
            "judge_model": score_result.get("judge_model", ""),
            "judge_backend": score_result.get("judge_backend", ""),
            "judge_samples": score_result.get("judge_samples", 0),
            "judge_fallback_count": score_result.get("judge_fallback_count", 0),
            "judge_agreement_rate": score_result.get("judge_agreement_rate", 0.0),
            "exact_match": score_result.get("exact_match", 0.0),
            "token_f1": score_result.get("token_f1", 0.0),
            "yes_no_accuracy": score_result.get("yes_no_accuracy", 0.0),
            "yes_no_count": score_result.get("yes_no_count", 0),
            "answer_mode": score_result.get("answer_mode", "multiple_choice"),
            "completion_rate": completion_rate,
            "parse_rate": parse_rate,
            "valid_output_rate": valid_output_rate,
            "placeholder_rate": placeholder_rate,
            "placeholder_reasons": placeholder_reasons,
            "inference_mode": inference_mode,
            "model_call_detected": bool(model_call_detected),
            "model_call_evidence": list(model_call_evidence or []),
            "smoke_forward_passed": smoke_forward_passed,
            "postprocess_valid": postprocess_valid,
            "postprocess_info": postprocess_info or {},
            "s2_components": s2_components,
            "s3_penalties": s3_penalties,
            "s4_penalties": s4_penalties,
            "medal_tier": medal_result["tier"],
            "medal_name": medal_result["name"],
            "counts": counts,
            "breakdown": score_result.get("breakdown", {}),
        },
        "format": {
            "submission_format_valid": format_result.get("submission_format_valid", False),
            "output_format_valid": format_result.get("output_format_valid", False),
        },
        "aggregate": {
            "rating": rating,
            "resolved": is_resolved(rating),
            "overall_score": overall_score,
            "agentic_score": workflow_score,
            "task_score": task_score,
            "progress_rate": round((completion_rate + parse_rate) / 2.0, 4),
            "active_steps": active_steps,
            # fail_rate = fraction of expected samples that produced no
            # valid answer (1 - completion_rate). Matches seg's definition
            # (eval_seg/aggregate.py: 1 - inference_completes) so the two
            # benchmarks report the same semantic.
            "fail_rate": round(max(0.0, 1.0 - completion_rate), 4),
        },
        "per_question": score_result.get("per_question", {}),
    }
