"""Guard against regression of the S1/S3 None-step zeroing bug.

compute_workflow_score must renormalize over active (non-None) steps so that a
run without the workflow judge is not silently capped at sum(active_weights).
"""

from __future__ import annotations

from aggregate import STEP_WEIGHTS, compute_workflow_score


def test_all_steps_present_matches_weighted_mean():
    scores = {"s1": 1.0, "s2": 1.0, "s3": 1.0, "s4": 1.0, "s5": 1.0}
    score, active = compute_workflow_score(scores)
    assert score == 1.0
    assert set(active) == set(STEP_WEIGHTS)


def test_none_steps_renormalize_instead_of_zeroing():
    # s1 / s3 unset (typical of run_eval without --llm-judge).
    scores = {"s1": None, "s2": 1.0, "s3": None, "s4": 1.0, "s5": 1.0}
    score, active = compute_workflow_score(scores)
    # Before fix this was 0.40 (active weights / total). After fix: 1.0.
    assert score == 1.0
    assert set(active) == {"s2", "s4", "s5"}


def test_partial_scores_renormalize_proportionally():
    scores = {"s1": None, "s2": 0.5, "s3": None, "s4": 1.0, "s5": 0.0}
    # weights: s2=0.15, s4=0.15, s5=0.10 → active_sum=0.40
    # numerator = 0.15*0.5 + 0.15*1.0 + 0.10*0.0 = 0.225
    # score = 0.225 / 0.40 = 0.5625
    score, _ = compute_workflow_score(scores)
    assert score == 0.5625


def test_all_none_returns_zero():
    score, active = compute_workflow_score({"s1": None, "s2": None, "s3": None, "s4": None, "s5": None})
    assert score == 0.0
    assert active == []
