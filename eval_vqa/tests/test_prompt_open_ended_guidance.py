"""BUG-047: open-ended prompts must pre-warn agents about ≤5-word answers."""

from __future__ import annotations

from benchmark_runner import build_tier_system_prompt


def test_open_ended_task_includes_short_answer_contract():
    prompt = build_tier_system_prompt(
        task_id="pathvqa-task",
        tier="lite",
        question_ids=["q1", "q2"],
        subset="test",
        sample_limit=None,
    )
    assert "Open-ended answer contract" in prompt
    assert "≤ 5 words" in prompt
    # Both of S1 planning and S2 smoke are named so the contract anchors early.
    assert "Plan S1" in prompt and "S2 smoke" in prompt


def test_mcq_task_omits_open_ended_block():
    prompt = build_tier_system_prompt(
        task_id="medxpertqa-mm-task",
        tier="lite",
        question_ids=["q1"],
        subset="test",
        sample_limit=None,
    )
    assert "Open-ended answer contract" not in prompt
    assert "≤ 5 words" not in prompt


def test_open_ended_applies_across_all_open_tasks():
    for task_id in ("pathvqa-task", "vqa-rad-task", "slake-task"):
        prompt = build_tier_system_prompt(
            task_id=task_id,
            tier="lite",
            question_ids=["q1"],
            subset="test",
            sample_limit=None,
        )
        assert "Open-ended answer contract" in prompt, task_id
