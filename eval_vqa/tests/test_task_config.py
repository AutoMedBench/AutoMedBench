"""Sanity checks for task discovery and tier config."""

from __future__ import annotations

from task_loader import discover_tasks, load_task_config
from tier_config import get_tier_config


EXPECTED_TASKS = {
    "medxpertqa-mm-task",
    "pathvqa-task",
    "vqa-rad-task",
    "medframeqa-task",
}


def test_tasks_discovered():
    tasks = set(discover_tasks().keys())
    missing = EXPECTED_TASKS - tasks
    assert not missing, f"missing task dirs: {missing}"


def test_task_configs_parse():
    for tid in EXPECTED_TASKS:
        cfg = load_task_config(tid)
        assert cfg["task_id"] == tid
        assert cfg.get("answer_mode") in {"multiple_choice", "open_ended"}


def test_step_weights_align_with_seg():
    w = get_tier_config("lite").step_weights
    assert w == {"s1": 0.25, "s2": 0.15, "s3": 0.35, "s4": 0.15, "s5": 0.10}
    assert abs(sum(w.values()) - 1.0) < 1e-9
