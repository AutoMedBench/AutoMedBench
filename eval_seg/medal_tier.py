#!/usr/bin/env python3
"""Result tier assignment based on the task's primary Dice score.

The input score is task-dependent:
  - Binary tasks (kidney, liver, pancreas): mean lesion Dice
  - Multi-class tasks (feta): macro-mean Dice across foreground tissues

Three tiers:
  Good — meets or exceeds the good-result threshold (score >= 0.70)
  Okay — above baseline but below good (0.30 <= score < 0.70)
  Fail — below baseline (score < 0.30)
"""

GOOD_THRESHOLD = 0.70
BASELINE_THRESHOLD = 0.30

TIERS = {
    2: {"name": "good", "threshold": GOOD_THRESHOLD},
    1: {"name": "okay", "threshold": BASELINE_THRESHOLD},
}


def assign_tier(score: float) -> dict:
    """Assign result tier based on the task's primary Dice score.

    For binary tasks, ``score`` is the mean lesion Dice. For multi-class
    tasks, it is the macro-mean Dice across foreground tissues. Returns
    ``{"tier": 0..2, "name": "fail"|"okay"|"good"}``.
    """
    for tier_num in [2, 1]:
        t = TIERS[tier_num]
        if score >= t["threshold"]:
            return {"tier": tier_num, "name": t["name"]}

    return {"tier": 0, "name": "fail"}
