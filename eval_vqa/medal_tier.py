#!/usr/bin/env python3
"""Result tier assignment for VQA accuracy thresholds."""

GOOD_THRESHOLD = 0.40
BASELINE_THRESHOLD = 0.25

TIERS = {
    2: {"name": "good", "accuracy": GOOD_THRESHOLD},
    1: {"name": "okay", "accuracy": BASELINE_THRESHOLD},
}


def assign_tier(accuracy: float) -> dict:
    for tier_num in (2, 1):
        tier = TIERS[tier_num]
        if accuracy >= tier["accuracy"]:
            return {"tier": tier_num, "name": tier["name"]}
    return {"tier": 0, "name": "fail"}
