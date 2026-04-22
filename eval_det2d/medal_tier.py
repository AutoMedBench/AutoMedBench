#!/usr/bin/env python3
"""Result tier assignment based on detection mAP thresholds.

Three tiers:
  Good — meets or exceeds the good-result threshold
  Okay — above baseline but below good
  Fail — below baseline
"""

GOOD_THRESHOLD = 0.60
BASELINE_THRESHOLD = 0.30

TIERS = {
    2: {"name": "good", "mAP": GOOD_THRESHOLD},
    1: {"name": "okay", "mAP": BASELINE_THRESHOLD},
}


def assign_tier(map_score: float) -> dict:
    """Assign result tier based on detection mAP. Returns tier number and name."""
    for tier_num in [2, 1]:
        t = TIERS[tier_num]
        if map_score >= t["mAP"]:
            return {"tier": tier_num, "name": t["name"]}

    return {"tier": 0, "name": "fail"}
