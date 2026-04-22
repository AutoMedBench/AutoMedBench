#!/usr/bin/env python3
"""Tier presets for eval_report_gen."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TierConfig:
    name: str
    model_selection: str
    provide_requirements_txt: bool
    require_plan_md: bool
    require_plan_png: bool
    generate_summary_plots: bool
    step_weights: dict = field(default_factory=dict)


COMMON_WEIGHTS = {
    "s1": 0.25,
    "s2": 0.15,
    "s3": 0.35,
    "s4": 0.15,
    "s5": 0.10,
}


TIER_PRESETS = {
    "lite": TierConfig(
        name="lite",
        model_selection="guided",
        provide_requirements_txt=True,
        require_plan_md=True,
        require_plan_png=False,
        generate_summary_plots=False,
        step_weights=COMMON_WEIGHTS,
    ),
    "standard": TierConfig(
        name="standard",
        model_selection="family_search",
        provide_requirements_txt=False,
        require_plan_md=True,
        require_plan_png=True,
        generate_summary_plots=False,
        step_weights=COMMON_WEIGHTS,
    ),
    "pro": TierConfig(
        name="pro",
        model_selection="open_research",
        provide_requirements_txt=False,
        require_plan_md=True,
        require_plan_png=True,
        generate_summary_plots=True,
        step_weights=COMMON_WEIGHTS,
    ),
}


def get_tier_config(tier: str) -> TierConfig:
    if tier not in TIER_PRESETS:
        raise ValueError(f"Unknown tier '{tier}'. Available: {sorted(TIER_PRESETS)}")
    return TIER_PRESETS[tier]
