#!/usr/bin/env python3
"""Tier configuration for Lite / Standard / Pro benchmark modes.

Each tier controls:
  - S1 model selection strategy (hardcoded architecture vs. candidates vs. open research)
  - S2 environment setup (provided requirements.txt vs. agent figures deps)
  - S4 post-processing (enabled or disabled)
  - Skill blocks embedded in the system prompt
  - Artifact requirements (plan.md, plan.png, summary plots)
  - Step weights for scoring
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TierConfig:
    """Immutable tier configuration."""
    name: str                                     # "lite", "standard", "pro"

    # S1 — Model selection
    model_selection: str                          # "hardcoded" | "softcode" | "research"

    # S2 — Environment
    provide_requirements_txt: bool                # Lite: True, Standard/Pro: False

    # Artifact requirements
    require_plan_md: bool                         # all tiers: True
    require_plan_png: bool                        # Lite: False, Standard/Pro: True

    # S4 — Post-processing
    require_postprocessing: bool                  # Lite/Standard: False, Pro: True

    # Summary plots (Pro only)
    generate_summary_plots: bool

    # Step weights for scoring (all S1-S5 always active)
    step_weights: dict = field(default_factory=dict)


# -------------------------------------------------------------------
# Tier presets
# -------------------------------------------------------------------

# Step weights per tier — S3 (debug) is the heaviest for all tiers.
# Postprocessing only factors into pro scoring.
_WEIGHTS_LITE_STANDARD = {
    "s1": 0.25, "s2": 0.15, "s3": 0.35, "s4": 0.15, "s5": 0.10,
}
_WEIGHTS_PRO = {
    "s1": 0.25, "s2": 0.15, "s3": 0.35, "s4": 0.15, "s5": 0.10,
}

TIER_PRESETS = {
    "lite": TierConfig(
        name="lite",
        model_selection="hardcoded",
        provide_requirements_txt=True,
        require_plan_md=True,
        require_plan_png=False,
        require_postprocessing=False,
        generate_summary_plots=False,
        step_weights=_WEIGHTS_LITE_STANDARD,
    ),
    "standard": TierConfig(
        name="standard",
        model_selection="softcode",
        provide_requirements_txt=False,
        require_plan_md=True,
        require_plan_png=True,
        require_postprocessing=False,
        generate_summary_plots=False,
        step_weights=_WEIGHTS_LITE_STANDARD,
    ),
    "pro": TierConfig(
        name="pro",
        model_selection="research",
        provide_requirements_txt=False,
        require_plan_md=True,
        require_plan_png=True,
        require_postprocessing=False,
        generate_summary_plots=False,
        step_weights=_WEIGHTS_PRO,
    ),
}


def get_tier_config(tier: str) -> TierConfig:
    """Return TierConfig for the given tier name."""
    if tier not in TIER_PRESETS:
        raise ValueError(
            f"Unknown tier '{tier}'. Available: {list(TIER_PRESETS.keys())}"
        )
    return TIER_PRESETS[tier]
