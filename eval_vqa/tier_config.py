#!/usr/bin/env python3
"""Tier configuration for VQA benchmark modes."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from task_loader import load_model_info, load_standard_candidate_specs, render_standard_candidate_summary
except ModuleNotFoundError:
    from .task_loader import load_model_info, load_standard_candidate_specs, render_standard_candidate_summary


@dataclass(frozen=True)
class TierConfig:
    name: str
    model_selection: str
    provide_requirements_txt: bool
    require_plan_md: bool
    require_plan_png: bool
    supported_models: tuple[str, ...]
    step_weights: dict[str, float] = field(default_factory=dict)


_WEIGHTS = {
    "s1": 0.25,
    "s2": 0.15,
    "s3": 0.35,
    "s4": 0.15,
    "s5": 0.10,
}


TIER_PRESETS = {
    "lite": TierConfig(
        name="lite",
        model_selection="hardcoded",
        provide_requirements_txt=True,
        require_plan_md=True,
        require_plan_png=False,
        supported_models=("microsoft/llava-med-v1.5-mistral-7b",),
        step_weights=_WEIGHTS,
    ),
    "standard": TierConfig(
        name="standard",
        model_selection="bounded_candidates",
        provide_requirements_txt=False,
        require_plan_md=True,
        require_plan_png=False,
        supported_models=(
            "microsoft/llava-med-v1.5-mistral-7b",
            "UCSC-VLAA/MedVLThinker-3B-RL_m23k",
            "MedVLSynther/MedVLSynther-3B-RL_13K",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "google/gemma-4-E2B-it",
            "google/gemma-4-E4B-it",
        ),
        step_weights=_WEIGHTS,
    ),
}


def get_tier_config(tier: str) -> TierConfig:
    if tier not in TIER_PRESETS:
        raise ValueError(f"Unknown tier '{tier}'. Available: {sorted(TIER_PRESETS)}")
    return TIER_PRESETS[tier]


def get_task_model_info(task_id: str) -> dict:
    model_info = load_model_info(task_id)
    lite = model_info.get("lite", {})
    candidate_models = load_standard_candidate_specs(task_id)
    return {
        "lite_model": lite.get("model_name"),
        "lite_model_spec": lite,
        "standard_candidates": [item.get("model_name") for item in candidate_models if item.get("model_name")],
        "standard_candidate_specs": candidate_models,
        "model_families": {
            item.get("model_name"): item.get("family")
            for item in candidate_models
            if item.get("model_name")
        },
        "selection_guidance": model_info.get("selection_guidance", []),
        "standard_candidate_summary": render_standard_candidate_summary(task_id),
    }
