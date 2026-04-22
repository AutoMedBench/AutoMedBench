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
# Per-task model information
# -------------------------------------------------------------------

TASK_MODEL_INFO = {
    "kidney": {
        # Lite — best-performing model + verified site
        # SOTA: KagglingFace/nnUNet-KiTS19-3d-lowres-50epochs
        #   Validated: organ Dice=0.9622, lesion Dice=0.8605 (gpt-5.4, 2026-04-12)
        "model_architecture": "nnU-Net v2 (KiTS19 3d_lowres)",
        "model_description": (
            "SOTA checkpoint: huggingface.co/KagglingFace/nnUNet-KiTS19-3d-lowres-50epochs. "
            "nnU-Net v2 trained on KiTS19 with 3d_lowres config, fold 1, 50 epochs. "
            "Has kidney organ + kidney tumor labels. Use nnUNetPredictor from "
            "nnunetv2.inference.predict_from_raw_data to run inference. "
            "NOTE: nnU-Net env vars (nnUNet_raw, nnUNet_preprocessed, nnUNet_results) "
            "are not required for inference-only mode — use predict_from_files() or "
            "predict_single_npy_array() directly."
        ),
        "requirements_txt": "requirements_kidney.txt",

        # Standard — model families with verified sources
        "model_range": [
            "- **nnU-Net pre-trained models** — search for nnU-Net checkpoints "
            "trained on KiTS (kidney + tumor labels). Both v1 and v2 format "
            "checkpoints exist online. Standalone PyTorch models (e.g., "
            "MLPerf 3D-UNet KiTS19) are also available",
            "- **MONAI Bundles / VISTA3D** — check the MONAI Model Zoo on "
            "HuggingFace (huggingface.co/MONAI). VISTA3D has kidney organ labels "
            "but verify it has kidney TUMOR labels before using",
        ],

        # Pro — tumor-focused clinical background only
        "clinical_background": (
            "Kidney tumors in CT include renal cell carcinoma (RCC), "
            "the most common kidney malignancy. Your score depends "
            "primarily on tumor/lesion Dice — a model that segments "
            "kidneys but misses tumors will score near zero. You need a "
            "model trained specifically on tumor-annotated data. "
            "Note: tools like TotalSegmentator are general organ segmenters — "
            "they segment kidneys but do NOT detect tumors."
        ),
    },
    "liver": {
        # Lite — best-performing model + verified site
        "model_architecture": "MONAI VISTA3D",
        "model_description": (
            "Available on HuggingFace (huggingface.co/MONAI). "
            "Has hepatic_tumor and liver_tumor labels for tumor detection, "
            "plus liver organ labels. Use the MONAI bundle API to download."
        ),
        "requirements_txt": "requirements_liver.txt",

        # Standard — model families with verified sources
        "model_range": [
            "- **nnU-Net pre-trained models** — search for nnU-Net checkpoints "
            "trained on LiTS (liver + tumor labels). Both v1 and v2 format "
            "checkpoints exist online",
            "- **MONAI Bundles / VISTA3D** — check the MONAI Model Zoo on "
            "HuggingFace (huggingface.co/MONAI). VISTA3D has hepatic_tumor and "
            "liver_tumor labels for tumor detection, plus liver organ labels",
        ],

        # Pro — tumor-focused clinical background only
        "clinical_background": (
            "Liver tumors in CT include hepatocellular carcinoma (HCC), "
            "metastases from colorectal/breast/lung cancers, and "
            "cholangiocarcinoma. Your score depends primarily on "
            "tumor/lesion Dice — a model that segments livers but misses "
            "tumors will score near zero. You need a model trained "
            "specifically on tumor-annotated data. "
            "Note: tools like TotalSegmentator are general organ segmenters — "
            "they segment livers but do NOT detect tumors."
        ),
    },
    "pancreas": {
        # Lite — best-performing model + verified site
        "model_architecture": "MONAI VISTA3D",
        "model_description": (
            "Available on HuggingFace (huggingface.co/MONAI). "
            "Has pancreatic_tumor (label 24) for tumor detection, plus "
            "pancreas organ labels. Use the MONAI bundle API to download."
        ),
        "requirements_txt": "requirements_pancreas.txt",

        # Standard — model families with verified sources
        "model_range": [
            "- **nnU-Net pre-trained models** — search for nnU-Net checkpoints "
            "trained on MSD Pancreas (pancreas + tumor labels). Both v1 and v2 "
            "format checkpoints exist online",
            "- **MONAI Bundles** — check the MONAI Model Zoo on HuggingFace "
            "(huggingface.co/MONAI) for pancreas tumor segmentation bundles",
        ],

        # Pro — tumor-focused clinical background only
        "clinical_background": (
            "Pancreatic tumors in CT include pancreatic ductal "
            "adenocarcinoma (PDAC), the most lethal GI malignancy. "
            "Pancreatic tumor segmentation is extremely difficult — "
            "even SOTA methods achieve only ~0.52 Dice. Your score "
            "depends primarily on tumor/lesion Dice — a model that "
            "segments pancreas but misses tumors will score near zero. "
            "You need a model trained specifically on tumor-annotated data. "
            "Note: tools like TotalSegmentator are general organ segmenters — "
            "they segment pancreas but do NOT detect tumors."
        ),
    },
}


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
        generate_summary_plots=True,
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


def get_task_model_info(task: str) -> dict:
    """Return per-task model info dict."""
    if task not in TASK_MODEL_INFO:
        raise ValueError(
            f"Unknown task '{task}'. Available: {list(TASK_MODEL_INFO.keys())}"
        )
    return TASK_MODEL_INFO[task]
