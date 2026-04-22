#!/usr/bin/env python3
"""Real agent benchmark: agent plans, codes, and executes segmentation itself.

The agent gets ONE tool (execute_code) and must follow S1-S5 to produce
organ + lesion masks from raw CT scans.

Three difficulty tiers:
  --tier lite      Hard-coded SOTA model, provided requirements.txt, skills for S1-S3
  --tier standard  Soft-coded model range, skill for S1
  --tier pro       No hints, deep research, post-processing, summary plots

Usage:
    python benchmark_runner.py --agent claude-opus-4-6 --task kidney --tier lite
    python benchmark_runner.py --agent claude-opus-4-6 --task kidney --tier standard
    python benchmark_runner.py --agent claude-opus-4-6 --task kidney --tier pro
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import requests
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Only add pip_packages/ fallback when numpy is not already importable
# (e.g., base env without conda deps). Under conda envs like 'smile',
# pip_packages/numpy shadows the native C-extension build and crashes.
try:
    import numpy  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pip_packages"))
from detail_report import generate_detail_report, print_detail_report
from run_eval import run_eval
from tier_config import get_tier_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # MedAgentsBench/
CONFIG_PATH = os.path.join(SCRIPT_DIR, "agent_config.yaml")
BASE_CONDA_ENV = "smile"  # base env with PyTorch, MONAI, nnU-Net, etc.

# Task configuration is loaded from per-task folders (eval_seg/<task-id>/).
# See task_loader.py for the loader. Legacy task names (kidney, liver,
# pancreas) are auto-mapped to <name>-seg-task folders.
from task_loader import (load_task_config, load_model_info, load_skill,
                         load_requirements_path, get_task_data_root,
                         discover_patients as _discover_patients_from_loader,
                         discover_tasks)

# ==================================================================
# System prompt — tier-aware composable blocks
# Templates are stored in eval_seg/prompts/ for readability.
# The Python strings below are loaded from those files at import time.
# If a file is missing, the inline fallback is used.
# ==================================================================

def _load_prompt(relpath: str, fallback: str = "") -> str:
    """Load a prompt template file. Returns fallback if not found."""
    full = os.path.join(SCRIPT_DIR, relpath)
    if os.path.isfile(full):
        with open(full) as f:
            return f.read()
    return fallback

_PREAMBLE = """\
You are a medical AI agent. You have access to a Linux coding environment with \
an NVIDIA H100 80GB GPU.

## CRITICAL — Sandbox Isolation Rules (VIOLATION = IMMEDIATE DISQUALIFICATION)
You are running inside an ISOLATED environment. You MUST obey these rules:
  - You may ONLY access files under your data directory: {data_dir}/
  - You may ONLY write files under your workspace: {output_dir}/
  - You may use pip/conda to install packages and download models from the internet.
  - You MUST NOT scan, read, or list any path outside these two directories.
  - You MUST NOT run: find /, ls /, os.walk('/'), os.scandir('/'), or any broad filesystem search.
  - You MUST NOT use relative path traversal (../..) to escape your sandbox.
  - You MUST NOT use pathlib, subprocess with list arguments, string concatenation, \
base64 decoding, or any other technique to construct forbidden paths dynamically.
  - You MUST NOT read benchmark harness code, evaluation scripts, or configuration files.
  - You MUST NOT access: /proc, /sys, /etc, /var, /home, /app, or any system path.
  - To find pre-trained models, use pip install, HuggingFace Hub API, or direct \
URL downloads — NEVER search the local filesystem outside your workspace.

### Downloads
  ALL downloads (models, weights, packages, caches) MUST go into {output_dir}/. \
Set environment variables before downloading:
  export HF_HOME={output_dir}/.cache/huggingface
  export TORCH_HOME={output_dir}/.cache/torch
  export XDG_CACHE_HOME={output_dir}/.cache
  export PIP_CACHE_DIR={output_dir}/.cache/pip
  NEVER download to ~/.cache, /tmp, /root, or any location outside {output_dir}/.

### Violation Penalty
  The FIRST sandbox violation triggers a WARNING — your code is blocked but \
you may continue. Fix your approach immediately.
  The SECOND violation triggers IMMEDIATE TERMINATION. The consequences are \
severe and non-negotiable:
  - All S-step scores from the violation point onward are set to 0.
  - Clinical score (organ Dice + lesion Dice) is set to 0.
  - Overall rating is set to F (Disqualified).
  - The task is marked as unresolved.
  There is NO benefit to attempting forbidden filesystem access — it always \
results in a worse outcome than honest failure. Focus on downloading models \
from the internet and writing your own code.

## Task
{task_description}

## Data
{modality} scans ({input_format}) are located under:
{data_dir}/

Each subfolder is one patient (the folder name is the patient ID). \
Each patient has a file named `{input_filename}`. \
List the directory to discover all patient IDs before starting.

## Required Outputs  (all under {output_dir}/)
  agents_outputs/<patient_id>/organ.nii.gz   -- binary {organ} mask (0 or 1), same shape as input scan
  agents_outputs/<patient_id>/lesion.nii.gz  -- binary lesion/tumor mask (0 or 1), same shape as input scan
  agents_outputs/agents_decision.csv         -- (optional) CSV with columns: patient_id, organ, lesion_present
"""

# ---- Environment block per tier ----

_ENV_LITE = """\
## Environment
You start with a base environment containing:
Python 3, PyTorch 2.6 + CUDA, MONAI 1.5.0 (monai.bundle for model zoo), \
nnU-Net v2.6.0, nibabel, numpy, scipy, scikit-image, pandas.
Use this base environment for research in S1. In S2, create a virtual \
environment (python -m venv) under {output_dir}/env/ and install \
dependencies from the provided requirements file:
  {requirements_txt_path}
You may install additional packages if needed. Do NOT install packages \
into the base environment.
"""

_ENV_STANDARD = """\
## Environment
You start with a base environment containing:
Python 3, PyTorch 2.6 + CUDA, MONAI 1.5.0 (monai.bundle for model zoo), \
nnU-Net v2.6.0, nibabel, numpy, scipy, scikit-image, pandas.
Use the base environment directly — all key packages are pre-installed. \
If you need extra packages, pip install them into the base environment. \
Internet access is available — you can download model weights and pip \
install packages.
"""

_ENV_PRO = """\
## Environment
You start with a base environment containing:
Python 3, PyTorch 2.6 + CUDA, MONAI 1.5.0 (monai.bundle for model zoo), \
nnU-Net v2.6.0, nibabel, numpy, scipy, scikit-image, pandas.
Use the base environment directly — all key packages are pre-installed. \
If you need extra packages, pip install them into the base environment. \
Internet access is available — you can download model weights, pip \
install packages, and search the web via code (e.g. scraping, search \
libraries). You are encouraged to do so, especially during S1.
"""

# ---- S1 block per tier ----

_S1_LITE = """\
S1  RESEARCH    Find and understand the model you will use.
  Your target model is **{model_architecture}**. {model_description}
  1. Download the pre-trained weights from the specified source.
  2. CRITICAL: Verify the checkpoint has TUMOR labels (not just organ or \
cyst labels). Inspect the label map. You need a model trained specifically \
on tumor-annotated data.
  3. Check its expected input format (spacing, orientation, intensity range).
  4. Save your findings to {output_dir}/plan/plan.md — include: model name, \
checkpoint source, label map, and inference pipeline.
"""

_S1_STANDARD = """\
S1  PLAN        Research and choose the best model from the following families.
  Explore these model families for {organ} organ and TUMOR segmentation:
{model_range}
  1. For each family, search for available pre-trained models. Find specific \
checkpoints that can be downloaded.
  2. For each candidate found, verify via code: (a) weights can be downloaded \
in this environment, (b) its label map covers {organ} organ AND {organ} \
TUMOR structures. CRITICAL: general-purpose organ segmentation models \
(e.g., TotalSegmentator) typically do NOT have tumor labels and will score \
near zero on lesion Dice. You need a model trained specifically on \
tumor-annotated data.
  3. Compare candidates on tumor label coverage, reported accuracy, setup \
complexity, and feasibility in this environment.
  4. Select the best option. Justify your choice — explain why it beats the \
alternatives for this specific task. Prioritize tumor detection capability.
  5. Save your plan to {output_dir}/plan/plan.md — include: candidates \
evaluated, comparison table, final decision, and the pipeline you will \
follow (which model for organ, which for tumor, preprocessing steps).
  6. Generate a flowchart visualization of your plan and save it as \
{output_dir}/plan/plan.png (use matplotlib or any plotting library).

  Skill — How to search and compare models (examples only — use any approach that works):
  ```python
  # Search HuggingFace for models
  import requests
  resp = requests.get("https://huggingface.co/api/models",
                      params={{"search": "{organ} tumor segmentation", "limit": 10}})
  for model in resp.json():
      print(model["modelId"], model.get("tags", []))

  # Download model weights from a URL
  import urllib.request
  urllib.request.urlretrieve(
      "https://example.com/model/weights.zip",
      "{output_dir}/model/weights.zip")

  # Check MONAI Model Zoo bundles
  from monai.bundle import download
  # Try: monai.bundle.download("bundle_name", bundle_dir="{output_dir}/model")
  # Inspect: configs/inference.json, docs/labels.json for label maps
  ```

  ```markdown
  # Comparison table template for plan.md:
  | Model | Organ Labels | TUMOR Labels | Download Size | Ease of Setup | Notes |
  |-------|-------------|--------------|---------------|---------------|-------|
  | ...   | ...         | ...          | ...           | ...           | ...   |
  ```
"""

_S1_PRO = """\
S1  PLAN        This is a head-to-head competition against other AI agents.
  Your final score depends entirely on the quality of the model(s) you find. \
Every other agent is solving this exact same task right now. The agent that \
finds the best model wins. Settling for the first working model guarantees \
a low ranking.

  Your objective: find the absolute best pre-trained model(s) for {organ} \
organ and TUMOR segmentation in {modality} scans.

  Rules:
  - Do NOT rely on what you think you know. Search actively. Your training \
data is stale — models, checkpoints, and repositories change constantly.
  - Do NOT stop at the first model that works. You MUST find and evaluate \
at least 3 distinct candidates before choosing. Agents that skip \
research and jump to inference will be scored harshly.
  - Search broadly: HuggingFace, GitHub, MONAI Model Zoo, package \
repositories, challenge leaderboards, project pages. Any source is valid.
  - TUMOR detection is what separates winning agents from losing ones. \
A model that segments the organ but misses tumors will score near zero \
on the most heavily weighted metric (lesion Dice).
  - You need a model trained specifically on tumor-annotated data. \
General-purpose organ segmentation tools do NOT have tumor labels.

  Steps:
  1. Conduct a wide research across available sources. Download \
nothing yet — just catalog what exists, what labels each model covers, \
and where the weights live. Be efficient — spend no more than 10-15 \
tool calls on research before making a decision.
  2. For every candidate, verify programmatically: (a) weights are actually \
downloadable, (b) label map covers BOTH {organ} organ AND {organ} \
lesions/tumors, (c) compatible with the {modality} data in this dataset.
  3. If no single model covers both, design a multi-model pipeline.
  4. Rank all candidates by expected lesion segmentation quality. Pick the \
best. Justify with hard evidence — Dice scores, challenge rankings, \
label coverage.
  5. Save your plan to {output_dir}/plan/plan.md — include: every candidate \
found, full comparison, final decision with justification, and your \
complete pipeline design.
  6. Generate a flowchart of your pipeline as {output_dir}/plan/plan.png.
"""

# ---- S2 block per tier ----

_S2_LITE = """\
S2  SETUP       Set up your environment and load the model.
  1. Create a virtual environment under {output_dir}/env/ using \
`python -m venv --system-site-packages`. Activate it.
  2. Install the base dependencies: `pip install -r {requirements_txt_path}`
     You may install additional packages if needed.
  3. Download model weights and any required configs.
  4. Load the model into GPU memory and confirm it initializes without errors.
  5. Verify compatibility with the scan data (check one scan's shape, spacing, \
and intensity range against model expectations).

  Skill — How to set up the environment (examples only — use any approach that works):
  ```bash
  # Create venv with access to system packages (PyTorch, MONAI, etc.)
  python -m venv --system-site-packages {output_dir}/env
  source {output_dir}/env/bin/activate

  # Install base dependencies
  pip install -r {requirements_txt_path}

  # Install additional packages if needed
  pip install <package_name>
  ```

  ```python
  # Download model from HuggingFace
  from huggingface_hub import snapshot_download
  model_dir = snapshot_download("REPO_ID", local_dir="{output_dir}/model")

  # Load and verify on GPU
  import torch
  device = torch.device("cuda")
  model = ...  # model-specific loading
  model.to(device)
  model.eval()
  print(f"Model loaded on {{device}}")
  ```
"""

_S2_STANDARD_PRO = """\
S2  SETUP       Set up your environment and download the chosen model(s).
  1. Create a virtual environment for this run under {output_dir}/env/ \
using `python -m venv`. Activate it and install any additional packages \
your chosen model requires. Use this venv for all subsequent steps.
  2. Download model weights, configs, and any required dependencies.
  3. Load the model into GPU memory and confirm it initializes without errors.
  4. Check the model's expected input format (spacing, orientation, intensity \
range) and verify compatibility with the scan data.
"""

# ---- S3 block (shared for all tiers) ----

_S3_ALL_LITE_STANDARD = """\
S3  VALIDATE    Run inference on ONE patient as a sanity check.
  1. Pick one patient and run the full inference pipeline end-to-end.
  2. You MUST use GPU (CUDA) for inference. Verify with torch.cuda.is_available(). \
Load the model onto GPU (e.g. model.cuda() or device='cuda'). If torch.load uses \
map_location, set it to torch.device('cuda'). Only fall back to CPU if CUDA is \
genuinely unavailable — never force CPU when a GPU is present.
  3. Verify the output:
     - Shape matches the original input scan
     - Values are binary (0 and 1 only)
     - Organ mask has a reasonable voxel count (> 1000)
     - CRITICAL — Lesion mask check: compute lesion_voxel_ratio = \
lesion_mask.sum() / organ_mask.sum(). Print this ratio. If it is exactly 0.0, \
your model is NOT detecting tumors — go back to S1 and pick a model with \
actual tumor labels. A working tumor model should produce lesion_voxel_ratio \
between 0.01 and 0.6 on most patients. A ratio of 0.0 means FAILURE.
  4. If the output looks wrong or lesion_voxel_ratio is 0, debug and fix \
the pipeline before proceeding. Do NOT continue to S4 with a model that \
produces empty lesion masks.
  5. If you have been stuck debugging for more than 600 seconds without a \
working pipeline, abandon the current approach and go back to S1 to choose \
a different model or method.
"""

_S3_ALL_PRO = """\
S3  VALIDATE    Run inference on ONE patient as a sanity check.
  1. Pick one patient and run the full inference pipeline end-to-end.
  2. Make full use of available computational resources (check nvidia-smi).
  3. Verify the output:
     - Shape matches the original input scan
     - Values are binary (0 and 1 only)
     - Organ mask has a reasonable voxel count (> 1000)
     - CRITICAL — Lesion mask check: compute lesion_voxel_ratio = \
lesion_mask.sum() / organ_mask.sum(). Print this ratio. If it is exactly 0.0, \
your model is NOT detecting tumors — go back to S1 and pick a model with \
actual tumor labels. A working tumor model should produce lesion_voxel_ratio \
between 0.01 and 0.6 on most patients. A ratio of 0.0 means FAILURE.
  4. If the output looks wrong or lesion_voxel_ratio is 0, debug and fix \
the pipeline before proceeding. Do NOT continue to S4 with a model that \
produces empty lesion masks.
  5. If you have been stuck debugging for more than 600 seconds without a \
working pipeline, abandon the current approach and go back to S1 to choose \
a different model or method.
"""

# NOTE: S3 skill block removed — now loaded from task folder (lite_s3.md / standard_s3.md)

# ---- S4 block per tier ----

_S4_LITE_STANDARD = """\
S4  INFERENCE   Run inference on ALL patients.
  1. Run the pipeline on every patient in the dataset. Save organ.nii.gz \
and lesion.nii.gz under agents_outputs/<patient_id>/ for each patient.
  2. You MUST use GPU (CUDA) for inference — the same as S3. Ensure model \
and data are on GPU. Never use map_location='cpu' when a GPU is available.
  3. Maximize throughput — if the pipeline supports batched inference, use it.
  4. Print progress so the log captures which patient is being processed.
  5. After all patients are done, confirm that every patient has both output files.
  Do NOT apply post-processing to the masks. Use the raw model outputs directly.
"""

_S4_PRO = """\
S4  INFERENCE   Run inference on ALL patients.
  1. Run the pipeline on every patient in the dataset. Save organ.nii.gz \
and lesion.nii.gz under agents_outputs/<patient_id>/ for each patient.
  2. Make full use of available computational resources for maximum throughput.
  3. If the pipeline supports batched inference, use it.
  4. Print progress so the log captures which patient is being processed.
  5. After all patients are done, confirm that every patient has both output files.
  Do NOT apply post-processing to the masks. Use the raw model outputs directly.
"""

# ---- S5 block (shared) ----

_S5_ALL = """\
S5  SUBMIT      Verify outputs and submit results.
  1. Confirm that every patient has both organ.nii.gz and lesion.nii.gz \
under agents_outputs/<patient_id>/.
  2. Verify no files are missing or corrupted.
  3. (Optional) Write a decision CSV to {output_dir}/agents_outputs/agents_decision.csv \
with columns: patient_id, organ, lesion_present (0 or 1). One row per patient. \
Set lesion_present=1 if the lesion mask has non-trivial foreground voxels, 0 otherwise. \
This CSV is not scored but is useful for analysis.
  4. Call `submit_results` when everything is saved and verified.
"""

# ---- Important block per tier ----

_IMPORTANT_LITE = """\
## Important
- This is an INFERENCE-ONLY benchmark. Load pre-trained weights — do NOT \
train or fine-tune any model.
- Masks MUST be binary (0 and 1 only) and match the CT spatial dimensions exactly.
- Process ALL patients found in the data directory — missing ANY patient means automatic failure (Rating F, zero Dice credit).
- Print progress so the log captures what's happening.
"""

_IMPORTANT_STANDARD = """\
## Important
- This is an INFERENCE-ONLY benchmark. Find pre-trained models and load their \
weights — do NOT train or fine-tune any model.
- Masks MUST be binary (0 and 1 only) and match the CT spatial dimensions exactly.
- Process ALL patients found in the data directory — missing ANY patient means automatic failure (Rating F, zero Dice credit).
- If a model does not have a {organ} lesion/tumor class, consider alternative \
approaches (e.g. different model, combining models, or using available labels as proxy).
- Print progress so the log captures what's happening.
"""

_IMPORTANT_PRO = """\
## Important
- This is an INFERENCE-ONLY benchmark. Find pre-trained models and load their \
weights — do NOT train or fine-tune any model.
- Masks MUST be binary (0 and 1 only) and match the CT spatial dimensions exactly.
- Process ALL patients found in the data directory — missing ANY patient means automatic failure (Rating F, zero Dice credit).
- If a model does not have a {organ} lesion/tumor class, consider alternative \
approaches (e.g. different model, combining models, or using available labels as proxy).
- You are competing against other agents. The winning strategy is NOT speed — \
it is finding the best model. An agent that spends 5 minutes on research \
and picks a mediocre model will lose to one that spends 15 minutes and \
finds the right model.
- Organ-only models will score poorly. Lesion Dice is the decisive metric.
- Print progress so the log captures what's happening.
"""

# ==================================================================
# Multi-class task variants (e.g. FeTA fetal brain multi-tissue seg).
# These are used ONLY when task_cfg["task_type"] == "multiclass".
# Binary organ/lesion tasks (kidney, liver, pancreas) are UNAFFECTED.
# ==================================================================

_REQUIRED_OUTPUTS_MC = """\
## Required Outputs  (all under {output_dir}/)
  agents_outputs/<patient_id>/{output_filename}   -- SINGLE multi-class label map, \
integer values in {{0}} ∪ tissue_labels ({tissue_labels_brief}), same shape as input scan
"""

# Multiclass tasks score clinical quality as macro-mean Dice across tissues —
# not organ + lesion — so the violation-penalty bullet must match.
_VIOLATION_CLINICAL_BINARY = "  - Clinical score (organ Dice + lesion Dice) is set to 0."
_VIOLATION_CLINICAL_MC = "  - Clinical score (mean Dice across tissues) is set to 0."

_S1_LITE_MC = """\
S1  RESEARCH    Find and understand the model you will use.
  This task is a MULTI-TISSUE segmentation problem. Your final clinical \
score is the MEAN Dice across the {num_foreground_classes} foreground tissue \
classes: {tissue_labels_brief}.
  Your target model is **{model_architecture}**. {model_description}
  1. Download the pre-trained weights from the specified source.
  2. CRITICAL: verify the checkpoint's label scheme actually covers all \
{num_foreground_classes} target tissues. Missing any class means 0 Dice on \
that class and a lower mean. If the model uses a different label numbering, \
you will need to REMAP at inference time.
  3. Check its expected input format (spacing, orientation, intensity range).
  4. Save your findings to {output_dir}/plan/plan.md — include: model name, \
checkpoint source, label map (source → target), and inference pipeline.
"""

_S1_STANDARD_MC = """\
S1  PLAN        Research and choose the best model from the following families.
  This task is a MULTI-TISSUE segmentation problem. Your final clinical \
score is the MEAN Dice across the {num_foreground_classes} foreground tissue \
classes: {tissue_labels_brief}.
  Explore these model families for multi-tissue {organ} segmentation in {modality}:
{model_range}
  1. For each family, search for available pre-trained models. Find specific \
checkpoints that can be downloaded.
  2. For each candidate found, verify via code: (a) weights can be downloaded \
in this environment, (b) its label scheme covers all {num_foreground_classes} \
target tissues. CRITICAL: models with a different label numbering (e.g., a \
dHCP parcellator with 80+ regions, or an adult brain atlas) must be remapped \
at inference time. Missing classes score 0 Dice and pull the mean down.
  3. Compare candidates on: tissue coverage, reported mean Dice on this task, \
setup complexity, feasibility in this environment.
  4. Select the best option. Justify your choice — explain why it beats the \
alternatives for this specific task.
  5. Save your plan to {output_dir}/plan/plan.md — include: candidates \
evaluated, comparison table (per-tissue Dice where reported), final decision, \
and the pipeline you will follow (model, preprocessing, label remapping).
  6. Generate a flowchart visualization of your plan and save it as \
{output_dir}/plan/plan.png (use matplotlib or any plotting library).

  Skill — How to search and compare models (examples only — use any approach that works):
  ```python
  # Search HuggingFace for models
  import requests
  resp = requests.get("https://huggingface.co/api/models",
                      params={{"search": "{organ} multi-tissue segmentation", "limit": 10}})
  for model in resp.json():
      print(model["modelId"], model.get("tags", []))

  # Download model weights from a URL
  import urllib.request
  urllib.request.urlretrieve(
      "https://example.com/model/weights.zip",
      "{output_dir}/model/weights.zip")

  # Check MONAI Model Zoo bundles
  from monai.bundle import download
  ```

  ```markdown
  # Comparison table template for plan.md:
  | Model | Covers all target tissues? | Reported mean Dice | Download Size | Setup | Notes |
  |-------|---------------------------|--------------------|---------------|-------|-------|
  | ...   | ...                       | ...                | ...           | ...   | ...   |
  ```
"""

_S3_ALL_LITE_STANDARD_MC = """\
S3  VALIDATE    Run inference on ONE patient as a sanity check.
  1. Pick one patient and run the full inference pipeline end-to-end.
  2. You MUST use GPU (CUDA) for inference. Verify with torch.cuda.is_available(). \
Load the model onto GPU (e.g. model.cuda() or device='cuda'). If torch.load uses \
map_location, set it to torch.device('cuda'). Only fall back to CPU if CUDA is \
genuinely unavailable — never force CPU when a GPU is present.
  3. Verify the output:
     - Shape matches the original input scan (resample back to input geometry \
if your model runs at a fixed internal spacing — use nearest neighbour for labels).
     - Output is a SINGLE label map saved as `{output_filename}` (not one file per class).
     - Integer values are a subset of {{0}} ∪ target tissues ({tissue_labels_brief}).
     - Per-tissue voxel counts are all non-zero on a healthy reconstruction. \
If any target tissue is empty, your label mapping is broken — fix before S4.
  4. If the output looks wrong, debug and fix the pipeline before proceeding. \
Do NOT continue to S4 with a broken label scheme.
  5. If you have been stuck debugging for more than 600 seconds without a \
working pipeline, abandon the current approach and go back to S1 to choose \
a different model or method.
"""

_S3_ALL_PRO_MC = _S3_ALL_LITE_STANDARD_MC.replace(
    "  2. You MUST use GPU (CUDA) for inference. Verify with torch.cuda.is_available(). \\\nLoad the model onto GPU (e.g. model.cuda() or device='cuda'). If torch.load uses \\\nmap_location, set it to torch.device('cuda'). Only fall back to CPU if CUDA is \\\ngenuinely unavailable — never force CPU when a GPU is present.",
    "  2. Make full use of available computational resources (check nvidia-smi).",
)

_S4_LITE_STANDARD_MC = """\
S4  INFERENCE   Run inference on ALL patients.
  1. Run the pipeline on every patient in the dataset. Save the multi-class \
label map as `agents_outputs/<patient_id>/{output_filename}` for each patient.
  2. You MUST use GPU (CUDA) for inference — same as S3.
  3. Maximize throughput — if the pipeline supports batched inference, use it.
  4. Print progress so the log captures which patient is being processed.
  5. After all patients are done, confirm that every patient has the output file.
  Do NOT apply post-processing to the label map. Use raw model outputs directly.
"""

_S4_PRO_MC = """\
S4  INFERENCE   Run inference on ALL patients.
  1. Run the pipeline on every patient in the dataset. Save the multi-class \
label map as `agents_outputs/<patient_id>/{output_filename}` for each patient.
  2. Make full use of available computational resources for maximum throughput.
  3. If the pipeline supports batched inference, use it.
  4. Print progress so the log captures which patient is being processed.
  5. After all patients are done, confirm that every patient has the output file.
  Do NOT apply post-processing to the label map. Use raw model outputs directly.
"""

_S5_ALL_MC = """\
S5  SUBMIT      Verify outputs and submit results.
  1. Confirm that every patient has `{output_filename}` under \
agents_outputs/<patient_id>/.
  2. Verify no files are missing or corrupted. Check that label values are \
integers within {{0}} ∪ target tissues for each file.
  3. Call `submit_results` when everything is saved and verified.
"""

_IMPORTANT_LITE_MC = """\
## Important
- This is an INFERENCE-ONLY multi-tissue segmentation benchmark. Load \
pre-trained weights — do NOT train or fine-tune any model.
- Output MUST be a single multi-class label map per patient (`{output_filename}`), \
with integer values in {{0}} ∪ target tissues, matching the input scan shape exactly.
- Process ALL patients found in the data directory — missing ANY patient means \
automatic failure (Rating F, zero Dice credit).
- Final clinical score = MEAN Dice across the {num_foreground_classes} foreground \
tissues. Small/hard tissues pull the mean down — coverage matters.
- Print progress so the log captures what's happening.
"""

_IMPORTANT_STANDARD_MC = """\
## Important
- This is an INFERENCE-ONLY multi-tissue segmentation benchmark. Find \
pre-trained models and load their weights — do NOT train or fine-tune.
- Output MUST be a single multi-class label map per patient (`{output_filename}`), \
with integer values in {{0}} ∪ target tissues, matching the input scan shape exactly.
- Process ALL patients — missing any patient = Rating F, zero credit.
- Final clinical score = MEAN Dice across the {num_foreground_classes} foreground \
tissues. Models that do not cover all target classes will score 0 on the missing \
ones and pull the mean down — prefer models trained on this exact label scheme.
- Print progress so the log captures what's happening.
"""

_IMPORTANT_PRO_MC = """\
## Important
- This is an INFERENCE-ONLY multi-tissue segmentation benchmark. Find \
pre-trained models and load their weights — do NOT train or fine-tune.
- Output MUST be a single multi-class label map per patient (`{output_filename}`), \
with integer values in {{0}} ∪ target tissues, matching the input scan shape exactly.
- Process ALL patients — missing any patient = Rating F, zero credit.
- Final clinical score = MEAN Dice across the {num_foreground_classes} foreground \
tissues. Missing classes = 0 on that class — label scheme coverage is decisive.
- You are competing against other agents. The winning strategy is NOT speed — it is \
finding the best model that covers all target tissues.
- Print progress so the log captures what's happening.
"""


# ---- Kickoff messages ----

_KICKOFF = {
    "lite": "Begin. The model architecture has been chosen for you. "
            "Research it, then follow S1 through S5.",
    "standard": "Begin. Choose from the candidate model families, "
                "then follow S1 through S5.",
    "pro": "Begin. Follow S1 through S5.",
}

# Override constants from prompt files (if they exist).
# This keeps the inline strings as fallback but prefers the files.
_PREAMBLE = _load_prompt("prompts/common/preamble.md", _PREAMBLE)
_ENV_LITE = _load_prompt("prompts/common/env_lite.md", _ENV_LITE)
_ENV_STANDARD = _load_prompt("prompts/common/env_standard.md", _ENV_STANDARD)
_ENV_PRO = _load_prompt("prompts/common/env_pro.md", _ENV_PRO)
_S1_LITE = _load_prompt("prompts/s1_plan/lite.md", _S1_LITE)
_S1_STANDARD = _load_prompt("prompts/s1_plan/standard.md", _S1_STANDARD)
_S1_PRO = _load_prompt("prompts/s1_plan/pro.md", _S1_PRO)
_S2_LITE = _load_prompt("prompts/s2_setup/lite.md", _S2_LITE)
_S2_STANDARD_PRO = _load_prompt("prompts/s2_setup/standard_pro.md", _S2_STANDARD_PRO)
_S3_ALL_LITE_STANDARD = _load_prompt("prompts/s3_validate/lite_standard.md", _S3_ALL_LITE_STANDARD)
_S3_ALL_PRO = _load_prompt("prompts/s3_validate/pro.md", _S3_ALL_PRO)
_S4_LITE_STANDARD = _load_prompt("prompts/s4_inference/lite_standard.md", _S4_LITE_STANDARD)
_S4_PRO = _load_prompt("prompts/s4_inference/pro.md", _S4_PRO)
_S5_ALL = _load_prompt("prompts/s5_submit/all.md", _S5_ALL)
_IMPORTANT_LITE = _load_prompt("prompts/common/important_lite.md", _IMPORTANT_LITE)
_IMPORTANT_STANDARD = _load_prompt("prompts/common/important_standard.md", _IMPORTANT_STANDARD)
_IMPORTANT_PRO = _load_prompt("prompts/common/important_pro.md", _IMPORTANT_PRO)


def build_tier_system_prompt(tier_config, task_cfg, model_info,
                             data_dir, output_dir, task_id=None):
    """Assemble the full system prompt from tier-aware composable blocks.

    Skills are loaded from the task folder (eval_seg/<task_id>/) when
    task_id is provided. Otherwise falls back to the hardcoded blocks
    (legacy mode for backward compatibility).

    If the task's config.yaml declares ``task_type: multiclass`` the
    multi-tissue prompt variants are used instead of the binary
    organ/lesion ones. Binary tasks (kidney, liver, pancreas) are
    unaffected.
    """
    organ = task_cfg["organ"]
    task_config = load_task_config(task_id) if task_id else {}
    modality = task_config.get("modality", "CT")
    input_filename = task_config.get("input_filename", "ct.nii.gz")
    input_format = "NIfTI .nii.gz"
    lesion_ratio_min = task_config.get("lesion_ratio_min", 0.01)

    is_multiclass = task_config.get("task_type") == "multiclass"
    output_filename = task_config.get("output_filename", "dseg.nii.gz")
    tissue_labels = task_config.get("tissue_labels") or {}
    # Render "1=eCSF, 2=GM, ..." for prompt interpolation
    tissue_labels_brief = ", ".join(
        f"{k}={v}" for k, v in sorted(
            ((int(k), v) for k, v in tissue_labels.items()), key=lambda x: x[0]
        )
    ) if tissue_labels else ""
    num_foreground_classes = len(tissue_labels)

    fmt = {
        "task_description": task_cfg["task_description"],
        "data_dir": data_dir,
        "output_dir": output_dir,
        "organ": organ,
        "modality": modality,
        "input_filename": input_filename,
        "input_format": input_format,
        "lesion_ratio_min": lesion_ratio_min,
        "output_filename": output_filename,
        "tissue_labels_brief": tissue_labels_brief,
        "num_foreground_classes": num_foreground_classes,
    }

    tier = tier_config.name

    # ── Helper to load a skill from the task folder ──
    def _skill(filename):
        """Load a skill .md file from the task folder and format it."""
        if not task_id:
            return ""
        raw = load_skill(task_id, filename)
        if not raw:
            return ""
        # Simple {var} substitution (double-braces {{ }} for Python format strings
        # in skill files are left as-is since they're shown to the agent as code)
        try:
            return raw.format(**fmt)
        except (KeyError, IndexError):
            return raw  # skill file may have unresolvable placeholders — show as-is

    # Preamble — swap the "Required Outputs" block for multi-class tasks.
    # Substitute on the UNFORMATTED template so the single .format() call at
    # the end handles all placeholders without double-escaping.
    preamble = _PREAMBLE
    if is_multiclass:
        import re as _re
        preamble = _re.sub(
            r"## Required Outputs.*?(?=\Z)",
            _REQUIRED_OUTPUTS_MC.rstrip() + "\n",
            preamble,
            flags=_re.DOTALL,
        )
        # Swap the binary clinical-score wording in the Violation Penalty
        # bullet to the multiclass phrasing. Binary tasks untouched.
        preamble = preamble.replace(
            _VIOLATION_CLINICAL_BINARY, _VIOLATION_CLINICAL_MC
        )
    parts = [preamble.format(**fmt)]

    # Environment block
    if tier == "lite":
        req_src = load_requirements_path(task_id) if task_id else ""
        if not req_src:
            req_src = os.path.join(SCRIPT_DIR, "data", organ,
                                   model_info.get("requirements_txt", ""))
        req_dest = os.path.join(output_dir, "requirements.txt")
        if os.path.isfile(req_src) and not os.path.isfile(req_dest):
            import shutil
            shutil.copy2(req_src, req_dest)
        fmt["requirements_txt_path"] = os.path.join(output_dir, "requirements.txt")
        parts.append(_ENV_LITE.format(**fmt))
    elif tier == "standard":
        parts.append(_ENV_STANDARD.format(**fmt))
    else:
        parts.append(_ENV_PRO.format(**fmt))

    # Workflow header
    parts.append("## Workflow  (S1 -> S5, follow in order)")

    # S1
    if tier == "lite":
        fmt["model_architecture"] = model_info["model_architecture"]
        fmt["model_description"] = model_info["model_description"]
        tpl = _S1_LITE_MC if is_multiclass else _S1_LITE
        parts.append(tpl.format(**fmt))
        skill_s1 = _skill("lite_s1.md")
        if skill_s1:
            parts.append("\n" + skill_s1)
    elif tier == "standard":
        fmt["model_range"] = "\n".join(
            f"  - {line}" for line in model_info["model_range"]
        )
        tpl = _S1_STANDARD_MC if is_multiclass else _S1_STANDARD
        parts.append(tpl.format(**fmt))
        skill_s1 = _skill("standard_s1.md")
        if skill_s1:
            parts.append("\n" + skill_s1)
    else:
        fmt["modality"] = modality
        parts.append(_S1_PRO.format(**fmt))

    # S2
    if tier == "lite":
        parts.append(_S2_LITE.format(**fmt))
        skill_s2 = _skill("lite_s2.md")
        if skill_s2:
            parts.append("\n" + skill_s2)
    else:
        parts.append(_S2_STANDARD_PRO.format(**fmt))

    # S3
    if tier in ("lite", "standard"):
        tpl_s3 = _S3_ALL_LITE_STANDARD_MC if is_multiclass else _S3_ALL_LITE_STANDARD
        s3 = tpl_s3.format(**fmt)
        # Load S3 skill from task folder
        skill_name = "lite_s3.md" if tier == "lite" else "standard_s3.md"
        skill_s3 = _skill(skill_name)
        if skill_s3:
            s3 += "\n" + skill_s3
    else:
        tpl_s3_pro = _S3_ALL_PRO_MC if is_multiclass else _S3_ALL_PRO
        s3 = tpl_s3_pro.format(**fmt)
    parts.append(s3)

    # S4
    if tier in ("lite", "standard"):
        tpl_s4 = _S4_LITE_STANDARD_MC if is_multiclass else _S4_LITE_STANDARD
        parts.append(tpl_s4.format(**fmt))
    else:
        tpl_s4_pro = _S4_PRO_MC if is_multiclass else _S4_PRO
        parts.append(tpl_s4_pro.format(**fmt))

    # S5
    tpl_s5 = _S5_ALL_MC if is_multiclass else _S5_ALL
    parts.append(tpl_s5.format(**fmt))

    # Important
    if tier == "lite":
        tpl_imp = _IMPORTANT_LITE_MC if is_multiclass else _IMPORTANT_LITE
    elif tier == "standard":
        tpl_imp = _IMPORTANT_STANDARD_MC if is_multiclass else _IMPORTANT_STANDARD
    else:
        tpl_imp = _IMPORTANT_PRO_MC if is_multiclass else _IMPORTANT_PRO
    parts.append(tpl_imp.format(**fmt))

    return "\n".join(parts)


# ==================================================================
# Tool schemas (OpenAI format)
# ==================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Execute Python or bash code in your isolated GPU environment. "
                "Pre-installed: PyTorch, MONAI, nnU-Net, nibabel, etc. "
                "You can pip install additional packages. "
                "Returns stdout and stderr. No timeout on execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "bash"],
                        "description": "python or bash",
                    },
                    "code": {
                        "type": "string",
                        "description": "The code to execute",
                    },
                },
                "required": ["language", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_results",
            "description": (
                "Call this when all outputs are saved and verified. "
                "Signals that the agent has completed S5 and is done."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# ==================================================================
# Code execution
# ==================================================================

# Paths the agent must NOT access (isolation)
BLOCKED_PATHS = [
    os.path.join(SCRIPT_DIR, "dummy_agents"),
    os.path.join(SCRIPT_DIR, "results"),
    # Block private dirs for all tasks (auto-discovered from task folders)
    *[os.path.join(get_task_data_root(tid), "private")
      for tid in discover_tasks() if os.path.isdir(get_task_data_root(tid))],
    # Block task folder configs (model info, skills — agent must not read these)
    *[path for path in discover_tasks().values()],
    # Block other agents' runs — prevent cheating by reading plans or masks
    os.path.join(SCRIPT_DIR, "runs"),
    os.path.join(SCRIPT_DIR, "runs_archive"),
    # Block reading the benchmark harness itself (contains isolation rules,
    # tier configs, scoring logic — agent could reverse-engineer bypass)
    os.path.join(SCRIPT_DIR, "benchmark_runner.py"),
    os.path.join(SCRIPT_DIR, "dice_scorer.py"),
    os.path.join(SCRIPT_DIR, "format_checker.py"),
    os.path.join(SCRIPT_DIR, "medal_tier.py"),
    os.path.join(SCRIPT_DIR, "aggregate.py"),
    os.path.join(SCRIPT_DIR, "failure_classifier.py"),
    os.path.join(SCRIPT_DIR, "detail_report.py"),
    os.path.join(SCRIPT_DIR, "run_eval.py"),
    os.path.join(SCRIPT_DIR, "llm_judge.py"),
    os.path.join(SCRIPT_DIR, "agent_config.yaml"),
]

# Filesystem escape patterns — agent must NEVER scan outside its sandbox.
# These are checked with regex word-boundary matching to avoid false positives
# (e.g., "ls /data/public" should NOT match "ls /").
import re

BLOCKED_ESCAPE_REGEXES = [
    # -- Broad filesystem scans: find --
    r"find\s+/\s",           # find / ...
    r"find\s+/\n",           # find / at end of line
    r"find\s+/\"",           # find /"
    r"find\s+/'",            # find /'
    r"find\s+/$",            # find / at end of string
    r"find\s+/lustre\b",    # find /lustre
    r"find\s+/home\b",      # find /home
    r"find\s+/opt\b",       # find /opt
    r"find\s+/usr\b",       # find /usr
    r"find\s+/root\b",      # find /root
    r"find\s+/tmp\b",       # find /tmp

    # -- Broad filesystem scans: ls (with optional flags like ls -la /) --
    r"\bls\s+(-[a-zA-Z]+\s+)?/\s",     # ls / ...
    r"\bls\s+(-[a-zA-Z]+\s+)?/\n",     # ls / at EOL
    r"\bls\s+(-[a-zA-Z]+\s+)?/\"",     # ls /"
    r"\bls\s+(-[a-zA-Z]+\s+)?/'",      # ls /'
    r"\bls\s+(-[a-zA-Z]+\s+)?/$",      # ls / at end of string
    r"\bls\s+(-[a-zA-Z]+\s+)?/lustre\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/home\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/etc\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/opt\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/usr\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/root\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/tmp\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/var\b",

    # -- Broad filesystem scans: tree --
    r"\btree\s+/\s",
    r"\btree\s+/\n",
    r"\btree\s+/$",
    r"\btree\s+/[a-z]",     # tree /anything

    # -- Broad filesystem scans: du --
    r"\bdu\s+(-[a-zA-Z]+\s+)?/\s",
    r"\bdu\s+(-[a-zA-Z]+\s+)?/$",

    # -- Relative path traversal (parent directory escape) --
    r"\.\./\.\.",                        # ../../  (two levels up = always escaping)
    r"os\.path\.join\([^)]*\.\.",        # os.path.join(x, '..')

    # -- Python filesystem APIs: root traversal --
    r"os\.walk\s*\(\s*['\"]\/['\"]",          # os.walk('/')
    r"os\.listdir\s*\(\s*['\"]\/['\"]",       # os.listdir('/')
    r"os\.scandir\s*\(\s*['\"]\/['\"]",       # os.scandir('/')
    # glob.glob('/') but NOT glob.glob('/lustre/...') (agent's allowed paths)
    # or glob.glob('/data/...') / glob.glob('/workspace/...') (Docker paths)
    r"glob\.glob\s*\(\s*['\"]\/(?!lustre\/|data\/|workspace\/)",
    r"glob\.iglob\s*\(\s*['\"]\/(?!lustre\/|data\/|workspace\/)",

    # -- Python filesystem APIs: parent traversal --
    r"os\.walk\s*\(\s*['\"]\.\.[\/'\"\\]",    # os.walk('..')
    r"os\.listdir\s*\(\s*['\"]\.\.[\/'\"\\]", # os.listdir('..')
    r"os\.scandir\s*\(\s*['\"]\.\.[\/'\"\\]", # os.scandir('..')

    # -- pathlib traversal --
    r"Path\s*\(\s*['\"]\/['\"\)]",            # Path('/')
    r"Path\s*\(\s*['\"]\.\.[\/'\"\\]",        # Path('..')
    r"\.parent\s*\.parent",                    # path.parent.parent

    # -- System paths --
    r"\/proc\/",             # procfs
    r"\/sys\/",              # sysfs
    r"\/etc\/passwd",        # system files
    r"\/etc\/shadow",        # system files
    r"\/etc\/hosts",         # system files

    # -- Container/namespace escape --
    r"docker\.sock",         # docker socket
    r"\bnsenter\b",          # namespace enter
    r"\bchroot\b",           # change root
    r"\/var\/run\/",         # runtime sockets
    r"\/proc\/1\/root",      # container escape

    # -- Recon commands --
    r"\blocate\s+",          # locate command
    r"\bwhereis\s+",         # whereis command

    # -- String construction evasion --
    r"chr\s*\(\s*47\s*\)",                    # chr(47) == '/'
    r"\\x2f",                                  # hex escape for '/'
    r"b64decode|b64encode",                    # base64 evasion
    r"codecs\.decode",                         # codec-based evasion
    r"__import__\s*\(\s*['\"]os['\"]",        # dynamic import of os
    r"getattr\s*\(\s*os\s*,",                 # getattr(os, 'listdir')

    # -- subprocess with list args (evades string pattern matching) --
    r"subprocess\.\w+\(\s*\[\s*['\"]find['\"]",  # subprocess.run(['find'...
    r"subprocess\.\w+\(\s*\[\s*['\"]ls['\"]",    # subprocess.run(['ls'...
    r"subprocess\.\w+\(\s*\[\s*['\"]tree['\"]",  # subprocess.run(['tree'...

    # ── Red-team discovered bypasses (runtime path construction) ──

    # os.path.sep used to construct paths dynamically
    # NOTE: os.sep removed — false positive in legitimate code (e.g. tree walkers).
    # os.path.sep is kept as it's more deliberate path construction.
    r"\bos\.path\.sep\b",                      # os.path.sep

    # bytes/bytearray with ASCII codes to construct paths
    r"\bbytes\s*\(\s*\[",                      # bytes([...])
    r"\bbytearray\s*\(\s*\[",                  # bytearray([...])

    # struct.pack to construct path bytes
    r"\bstruct\.pack\b",

    # binascii (alternative to blocked base64)
    r"\bbinascii\.",

    # importlib.util to load arbitrary files as modules
    r"\bimportlib\.util\b",

    # bash printf for path construction
    r"\bprintf\s+['\"]%s['\"]",               # printf '%s' ... (path concat)
    r"\$\(printf\b",                           # $(printf ...) subshell

    # bash variable expansion tricks
    r"\$\{[a-z]\}\$\{[a-z]\}",               # ${x}${y}${z} variable concat
    r"\bsource\s+\/dev\/stdin\b",             # source /dev/stdin (code injection)
    r"\beval\s+\"\$",                          # eval "$..." (variable eval)
    r"\bIFS\s*=",                              # IFS manipulation

    # Python string manipulation to morph allowed paths into forbidden
    r"\.replace\s*\(\s*['\"]public['\"]",     # .replace('public', 'private')
    r"\.replace\s*\(\s*['\"]workspace['\"]",  # .replace('workspace', ...)

    # os.path.join with unpacked list (hides path segments)
    r"os\.path\.join\s*\(\s*\*",              # os.path.join(*list)

    # map(chr, ...) to construct path from int list
    r"\bmap\s*\(\s*chr\b",                    # map(chr, [47,...])
    r"chr\s*\(\s*[a-z_]\w*\s*\)",             # chr(c) — variable arg

    # bytes.fromhex
    r"\bbytes\.fromhex\b",

    # os.popen (can execute shell commands with constructed paths)
    r"\bos\.popen\b",

    # unicode named characters for path separators
    r"\\N\{SOLIDUS\}",

    # pathlib division operator to build paths
    r"Path\s*\(\s*['\"]['\"]?\s*\)\s*\/",
    r"Path\s*\(\s*\)\s*\/",

    # Step-by-step directory traversal
    r"os\.chdir\s*\(\s*['\"]\/['\"]",         # os.chdir('/')
    r"os\.chdir\s*\(\s*['\"]\.\.?['\"]",      # os.chdir('..')
    r"\bos\.fchdir\b",
    r"\bcd\s+/workspace\s*&&\s*cd\s+\.\.",
    r"\bcd\s+\.\.\s",
    r"\bcd\s+/\s",
    r"\bcd\s+/$",

    # ast/compile/exec for code generation evasion
    r"\bast\.parse\b",
    r"\bcompile\s*\(",
    r"\bexec\s*\(",

    # importlib.import_module
    r"\bimportlib\.import_module\b",

    # Extracting slash from runtime values
    r"\bos\.altsep\b",
    r"\bsys\.exec_prefix\b",
]
_ESCAPE_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE)
              for p in BLOCKED_ESCAPE_REGEXES]


def _check_isolation(code: str, extra_blocked: list = None,
                     blocked_keywords: list = None,
                     allowed_roots: list = None) -> str:
    """Return error string if code references blocked paths, else empty.

    Enforces strict sandboxing:
      1. Block explicit forbidden paths
      2. Block filesystem escape patterns (find /, ls /, os.walk('/'), etc.)
      3. Block forbidden keywords
      4. Block ground_truth references
    """
    # Check explicit blocked paths
    all_blocked = BLOCKED_PATHS + (extra_blocked or [])
    for bp in all_blocked:
        if bp in code:
            return f"BLOCKED: code references restricted path {bp}"

    # Check escape patterns (regex-based to avoid false positives)
    for rx in _ESCAPE_RE:
        m = rx.search(code)
        if m:
            return (f"BLOCKED: sandbox escape detected — matched '{m.group()}'. "
                    f"You may ONLY access your data directory and workspace.")

    # Keyword-level blocking (e.g., "tier_config" for Pro)
    for kw in (blocked_keywords or []):
        if kw in code:
            return f"BLOCKED: code references restricted keyword '{kw}'"

    # Block reading ground_truth.csv
    if "ground_truth" in code:
        return "BLOCKED: code references ground_truth (private data)"

    # Anti-tampering — block attempts to disable the runtime sandbox
    _anti_tamper = [
        r"_sandbox_hook",
        r"_sandbox_check",
        r"sys\.addaudithook",
        r"addaudithook",
        r"PySys_AddAuditHook",
        r"PySys_Audit",
        r"ctypes.*pythonapi",
        r"sandbox.*preamble",
        r"SANDBOX_PREAMBLE",
    ]
    for pat in _anti_tamper:
        if re.search(pat, code, re.IGNORECASE):
            return ("BLOCKED: anti-tampering — code attempts to reference "
                    "or disable the sandbox enforcement layer.")

    return ""


# ── Runtime audit hook preamble (injected into Python scripts) ────────
import textwrap as _textwrap

_SANDBOX_PREAMBLE_PY = _textwrap.dedent("""\
    import sys as _sys, os as _os

    # Force all downloads/caches into workspace (override, not setdefault,
    # so inherited env vars from conda/shell can't redirect to /root/.cache).
    _ws = _os.environ.get('WORKSPACE_DIR', '/workspace')
    _os.environ['HF_HOME'] = _os.path.join(_ws, '.cache', 'huggingface')
    _os.environ['HF_TOKEN'] = _os.environ.get('HF_TOKEN', 'REDACTED_HF_TOKEN')
    _os.environ['HUGGING_FACE_HUB_TOKEN'] = _os.environ['HF_TOKEN']
    _os.environ['TORCH_HOME'] = _os.path.join(_ws, '.cache', 'torch')
    _os.environ['XDG_CACHE_HOME'] = _os.path.join(_ws, '.cache')
    _os.environ['PIP_CACHE_DIR'] = _os.path.join(_ws, '.cache', 'pip')
    _os.environ['MONAI_DATA_DIRECTORY'] = _os.path.join(_ws, '.cache', 'monai')
    _os.environ['MPLCONFIGDIR'] = _os.path.join(_ws, '.cache', 'matplotlib')
    _os.environ['TMPDIR'] = _os.path.join(_ws, '.cache', 'tmp')
    _os.environ['TORCHINDUCTOR_CACHE_DIR'] = _os.path.join(_ws, '.cache', 'torchinductor')
    _os.makedirs(_os.path.join(_ws, '.cache', 'tmp'), exist_ok=True)
    _os.makedirs(_os.path.join(_ws, '.cache', 'torchinductor'), exist_ok=True)
    _os.makedirs(_os.path.join(_ws, '.cache'), exist_ok=True)

    # Fix torch.load weights_only default (PyTorch 2.6+) so agent never
    # needs to patch it — preserves original map_location (keeps GPU).
    try:
        import torch as _torch
        _orig_torch_load = _torch.load
        def _patched_torch_load(*_a, **_kw):
            _kw.setdefault('weights_only', False)
            return _orig_torch_load(*_a, **_kw)
        _torch.load = _patched_torch_load
    except ImportError:
        pass

    def _sandbox_hook(_event, _args):
        _FORBIDDEN_PREFIXES = ('/data/private', '/eval/', '/results/')
        # Write-only forbidden prefixes: block mkdir/rename/remove/copy but
        # allow reads (many libraries legitimately read from /tmp or /root).
        _WRITE_FORBIDDEN_PREFIXES = ('/root/', '/tmp/')
        _AUDIT_EVENTS = ('open', 'os.listdir', 'os.scandir',
                         'os.chdir', 'os.mkdir', 'os.rename',
                         'os.remove', 'os.symlink',
                         'shutil.copyfile', 'shutil.copytree',
                         'shutil.rmtree')
        _WRITE_EVENTS = ('os.mkdir', 'os.rename', 'os.remove',
                         'os.symlink', 'shutil.copyfile',
                         'shutil.copytree', 'shutil.rmtree')
        if _event in _AUDIT_EVENTS and _args:
            _path = str(_args[0])
            try:
                _resolved = _os.path.realpath(_path)
            except Exception:
                _resolved = _path
            for _fp in _FORBIDDEN_PREFIXES:
                if _resolved.startswith(_fp) or _path.startswith(_fp):
                    _sys.stderr.write(
                        f"SANDBOX VIOLATION: access to {_resolved} is FORBIDDEN.\\n"
                        f"All remaining scores will be zeroed. Rating = F.\\n")
                    _sys.stderr.flush()
                    _os._exit(99)
            # Block writes (but not reads) to /root/ and /tmp/
            if _event in _WRITE_EVENTS or (_event == 'open' and len(_args) > 1 and
                    any(c in str(_args[1]) for c in ('w', 'a', 'x'))):
                for _fp in _WRITE_FORBIDDEN_PREFIXES:
                    if _resolved.startswith(_fp) or _path.startswith(_fp):
                        _sys.stderr.write(
                            f"SANDBOX VIOLATION: write to {_resolved} is FORBIDDEN.\\n"
                            f"Downloads/caches must go into $WORKSPACE_DIR.\\n")
                        _sys.stderr.flush()
                        _os._exit(99)

    _sys.addaudithook(_sandbox_hook)
    del _sandbox_hook
    # ── end sandbox preamble ──
""")

_SANDBOX_PREAMBLE_BASH = _textwrap.dedent("""\
    # ── sandbox preamble ──
    _WS="${WORKSPACE_DIR:-/workspace}"
    export HF_HOME="${_WS}/.cache/huggingface"
    export HF_TOKEN="${HF_TOKEN:-REDACTED_HF_TOKEN}"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export TORCH_HOME="${_WS}/.cache/torch"
    export XDG_CACHE_HOME="${_WS}/.cache"
    export PIP_CACHE_DIR="${_WS}/.cache/pip"
    export MONAI_DATA_DIRECTORY="${_WS}/.cache/monai"
    export MPLCONFIGDIR="${_WS}/.cache/matplotlib"
    export TMPDIR="${_WS}/.cache/tmp"
    export TORCHINDUCTOR_CACHE_DIR="${_WS}/.cache/torchinductor"
    mkdir -p "${_WS}/.cache/tmp" "${_WS}/.cache/torchinductor" "${_WS}/.cache"
    _sandbox_check() {
        local resolved
        for arg in "$@"; do
            resolved=$(readlink -f "$arg" 2>/dev/null || echo "$arg")
            case "$resolved" in
                /data/private*|/eval/*|/results/*)
                    echo "SANDBOX VIOLATION: access to $resolved is FORBIDDEN." >&2
                    echo "All remaining scores will be zeroed. Rating = F." >&2
                    exit 99
                    ;;
            esac
        done
    }
    _sandbox_write_check() {
        local resolved
        for arg in "$@"; do
            resolved=$(readlink -f "$arg" 2>/dev/null || echo "$arg")
            case "$resolved" in
                /root/*|/tmp/*)
                    echo "SANDBOX VIOLATION: write to $resolved is FORBIDDEN." >&2
                    echo "Downloads/caches must go into the workspace dir." >&2
                    exit 99
                    ;;
            esac
        done
    }
    cat()  { _sandbox_check "$@"; command cat "$@"; }
    head() { _sandbox_check "$@"; command head "$@"; }
    tail() { _sandbox_check "$@"; command tail "$@"; }
    less() { _sandbox_check "$@"; command less "$@"; }
    more() { _sandbox_check "$@"; command more "$@"; }
    cp()   { _sandbox_check "$@"; _sandbox_write_check "$@"; command cp "$@"; }
    mv()   { _sandbox_check "$@"; _sandbox_write_check "$@"; command mv "$@"; }
    ln()   { _sandbox_check "$@"; _sandbox_write_check "$@"; command ln "$@"; }
    mkdir() { _sandbox_write_check "$@"; command mkdir "$@"; }
    rm()    { _sandbox_write_check "$@"; command rm "$@"; }
    tee()   { _sandbox_write_check "$@"; command tee "$@"; }
    # ── end sandbox preamble ──
""")


def execute_code(language: str, code: str, cwd: str,
                 conda_env: str = None, timeout: int = None,
                 extra_blocked: list = None,
                 blocked_keywords: list = None) -> dict:
    """Run code in an isolated conda env, return stdout + stderr.

    Python scripts get a sys.addaudithook() preamble injected.
    Bash scripts get function wrappers for cat/head/tail/etc.
    """
    env_name = conda_env or BASE_CONDA_ENV

    # Isolation check (static layers 1+2)
    violation = _check_isolation(code, extra_blocked=extra_blocked,
                                 blocked_keywords=blocked_keywords)
    if violation:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": violation,
        }

    # Inject runtime sandbox preamble (layer 3)
    if language == "python":
        full_code = _SANDBOX_PREAMBLE_PY + code
    else:
        full_code = _SANDBOX_PREAMBLE_BASH + code

    # Write code to temp file to avoid shell quoting issues
    suffix = ".py" if language == "python" else ".sh"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, dir=cwd, delete=False
    ) as f:
        f.write(full_code)
        script_path = f.name

    try:
        # Use --prefix for path-based envs, -n for named envs
        if os.sep in env_name:
            env_flag = ["--prefix", env_name]
        else:
            env_flag = ["-n", env_name]

        if language == "python":
            cmd = ["conda", "run"] + env_flag + ["python3", script_path]
        else:
            cmd = ["conda", "run"] + env_flag + ["bash", script_path]

        # Pass WORKSPACE_DIR so the sandbox preamble resolves cache
        # paths to the actual output directory, not the /workspace default.
        # NOTE: Do NOT override HOME — it breaks conda env resolution
        # (conda needs ~/.condarc to discover named environments).
        # Instead we rely on force-set cache env vars in the preamble +
        # the sandbox audit hook to block writes to /root/ and /tmp/.
        run_env = os.environ.copy()
        run_env['WORKSPACE_DIR'] = cwd

        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=cwd, env=run_env,
        )
        stdout = proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout
        stderr = proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr

        # Check if the runtime sandbox killed the process (exit code 99)
        if proc.returncode == 99 and "SANDBOX VIOLATION" in stderr:
            return {
                "exit_code": -1,
                "stdout": stdout,
                "stderr": f"BLOCKED: {stderr.strip()}",
            }

        return {
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"TIMEOUT: execution exceeded {timeout}s",
        }
    except Exception as e:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Execution error: {e}",
        }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass



# NOTE: Heuristic S1-S3 scorer removed. S1-S3 are now scored exclusively
# by the LLM judge. If the judge fails, S1-S3 remain None.


def check_submission(output_dir: str, patients: list) -> dict:
    """Quick check that expected output files exist."""
    missing_masks = []
    for pid in patients:
        for fname in ("organ.nii.gz", "lesion.nii.gz"):
            p = os.path.join(output_dir, "agents_outputs", pid, fname)
            if not os.path.isfile(p):
                missing_masks.append(f"{pid}/{fname}")
    return {
        "missing_masks": missing_masks,
        "complete": len(missing_masks) == 0,
    }


# ==================================================================
# Tool-call summary builder
# ==================================================================

def _code_description(code: str) -> str:
    """Extract first comment line from code as a human-readable description."""
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            desc = stripped.lstrip("#").strip()
            if desc:
                return desc[:80]
    return ""


def _classify_phase(code: str, messages: list, turn: int,
                    patients: list, prev_phase: str) -> str:
    """Heuristic: assign a tool call to S1–S5 based on code content.

    Priority:
      1. Explicit "S1:"–"S5:" labels in code comments (agents often write these)
      2. Keyword heuristics on the code content
      3. Carry forward from previous call
    """
    code_lower = code.lower()
    desc = _code_description(code).lower()

    # (1) Explicit step labels in comment lines — highest priority
    import re
    step_refs = re.findall(r'\bs([1-5])\b', desc)
    if step_refs:
        return f"S{step_refs[0]}"

    # Also check first 3 code lines for step labels
    first_lines = "\n".join(code.split("\n")[:3]).lower()
    step_refs = re.findall(r'#\s*s([1-5])\b', first_lines)
    if step_refs:
        return f"S{step_refs[0]}"

    # (2) Keyword heuristics on code content
    if "submit_results" in code_lower:
        return "S5"
    if "agents_decision" in code_lower and ("to_csv" in code_lower
                                            or "write" in code_lower):
        return "S5"

    # Batch inference: iterates over many patients (not just mkdir)
    pid_refs = sum(1 for pid in patients if pid in code)
    is_mkdir_only = "makedirs" in code_lower or "mkdir" in code_lower
    if not is_mkdir_only and (
        pid_refs >= 4 or "for pid in" in code_lower
        or "for patient" in code_lower
    ):
        return "S4"

    # Post-processing after inference
    if prev_phase == "S4" and any(kw in code_lower for kw in [
        "post-process", "postprocess", "connected_component",
        "verify", "statistics", "all outputs",
    ]):
        return "S4"

    # Environment setup
    if "venv" in code_lower or "python -m venv" in code_lower:
        return "S2"
    if "pip install" in code_lower and prev_phase in ("", "S1", "S2"):
        return "S2"

    # Plan creation
    if "plan.md" in code_lower or "plan.png" in code_lower or "flowchart" in code_lower:
        return "S1"

    # Model research / search (only if still in early phases)
    if prev_phase in ("", "S1") and any(kw in code_lower for kw in [
        "urllib.request", "bundle", "model zoo", "search",
        "zenodo", "huggingface", "label map", "class_map",
        "curl ", "check",
    ]):
        return "S1"

    # Single patient inference (after setup) → S3
    if prev_phase == "S2" and pid_refs >= 1:
        return "S3"

    # Carry forward previous phase if nothing else matches
    return prev_phase or "S1"


def _build_tool_summary(code_executions: list, submitted: bool,
                        messages: list, patients: list) -> dict:
    """Build a rich tool-call summary with per-call log and phase breakdown."""

    # --- Per-call log ---
    call_log = []
    phase = ""
    for i, ex in enumerate(code_executions):
        phase = _classify_phase(
            ex.get("code", ""), messages, ex.get("turn", 0),
            patients, phase,
        )
        call_log.append({
            "seq": i + 1,
            "turn": ex.get("turn"),
            "phase": phase,
            "language": ex.get("language"),
            "exit_code": ex.get("exit_code"),
            "exec_time_s": ex.get("exec_time_s"),
            "description": _code_description(ex.get("code", "")),
        })

    if submitted:
        call_log.append({
            "seq": len(code_executions) + 1,
            "turn": call_log[-1]["turn"] + 1 if call_log else 1,
            "phase": "S5",
            "language": None,
            "exit_code": 0,
            "exec_time_s": None,
            "description": "submit_results",
        })

    # --- Phase summary ---
    phase_stats = {}
    for entry in call_log:
        p = entry["phase"]
        if p not in phase_stats:
            phase_stats[p] = {"calls": 0, "errors": 0, "total_exec_s": 0.0}
        phase_stats[p]["calls"] += 1
        if entry["exit_code"] and entry["exit_code"] != 0:
            phase_stats[p]["errors"] += 1
        if entry["exec_time_s"]:
            phase_stats[p]["total_exec_s"] += entry["exec_time_s"]

    # Round exec times
    for ps in phase_stats.values():
        ps["total_exec_s"] = round(ps["total_exec_s"], 1)

    # --- Failures ---
    failures = []
    for entry in call_log:
        if entry["exit_code"] and entry["exit_code"] != 0:
            failures.append({
                "seq": entry["seq"],
                "phase": entry["phase"],
                "description": entry["description"],
            })

    return {
        "total": len(call_log),
        "by_tool": {
            "execute_code": len(code_executions),
            "submit_results": 1 if submitted else 0,
        },
        "errors": sum(1 for c in code_executions if c["exit_code"] != 0),
        "call_log": call_log,
        "phase_summary": phase_stats,
        "failures": failures,
    }


# ==================================================================
# OpenRouter API
# ==================================================================

def call_api(api_key, model, system, messages, tools,
             temperature=0.0, reasoning=True, base_url=None):
    """Call LLM API with tool use + reasoning."""
    endpoint = (base_url.rstrip("/") + "/chat/completions" if base_url
                else "https://openrouter.ai/api/v1/chat/completions")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if reasoning and not base_url:
        payload["reasoning"] = {"enabled": True}

    resp = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")

    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning_details = (msg.get("reasoning_details")
                         or msg.get("reasoning_content"))

    tc = []
    if msg.get("tool_calls"):
        for t in msg["tool_calls"]:
            args = t["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            tc.append({
                "id": t["id"],
                "name": t["function"]["name"],
                "arguments": args,
            })

    usage = data.get("usage", {})
    return {
        "content": content,
        "tool_calls": tc,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "finish_reason": choice.get("finish_reason", "stop"),
        "reasoning_details": reasoning_details,
    }


# ==================================================================
# Main runner
# ==================================================================

class BenchmarkRunner:
    def __init__(self, agent_name: str, task: str, tier: str = "pro",
                 llm_judge: bool = False, online_judge: bool = False,
                 judge_model_path: str = None, judge_vllm_url: str = None,
                 output_dir: str = None):
        self.tier = get_tier_config(tier)
        self.llm_judge = llm_judge
        self.online_judge = online_judge
        self.judge_model_path = judge_model_path
        self.judge_vllm_url = judge_vllm_url

        with open(CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)

        # Load task from task folder (supports legacy names like "kidney")
        self.task_id = task
        task_config = load_task_config(task)
        self.task_id = task_config["_task_id"]  # resolved name
        model_info_dict = load_model_info(self.task_id)

        # Extract tier-specific model info in the format the prompt builder expects
        tier_mi = model_info_dict.get(tier, model_info_dict.get("lite", {}))
        self.model_info = tier_mi
        # For prompt builder: merge all tier info into one dict (legacy compat)
        self.model_info_all = {
            "model_architecture": model_info_dict.get("lite", {}).get("model_architecture", ""),
            "model_description": model_info_dict.get("lite", {}).get("model_description", ""),
            "model_range": model_info_dict.get("standard", {}).get("model_range", []),
            "clinical_background": model_info_dict.get("pro", {}).get("clinical_background", ""),
        }

        self.task_cfg = {
            "organ": task_config["organ"],
            "task_description": task_config["task_description"],
        }
        # Keep the full task config for downstream multi-class branching
        # (does not affect binary tasks — consumers default to binary).
        self.task_config_full = task_config
        self.organ = task_config["organ"]
        self.data_root = task_config["_data_root"]
        self.task = task_config.get("organ", task)  # for backward compat in reports

        self.agent_cfg = self.config["agents"][agent_name]
        self.agent_name = agent_name
        self.patients = _discover_patients_from_loader(self.task_id)
        if not self.patients:
            sys.exit(f"No patients found in {self.data_root}/public/. "
                     f"Stage data before running the benchmark.")
        self.model = self.agent_cfg["model"]
        self.api_key = self.agent_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = self.agent_cfg.get("base_url")
        self.reasoning = self.agent_cfg.get("reasoning", True)

        # Time limit from task config (hidden from agent)
        self.max_turns = 999999
        self.max_time_s = task_config.get("time_limit_s", 3600)

        # Paths — user-supplied --output-dir or auto-generated
        import random as _random
        self._real_data_dir = os.path.join(self.data_root, "public")
        _run_tag = time.strftime("%y%m%d") + "-" + "".join(
            _random.choices("0123456789abcdef", k=6))
        if output_dir:
            # User controls the directory; append run tag as subdirectory
            self.run_dir = os.path.join(output_dir, _run_tag)
        else:
            # Legacy auto-generated path
            self.run_dir = os.path.join(
                SCRIPT_DIR, "runs", tier, agent_name, task, _run_tag,
            )
        # Process logs (conversation, trace, tool calls, reports)
        self.process_dir = os.path.join(self.run_dir, "process")
        os.makedirs(self.process_dir, exist_ok=True)
        # Archive dir for results (copied from /workspace after run)
        self._real_output_dir = os.path.join(self.run_dir, "outputs")
        os.makedirs(self._real_output_dir, exist_ok=True)

        # ── Sandbox abstraction ──────────────────────────────────────
        # The agent NEVER sees real host paths.  It uses:
        #   /data/public  → symlink to the task's public data
        #   /workspace/run_<id>  → per-run isolated output dir
        # After the run, results are copied to the archive
        # (self._real_output_dir) for persistence, then the run dir
        # is cleaned up.
        #
        # Each run gets its own subdirectory under /workspace to prevent
        # race conditions when multiple runs execute concurrently on the
        # same host.  The run_id includes PID to guarantee uniqueness
        # even if two runs start within the same second.
        self.data_dir = "/data/public"
        _ws_id = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        self.output_dir = os.path.join("/workspace", f"run_{_ws_id}")

        # Set up /data/public symlink (safe — /data/ is not pre-existing)
        os.makedirs("/data", exist_ok=True)
        if os.path.islink(self.data_dir):
            os.unlink(self.data_dir)
        if not os.path.exists(self.data_dir):
            os.symlink(self._real_data_dir, self.data_dir)

        # Create a fresh per-run workspace (no cleanup needed — it's new)
        import shutil as _shutil
        if os.path.exists(self.output_dir):
            _shutil.rmtree(self.output_dir)
        os.makedirs(os.path.join(self.output_dir, "agents_outputs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plan"), exist_ok=True)

        # Tier-specific isolation: block other tiers' runs and
        # tier-specific data files the agent shouldn't see.
        self.extra_blocked = []
        other_tiers = [t for t in ("lite", "standard", "pro") if t != tier]
        for ot in other_tiers:
            self.extra_blocked.append(
                os.path.join(SCRIPT_DIR, "runs", ot)
            )
        # Pro/Standard should not see the provided requirements.txt
        if tier in ("pro", "standard"):
            # Block all task folder requirements files
            for t_id, t_path in discover_tasks().items():
                req_file = os.path.join(t_path, "requirements.txt")
                if os.path.isfile(req_file):
                    self.extra_blocked.append(req_file)
        # Block listing /workspace/ root (could reveal other concurrent runs)
        # Agent should only access its own /workspace/run_<id>/ dir
        self.extra_blocked.append("ls /workspace\n")
        self.extra_blocked.append("ls -l /workspace\n")
        self.extra_blocked.append("os.listdir(\"/workspace\")")
        self.extra_blocked.append("os.listdir('/workspace')")
        # Blocked keywords — patterns that shouldn't appear in agent code
        self.blocked_keywords = []
        # Pro should not read tier_config (contains model hints for other tiers)
        if tier == "pro":
            self.extra_blocked.append(
                os.path.join(SCRIPT_DIR, "tier_config.py")
            )
            self.blocked_keywords.append("tier_config")
        # Standard/Pro should not read requirements files by name
        if tier in ("pro", "standard"):
            self.blocked_keywords.append("requirements_kidney")
            self.blocked_keywords.append("requirements_liver")

        # System prompt — tier-aware, skills loaded from task folder
        self.system = build_tier_system_prompt(
            self.tier, self.task_cfg, self.model_info_all,
            self.data_dir, self.output_dir, task_id=self.task_id,
        )

    def run(self) -> dict:
        # Auto-save run.log inside the run_dir (no external tee needed)
        _run_log_path = os.path.join(self.run_dir, "run.log")
        _run_log_f = open(_run_log_path, "w")
        _orig_stdout = sys.stdout

        class _Tee:
            """Write to both stdout and the run log file."""
            def __init__(self, *streams):
                self._streams = streams
            def write(self, data):
                for s in self._streams:
                    s.write(data)
                    s.flush()
            def flush(self):
                for s in self._streams:
                    s.flush()

        sys.stdout = _Tee(_orig_stdout, _run_log_f)

        kickoff = _KICKOFF[self.tier.name]
        messages = [{"role": "user", "content": kickoff}]

        t_start = time.time()
        total_in = 0
        total_out = 0
        api_calls = 0
        submitted = False
        isolation_breach = False       # set True on 2nd violation → zero all scores
        violation_warnings = 0         # count of warnings issued (kill at 2)
        breach_turn = None             # turn at which the fatal breach occurred
        breach_detail = ""             # violation description
        code_executions = []
        trace = []  # per-event trace — written to trace.jsonl

        # Open trace file for streaming writes (zero cost to agent context)
        trace_path = os.path.join(self.process_dir, "trace.jsonl")
        trace_f = open(trace_path, "w")

        # Detailed tool call log — every call with full code + output
        tool_log_path = os.path.join(self.process_dir, "tool_calls.jsonl")
        tool_log_f = open(tool_log_path, "w")

        def _log_tool_call(turn, name, arguments, result, exec_time_s=None):
            """Write one detailed tool call entry."""
            entry = {
                "ts": round(time.time() - t_start, 2),
                "turn": turn,
                "tool": name,
                "arguments": arguments,
                "result": result,
            }
            if exec_time_s is not None:
                entry["exec_time_s"] = exec_time_s
            tool_log_f.write(json.dumps(entry, default=str) + "\n")
            tool_log_f.flush()

        def _trace(event_type, data):
            """Append one event to the trace file immediately."""
            entry = {"ts": round(time.time() - t_start, 2),
                     "type": event_type, **data}
            trace.append(entry)
            trace_f.write(json.dumps(entry, default=str) + "\n")
            trace_f.flush()

        print(f"\n{'='*60}")
        print(f"  MedAgentsBench — {self.tier.name.upper()} tier")
        print(f"  Agent: {self.agent_name}  Model: {self.model}")
        print(f"  Task: {self.task}  Patients: {len(self.patients)}")
        print(f"  Output: {self.run_dir}")
        print(f"{'='*60}\n")

        for turn in range(self.max_turns):
            elapsed = time.time() - t_start
            if elapsed > self.max_time_s:
                print(f"\n[Runner] TIME LIMIT ({self.max_time_s}s) — stopping.")
                break

            # API call
            try:
                resp = call_api(
                    self.api_key, self.model, self.system, messages,
                    TOOLS, reasoning=self.reasoning,
                    base_url=self.base_url,
                )
            except Exception as e:
                print(f"\n[Runner] API ERROR: {e}")
                # Retry once after 5s
                time.sleep(5)
                try:
                    resp = call_api(
                        self.api_key, self.model, self.system, messages,
                        TOOLS, reasoning=self.reasoning,
                        base_url=self.base_url,
                    )
                except Exception as e2:
                    print(f"[Runner] RETRY FAILED: {e2} — stopping.")
                    break

            api_calls += 1
            total_in += resp["input_tokens"]
            total_out += resp["output_tokens"]

            _trace("api_call", {
                "turn": turn + 1,
                "input_tokens": resp["input_tokens"],
                "output_tokens": resp["output_tokens"],
                "finish_reason": resp["finish_reason"],
                "tool_calls": [tc["name"] for tc in resp["tool_calls"]],
                "content_preview": (resp["content"] or "")[:300],
            })

            # Print agent text
            if resp["content"]:
                preview = resp["content"][:200].replace("\n", " ")
                print(f"  [Turn {turn+1} | {elapsed:.0f}s] {preview}...")

            if not resp["tool_calls"]:
                if resp["finish_reason"] == "length":
                    print(f"  [Turn {turn+1}] WARNING: response truncated (finish_reason=length). Retrying...")
                    messages.append({"role": "assistant", "content": resp["content"] or ""})
                    messages.append({"role": "user", "content": "Your previous response was truncated. Please continue and use the execute_code tool to run your code."})
                    continue
                print(f"  [Turn {turn+1}] No tool calls — agent stopped.")
                break

            # Build assistant message for conversation
            asst_msg = {"role": "assistant", "content": resp["content"] or None}
            if resp["reasoning_details"]:
                asst_msg["reasoning_details"] = resp["reasoning_details"]
            if resp["tool_calls"]:
                asst_msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(tc["arguments"])}}
                    for tc in resp["tool_calls"]
                ]
            messages.append(asst_msg)

            # Execute tool calls
            for tc in resp["tool_calls"]:
                name = tc["name"]
                args = tc["arguments"]

                if name == "execute_code":
                    lang = args.get("language", "python")
                    code = args.get("code", "")
                    code_preview = code[:80].replace("\n", "\\n")
                    print(f"  [Code:{lang}] {code_preview}...")

                    t_exec = time.time()
                    remaining = max(60, int(self.max_time_s - (t_exec - t_start)))
                    result = execute_code(
                        lang, code, cwd=self.output_dir,
                        extra_blocked=self.extra_blocked,
                        blocked_keywords=self.blocked_keywords,
                        timeout=remaining,
                    )
                    exec_time = time.time() - t_exec

                    exec_entry = {
                        "turn": turn + 1,
                        "language": lang,
                        "code": code,
                        "exit_code": result["exit_code"],
                        "exec_time_s": round(exec_time, 1),
                        "stdout_preview": result["stdout"][:200],
                    }
                    code_executions.append(exec_entry)

                    _trace("code_exec", {
                        "turn": turn + 1,
                        "language": lang,
                        "exit_code": result["exit_code"],
                        "exec_time_s": round(exec_time, 1),
                        "code_lines": code.count("\n") + 1,
                        "stdout_len": len(result["stdout"]),
                        "stderr_len": len(result["stderr"]),
                    })

                    # ISOLATION VIOLATION — 1 warning, then kill
                    if result["exit_code"] == -1 and "BLOCKED" in result.get("stderr", ""):
                        violation_warnings += 1

                        if violation_warnings < 2:
                            # ── First violation: WARNING ──
                            warning_msg = (
                                f"⚠ WARNING ({violation_warnings}/1): SANDBOX VIOLATION at turn {turn+1}.\n"
                                f"Violation: {result['stderr']}\n\n"
                                f"Your code was BLOCKED and NOT executed. You may continue,\n"
                                f"but the NEXT violation will IMMEDIATELY TERMINATE your run:\n"
                                f"  - All remaining S-step scores → 0\n"
                                f"  - Clinical score → 0\n"
                                f"  - Rating → F (Disqualified)\n"
                                f"Fix your approach and stay within /data/public/ and /workspace/."
                            )
                            print(f"\n  *** VIOLATION WARNING ({violation_warnings}/1) at turn {turn+1} ***")
                            print(f"  {result['stderr']}")
                            print(f"  Agent warned — next violation = kill.\n")
                            _trace("violation_warning", {
                                "turn": turn + 1,
                                "warning_number": violation_warnings,
                                "violation": result["stderr"],
                            })
                            # Return warning as the tool result so agent sees it
                            result["stderr"] = warning_msg
                            tool_result = json.dumps(result)
                            _log_tool_call(turn + 1, name, args, result,
                                           exec_time_s=round(exec_time, 1))
                            # Continue the loop — don't break
                        else:
                            # ── Second violation: KILL ──
                            isolation_breach = True
                            breach_turn = turn + 1
                            breach_detail = result["stderr"]
                            print(f"\n  *** ISOLATION BREACH at turn {breach_turn} (2nd violation) ***")
                            print(f"  {breach_detail}")
                            print(f"  KILLED — all remaining S-step scores and clinical score → 0.\n")
                            _trace("isolation_breach", {
                                "turn": breach_turn,
                                "violation": breach_detail,
                                "penalty": "zero_all_remaining_scores",
                                "prior_warnings": violation_warnings - 1,
                            })
                            _log_tool_call(turn + 1, name, args, result,
                                           exec_time_s=round(exec_time, 1))
                            submitted = False
                            break

                    status = "OK" if result["exit_code"] == 0 else f"FAIL(rc={result['exit_code']})"
                    print(f"           {status} ({exec_time:.1f}s)")
                    if result["stdout"].strip():
                        for line in result["stdout"].strip().split("\n")[-5:]:
                            print(f"           > {line[:100]}")
                    if result["exit_code"] != 0 and result["stderr"]:
                        for line in result["stderr"].strip().split("\n")[-3:]:
                            print(f"           ! {line[:100]}")

                    tool_result = json.dumps(result)
                    _log_tool_call(turn + 1, name, args, result,
                                   exec_time_s=round(exec_time, 1))

                elif name == "submit_results":
                    check = check_submission(self.output_dir, self.patients)
                    print(f"  [Submit] complete={check['complete']} "
                          f"missing={len(check['missing_masks'])}")
                    tool_result = json.dumps(check)
                    submitted = True
                    _trace("submit", check)
                    _log_tool_call(turn + 1, name, args, check)
                else:
                    tool_result = json.dumps({"error": f"Unknown tool: {name}"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

            if submitted:
                print(f"\n  [Turn {turn+1}] Agent called submit_results — done.")
                break

            # Check if fatal breach (2nd violation) caused inner break
            if isolation_breach:
                print(f"  [Runner] Isolation breach (2nd violation) — skipping to evaluation.")
                break

        wall_time = time.time() - t_start

        _trace("run_end", {
            "wall_time_s": round(wall_time, 2),
            "api_calls": api_calls,
            "total_in": total_in,
            "total_out": total_out,
            "submitted": submitted,
        })
        trace_f.close()
        tool_log_f.close()

        # --- Save conversation log ---
        conv_path = os.path.join(self.process_dir, "conversation.json")
        with open(conv_path, "w") as f:
            json.dump({
                "agent": self.agent_name,
                "model": self.model,
                "task": self.task,
                "tier": self.tier.name,
                "system_prompt": self.system,
                "messages": messages,
                "code_executions": code_executions,
                "trace_events": len(trace),
            }, f, indent=2, default=str)

        # --- Run evaluation ---
        print(f"\n[Runner] Running evaluation...")
        gt_dir = os.path.join(self.data_root, "private")
        agent_out = os.path.join(self.output_dir, "agents_outputs")
        public_dir = os.path.join(self.data_root, "public")

        eval_report = run_eval(
            gt_dir=gt_dir,
            agent_dir=agent_out,
            public_dir=public_dir,
            patient_ids=self.patients,
            task_cfg=self.task_config_full,
        )

        # S1-S3 are scored by the LLM judge (required).
        # Set to None here — they MUST be filled by the judge below.
        from aggregate import compute_workflow_score, assign_rating
        tier_weights = self.tier.step_weights or None

        # ── ISOLATION BREACH PENALTY ──────────────────────────────────
        # If the agent attempted a sandbox violation, zero all remaining
        # S-step scores from the breach point onward, and set clinical
        # score to 0.  This makes violations strictly worse than honest
        # failure — there is no incentive to cheat.
        if isolation_breach:
            print(f"\n[Runner] *** ISOLATION BREACH PENALTY ***")
            print(f"[Runner] Breach at turn {breach_turn}: {breach_detail}")
            print(f"[Runner] Zeroing all remaining step scores and clinical score.")

            # Determine which step the agent was in at breach time.
            # Steps map roughly to execution phases by turn count, but
            # we take the conservative approach: zero ALL step scores
            # from the step that was active at breach time onward.
            # Since we can't reliably determine the exact step, zero
            # everything after S1 if breach was early, or just nuke all.
            step_order = ["s1", "s2", "s3", "s4", "s5"]

            # Heuristic: which step was the agent in at breach?
            # S1 = turns 1-5, S2 = turns 6-10, S3 = after that...
            # Conservative: zero from the breached step onward.
            if breach_turn <= 5:
                zero_from = 0   # zero all steps including S1
            elif breach_turn <= 10:
                zero_from = 1   # zero S2 onward (keep partial S1)
            elif breach_turn <= 15:
                zero_from = 2   # zero S3 onward
            else:
                zero_from = 3   # zero S4 onward

            for i in range(zero_from, len(step_order)):
                eval_report["step_scores"][step_order[i]] = 0.0

            # Clinical score = 0 (any output produced via cheating is worthless)
            eval_report["aggregate"]["clinical_score"] = 0.0
            eval_report["metrics"]["organ_dice"] = 0.0
            eval_report["metrics"]["lesion_dice"] = 0.0
            eval_report["metrics"]["medal_tier"] = 0
            eval_report["metrics"]["medal_name"] = "disqualified"

            # Recompute agentic/overall with zeroed steps
            wf, active = compute_workflow_score(eval_report["step_scores"],
                                                weights=tier_weights)
            eval_report["aggregate"]["agentic_score"] = wf
            eval_report["aggregate"]["active_steps"] = active
            eval_report["aggregate"]["overall_score"] = 0.0
            eval_report["aggregate"]["rating"] = "F"
            eval_report["aggregate"]["resolved"] = False

            # Tag the report with breach metadata
            eval_report["isolation_breach"] = {
                "breached": True,
                "turn": breach_turn,
                "detail": breach_detail,
                "penalty": "All remaining step scores zeroed; clinical score = 0; rating = F",
            }

            print(f"[Runner] Final scores after penalty:")
            for s in step_order:
                print(f"  {s.upper()} = {eval_report['step_scores'][s]}")
            print(f"  Clinical = 0.0  |  Rating = F  |  Resolved = False")

        # --- LLM Judge (required — scores S1-S3) ---
        if not isolation_breach:
            print(f"\n[Runner] Running LLM judge "
                  f"({'online: Claude Opus 4.7' if self.online_judge else 'offline: DeepSeek-R1-Distill-Qwen-32B'})...")
            try:
                from llm_judge import create_judge
                judge_kwargs = {}
                if self.judge_model_path:
                    judge_kwargs["model_path"] = self.judge_model_path
                if self.judge_vllm_url:
                    judge_kwargs["base_url"] = self.judge_vllm_url
                judge = create_judge(online=self.online_judge, **judge_kwargs)

                conversation_for_judge = {
                    "agent": self.agent_name,
                    "model": self.model,
                    "task": self.task,
                    "tier": self.tier.name,
                    "messages": messages,
                    "code_executions": code_executions,
                }
                verdict = judge.judge(conversation_for_judge, eval_report, self.task)
                eval_report["llm_judge"] = verdict.to_dict()

                # Set S1-S3 from judge
                eval_report["step_scores"]["s1"] = verdict.s1_plan_score
                eval_report["step_scores"]["s2"] = verdict.s2_setup_score
                eval_report["step_scores"]["s3"] = verdict.s3_validate_score

                # Recompute aggregate with judge scores
                wf, active = compute_workflow_score(eval_report["step_scores"],
                                                    weights=tier_weights)
                eval_report["aggregate"]["agentic_score"] = wf
                eval_report["aggregate"]["active_steps"] = active
                eval_report["aggregate"]["rating"] = assign_rating(
                    0,
                    medal_tier=eval_report["metrics"].get("medal_tier", 0),
                    format_valid=eval_report["format"].get("submission_format_valid", False),
                )

                print(f"[Runner] Judge S1={verdict.s1_plan_score:.2f} "
                      f"S2={verdict.s2_setup_score:.2f} S3={verdict.s3_validate_score:.2f}")
            except Exception as e:
                # Judge is required — if it fails, S1-S3 stay None.
                # The run report will be generated but S1-S3 are unscored.
                print(f"[Runner] JUDGE FAILED: {e}")
                print(f"[Runner] S1-S3 will be None — run is incomplete without judge.")
                eval_report["llm_judge"] = {"error": str(e)}

        # --- Build detail report ---
        pricing = self.config.get("pricing", {}).get(self.agent_name, {})
        in_price = pricing.get("input_per_1m", 0)
        out_price = pricing.get("output_per_1m", 0)
        cost = (total_in * in_price + total_out * out_price) / 1_000_000

        runtime = {
            "wall_time_s": round(wall_time, 2),
            "api_calls": api_calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "estimated_cost_usd": round(cost, 4),
            "code_executions": len(code_executions),
        }

        tool_summary = _build_tool_summary(
            code_executions, submitted, messages, self.patients,
        )

        # Extract judge verdict dict if available
        judge_verdict = eval_report.get("llm_judge")
        if isinstance(judge_verdict, dict) and "error" in judge_verdict:
            judge_verdict = None  # judge failed, don't use

        detail = generate_detail_report(
            eval_report=eval_report,
            runtime=runtime,
            agent_name=self.agent_name,
            model=self.model,
            task=self.task,
            tool_summary=tool_summary,
            judge_verdict=judge_verdict,
            tier=self.tier.name,
        )

        # Save
        report_path = os.path.join(self.run_dir, "detail_report.json")
        with open(report_path, "w") as f:
            json.dump(detail, f, indent=2)

        # Pro tier: generate summary plots
        if self.tier.generate_summary_plots:
            try:
                from summary_plots import generate_summary_plots
                plot_dir = os.path.join(self.process_dir, "plots")
                plot_paths = generate_summary_plots(detail, plot_dir)
                print(f"[Runner] Plots   -> {plot_dir} ({len(plot_paths)} files)")
            except Exception as e:
                print(f"[Runner] Warning: summary plots failed: {e}")

        print_detail_report(detail)
        print(f"[Runner] Outputs -> {self._real_output_dir}")
        print(f"[Runner] Report  -> {report_path}")
        print(f"[Runner] Conv    -> {conv_path}")
        print(f"[Runner] Trace   -> {trace_path}")
        print(f"[Runner] Tools   -> {tool_log_path}")

        # --- Archive results from per-run workspace to run dir, then clean ---
        import shutil as _shutil
        _ws = self.output_dir  # /workspace/run_<id>
        _archive = self._real_output_dir  # runs/.../outputs/

        # Copy agents_outputs and plan to archive
        for subdir in ("agents_outputs", "plan"):
            src = os.path.join(_ws, subdir)
            dst = os.path.join(_archive, subdir)
            if os.path.isdir(src) and os.listdir(src):
                if os.path.isdir(dst):
                    _shutil.rmtree(dst)
                _shutil.copytree(src, dst)
        # Copy tier_prompt.txt and agents_decision.csv
        for fname in ("tier_prompt.txt", "agents_decision.csv"):
            src = os.path.join(_ws, fname)
            if os.path.isfile(src):
                _shutil.copy2(src, os.path.join(_archive, fname))

        # Clean the entire per-run workspace directory
        try:
            _shutil.rmtree(_ws)
            print(f"[Runner] Cleaned {_ws}")
        except OSError as e:
            print(f"[Runner] Warning: could not remove {_ws}: {e}")

        print(f"[Runner] Archived -> {_archive}")

        # Restore stdout and close run log
        sys.stdout = _orig_stdout
        _run_log_f.close()

        return detail


# ==================================================================
# CLI
# ==================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MedAgentsBench benchmark — agent codes its own solution")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--task", required=True,
                        help="Task ID (e.g., kidney-seg-task) or legacy name (kidney, liver, pancreas)")
    parser.add_argument("--tier", default="pro", choices=["lite", "standard", "pro"],
                        help="Benchmark tier (lite/standard/pro, default: pro)")
    # LLM Judge (required — scores S1-S3)
    parser.add_argument("--offline-judge", action="store_true",
                        help="Use local DeepSeek model instead of online Claude")
    parser.add_argument("--judge-model-path", default=None,
                        help="Local model path for offline judge")
    parser.add_argument("--judge-vllm-url", default=None,
                        help="URL of running vLLM server for offline judge")
    parser.add_argument("--output-dir", default=None,
                        help="Parent directory for run output (run tag appended automatically)")
    args = parser.parse_args()

    runner = BenchmarkRunner(
        agent_name=args.agent,
        task=args.task,
        tier=args.tier,
        llm_judge=True,
        online_judge=not args.offline_judge,
        judge_model_path=args.judge_model_path,
        judge_vllm_url=args.judge_vllm_url,
        output_dir=args.output_dir,
    )
    runner.run()


if __name__ == "__main__":
    main()
