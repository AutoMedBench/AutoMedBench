#!/usr/bin/env python3
"""Load task configuration and skills from per-task folders.

Each task lives in eval_seg/<task-id>/ with:
  config.yaml       — organ, modality, input_filename, time_limit, etc.
  model_info.yaml   — lite/standard/pro model info
  requirements.txt  — for lite tier
  lite_s1.md        — S1 skill for lite
  lite_s2.md        — S2 skill for lite
  lite_s3.md        — S3 skill for lite/standard
  standard_s1.md    — S1 skill for standard
  standard_s3.md    — S3 skill for standard

To add a new task: create a folder, fill the files. No Python changes.
"""

import os
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def discover_tasks() -> dict:
    """Auto-discover all task folders under eval_seg/.

    Returns dict mapping task_id → task folder path.
    A valid task folder must contain config.yaml.
    """
    tasks = {}
    for name in sorted(os.listdir(SCRIPT_DIR)):
        task_dir = os.path.join(SCRIPT_DIR, name)
        if os.path.isdir(task_dir) and os.path.isfile(os.path.join(task_dir, "config.yaml")):
            tasks[name] = task_dir
    return tasks


def load_task_config(task_id: str) -> dict:
    """Load config.yaml for a task. Returns the parsed dict."""
    tasks = discover_tasks()
    if task_id not in tasks:
        # Try legacy names: "kidney" → "kidney-seg-task"
        legacy = f"{task_id}-seg-task"
        if legacy in tasks:
            task_id = legacy
        else:
            available = list(tasks.keys())
            raise ValueError(f"Unknown task '{task_id}'. Available: {available}")

    task_dir = tasks[task_id]
    with open(os.path.join(task_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)

    # Add derived paths
    config["_task_dir"] = task_dir
    config["_task_id"] = task_id
    data_dir_name = config.get("data_dir_name", "")
    config["_data_root"] = os.path.join(PROJECT_DIR, "data", data_dir_name)

    return config


def load_model_info(task_id: str) -> dict:
    """Load model_info.yaml for a task."""
    config = load_task_config(task_id)
    model_info_path = os.path.join(config["_task_dir"], "model_info.yaml")
    with open(model_info_path) as f:
        return yaml.safe_load(f)


def load_skill(task_id: str, filename: str) -> str:
    """Load a skill file (e.g., 'lite_s1.md') as a string.

    Returns empty string if file doesn't exist (Pro tier has no skills).
    """
    config = load_task_config(task_id)
    skill_path = os.path.join(config["_task_dir"], filename)
    if not os.path.isfile(skill_path):
        return ""
    with open(skill_path) as f:
        return f.read()


def load_requirements_path(task_id: str) -> str:
    """Return absolute path to requirements.txt for lite tier.

    Returns empty string if not found.
    """
    config = load_task_config(task_id)
    req_path = os.path.join(config["_task_dir"], "requirements.txt")
    if os.path.isfile(req_path):
        return req_path
    return ""


def get_task_data_root(task_id: str) -> str:
    """Return path to the task's data directory."""
    config = load_task_config(task_id)
    return config["_data_root"]


def discover_patients(task_id: str) -> list:
    """Auto-discover patient IDs from the task's data directory.

    Scans data_root/public/ for subdirectories containing the task's
    input_filename (e.g., ct.nii.gz, t1.nii.gz).
    """
    config = load_task_config(task_id)
    input_filename = config.get("input_filename", "ct.nii.gz")
    public_dir = os.path.join(config["_data_root"], "public")
    if not os.path.isdir(public_dir):
        return []
    patients = sorted([
        d for d in os.listdir(public_dir)
        if os.path.isdir(os.path.join(public_dir, d))
        and os.path.exists(os.path.join(public_dir, d, input_filename))
    ])
    return patients


# ---------------------------------------------------------------
# Convenience: build full task info dict for the runner
# ---------------------------------------------------------------

def load_full_task(task_id: str) -> dict:
    """Load everything needed for a task run.

    Returns a dict with:
      config     — from config.yaml
      model_info — from model_info.yaml
      skills     — dict of {filename: content} for all skill files
      patients   — list of patient IDs
      data_root  — path to data directory
    """
    config = load_task_config(task_id)
    model_info = load_model_info(task_id)
    task_dir = config["_task_dir"]

    # Load all skill files
    skills = {}
    for fname in sorted(os.listdir(task_dir)):
        if fname.endswith(".md"):
            with open(os.path.join(task_dir, fname)) as f:
                skills[fname] = f.read()

    return {
        "config": config,
        "model_info": model_info,
        "skills": skills,
        "patients": discover_patients(task_id),
        "data_root": config["_data_root"],
    }
