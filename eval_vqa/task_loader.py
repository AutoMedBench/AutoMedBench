#!/usr/bin/env python3
"""Load VQA task configuration and staged question metadata."""

from __future__ import annotations

import json
import os
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TASKS_DIR = os.path.join(SCRIPT_DIR, "tasks")


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_yaml_file(path: str) -> dict[str, Any]:
    return _load_yaml(path)


def discover_tasks() -> dict[str, str]:
    tasks: dict[str, str] = {}
    if not os.path.isdir(TASKS_DIR):
        return tasks
    for name in sorted(os.listdir(TASKS_DIR)):
        task_dir = os.path.join(TASKS_DIR, name)
        if os.path.isdir(task_dir) and os.path.isfile(os.path.join(task_dir, "config.yaml")):
            tasks[name] = task_dir
    return tasks


def _resolve_task_id(task_id: str) -> str:
    tasks = discover_tasks()
    if task_id in tasks:
        return task_id
    available = sorted(tasks)
    raise ValueError(f"Unknown task '{task_id}'. Available: {available}")


def load_task_config(task_id: str) -> dict[str, Any]:
    task_id = _resolve_task_id(task_id)
    task_dir = discover_tasks()[task_id]
    config = _load_yaml(os.path.join(task_dir, "config.yaml"))
    config["_task_dir"] = task_dir
    config["_task_id"] = task_id
    config["_data_root"] = os.path.join(PROJECT_DIR, "data", config.get("data_dir_name", ""))
    return config


def load_model_info(task_id: str) -> dict[str, Any]:
    config = load_task_config(task_id)
    return _load_yaml(os.path.join(config["_task_dir"], "model_info.yaml"))


def load_standard_candidate_specs(task_id: str) -> list[dict[str, Any]]:
    model_info = load_model_info(task_id)
    standard = model_info.get("standard", {})
    return [dict(item) for item in standard.get("candidate_models", []) if item.get("model_name")]


def render_standard_candidate_summary(task_id: str) -> str:
    lines: list[str] = []
    for candidate in load_standard_candidate_specs(task_id):
        model_name = str(candidate.get("model_name"))
        family = str(candidate.get("family") or "unknown family")
        accessibility = str(candidate.get("accessibility") or ("gated" if candidate.get("gated") else "open"))
        priority = candidate.get("selection_priority")
        prefix = f"{priority}. " if priority is not None else "- "
        lines.append(f"{prefix}`{model_name}`")
        lines.append(f"  family: {family}")
        lines.append(f"  access: {accessibility}")
        if candidate.get("notes"):
            lines.append(f"  notes: {candidate['notes']}")
    return "\n".join(lines)


def load_skill(task_id: str, filename: str) -> str:
    config = load_task_config(task_id)
    path = os.path.join(config["_task_dir"], filename)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def load_requirements_path(task_id: str) -> str:
    config = load_task_config(task_id)
    path = os.path.join(config["_task_dir"], "requirements.txt")
    return path if os.path.isfile(path) else ""


def get_task_data_root(task_id: str) -> str:
    return load_task_config(task_id)["_data_root"]


def load_question(public_dir: str, question_id: str) -> dict[str, Any]:
    path = os.path.join(public_dir, question_id, "question.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_question_ids(task_id: str, split: str | None = None) -> list[str]:
    config = load_task_config(task_id)
    public_dir = os.path.join(config["_data_root"], "public")
    if not os.path.isdir(public_dir):
        return []

    question_ids: list[str] = []
    for name in sorted(os.listdir(public_dir)):
        sample_dir = os.path.join(public_dir, name)
        question_path = os.path.join(sample_dir, config.get("input_filename", "question.json"))
        if not os.path.isdir(sample_dir) or not os.path.isfile(question_path):
            continue
        if split:
            try:
                question = load_question(public_dir, name)
            except json.JSONDecodeError:
                continue
            if question.get("split") != split:
                continue
        question_ids.append(name)
    return question_ids


def load_subset_ids(task_id: str, subset_name: str) -> list[str]:
    config = load_task_config(task_id)
    subset_path = os.path.join(config["_data_root"], f"{subset_name}_ids.txt")
    if not os.path.isfile(subset_path):
        return []
    with open(subset_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_full_task(task_id: str) -> dict[str, Any]:
    config = load_task_config(task_id)
    task_dir = config["_task_dir"]
    skills: dict[str, str] = {}
    for filename in sorted(os.listdir(task_dir)):
        if filename.endswith(".md"):
            with open(os.path.join(task_dir, filename), "r", encoding="utf-8") as handle:
                skills[filename] = handle.read()
    return {
        "config": config,
        "model_info": load_model_info(task_id),
        "skills": skills,
        "question_ids": discover_question_ids(task_id),
        "data_root": config["_data_root"],
    }


def resolve_agent_config_path() -> str:
    local_path = os.path.join(SCRIPT_DIR, "agent_config.yaml")
    if not os.path.isfile(local_path):
        fallback_path = os.path.join(PROJECT_DIR, "eval_seg", "agent_config.yaml")
        if os.path.isfile(fallback_path):
            return fallback_path
        raise FileNotFoundError("No VQA or shared agent_config.yaml file was found.")

    config = _load_yaml(local_path)
    shared_config = config.get("shared_config")
    if not shared_config:
        return local_path

    resolved = os.path.normpath(os.path.join(SCRIPT_DIR, shared_config))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Shared agent config does not exist: {resolved}")
    return resolved
