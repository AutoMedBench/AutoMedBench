#!/usr/bin/env python3
"""Task discovery and loading for eval_report_gen."""

from __future__ import annotations

from pathlib import Path

try:
    from .config_io import load_config
except ImportError:
    from config_io import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def discover_tasks() -> dict[str, Path]:
    tasks = {}
    for child in sorted(SCRIPT_DIR.iterdir()):
        if child.is_dir() and (child / "config.yaml").is_file():
            tasks[child.name] = child
    return tasks


def _resolve_task_id(task_id: str) -> str:
    tasks = discover_tasks()
    if task_id in tasks:
        return task_id
    raise ValueError(f"Unknown task '{task_id}'. Available: {sorted(tasks)}")


def load_task_config(task_id: str) -> dict:
    task_id = _resolve_task_id(task_id)
    config = load_config(SCRIPT_DIR / task_id / "config.yaml")
    config["_task_dir"] = str(SCRIPT_DIR / task_id)
    config["_task_id"] = task_id
    config["_data_root"] = str(REPO_ROOT / "data" / config["data_dir_name"] / config["staged_split_name"])
    return config


def load_model_info(task_id: str) -> dict:
    task = load_task_config(task_id)
    return load_config(Path(task["_task_dir"]) / "model_info.yaml")


def load_skill(task_id: str, filename: str) -> str:
    task = load_task_config(task_id)
    skill_path = Path(task["_task_dir"]) / filename
    return skill_path.read_text(encoding="utf-8") if skill_path.is_file() else ""


def load_requirements_path(task_id: str) -> str:
    task = load_task_config(task_id)
    req_path = Path(task["_task_dir"]) / "requirements.txt"
    return str(req_path) if req_path.is_file() else ""


def get_task_data_root(task_id: str) -> str:
    return load_task_config(task_id)["_data_root"]


def discover_cases(task_id: str) -> list[str]:
    task = load_task_config(task_id)
    public_root = Path(task["_data_root"]) / "public"
    if not public_root.is_dir():
        return []
    return sorted([child.name for child in public_root.iterdir() if child.is_dir()])


def load_full_task(task_id: str) -> dict:
    task = load_task_config(task_id)
    task_dir = Path(task["_task_dir"])
    skills = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(task_dir.glob("*.md"))
    }
    return {
        "config": task,
        "model_info": load_model_info(task_id),
        "skills": skills,
        "cases": discover_cases(task_id),
        "data_root": task["_data_root"],
    }
