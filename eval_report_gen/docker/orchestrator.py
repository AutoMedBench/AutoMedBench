#!/usr/bin/env python3
"""Docker orchestrator for eval_report_gen."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import sys

DOCKER_DIR = Path(__file__).resolve().parent
EVAL_DIR = DOCKER_DIR.parent
REPO_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from config_io import load_config
from task_loader import discover_cases, get_task_data_root, load_task_config


CONFIG_PATH = EVAL_DIR / "agent_config.yaml"
AGENT_IMAGE = "eval-report-gen-agent"
EVAL_IMAGE = "eval-report-gen-eval"


def agent_run_cmd(run_dir: str | Path, task_id: str, tier: str, agent_name: str, config_path: str | Path = CONFIG_PATH) -> list[str]:
    run_dir = Path(run_dir)
    cfg = load_config(config_path)
    task = load_task_config(task_id)
    data_root = Path(get_task_data_root(task_id))
    host_public = data_root / "public"
    host_outputs = run_dir / "outputs"
    host_api_keys = REPO_ROOT / "eval_report_gen" / "api_keys"
    host_outputs.mkdir(parents=True, exist_ok=True)

    return [
        "docker", "run", "--rm",
        "--network", "bridge",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=1g",
        "-v", f"{host_public}:/data/public:ro",
        "-v", f"{host_outputs}:/results:rw",
        "-v", f"{host_api_keys}:/app/eval_report_gen/api_keys:ro",
        "-e", f"AGENT={agent_name}",
        "-e", f"TASK={task_id}",
        "-e", f"TIER={tier}",
        "-e", f"OUTPUT_DIR=/results",
        AGENT_IMAGE,
    ]


def eval_run_cmd(run_dir: str | Path, task_id: str, agent_name: str) -> list[str]:
    run_dir = Path(run_dir)
    data_root = Path(get_task_data_root(task_id))
    host_private = data_root / "private"
    host_public = data_root / "public"
    host_agent_outputs = run_dir / "outputs" / "agent_outputs"
    host_results = run_dir

    return [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=512m",
        "-v", f"{host_public}:/data/public:ro",
        "-v", f"{host_private}:/data/private:ro",
        "-v", f"{host_agent_outputs}:/agent_outputs:ro",
        "-v", f"{host_results}:/results:rw",
        "-e", f"TASK={task_id}",
        "-e", f"AGENT_NAME={agent_name}",
        EVAL_IMAGE,
    ]


def build_images(agent_only: bool = False, eval_only: bool = False) -> None:
    if not eval_only:
        subprocess.run(
            ["docker", "build", "-t", AGENT_IMAGE, "-f", str(DOCKER_DIR / "agent" / "Dockerfile.agent"), str(REPO_ROOT)],
            check=True,
        )
    if not agent_only:
        subprocess.run(
            ["docker", "build", "-t", EVAL_IMAGE, "-f", str(DOCKER_DIR / "eval" / "Dockerfile.eval"), str(REPO_ROOT)],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Docker orchestrator for eval_report_gen")
    parser.add_argument("--agent", help="Agent name")
    parser.add_argument("--task", default="mimic-cxr-report-task")
    parser.add_argument("--tier", default="pro", choices=["lite", "standard", "pro"])
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.build_only:
        build_images()
        return
    if not args.agent:
        parser.error("--agent is required unless using --build-only")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = EVAL_DIR / "runs" / args.tier / args.agent / args.task / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_cmd = agent_run_cmd(run_dir, args.task, args.tier, args.agent)
    evaluator_cmd = eval_run_cmd(run_dir, args.task, args.agent)

    if args.dry_run:
        print(json.dumps({"agent_cmd": agent_cmd, "eval_cmd": evaluator_cmd}, indent=2))
        return

    subprocess.run(agent_cmd, check=True)
    subprocess.run(evaluator_cmd, check=True)


if __name__ == "__main__":
    main()
