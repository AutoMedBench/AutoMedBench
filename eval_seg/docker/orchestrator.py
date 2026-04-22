#!/usr/bin/env python3
"""MedAgentsBench Docker Orchestrator.

Sequences isolated agent and evaluation containers:
  1. Pre-render system prompt with container paths
  2. Launch agent container (GPU, network)
  3. Wait for agent to finish (or timeout)
  4. Launch eval container (CPU-only, no network)
  5. Collect detail_report.json

Usage:
    python orchestrator.py --agent claude-opus-4-6 --task kidney --tier lite
    python orchestrator.py --agent claude-opus-4-6 --task kidney --tier pro --judge online
    python orchestrator.py --build-only   # just build images
    python orchestrator.py --dry-run ...  # print docker commands without running
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Resolve paths relative to eval_seg/
DOCKER_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SEG_DIR = os.path.dirname(DOCKER_DIR)

sys.path.insert(0, EVAL_SEG_DIR)
from tier_config import get_tier_config, get_task_model_info

CONFIG_PATH = os.path.join(EVAL_SEG_DIR, "agent_config.yaml")

# Container image names
AGENT_IMAGE = "medagentsbench-agent"
EVAL_IMAGE = "medagentsbench-eval"


PROJECT_DIR = os.path.dirname(EVAL_SEG_DIR)  # MedAgentsBench/

# Per-task data roots — each task has its own dataset directory.
TASK_DATA = {
    "kidney": os.path.join(PROJECT_DIR, "data", "CruzAbdomen_Kidney"),
    "liver": os.path.join(PROJECT_DIR, "data", "CruzAbdomen_Liver"),
    "pancreas": os.path.join(PROJECT_DIR, "data", "CruzAbdomen_Pancreas"),
}


def discover_patients(task: str) -> list:
    """Find patient IDs from the task's public data directory."""
    data_root = TASK_DATA[task]
    public_dir = os.path.join(data_root, "public")
    if not os.path.isdir(public_dir):
        sys.exit(f"Data directory not found: {public_dir}")
    patients = sorted([
        d for d in os.listdir(public_dir)
        if os.path.isdir(os.path.join(public_dir, d))
        and not d.startswith(".")
    ])
    if not patients:
        sys.exit(f"No patients found in {public_dir}")
    return patients


def build_images(agent_only=False, eval_only=False):
    """Build Docker images."""
    if not eval_only:
        print(f"[Orchestrator] Building {AGENT_IMAGE}...")
        agent_ctx = os.path.join(DOCKER_DIR, "agent")
        subprocess.run([
            "docker", "build",
            "-t", AGENT_IMAGE,
            "-f", os.path.join(agent_ctx, "Dockerfile.agent"),
            agent_ctx,
        ], check=True)
        print(f"[Orchestrator] {AGENT_IMAGE} built.")

    if not agent_only:
        print(f"[Orchestrator] Building {EVAL_IMAGE}...")
        # Eval Dockerfile copies from eval_seg/ root
        subprocess.run([
            "docker", "build",
            "-t", EVAL_IMAGE,
            "-f", os.path.join(DOCKER_DIR, "eval", "Dockerfile.eval"),
            EVAL_SEG_DIR,
        ], check=True)
        print(f"[Orchestrator] {EVAL_IMAGE} built.")


def render_system_prompt(tier_config, task: str, model_info: dict) -> str:
    """Render the system prompt with container-internal paths."""
    # Import the prompt builder from the host codebase
    sys.path.insert(0, EVAL_SEG_DIR)
    from benchmark_runner import build_tier_system_prompt, TASK_CONFIG

    task_cfg = TASK_CONFIG[task]

    # Use container-internal paths
    return build_tier_system_prompt(
        tier_config, task_cfg, model_info,
        data_dir="/data/public",
        output_dir="/workspace",
    )


def run_agent_container(run_dir: str, task: str, tier: str,
                        agent_cfg: dict, agent_name: str,
                        patients: list, gpu_id: int = 0,
                        timeout: int = 3600, dry_run: bool = False):
    """Launch the agent container."""
    data_root = TASK_DATA[task]
    host_data_public = os.path.join(data_root, "public")
    host_outputs = os.path.join(run_dir, "outputs")

    patient_ids = ",".join(patients)
    model = agent_cfg["model"]
    api_key = agent_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    reasoning = str(agent_cfg.get("reasoning", True)).lower()

    cmd = [
        "docker", "run", "--rm",
        f"--gpus", f'"device={gpu_id}"',
        "--network", "bridge",
        "--memory", "64g",
        "--shm-size", "16g",
        "--pids-limit", "4096",
        "--security-opt", "no-new-privileges",
        # Read-only root + writable tmp
        "--read-only",
        "--tmpfs", "/tmp:rw,size=10g",
        # Mounts
        "-v", f"{host_data_public}:/data/public:ro",
        "-v", f"{host_outputs}:/workspace:rw",
        # Environment
        "-e", f"AGENT_NAME={agent_name}",
        "-e", f"MODEL={model}",
        "-e", f"API_KEY={api_key}",
        "-e", f"TASK={task}",
        "-e", f"TIER={tier}",
        "-e", f"PATIENT_IDS={patient_ids}",
        "-e", f"REASONING={reasoning}",
        # Image
        AGENT_IMAGE,
    ]

    if dry_run:
        print("[DRY RUN] Agent container command:")
        print("  " + " \\\n    ".join(cmd))
        return 0

    print(f"[Orchestrator] Launching agent container (timeout={timeout}s)...")
    try:
        proc = subprocess.run(cmd, timeout=timeout)
        return proc.returncode
    except subprocess.TimeoutExpired:
        print(f"[Orchestrator] Agent container timed out after {timeout}s")
        # Force kill
        subprocess.run(["docker", "kill", AGENT_IMAGE], capture_output=True)
        return -1


def run_eval_container(run_dir: str, task: str, agent_name: str,
                       patients: list, dry_run: bool = False):
    """Launch the eval container."""
    data_root = TASK_DATA[task]
    host_data_private = os.path.join(data_root, "private")
    host_data_public = os.path.join(data_root, "public")
    host_agent_outputs = os.path.join(run_dir, "outputs", "agents_outputs")

    patient_ids = ",".join(patients)

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=1g",
        "--memory", "8g",
        "--security-opt", "no-new-privileges",
        # Mounts
        "-v", f"{host_data_private}:/data/private:ro",
        "-v", f"{host_data_public}:/data/public:ro",
        "-v", f"{host_agent_outputs}:/agent_outputs:ro",
        "-v", f"{run_dir}:/results:rw",
        # Environment
        "-e", f"PATIENT_IDS={patient_ids}",
        "-e", f"TASK={task}",
        "-e", f"AGENT_NAME={agent_name}",
        # Image
        EVAL_IMAGE,
    ]

    if dry_run:
        print("[DRY RUN] Eval container command:")
        print("  " + " \\\n    ".join(cmd))
        return 0

    print(f"[Orchestrator] Launching eval container (no network, CPU-only)...")
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="MedAgentsBench Docker Orchestrator")
    parser.add_argument("--agent", help="Agent name from agent_config.yaml")
    parser.add_argument("--task", choices=list(TASK_DATA.keys()),
                        help="Task: kidney, liver, pancreas")
    parser.add_argument("--tier", default="pro", choices=["lite", "standard", "pro"])
    parser.add_argument("--judge", default="none", choices=["none", "online", "offline"])
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--timeout", type=int, default=3600, help="Wall-clock timeout (s)")
    parser.add_argument("--config", default=CONFIG_PATH, help="Agent config YAML")
    parser.add_argument("--build-only", action="store_true", help="Only build images")
    parser.add_argument("--dry-run", action="store_true", help="Print commands, don't run")
    args = parser.parse_args()

    # Build-only mode
    if args.build_only:
        build_images()
        return

    if not args.agent or not args.task:
        parser.error("--agent and --task are required")

    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.agent not in config["agents"]:
        sys.exit(f"Unknown agent '{args.agent}'. "
                 f"Available: {list(config['agents'].keys())}")
    agent_cfg = config["agents"][args.agent]

    # Discover patients
    patients = discover_patients(args.task)
    tier_config = get_tier_config(args.tier)
    model_info = get_task_model_info(args.task)

    # Create run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        EVAL_SEG_DIR, "runs", args.tier, args.agent, args.task, timestamp,
    )
    outputs_dir = os.path.join(run_dir, "outputs")
    os.makedirs(os.path.join(outputs_dir, "agents_outputs"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "plan"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "process"), exist_ok=True)

    # Render system prompt with container paths
    system_prompt = render_system_prompt(tier_config, args.task, model_info)
    prompt_path = os.path.join(outputs_dir, "tier_prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(system_prompt)

    print(f"\n{'='*60}")
    print(f"  MedAgentsBench Docker Orchestrator")
    print(f"  Agent: {args.agent}  Task: {args.task}  Tier: {args.tier}")
    print(f"  Patients: {len(patients)}")
    print(f"  Run dir: {run_dir}")
    print(f"  GPU: {args.gpu_id}  Timeout: {args.timeout}s")
    print(f"{'='*60}\n")

    # --- Phase 1: Agent container ---
    agent_rc = run_agent_container(
        run_dir=run_dir,
        task=args.task,
        tier=args.tier,
        agent_cfg=agent_cfg,
        agent_name=args.agent,
        patients=patients,
        gpu_id=args.gpu_id,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )

    isolation_breach = (agent_rc == 99)
    if isolation_breach and not args.dry_run:
        print(f"[Orchestrator] *** ISOLATION BREACH *** Agent killed (exit 99).")
        print(f"[Orchestrator] All remaining step scores → 0, clinical → 0, rating → F.")
    elif agent_rc != 0 and not args.dry_run:
        print(f"[Orchestrator] Agent container exited with code {agent_rc}")

    # Move process logs from outputs/process/ to run_dir/process/
    src_process = os.path.join(outputs_dir, "process")
    dst_process = os.path.join(run_dir, "process")
    if os.path.isdir(src_process):
        for fname in os.listdir(src_process):
            src = os.path.join(src_process, fname)
            dst = os.path.join(dst_process, fname)
            if os.path.isfile(src):
                os.rename(src, dst)

    # --- Phase 2: Eval container ---
    eval_rc = run_eval_container(
        run_dir=run_dir,
        task=args.task,
        agent_name=args.agent,
        patients=patients,
        dry_run=args.dry_run,
    )

    if eval_rc != 0 and not args.dry_run:
        print(f"[Orchestrator] Eval container exited with code {eval_rc}")

    # --- Results ---
    report_path = os.path.join(run_dir, "detail_report.json")
    if os.path.isfile(report_path):
        with open(report_path) as f:
            report = json.load(f)

        # Apply isolation breach penalty if the agent was killed
        if isolation_breach:
            agg = report.get("aggregate", {})
            steps = report.get("step_scores", report.get("agentic_score", {}).get("step_scores", {}))
            for s in ("s1", "s2", "s3", "s4", "s5"):
                if s in steps:
                    steps[s] = 0.0
            agg["agentic_score"] = 0.0
            agg["clinical_score"] = 0.0
            agg["overall_score"] = 0.0
            agg["rating"] = "F"
            agg["resolved"] = False
            report["isolation_breach"] = {
                "breached": True,
                "penalty": "All step scores zeroed; clinical = 0; rating = F (Disqualified)",
            }
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

        agg = report.get("aggregate", {})
        print(f"\n{'='*60}")
        print(f"  RESULTS")
        if isolation_breach:
            print(f"  *** DISQUALIFIED — ISOLATION BREACH ***")
        print(f"  Rating:   {agg.get('rating', '?')}")
        print(f"  Resolved: {agg.get('resolved', False)}")
        print(f"  Overall:  {agg.get('overall_score', 0):.4f}")
        print(f"  Agentic:  {agg.get('agentic_score', 0):.4f}")
        print(f"  Clinical: {agg.get('clinical_score', 0):.4f}")
        print(f"  Report:   {report_path}")
        print(f"{'='*60}")
    elif not args.dry_run:
        print(f"[Orchestrator] WARNING: No detail_report.json found at {report_path}")

    # --- Phase 3: Judge (optional) ---
    if args.judge != "none" and not args.dry_run:
        print(f"[Orchestrator] LLM Judge ({args.judge}) — not yet implemented in Docker mode.")
        # TODO: launch judge container


if __name__ == "__main__":
    main()
