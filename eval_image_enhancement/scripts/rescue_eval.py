#!/usr/bin/env python3
"""Rescue eval — re-run the eval container on cells where the original eval
was killed by the 1800s timeout bug (detail_report has tier_name=no_report
but workspace/agents_outputs still holds the 100 enhanced.npy files).

The agent_outputs and reference data are intact; only scoring needs to redo.

Usage:
    python rescue_eval.py <run_root> [max_parallel=4]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO       = Path(__file__).resolve().parent.parent.parent
IE_DIR     = REPO / "eval_image_enhancement"
EVAL_IMAGE = "ie-eval:v5"
EVAL_TIMEOUT_S = 5400

TASK_DATA_DIR = {
    "ldct-denoising-task": "LDCT_SimNICT",
    "mri-sr-task":         "MRI_SR_SRMRI",
}


def needs_rescue(run_dir: Path) -> bool:
    rp = run_dir / "detail_report.json"
    if not rp.is_file():
        return False
    try:
        r = json.load(open(rp))
    except Exception:
        return False
    # Skeleton report written when eval didn't finish
    if (r.get("rating") or {}).get("tier_name") == "no_report":
        ao = run_dir / "workspace" / "agents_outputs"
        if ao.is_dir():
            n = sum(1 for _ in ao.rglob("*/enhanced.npy"))
            return n > 0
    return False


def extract_patient_ids(run_dir: Path) -> list[str]:
    ao = run_dir / "workspace" / "agents_outputs"
    pids = sorted([p.name for p in ao.iterdir() if p.is_dir()])
    return pids


def run_one_eval(run_dir: Path, task: str, tier: str, agent_name: str,
                 repeat_idx: int, patient_ids: list[str], gpu_id: int = 0) -> int:
    data_task_dir = REPO / "data" / TASK_DATA_DIR[task]
    private_dir   = data_task_dir / "private"
    public_dir    = data_task_dir / "public"
    agent_outputs = run_dir / "workspace" / "agents_outputs"

    name = f"rescue_eval_{int(time.time())}_{agent_name.replace('-','').replace('.','')}_{task.replace('-','')}_{tier}_r{repeat_idx}"

    cmd = [
        "docker", "run", "--rm",
        "--name", name,
        "--network", "none",
        "--gpus", f"device={gpu_id}",
        "--cpus", "4", "--memory", "8g",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=1g",
        "-v", f"{private_dir}:/data/private:ro",
        "-v", f"{public_dir}:/data/public:ro",
        "-v", f"{agent_outputs}:/agent_outputs:ro",
        "-v", f"{run_dir}:/results:rw",
        "-e", f"AGENT_NAME={agent_name}",
        "-e", f"TASK={task}",
        "-e", f"TIER={tier}",
        "-e", f"REPEAT_IDX={repeat_idx}",
        "-e", f"PATIENT_IDS={','.join(patient_ids)}",
        EVAL_IMAGE,
    ]
    logf = run_dir / "process" / "rescue_eval.log"
    logf.parent.mkdir(exist_ok=True)
    with open(logf, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc


SHORT_TO_AGENT = {
    "opus46":      "claude-opus-4-6",
    "gpt54":       "gpt-5.4",
    "gemini31pro": "gemini-3.1-pro",
    "glm5":        "glm-5",
    "minimax25":   "minimax-m2.5",
    "qwen35":      "qwen3.5-397b",
}


def parse_cell(run_dir: Path) -> tuple[str, str, str, int] | None:
    # run_dir (relative to run_root) = <task>/<tier>/<short>_r<i>
    parts = run_dir.parts
    if len(parts) < 3:
        return None
    dirname = parts[-1]
    tier = parts[-2]
    task = parts[-3]
    if "_r" not in dirname:
        return None
    short, ri = dirname.rsplit("_r", 1)
    try:
        repeat_idx = int(ri)
    except ValueError:
        return None
    agent_name = SHORT_TO_AGENT.get(short, short)
    return task, tier, agent_name, repeat_idx


def main() -> int:
    run_root = Path(sys.argv[1] if len(sys.argv) > 1
                    else str(REPO / "runs" / "matrix_v5_full")).resolve()
    max_parallel = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    # Find all rescue candidates
    candidates = []
    for task_dir in run_root.iterdir():
        if not task_dir.is_dir(): continue
        for tier_dir in task_dir.iterdir():
            if not tier_dir.is_dir(): continue
            for run_dir in tier_dir.iterdir():
                if run_dir.is_dir() and needs_rescue(run_dir):
                    candidates.append(run_dir)

    print(f"Found {len(candidates)} cells needing rescue eval")
    if not candidates:
        return 0

    active: list[tuple[subprocess.Popen, Path]] = []
    idx = 0
    t_start = time.time()

    while idx < len(candidates) or active:
        # Reap finished
        still = []
        for p, rd in active:
            if p.poll() is None:
                still.append((p, rd))
            else:
                dt = time.time() - t_start
                print(f"[{dt:.0f}s done] {rd.relative_to(run_root)} exit={p.returncode}")
        active = still

        # Launch more
        while idx < len(candidates) and len(active) < max_parallel:
            rd = candidates[idx]
            parsed = parse_cell(rd.relative_to(run_root))
            if parsed is None:
                print(f"  skip (parse fail): {rd}")
                idx += 1
                continue
            task, tier, agent_name, repeat_idx = parsed
            pids = extract_patient_ids(rd)
            gpu_id = idx % 8
            print(f"[launch GPU{gpu_id}] {rd.relative_to(run_root)}  patients={len(pids)}")
            p = run_one_eval(rd, task, tier, agent_name, repeat_idx, pids, gpu_id=gpu_id)
            active.append((p, rd))
            idx += 1
            time.sleep(3)

        time.sleep(10)

    print(f"\nDONE rescue in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
