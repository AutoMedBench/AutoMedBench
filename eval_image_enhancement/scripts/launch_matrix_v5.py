#!/usr/bin/env python3
"""v5 matrix launcher — calls orchestrator.py for each (agent, task, tier, repeat).

Uses Popen polling + per-GPU pool so kills don't orphan jobs. Only skips runs
that already have detail_report.json.

Usage:
    python launch_matrix_v5.py <run_root> [n_repeats=3] [max_parallel=6] [n_patients=100]

When n_patients != 100, per-run wall caps are scaled down proportionally
(but never below a floor of 600s lite / 900s standard).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ORCHESTRATOR = REPO / "eval_image_enhancement" / "docker" / "orchestrator.py"

AGENTS = ["claude-opus-4-6", "gpt-5.4", "gemini-3.1-pro",
          "glm-5", "minimax-m2.5", "qwen3.5-397b"]
SHORT = {
    "claude-opus-4-6": "opus46",
    "gpt-5.4":         "gpt54",
    "gemini-3.1-pro":  "gemini31pro",
    "glm-5":           "glm5",
    "minimax-m2.5":    "minimax25",
    "qwen3.5-397b":    "qwen35",
}
TASKS = {
    "ldct-denoising-task": {"n": 100, "cap_lite": 2700, "cap_std": 3600},
    "mri-sr-task":         {"n": 100, "cap_lite": 2400, "cap_std": 3000},
}
TIERS = ["lite", "standard"]


def build_jobs(run_root: Path, n_repeats: int,
               n_patients: int | None = None) -> list[dict]:
    jobs = []
    for task, tc in TASKS.items():
        full_n = tc["n"]
        actual_n = n_patients if n_patients is not None else full_n
        # Scale wall cap linearly with N, but floor at 600s (lite) / 900s (std)
        scale = actual_n / full_n if full_n else 1.0
        cap_lite = max(600, int(tc["cap_lite"] * scale))
        cap_std  = max(900, int(tc["cap_std"]  * scale))
        for tier in TIERS:
            cap = cap_lite if tier == "lite" else cap_std
            for agent in AGENTS:
                short = SHORT[agent]
                for r in range(n_repeats):
                    outd = run_root / task / tier / f"{short}_r{r}"
                    if (outd / "detail_report.json").is_file():
                        continue
                    jobs.append({
                        "task":    task,
                        "tier":    tier,
                        "agent":   agent,
                        "short":   short,
                        "repeat":  r,
                        "outdir":  outd,
                        "logf":    outd.parent / f"{short}_r{r}.log",
                        "cap":     cap,
                        "n":       actual_n,
                    })
    return jobs


def launch_one(job: dict, gpu_id: int) -> subprocess.Popen:
    job["outdir"].parent.mkdir(parents=True, exist_ok=True)
    logf = open(job["logf"], "w")
    cmd = [
        sys.executable, "-u", str(ORCHESTRATOR),
        "--agent",       job["agent"],
        "--task",        job["task"],
        "--tier",        job["tier"],
        "--n-patients",  str(job["n"]),
        "--max-turns",   "150",
        "--max-seconds", str(job["cap"]),
        "--gpu-id",      str(gpu_id),
        "--repeat-idx",  str(job["repeat"]),
        "--output-dir",  str(job["outdir"]),
    ]
    env = os.environ.copy()
    print(f"[launch GPU{gpu_id}] {job['task']}/{job['tier']}/"
          f"{job['short']}_r{job['repeat']}  cap={job['cap']}s")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)


def main() -> int:
    run_root = Path(sys.argv[1] if len(sys.argv) > 1
                    else str(REPO / "runs" / "matrix_v5")).resolve()
    n_repeats    = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_parallel = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    n_patients   = int(sys.argv[4]) if len(sys.argv) > 4 else None

    jobs = build_jobs(run_root, n_repeats, n_patients)
    total = len(jobs)
    print(f"run_root={run_root}")
    print(f"Total pending jobs: {total} (max_parallel={max_parallel}, "
          f"repeats={n_repeats}, n_patients={n_patients or 'full'})")

    if not jobs:
        print("Nothing to launch — all reports already present.")
        return 0

    active: list[tuple[subprocess.Popen, dict, int]] = []
    gpu_pool = list(range(8))
    t_start = time.time()

    while jobs or active:
        still = []
        for p, j, g in active:
            if p.poll() is None:
                still.append((p, j, g))
            else:
                dt = time.time() - t_start
                print(f"[{dt:.0f}s done] {j['task']}/{j['tier']}/"
                      f"{j['short']}_r{j['repeat']}  "
                      f"exit={p.returncode}  GPU{g} freed")
                gpu_pool.append(g)
        active = still

        while jobs and len(active) < max_parallel and gpu_pool:
            j = jobs.pop(0)
            g = gpu_pool.pop(0)
            p = launch_one(j, g)
            active.append((p, j, g))
            time.sleep(2)

        if active and not jobs:
            print(f"[{time.time() - t_start:.0f}s] all launched, "
                  f"{len(active)} active")
        time.sleep(15)

    print(f"\nALL DONE in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
