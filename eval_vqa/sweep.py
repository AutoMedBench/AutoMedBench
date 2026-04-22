#!/usr/bin/env python3
"""Parallel sweep driver for eval_vqa_v2.

Spawns one subprocess per (agent, repeat) slot in a fixed GPU pool, then
writes ``sweep_plan.json`` / ``sweep_status.json`` / ``sweep_aggregate.json`` /
``SUMMARY.md`` / ``tracker.md`` under ``--sweep-root``.  Inspired by
``vqa_hard/eval_vqa/benchmark_runner.py`` but trimmed to the agent-sweep path
and decoupled from the single-run runner module.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER_PATH = os.path.join(SCRIPT_DIR, "benchmark_runner.py")

POST_CLEANUP_PATTERNS = ["LLaVA-Med*"]


def post_cleanup_run_dir(run_dir: str, patterns: list[str]) -> list[str]:
    """Delete heavy agent-downloaded artifacts under ``<run_dir>/outputs/`` once
    scoring is done. Safe to call only after ``load_run_artifacts`` has read
    everything scoring needs (report.json / run_summary.json live at run root,
    not under outputs/, so they are not affected)."""
    outputs = os.path.join(run_dir, "outputs")
    removed: list[str] = []
    if not os.path.isdir(outputs):
        return removed
    for pat in patterns:
        for p in glob.glob(os.path.join(outputs, pat)):
            try:
                if os.path.islink(p):
                    os.unlink(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
                removed.append(os.path.basename(p))
            except OSError:
                pass
    return removed


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def preflight_shm(parallel_workers: int, path: str = "/dev/shm") -> tuple[bool, str]:
    """Check that /dev/shm has enough free space for parallel LLaVA-Med loads.

    LLaVA-Med 7B mmap'd from multiple workers has triggered Bus error when
    /dev/shm is tight (CLAUDE.md eval_vqa notes). Rule of thumb: ≥ 4 GiB per
    worker, with a floor of 4 GiB. Returns (ok, message).
    """
    if parallel_workers <= 1:
        return True, ""
    try:
        stat = os.statvfs(path)
    except (OSError, AttributeError):
        return True, f"preflight: cannot stat {path}, skipping shm check"
    free_gib = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    required_gib = max(4.0, 4.0 * parallel_workers)
    msg = (
        f"/dev/shm free={free_gib:.1f} GiB, required≈{required_gib:.1f} GiB "
        f"for {parallel_workers} workers"
    )
    return free_gib >= required_gib, msg


def save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def load_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_plan(
    agents: list[str],
    repeats: int,
    task: str,
    tier: str,
    subset: str,
    sample_limit: int | None,
    shared_hf_cache: str | None = None,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for agent in agents:
        for repeat_index in range(1, repeats + 1):
            run_id = f"{agent}-r{repeat_index:02d}"
            runs.append(
                {
                    "run_id": run_id,
                    "agent": agent,
                    "repeat": repeat_index,
                    "task": task,
                    "tier": tier,
                    "subset": subset,
                    "sample_limit": sample_limit,
                    "shared_hf_cache": shared_hf_cache,
                }
            )
    return {
        "generated_at": utc_now_iso(),
        "task": task,
        "tier": tier,
        "subset": subset,
        "repeats": repeats,
        "agents": agents,
        "sample_limit": sample_limit,
        "runs": runs,
    }


def build_command(run: dict[str, Any], run_dir: str) -> list[str]:
    cmd = [
        sys.executable,
        RUNNER_PATH,
        "--agent", run["agent"],
        "--task", run["task"],
        "--tier", run["tier"],
        "--subset", run["subset"],
        "--output-dir", run_dir,
    ]
    if run.get("sample_limit") is not None:
        cmd.extend(["--sample-limit", str(run["sample_limit"])])
    if run.get("shared_hf_cache"):
        cmd.extend(["--shared-hf-cache", run["shared_hf_cache"]])
    return cmd


def load_run_artifacts(run_dir: str) -> dict[str, Any]:
    return {
        "summary": load_json(os.path.join(run_dir, "outputs", "run_summary.json")),
        "report": load_json(os.path.join(run_dir, "report.json")),
    }


def write_tracker(sweep_root: str, status: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Sweep tracker\n")
    lines.append(f"Root: `{sweep_root}`")
    lines.append(f"Task/Tier/Subset: {status.get('task')} / {status.get('tier')} / {status.get('subset')}")
    lines.append(f"Agents: {', '.join(status.get('agent_names') or [])}")
    lines.append(f"Repeats: {status.get('repeats')}")
    lines.append(f"Parallel workers: {status.get('parallel_workers')}")
    lines.append(f"GPU devices: {', '.join(status.get('gpu_devices') or [])}")
    lines.append(f"Started: {status.get('started_at')}  |  Wall time: {status.get('wall_time_s')}s")
    lines.append("")
    lines.append("| Run | Agent | Rep | GPU | Status | Rating | Acc | Mode | PhRate | mCall | Smoke | Wall_s | Tokens | Cost |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for run in status.get("runs", []):
        def _s(v):
            return "-" if v is None else v
        lines.append(
            f"| {run.get('run_id')} | {run.get('agent')} | {run.get('repeat')} "
            f"| {_s(run.get('gpu_device'))} | {_s(run.get('status'))} "
            f"| {_s(run.get('rating'))} | {_s(run.get('accuracy'))} "
            f"| {_s(run.get('inference_mode'))} | {_s(run.get('placeholder_rate'))} "
            f"| {_s(run.get('model_call_detected'))} | {_s(run.get('smoke_forward_passed'))} "
            f"| {_s(run.get('wall_time_s'))} | {_s(run.get('total_tokens'))} "
            f"| {_s(run.get('estimated_cost_usd'))} |"
        )
    with open(os.path.join(sweep_root, "tracker.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def aggregate_runs(sweep_root: str, status: dict[str, Any]) -> dict[str, Any]:
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ratings: dict[str, int] = defaultdict(int)
    for run in status.get("runs", []):
        by_agent[run["agent"]].append(run)
        if run.get("rating"):
            ratings[run["rating"]] += 1

    agent_rows: list[dict[str, Any]] = []
    for agent, runs in by_agent.items():
        accs = [r["accuracy"] for r in runs if isinstance(r.get("accuracy"), (int, float))]
        completed = [r for r in runs if r.get("status") not in (None, "running", "missing_summary")]
        modes: dict[str, int] = defaultdict(int)
        for r in runs:
            mode = r.get("inference_mode")
            if mode:
                modes[mode] += 1
        agent_rows.append(
            {
                "agent": agent,
                "runs": len(runs),
                "completed": len(completed),
                "mean_accuracy": round(sum(accs) / len(accs), 4) if accs else None,
                "ratings": {r["run_id"]: r.get("rating") for r in runs},
                "inference_modes": dict(modes),
            }
        )

    return {
        "task": status.get("task"),
        "tier": status.get("tier"),
        "subset": status.get("subset"),
        "repeats": status.get("repeats"),
        "agent_names": status.get("agent_names"),
        "rating_counts": dict(ratings),
        "agents": agent_rows,
        "runs": list(status.get("runs", [])),
        "wall_time_s": status.get("wall_time_s"),
    }


def render_summary_md(aggregate: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Sweep SUMMARY",
        "",
        f"- Task: {aggregate.get('task')}",
        f"- Tier: {aggregate.get('tier')}",
        f"- Subset: {aggregate.get('subset')}",
        f"- Repeats: {aggregate.get('repeats')}",
        f"- Wall time: {aggregate.get('wall_time_s')}s",
        "",
        "## Ratings",
    ]
    for rating, count in sorted(aggregate.get("rating_counts", {}).items()):
        lines.append(f"- {rating}: {count}")
    lines.extend(
        [
            "",
            "## Per-agent",
            "",
            "| Agent | Runs | Completed | Mean accuracy | Modes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in aggregate.get("agents", []):
        modes = row.get("inference_modes") or {}
        modes_str = ", ".join(f"{k}:{v}" for k, v in sorted(modes.items())) or "-"
        lines.append(
            f"| {row['agent']} | {row['runs']} | {row['completed']} | "
            f"{row['mean_accuracy'] if row['mean_accuracy'] is not None else '-'} | {modes_str} |"
        )

    lines.extend(
        [
            "",
            "## Per-run",
            "",
            "| Run | Agent | Rep | Rating | Acc | Completion | Mode | PhRate | mCall | Smoke | Wall_s | Tokens | Cost | Failure |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in aggregate.get("runs", []) or []:
        def _s(v):
            return "-" if v is None else v
        lines.append(
            f"| {row.get('run_id')} | {row.get('agent')} | {row.get('repeat')} "
            f"| {_s(row.get('rating'))} | {_s(row.get('accuracy'))} "
            f"| {_s(row.get('completion_rate'))} | {_s(row.get('inference_mode'))} "
            f"| {_s(row.get('placeholder_rate'))} | {_s(row.get('model_call_detected'))} "
            f"| {_s(row.get('smoke_forward_passed'))} | {_s(row.get('wall_time_s'))} "
            f"| {_s(row.get('total_tokens'))} | {_s(row.get('estimated_cost_usd'))} "
            f"| {_s(row.get('primary_failure'))} |"
        )
    return "\n".join(lines) + "\n"


def execute(args: argparse.Namespace) -> int:
    agents = parse_csv(args.agents)
    gpu_devices = parse_csv(args.gpu_devices)
    if not agents:
        print("ERROR: --agents required (comma-separated)", file=sys.stderr)
        return 2
    if args.parallel_workers > 1 and args.parallel_workers > len(gpu_devices):
        print(
            f"ERROR: --parallel-workers={args.parallel_workers} but only "
            f"{len(gpu_devices)} --gpu-devices provided",
            file=sys.stderr,
        )
        return 2

    shm_ok, shm_msg = preflight_shm(args.parallel_workers)
    if shm_msg:
        print(f"[sweep] shm preflight: {shm_msg}", file=sys.stderr)
    if not shm_ok:
        if args.allow_low_shm:
            print(
                "[sweep] WARNING: low /dev/shm detected; continuing because "
                "--allow-low-shm was set. Expect possible Bus errors during "
                "LLaVA-Med load.",
                file=sys.stderr,
            )
        else:
            print(
                "ERROR: insufficient /dev/shm for parallel workers. Mitigations:\n"
                "  (a) reduce --parallel-workers,\n"
                "  (b) run under docker with --shm-size=8g,\n"
                "  (c) override with --allow-low-shm if you know what you're doing.",
                file=sys.stderr,
            )
            return 2

    sweep_root = os.path.abspath(args.sweep_root)
    os.makedirs(sweep_root, exist_ok=True)

    plan = build_plan(
        agents=agents,
        repeats=args.repeats,
        task=args.task,
        tier=args.tier,
        subset=args.subset,
        sample_limit=args.sample_limit,
        shared_hf_cache=args.shared_hf_cache,
    )
    save_json(os.path.join(sweep_root, "sweep_plan.json"), plan)

    status: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "started_at": utc_now_iso(),
        "run_root": sweep_root,
        "task": args.task,
        "tier": args.tier,
        "subset": args.subset,
        "repeats": args.repeats,
        "sample_limit": args.sample_limit,
        "agent_names": agents,
        "parallel_workers": args.parallel_workers,
        "gpu_devices": gpu_devices,
        "wall_time_s": 0.0,
        "runs": [],
    }
    status_path = os.path.join(sweep_root, "sweep_status.json")
    sweep_started = time.perf_counter()

    pending = list(plan["runs"])
    running: list[dict[str, Any]] = []
    free_gpus = list(gpu_devices)

    while pending or running:
        while pending and len(running) < args.parallel_workers:
            planned = pending.pop(0)
            run_dir = os.path.join(sweep_root, planned["run_id"])
            os.makedirs(run_dir, exist_ok=True)

            run_status = dict(planned)
            run_status["run_dir"] = run_dir
            run_status["started_at"] = utc_now_iso()
            run_status["status"] = "running"
            gpu_device = free_gpus.pop(0) if free_gpus else None
            if gpu_device is not None:
                run_status["gpu_device"] = gpu_device

            command = build_command(planned, run_dir)
            run_status["command"] = command
            run_env = os.environ.copy()
            if gpu_device is not None:
                run_env["CUDA_VISIBLE_DEVICES"] = gpu_device

            log_path = os.path.join(run_dir, "sweep_worker.log")
            log_handle = open(log_path, "w", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=SCRIPT_DIR, env=run_env, stdout=log_handle, stderr=subprocess.STDOUT
            )
            running.append(
                {
                    "planned": planned,
                    "run_dir": run_dir,
                    "run_status": run_status,
                    "started_wall": time.perf_counter(),
                    "process": process,
                    "gpu_device": gpu_device,
                    "log_handle": log_handle,
                }
            )
            status["runs"].append(run_status)
            status["wall_time_s"] = round(time.perf_counter() - sweep_started, 4)
            save_json(status_path, status)
            write_tracker(sweep_root, status)

        completed_idx: list[int] = []
        for idx, active in enumerate(running):
            rc = active["process"].poll()
            if rc is None:
                continue
            run_status = active["run_status"]
            run_status["returncode"] = rc
            run_status["ended_at"] = utc_now_iso()
            run_status["wall_time_s"] = round(time.perf_counter() - active["started_wall"], 4)

            artifacts = load_run_artifacts(active["run_dir"])
            summary = artifacts["summary"]
            report = artifacts["report"].get("eval_report") or artifacts["report"]
            if summary:
                run_status["status"] = summary.get("status", "unknown")
                run_status["completed_outputs"] = summary.get("completed_outputs", 0)
            else:
                run_status["status"] = "missing_summary" if rc == 0 else "error"
            if report:
                m = report.get("metrics", {}) or {}
                agg = report.get("aggregate", {}) or {}
                ss = report.get("step_scores", {}) or {}
                run_status["accuracy"] = m.get("accuracy")
                run_status["completion_rate"] = m.get("completion_rate")
                run_status["placeholder_rate"] = m.get("placeholder_rate")
                run_status["inference_mode"] = m.get("inference_mode")
                run_status["model_call_detected"] = m.get("model_call_detected")
                run_status["smoke_forward_passed"] = m.get("smoke_forward_passed")
                run_status["rating"] = agg.get("rating")
                run_status["resolved"] = agg.get("resolved")
                run_status["step_scores"] = ss
                failure = report.get("failure") or {}
                run_status["primary_failure"] = failure.get("primary_failure")
                runtime = (artifacts["summary"] or {}).get("runtime", {}) or {}
                run_status["total_tokens"] = runtime.get("total_tokens")
                run_status["estimated_cost_usd"] = runtime.get("estimated_cost_usd")

            if active["gpu_device"] is not None:
                free_gpus.append(active["gpu_device"])
            active["log_handle"].close()
            removed = post_cleanup_run_dir(active["run_dir"], POST_CLEANUP_PATTERNS)
            if removed:
                run_status["post_cleanup_removed"] = removed
            completed_idx.append(idx)

        for idx in reversed(completed_idx):
            del running[idx]

        status["wall_time_s"] = round(time.perf_counter() - sweep_started, 4)
        save_json(status_path, status)
        write_tracker(sweep_root, status)

        if running and not completed_idx:
            time.sleep(1.0)

    status["ended_at"] = utc_now_iso()
    status["wall_time_s"] = round(time.perf_counter() - sweep_started, 4)
    save_json(status_path, status)

    aggregate = aggregate_runs(sweep_root, status)
    aggregate["ended_at"] = status["ended_at"]
    save_json(os.path.join(sweep_root, "sweep_aggregate.json"), aggregate)
    with open(os.path.join(sweep_root, "SUMMARY.md"), "w", encoding="utf-8") as handle:
        handle.write(render_summary_md(aggregate))
    write_tracker(sweep_root, status)
    print(f"Sweep done. Root: {sweep_root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="eval_vqa_v2 parallel agent sweep")
    parser.add_argument("--task", required=True)
    parser.add_argument("--tier", choices=("lite", "standard"), default="lite")
    parser.add_argument("--subset", default="all")
    parser.add_argument("--agents", required=True, help="Comma-separated agent names")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--gpu-devices", default="", help="Comma-separated CUDA device IDs")
    parser.add_argument(
        "--allow-low-shm",
        action="store_true",
        help="Bypass /dev/shm preflight (risk of Bus error during LLaVA-Med load).",
    )
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--shared-hf-cache", default=None,
                        help="Shared HF cache dir reused across all runs to avoid redownloads.")
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
