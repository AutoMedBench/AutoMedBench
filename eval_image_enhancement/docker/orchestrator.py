#!/usr/bin/env python3
"""v5 Docker Orchestrator — chains agent + eval containers.

Sequences isolated containers for one cell (agent × task × tier × repeat):
  1. Render system prompt with container-internal paths
  2. Launch agent container (GPU, network bridge) — writes /workspace
  3. Wait for agent to finish (or timeout)
  4. Launch eval container (CPU-only, --network none) — writes detail_report.json
  5. Apply isolation-breach penalty if agent exited 99

Usage:
  python orchestrator.py --agent claude-opus-4-6 --task ldct-denoising-task \
      --tier lite --n-patients 100 --gpu-id 0 --repeat-idx 0 \
      --max-seconds 2700 --output-dir /path/to/run
  python orchestrator.py --build-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

THIS_DIR     = Path(__file__).resolve().parent           # .../docker
IE_DIR       = THIS_DIR.parent                            # .../eval_image_enhancement
REPO_ROOT    = IE_DIR.parent                              # .../MedAgentsBench
PROMPTS_DIR  = IE_DIR / "prompts"

AGENT_IMAGE = "ie-agent:v5"
EVAL_IMAGE  = "ie-eval:v5"

TASK_DATA_DIR = {
    "ldct-denoising-task": "LDCT_SimNICT",
    "mri-sr-task":         "MRI_SR_SRMRI",
}


# ---------------------------------------------------------------------------
# Prompt composition (mirrors benchmark_runner_v4.build_system_prompt)
# ---------------------------------------------------------------------------
def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _fmt(t: str, **kw) -> str:
    for k, v in kw.items():
        t = t.replace("{" + k + "}", str(v))
    return t


def build_kickoff(tier: str) -> str:
    text = _read(PROMPTS_DIR / "common" / "kickoff.md")
    m = re.search(rf"##\s+{re.escape(tier)}\s*\n(.+?)(?=\n##\s+|\Z)",
                  text, re.DOTALL)
    return m.group(1).strip() if m else f"Begin. Follow S1-S5 for tier={tier}."


def build_system_prompt(task_cfg: dict, tier: str, task_dir: Path,
                        data_dir: str, output_dir: str,
                        patient_ids: list[str]) -> str:
    common = PROMPTS_DIR / "common"
    s = PROMPTS_DIR
    fmt_vars = dict(
        data_dir=data_dir, output_dir=output_dir,
        task_description=task_cfg["task_description"],
        modality=task_cfg.get("modality", "image"),
        input_format=task_cfg.get("input_format", ".npy"),
        input_filename=task_cfg.get("input_filename", "input.npy"),
        # Container-internal path — the file is staged into /workspace by
        # the orchestrator so `pip install -r <path>` works without touching
        # the host filesystem.
        requirements_txt_path=f"{output_dir}/requirements.txt",
        task_type=task_cfg.get("task_type", "image enhancement"),
    )
    parts = [
        _fmt(_read(common / "preamble.md"), **fmt_vars),
        _fmt(_read(common / f"important_{tier}.md"), **fmt_vars),
        _fmt(_read(common / f"env_{tier}.md"), **fmt_vars),
        _fmt(_read(s / "s1_plan" / f"{tier}.md"), **fmt_vars),
    ]
    s2 = s / "s2_setup" / f"{tier}.md"
    if not s2.is_file():
        s2 = s / "s2_setup" / "standard_pro.md"
    parts.append(_fmt(_read(s2), **fmt_vars))
    parts.append(_fmt(_read(
        s / "s3_validate" / ("pro.md" if tier == "pro" else "lite_standard.md")
    ), **fmt_vars))
    parts.append(_fmt(_read(
        s / "s4_inference" / ("pro.md" if tier == "pro" else "lite_standard.md")
    ), **fmt_vars))
    parts.append(_fmt(_read(s / "s5_submit" / "all.md"), **fmt_vars))

    mi_path = task_dir / "model_info.yaml"
    if mi_path.is_file():
        mi = yaml.safe_load(open(mi_path)) or {}
        tier_mi = mi.get(tier, {})
        if tier == "standard" and "model_range" in tier_mi:
            mr = "\n".join(f"  - {item}" for item in tier_mi["model_range"])
            parts = [p.replace("{model_range}", mr) for p in parts]
        if tier == "lite" and "model_architecture" in tier_mi:
            parts = [p.replace("{model_architecture}",
                               tier_mi["model_architecture"]) for p in parts]
            parts = [p.replace("{model_description}",
                               tier_mi.get("model_description", "")) for p in parts]
        parts.append(f"\n## Model information ({tier} tier)")
        for k, v in tier_mi.items():
            if isinstance(v, list):
                parts.append(f"- **{k}**:")
                parts.extend([f"  - {it}" for it in v])
            else:
                parts.append(f"- **{k}**: {v}")

    if tier == "lite":
        for sname in ["lite_s1.md", "lite_s2.md", "lite_s3.md"]:
            p = task_dir / sname
            if p.is_file():
                parts.append(f"\n### Task skill — {sname}\n"
                             f"{_fmt(_read(p), **fmt_vars)}\n")
    elif tier == "standard":
        for sname in ["standard_s1.md", "standard_s3.md"]:
            p = task_dir / sname
            if p.is_file():
                parts.append(f"\n### Task skill — {sname}\n"
                             f"{_fmt(_read(p), **fmt_vars)}\n")

    parts.append(f"\n## Patients to process ({len(patient_ids)})\n"
                 f"{', '.join(patient_ids)}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Agent config / API key
# ---------------------------------------------------------------------------
def load_agent_config(agent_name: str) -> dict[str, Any]:
    cfg_path = REPO_ROOT / "eval_seg" / "agent_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    agents = cfg.get("agents", {})
    if agent_name not in agents:
        raise KeyError(f"agent {agent_name!r} not in {cfg_path}; "
                       f"have {list(agents)}")
    return agents[agent_name]


def resolve_api_key(agent_cfg: dict) -> str:
    env_name = agent_cfg.get("api_key_env")
    if env_name:
        k = os.environ.get(env_name, "")
        if k:
            return k
    bu = (agent_cfg.get("base_url") or "").lower()
    if "openrouter.ai" in bu:
        return os.environ.get("OPENROUTER_API_KEY", "")
    if "inference-api.nvidia.com" in bu:
        return os.environ.get("NVIDIA_API_KEY", "")
    return os.environ.get("OPENROUTER_API_KEY", "")


# ---------------------------------------------------------------------------
# Build images
# ---------------------------------------------------------------------------
def build_images(agent_only: bool = False, eval_only: bool = False) -> None:
    uid = os.getuid()
    gid = os.getgid()
    build_args = ["--build-arg", f"USER_UID={uid}",
                  "--build-arg", f"USER_GID={gid}"]

    if not eval_only:
        print(f"[orchestrator] Building {AGENT_IMAGE} (UID={uid}/GID={gid}) ...")
        agent_ctx = THIS_DIR / "agent"
        subprocess.run([
            "docker", "build",
            "-t", AGENT_IMAGE,
            "-f", str(agent_ctx / "Dockerfile.agent"),
            *build_args,
            str(agent_ctx),
        ], check=True)
        print(f"[orchestrator] {AGENT_IMAGE} built.")

    if not agent_only:
        print(f"[orchestrator] Building {EVAL_IMAGE} (UID={uid}/GID={gid}) ...")
        # Build context is eval_image_enhancement/ so the scorer + bands can be copied
        subprocess.run([
            "docker", "build",
            "-t", EVAL_IMAGE,
            "-f", str(THIS_DIR / "eval" / "Dockerfile.eval"),
            *build_args,
            str(IE_DIR),
        ], check=True)
        print(f"[orchestrator] {EVAL_IMAGE} built.")


# ---------------------------------------------------------------------------
# Container runs
# ---------------------------------------------------------------------------
def run_agent_container(*, workspace_dir: Path, agent_data_dir: Path,
                        task: str, tier: str, agent_name: str,
                        agent_cfg: dict, patient_ids: list[str],
                        gpu_id: int, repeat_idx: int,
                        max_turns: int, max_seconds: int,
                        container_name: str, dry_run: bool = False) -> int:
    api_key  = resolve_api_key(agent_cfg)
    base_url = agent_cfg.get("base_url") or "https://api.openai.com/v1"
    model    = agent_cfg["model"]
    hf_token = os.environ.get("HF_TOKEN", "")

    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--gpus", f"device={gpu_id}",
        "--network", "bridge",
        "--cpus", "4",
        "--memory", "64g",
        "--shm-size", "16g",
        "--pids-limit", "4096",
        "--security-opt", "no-new-privileges",
        # Hardening (eval_seg parity): immutable root + writable /tmp
        "--read-only",
        "--tmpfs", "/tmp:rw,size=10g",
        # Mounts: agent sees only public data + its workspace
        "-v", f"{agent_data_dir}:/data/public:ro",
        "-v", f"{workspace_dir}:/workspace:rw",
        # Env
        "-e", f"AGENT_NAME={agent_name}",
        "-e", f"MODEL={model}",
        "-e", f"API_KEY={api_key}",
        "-e", f"BASE_URL={base_url}",
        "-e", f"TASK={task}",
        "-e", f"TIER={tier}",
        "-e", f"REPEAT_IDX={repeat_idx}",
        "-e", f"PATIENT_IDS={','.join(patient_ids)}",
        "-e", f"MAX_TURNS={max_turns}",
        "-e", f"MAX_SECONDS={max_seconds}",
        "-e", f"HF_TOKEN={hf_token}",
        # With --read-only, /opt/conda is immutable. Route pip / caches to /workspace
        # so any `pip install` the agent triggers lands in a writable mount.
        "-e", "PYTHONUSERBASE=/workspace/.local",
        "-e", "PIP_USER=1",
        "-e", "PIP_CACHE_DIR=/workspace/.cache/pip",
        "-e", "HF_HOME=/workspace/.cache/huggingface",
        "-e", "TORCH_HOME=/workspace/.cache/torch",
        "-e", "XDG_CACHE_HOME=/workspace/.cache",
        AGENT_IMAGE,
    ]

    if dry_run:
        print("[dry-run] agent:\n  " + " \\\n    ".join(cmd))
        return 0

    # Add generous wall-time margin for container lifecycle
    docker_timeout = max_seconds + 600
    print(f"[orchestrator] launching agent container {container_name} "
          f"on GPU {gpu_id} (cap={max_seconds}s)")
    try:
        proc = subprocess.run(cmd, timeout=docker_timeout)
        return proc.returncode
    except subprocess.TimeoutExpired:
        print(f"[orchestrator] agent container exceeded {docker_timeout}s — "
              f"killing {container_name}")
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        return -1


def run_eval_container(*, run_dir: Path, agent_data_dir: Path,
                       private_dir: Path, task: str, tier: str,
                       agent_name: str, patient_ids: list[str],
                       repeat_idx: int, container_name: str,
                       dry_run: bool = False) -> int:
    agent_outputs_dir = run_dir / "workspace" / "agents_outputs"
    agent_outputs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--network", "none",
        # GPU access for fast LPIPS scoring (~50× speedup on 512×512 images).
        # Network is still isolated (--network none), so GPU ≠ exfiltration path.
        "--gpus", "all",
        "--cpus", "4",
        "--memory", "8g",
        "--security-opt", "no-new-privileges",
        # Hardening (eval_seg parity): immutable root + small /tmp
        "--read-only",
        "--tmpfs", "/tmp:rw,size=1g",
        # Mounts
        "-v", f"{private_dir}:/data/private:ro",
        "-v", f"{agent_data_dir}:/data/public:ro",
        "-v", f"{agent_outputs_dir}:/agent_outputs:ro",
        "-v", f"{run_dir}:/results:rw",
        # Env
        "-e", f"AGENT_NAME={agent_name}",
        "-e", f"TASK={task}",
        "-e", f"TIER={tier}",
        "-e", f"REPEAT_IDX={repeat_idx}",
        "-e", f"PATIENT_IDS={','.join(patient_ids)}",
        EVAL_IMAGE,
    ]

    if dry_run:
        print("[dry-run] eval:\n  " + " \\\n    ".join(cmd))
        return 0

    # 100p × 2 passes (raw + shuffle-NC) × LPIPS on CPU, with up to 8 evals
    # running in parallel, routinely needs >30 min per cell. Keep the timeout
    # generous to avoid killing a run that's still making progress.
    EVAL_TIMEOUT_S = 5400  # 90 minutes
    print(f"[orchestrator] launching eval container {container_name} (no-network, cap={EVAL_TIMEOUT_S}s)")
    try:
        proc = subprocess.run(cmd, timeout=EVAL_TIMEOUT_S)
        return proc.returncode
    except subprocess.TimeoutExpired:
        print(f"[orchestrator] eval container exceeded {EVAL_TIMEOUT_S}s — killing")
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agent")
    p.add_argument("--task", choices=list(TASK_DATA_DIR))
    p.add_argument("--tier", default="lite", choices=["lite", "standard", "pro"])
    p.add_argument("--n-patients", type=int, default=100)
    p.add_argument("--max-turns", type=int, default=150)
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--repeat-idx", type=int, default=0)
    p.add_argument("--output-dir")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.build_only:
        build_images()
        return 0

    if not args.agent or not args.task:
        p.error("--agent and --task are required")

    # GPU preflight
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=15,
    )
    gpus_found = smi.stdout.strip().splitlines()
    if args.gpu_id >= len(gpus_found):
        print(f"[preflight] GPU {args.gpu_id} not found. Visible: {gpus_found}")
        return 2
    print(f"[preflight] GPU {args.gpu_id}: {gpus_found[args.gpu_id]}")

    # Task + patients
    task_dir = IE_DIR / args.task
    with open(task_dir / "config.yaml") as f:
        task_cfg = yaml.safe_load(f)
    data_task_dir = REPO_ROOT / "data" / TASK_DATA_DIR[args.task]
    public_dir  = data_task_dir / "public"
    private_dir = data_task_dir / "private"
    if not public_dir.exists() or not private_dir.exists():
        print(f"[preflight] missing data at {data_task_dir}")
        return 2
    all_pids = sorted([d.name for d in public_dir.iterdir() if d.is_dir()])
    patient_ids = all_pids[: args.n_patients]
    print(f"[preflight] patients: {len(patient_ids)}")

    # Run directory
    run_tag = time.strftime("%Y%m%dT%H%M%S")
    run_dir = (Path(args.output_dir) if args.output_dir else
               REPO_ROOT / "runs" /
               f"v5_{args.task}_{args.tier}_{args.agent}_r{args.repeat_idx}_{run_tag}"
               ).resolve()
    workspace_dir     = run_dir / "workspace"
    process_dir       = run_dir / "process"
    agent_outputs_dir = workspace_dir / "agents_outputs"
    for d in (workspace_dir, agent_outputs_dir, process_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[preflight] run_dir={run_dir}")

    # Stage agent_data: copy only the N selected patients' public dirs
    agent_data = run_dir / "agent_data"
    if agent_data.exists():
        shutil.rmtree(agent_data)
    agent_data.mkdir(parents=True)
    for pid in patient_ids:
        shutil.copytree(public_dir / pid, agent_data / pid)

    # Render tier prompt — use container-internal paths
    data_dir_in_container   = "/data/public"
    output_dir_in_container = "/workspace"
    agent_cfg = load_agent_config(args.agent)
    sys_prompt = build_system_prompt(
        task_cfg=task_cfg, tier=args.tier, task_dir=task_dir,
        data_dir=data_dir_in_container,
        output_dir=output_dir_in_container,
        patient_ids=patient_ids,
    )
    (workspace_dir / "tier_prompt.txt").write_text(sys_prompt)
    (workspace_dir / "kickoff.txt").write_text(build_kickoff(args.tier))
    (process_dir / "system_prompt.md").write_text(sys_prompt)
    # Stage requirements.txt into workspace so `pip install -r
    # /workspace/requirements.txt` works inside the container.
    req_src = task_dir / "requirements.txt"
    if req_src.is_file():
        shutil.copy(req_src, workspace_dir / "requirements.txt")

    # Container names (deterministic, include epoch to avoid collision across repeats)
    epoch = int(time.time())
    base_name = (f"ie5_{args.task.replace('-','')}_{args.tier}_"
                 f"{args.agent.replace('-','').replace('.','')}"
                 f"_r{args.repeat_idx}_{epoch}")
    agent_container = f"{base_name}_agent"
    eval_container  = f"{base_name}_eval"

    # --- Phase 1: agent ---
    t_agent_start = time.time()
    agent_rc = run_agent_container(
        workspace_dir=workspace_dir,
        agent_data_dir=agent_data,
        task=args.task, tier=args.tier,
        agent_name=args.agent, agent_cfg=agent_cfg,
        patient_ids=patient_ids,
        gpu_id=args.gpu_id, repeat_idx=args.repeat_idx,
        max_turns=args.max_turns, max_seconds=args.max_seconds,
        container_name=agent_container, dry_run=args.dry_run,
    )
    agent_wall = time.time() - t_agent_start
    isolation_breach = (agent_rc == 99)
    if isolation_breach:
        print(f"[orchestrator] *** ISOLATION BREACH *** agent exit=99; "
              f"all step scores will be zeroed")
    elif agent_rc != 0 and not args.dry_run:
        print(f"[orchestrator] agent exited with code {agent_rc}")

    # --- Phase 2: eval ---
    t_eval_start = time.time()
    eval_rc = run_eval_container(
        run_dir=run_dir, agent_data_dir=agent_data,
        private_dir=private_dir,
        task=args.task, tier=args.tier, agent_name=args.agent,
        patient_ids=patient_ids, repeat_idx=args.repeat_idx,
        container_name=eval_container, dry_run=args.dry_run,
    )
    eval_wall = time.time() - t_eval_start
    if eval_rc != 0 and not args.dry_run:
        print(f"[orchestrator] eval container exited with code {eval_rc}")

    # --- Phase 3: collect + penalty ---
    report_path = run_dir / "detail_report.json"
    if report_path.is_file():
        with open(report_path) as f:
            report = json.load(f)

        # Always record orchestrator-level wallclock stats
        report["orchestrator"] = {
            "agent_wall_s":      round(agent_wall, 2),
            "eval_wall_s":       round(eval_wall, 2),
            "agent_exit_code":   agent_rc,
            "eval_exit_code":    eval_rc,
            "isolation_breach":  isolation_breach,
            "image_agent":       AGENT_IMAGE,
            "image_eval":        EVAL_IMAGE,
        }

        # Merge agent_summary.json (tokens, turns, etc.) — agent writes it
        # inside the container at /workspace/process/, which is
        # workspace_dir/process/ on the host.
        agent_summary_path = workspace_dir / "process" / "agent_summary.json"
        if agent_summary_path.is_file():
            with open(agent_summary_path) as f:
                report["agent_summary"] = json.load(f)

        if isolation_breach:
            report["rating"] = {"rating": "F", "tier_name": "disqualified",
                                "reasons": ["isolation_breach"],
                                "source": "penalty"}
            report["pass_rate"] = {
                "threshold_ssim": None, "classical_ssim": None,
                "n_passed": 0, "n_scored": 0, "pass_rate": 0.0,
            }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Reclaim disk: delete per-run caches after scoring is done.
        # Reports + agents_outputs + process logs stay for audit.
        for sub in (".cache", ".local", "env", "requirements.txt"):
            p = workspace_dir / sub
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.is_file():
                    p.unlink()
            except Exception:
                pass
        # Drop the agent_data copy — data is still available on the
        # dataset path; keeping per-run copies just burns disk.
        try:
            shutil.rmtree(agent_data, ignore_errors=True)
        except Exception:
            pass

        print(f"\n{'='*60}")
        print(f"  RESULTS  agent={args.agent}  task={args.task}  "
              f"tier={args.tier}  r={args.repeat_idx}")
        print(f"  rating   = {report.get('rating', {}).get('rating', '?')}")
        pr = report.get("pass_rate") or {}
        print(f"  pass_rate= {pr.get('pass_rate')}  "
              f"({pr.get('n_passed')}/{pr.get('n_scored')})")
        scr = report.get("scores") or {}
        if scr:
            print(f"  PSNR={scr.get('mean_psnr'):.3f}  "
                  f"SSIM={scr.get('mean_ssim'):.4f}  "
                  f"LPIPS={scr.get('mean_lpips'):.4f}")
        print(f"  agent_wall={agent_wall:.1f}s  eval_wall={eval_wall:.1f}s")
        print(f"  report: {report_path}")
        print(f"{'='*60}")
    elif not args.dry_run:
        print(f"[orchestrator] WARNING: no detail_report.json at {report_path}")
        # Still write a skeleton so the launcher can tell the cell ran
        skeleton = {
            "agent": args.agent, "task": args.task, "tier": args.tier,
            "repeat_idx": args.repeat_idx,
            "orchestrator": {
                "agent_wall_s": round(agent_wall, 2),
                "eval_wall_s":  round(eval_wall, 2),
                "agent_exit_code": agent_rc,
                "eval_exit_code":  eval_rc,
                "isolation_breach": isolation_breach,
            },
            "rating": {"rating": "F", "tier_name": "no_report"},
        }
        with open(report_path, "w") as f:
            json.dump(skeleton, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
