#!/usr/bin/env python3
"""v5 matrix aggregator — reads every detail_report.json and produces a wide
summary CSV + per-cell mean ± 95% bootstrap CI JSON.

Compared to the initial aggregator, this version:
  - separates F reasons: F_breach / F_timeout / F_no_output / F_below_band
  - reports valid_rate (fraction of repeats with scorable outputs)
  - reports bootstrap 95% CI for pass_rate and mean_psnr (10k resamples)

Usage:
    python aggregate_v5.py [run_root=/data3/wyh/code/MedAgentsBench/runs/matrix_v5_full]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

AGENTS_SHORT = ["opus46", "gpt54", "gemini31pro", "glm5", "minimax25", "qwen35"]
AGENT_LABEL = {
    "opus46":      "Claude Opus 4.6",
    "gpt54":       "GPT-5.4",
    "gemini31pro": "Gemini 3.1 Pro",
    "glm5":        "GLM-5",
    "minimax25":   "MiniMax 2.5",
    "qwen35":      "Qwen 3.5",
}
TIERS = ["lite", "standard"]
TASKS = ["ldct-denoising-task", "mri-sr-task"]


def classify_failure(r: dict) -> str:
    """Return one of: pass_A/B/C, F_breach, F_no_output, F_timeout, F_below_band, F_other."""
    rating = (r.get("rating") or {}).get("rating", "F")
    if rating in ("A", "B", "C"):
        return f"pass_{rating}"

    orch  = r.get("orchestrator")  or {}
    asum  = r.get("agent_summary") or {}
    fmt   = r.get("format")        or {}

    if orch.get("isolation_breach"):
        return "F_breach"
    n_valid = fmt.get("n_valid", 0) or 0
    n_total = fmt.get("n_patients") or r.get("n_patients") or 1
    completion = n_valid / max(1, n_total)
    if n_valid == 0:
        return "F_no_output"
    if not asum.get("submitted"):
        # Agent ran but never called submit (most common cause: wall timeout)
        return "F_timeout"
    return "F_below_band"


def load_run(run_dir: Path) -> dict | None:
    rp = run_dir / "detail_report.json"
    if not rp.is_file():
        return None
    try:
        r = json.load(open(rp))
    except Exception:
        return None

    pr = r.get("pass_rate") or {}
    scr = r.get("scores") or {}
    rating = (r.get("rating") or {}).get("rating", "F")
    agent_sum = r.get("agent_summary") or {}
    orch = r.get("orchestrator") or {}
    fmt = r.get("format") or {}

    # Whether this repeat yielded a scorable result (non-null scores)
    scorable = scr is not None and scr != {} and scr.get("mean_psnr") is not None

    # Agentic/overall fields are populated by compute_agentic_score.py
    agg = r.get("aggregate") or {}

    return {
        "pass_rate":          pr.get("pass_rate"),
        "n_passed":           pr.get("n_passed"),
        "n_scored":           pr.get("n_scored"),
        "mean_psnr":          scr.get("mean_psnr"),
        "mean_ssim":          scr.get("mean_ssim"),
        "mean_lpips":         scr.get("mean_lpips"),
        "rating":             rating,
        "failure_kind":       classify_failure(r),
        "scorable":           bool(scorable),
        "agentic_score":      agg.get("agentic_score"),
        "clinical_score":     agg.get("clinical_score"),
        "overall_score":      agg.get("overall_score"),
        "step_scores":        r.get("step_scores"),
        "turns":              agent_sum.get("turns"),
        "prompt_tokens":      agent_sum.get("prompt_tokens"),
        "completion_tokens":  agent_sum.get("completion_tokens"),
        "violations":         agent_sum.get("violations"),
        "submitted":          agent_sum.get("submitted"),
        "agent_wall_s":       orch.get("agent_wall_s"),
        "eval_wall_s":        orch.get("eval_wall_s"),
        "isolation_breach":   orch.get("isolation_breach", False),
        "n_valid":            fmt.get("n_valid"),
        "n_patients":         fmt.get("n_patients"),
    }


def bootstrap_ci(values: list[float], stat=np.mean,
                 n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float | None, float | None]:
    """Return (lo, hi) percentile bootstrap CI."""
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    stats = stat(samples, axis=1)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


def aggregate_cell(run_root: Path, task: str, tier: str, agent: str) -> dict:
    cell_dir = run_root / task / tier
    runs = []
    r = 0
    while True:
        rd = cell_dir / f"{agent}_r{r}"
        if not rd.is_dir():
            break
        loaded = load_run(rd)
        if loaded is not None:
            loaded["repeat"] = r
            runs.append(loaded)
        r += 1

    if not runs:
        return {"task": task, "tier": tier, "agent": agent, "n_repeats": 0}

    scorable = [x for x in runs if x.get("scorable")]
    n_scorable = len(scorable)

    def mean_and_ci(key, source=None):
        src = source if source is not None else scorable
        vals = [x[key] for x in src if x.get(key) is not None]
        if not vals:
            return None, None, None, None
        m = float(np.mean(vals))
        s = float(np.std(vals)) if len(vals) > 1 else 0.0
        lo, hi = bootstrap_ci(vals, n_boot=10000)
        return m, s, lo, hi

    # Core metrics (computed only over scorable repeats — null repeats treated as non-scorable)
    from collections import Counter
    failure_counts = Counter(x["failure_kind"] for x in runs)

    out = {
        "task":         task,
        "tier":         tier,
        "agent":        agent,
        "agent_label":  AGENT_LABEL.get(agent, agent),
        "n_repeats":    len(runs),
        "n_scorable":   n_scorable,
        "valid_rate":   round(n_scorable / max(1, len(runs)), 4),
        "ratings":      [x["rating"] for x in runs],
        "failure_counts": dict(failure_counts),
        "per_repeat":   runs,
    }

    for key in ["pass_rate", "mean_psnr", "mean_ssim", "mean_lpips",
                "agentic_score", "clinical_score", "overall_score"]:
        # agentic/clinical/overall exist on every repeat (even unscorable
        # ones get 0.0 from compute_agentic_score.py), so use full runs list
        src = runs if key in ("agentic_score", "clinical_score",
                              "overall_score") else scorable
        m, s, lo, hi = mean_and_ci(key, source=src)
        out[f"{key}_mean"]   = m
        out[f"{key}_std"]    = s
        out[f"{key}_ci_lo"]  = lo
        out[f"{key}_ci_hi"]  = hi

    # Efficiency stats over all repeats (turns/tokens/wall), regardless of scorable
    for key in ["turns", "prompt_tokens", "completion_tokens",
                "agent_wall_s", "eval_wall_s"]:
        vals = [x[key] for x in runs if x.get(key) is not None]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"]  = float(np.std(vals)) if len(vals) > 1 else 0.0

    # Grade stability: fraction of repeats with modal rating
    rating_counter = Counter(out["ratings"])
    modal_rating, modal_count = rating_counter.most_common(1)[0]
    out["modal_rating"]   = modal_rating
    out["rating_stable"]  = round(modal_count / max(1, len(runs)), 4)

    out["n_submitted"] = sum(1 for x in runs if x.get("submitted"))
    out["n_isolation"] = sum(1 for x in runs if x.get("isolation_breach"))
    return out


def main() -> int:
    run_root = Path(sys.argv[1] if len(sys.argv) > 1
                    else str(Path(__file__).resolve().parent.parent.parent
                             / "runs" / "matrix_v5_full")).resolve()
    # Fallback to the committed curated snapshot when raw runs/ tree was cleaned
    if not run_root.is_dir():
        fallback = (Path(__file__).resolve().parent.parent
                    / "results" / run_root.name)
        if fallback.is_dir():
            print(f"[aggregate_v5] runs/{run_root.name} missing — using "
                  f"committed snapshot at {fallback}")
            run_root = fallback
    if not run_root.is_dir():
        print(f"ERROR: {run_root} not found"); return 2
    print(f"run_root={run_root}")

    rows = []
    for task in TASKS:
        for tier in TIERS:
            for agent in AGENTS_SHORT:
                rows.append(aggregate_cell(run_root, task, tier, agent))

    # CSV — wide summary with CI + failure breakdown
    csv_path = run_root / "summary.csv"
    with open(csv_path, "w") as f:
        header = (
            "task,tier,agent,n_repeats,n_scorable,valid_rate,"
            "pass_rate_mean,pass_rate_ci_lo,pass_rate_ci_hi,"
            "agentic_score_mean,agentic_score_ci_lo,agentic_score_ci_hi,"
            "clinical_score_mean,clinical_score_ci_lo,clinical_score_ci_hi,"
            "overall_score_mean,overall_score_ci_lo,overall_score_ci_hi,"
            "psnr_mean,psnr_ci_lo,psnr_ci_hi,"
            "ssim_mean,ssim_ci_lo,ssim_ci_hi,"
            "lpips_mean,lpips_ci_lo,lpips_ci_hi,"
            "modal_rating,rating_stable,ratings,"
            "failure_counts,"
            "turns_mean,prompt_tokens_mean,completion_tokens_mean,agent_wall_s_mean\n"
        )
        f.write(header)
        for r in rows:
            def _f(k):
                v = r.get(k)
                return f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))
            f.write(",".join([
                r["task"], r["tier"], r["agent"],
                str(r.get("n_repeats", 0)),
                str(r.get("n_scorable", 0)),
                _f("valid_rate"),
                _f("pass_rate_mean"), _f("pass_rate_ci_lo"), _f("pass_rate_ci_hi"),
                _f("agentic_score_mean"), _f("agentic_score_ci_lo"), _f("agentic_score_ci_hi"),
                _f("clinical_score_mean"), _f("clinical_score_ci_lo"), _f("clinical_score_ci_hi"),
                _f("overall_score_mean"), _f("overall_score_ci_lo"), _f("overall_score_ci_hi"),
                _f("mean_psnr_mean"), _f("mean_psnr_ci_lo"), _f("mean_psnr_ci_hi"),
                _f("mean_ssim_mean"), _f("mean_ssim_ci_lo"), _f("mean_ssim_ci_hi"),
                _f("mean_lpips_mean"), _f("mean_lpips_ci_lo"), _f("mean_lpips_ci_hi"),
                str(r.get("modal_rating", "")),
                _f("rating_stable"),
                "|".join(r.get("ratings") or []),
                ";".join(f"{k}={v}" for k, v in (r.get("failure_counts") or {}).items()),
                _f("turns_mean"),
                _f("prompt_tokens_mean"), _f("completion_tokens_mean"),
                _f("agent_wall_s_mean"),
            ]) + "\n")
    print(f"wrote {csv_path}  ({len(rows)} cells)")

    json_path = run_root / "summary.json"
    with open(json_path, "w") as f:
        json.dump({"run_root": str(run_root), "cells": rows},
                  f, indent=2, default=str)
    print(f"wrote {json_path}")

    # Readable pass-rate matrix
    print(f"\nPASS_RATE  (mean [95% CI]) / modal_rating / valid_rate / failure_breakdown:")
    for task in TASKS:
        for tier in TIERS:
            print(f"\n{task}/{tier}:")
            for agent in AGENTS_SHORT:
                c = next(r for r in rows
                         if r["task"] == task and r["tier"] == tier
                         and r["agent"] == agent)
                pr  = c.get("pass_rate_mean")
                lo  = c.get("pass_rate_ci_lo")
                hi  = c.get("pass_rate_ci_hi")
                vr  = c.get("valid_rate", 0)
                mr  = c.get("modal_rating", "?")
                rs  = c.get("rating_stable", 0)
                fc  = c.get("failure_counts", {})
                n_rep = c.get("n_repeats", 0)
                if pr is not None:
                    pr_s = f"{pr:.3f} [{lo:.2f},{hi:.2f}]"
                else:
                    pr_s = "----"
                fc_s = ",".join(f"{k}={v}" for k, v in sorted(fc.items()))
                print(f"  {AGENT_LABEL[agent]:18s}  pass={pr_s:20s}  "
                      f"n_rep={n_rep:>2d} valid={vr:.2f} modal={mr}({rs:.2f}) "
                      f" failures=[{fc_s}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
