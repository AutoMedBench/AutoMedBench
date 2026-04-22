#!/usr/bin/env python3
"""Re-judge existing runs with the current S1-S5 rubric + new clinical formula.

Writes detail_report_new-rubrics.json alongside the original detail_report.json.
Original reports are never modified.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from llm_judge import OnlineJudge, JUDGE_SYSTEM_PROMPT
from aggregate import (
    STEP_WEIGHTS, compute_s4, compute_s5, compute_clinical_score,
    compute_workflow_score, compute_overall_score, assign_rating,
    is_resolved,
)
from medal_tier import assign_tier


STEP_WEIGHTS_LITE_STD = {"s1": 0.25, "s2": 0.15, "s3": 0.35, "s4": 0.15, "s5": 0.10}


def list_run_dirs(group_dir: Path):
    """Yield run subdirs that have both detail_report.json and conversation.json."""
    if not group_dir.is_dir():
        return
    for run in sorted(group_dir.iterdir()):
        if not run.is_dir():
            continue
        rpt = run / "detail_report.json"
        conv = run / "process" / "conversation.json"
        if rpt.exists() and conv.exists():
            yield run


def load_conversation(run_dir: Path) -> dict:
    """Load conversation.json, enrich with plan file existence flags."""
    conv = json.load(open(run_dir / "process" / "conversation.json"))
    # Pass through — the rubric uses the file existence check indirectly via
    # the conversation trace. No extra file flags needed.
    return conv


def recompute_scores(old_report: dict, verdict_dict: dict,
                     step_weights: dict = None) -> dict:
    """Build a new detail_report from the old one + a fresh judge verdict.

    Keeps header/runtime/tool_calls untouched. Recomputes clinical (new
    formula), agentic (weighted avg incl. fresh S1-S3), overall, rating.
    """
    weights = step_weights or STEP_WEIGHTS_LITE_STD

    # ---- Diagnostic metrics (unchanged — deterministic Dice) ----
    dm = old_report.get("diagnostic_metrics", {}) or {}
    organ_dice = float(dm.get("organ_dice") or 0.0)
    lesion_dice = float(dm.get("lesion_dice") or 0.0)

    # ---- Inference/format state (re-derive from old format block) ----
    fmt = old_report.get("format", {}) or {}
    masks_valid = bool(fmt.get("masks_valid") or fmt.get("submission_valid"))
    submission_valid = bool(fmt.get("submission_valid") or masks_valid)
    output_valid = 1.0 if masks_valid else 0.0

    # Completeness gate: if previously invalidated, keep invalidated.
    # We trust the old deterministic completion check.
    old_step_scores = (old_report.get("agentic_score") or {}).get("step_scores", {})
    old_s4 = old_step_scores.get("s4")
    # Back out inference_completes from old s4 = 0.5*ic + 0.5*ov
    if old_s4 is not None and output_valid == 1.0:
        inference_completes = max(0.0, min(1.0, 2.0 * old_s4 - output_valid))
    elif old_s4 is not None and output_valid == 0.0:
        inference_completes = max(0.0, min(1.0, 2.0 * old_s4))
    else:
        inference_completes = 1.0 if masks_valid else 0.0

    # ---- Fresh S1/S2/S3 from judge verdict ----
    s1 = float(verdict_dict.get("s1_plan_score", 0.0))
    s2 = float(verdict_dict.get("s2_setup_score", 0.0))
    s3 = float(verdict_dict.get("s3_validate_score", 0.0))

    # ---- Deterministic S4/S5 (same formula as aggregate.py) ----
    s4 = compute_s4(inference_completes, output_valid)
    has_valid_results = (inference_completes > 0) and (lesion_dice > 0 or organ_dice > 0)
    s5 = compute_s5(has_valid_results, bool(output_valid))

    step_scores = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}

    workflow_score, active_steps = compute_workflow_score(
        step_scores, weights=weights
    )
    clinical_score = compute_clinical_score(organ_dice, lesion_dice)
    overall_score = compute_overall_score(workflow_score, clinical_score)

    medal = assign_tier(lesion_dice)
    rating = assign_rating(overall_score, medal_tier=medal["tier"],
                           format_valid=masks_valid)
    resolved = is_resolved(rating)

    progress_rate = 1.0 if (inference_completes >= 0.9 and masks_valid) else (
        0.5 if (inference_completes >= 0.9 or masks_valid) else 0.0
    )

    step_failures = {
        "s1": verdict_dict.get("s1_failure"),
        "s2": verdict_dict.get("s2_failure"),
        "s3": verdict_dict.get("s3_failure"),
        "s4": verdict_dict.get("s4_failure"),
        "s5": verdict_dict.get("s5_failure"),
        "primary_failure": verdict_dict.get("detected_failure"),
        "failure_explanation": verdict_dict.get("failure_explanation", ""),
    }

    new_report = dict(old_report)  # shallow copy
    new_report["clinical_score"] = {
        "score": clinical_score,
        "organ_dice": organ_dice,
        "lesion_dice": lesion_dice,
        "auc": 0.0,
    }
    new_report["agentic_score"] = {
        "score": workflow_score,
        "step_scores": step_scores,
        "active_steps": active_steps,
        "progress_rate": progress_rate,
    }
    new_report["agentic_tier"] = {
        "rating": rating,
        "resolved": resolved,
        "overall_score": overall_score,
        "medal_tier": medal["tier"],
        "medal_name": medal["name"],
        "description": f"{medal['name'].capitalize()} result",
    }
    new_report["step_failures"] = step_failures
    new_report["llm_judge"] = {
        "s1_plan_score": s1,
        "s1a_plan_md": verdict_dict.get("s1a_plan_md", 0),
        "s1b_plan_pipeline": verdict_dict.get("s1b_plan_pipeline", 0),
        "s1c_lesion_model": verdict_dict.get("s1c_lesion_model", 0),
        "s1d_researched_3": verdict_dict.get("s1d_researched_3", 0),
        "s1e_plan_plot": verdict_dict.get("s1e_plan_plot", 0),
        "s1f_plot_pipeline": verdict_dict.get("s1f_plot_pipeline", 0),
        "s1_rationale": verdict_dict.get("s1_rationale", ""),
        "s2_setup_score": s2,
        "s2a_checkpoint_downloaded": verdict_dict.get("s2a_checkpoint_downloaded", 0),
        "s2b_compatibility_check": verdict_dict.get("s2b_compatibility_check", 0),
        "s2c_env_setup_success": verdict_dict.get("s2c_env_setup_success", 0),
        "s2d_env_fail_within_5": verdict_dict.get("s2d_env_fail_within_5", 0),
        "s2e_model_loaded": verdict_dict.get("s2e_model_loaded", 0),
        "s2_rationale": verdict_dict.get("s2_rationale", ""),
        "s3_validate_score": s3,
        "s3_rationale": verdict_dict.get("s3_rationale", ""),
        "tool_calling_score": verdict_dict.get("tool_calling_score", 0.0),
        "tool_calling_rationale": verdict_dict.get("tool_calling_rationale", ""),
        "clinical_reasoning_score": verdict_dict.get("clinical_reasoning_score", 0.0),
        "clinical_reasoning_rationale": verdict_dict.get("clinical_reasoning_rationale", ""),
        "overall_rationale": verdict_dict.get("overall_rationale", ""),
        "judge_model": verdict_dict.get("judge_model", ""),
        "judge_backend": verdict_dict.get("judge_backend", ""),
        "judge_latency_s": verdict_dict.get("judge_latency_s", 0.0),
    }
    new_report["rejudge_meta"] = {
        "rubric_version": "2026-04-sub-criteria",
        "clinical_formula": "0.50*organ + 0.50*lesion",
        "step_weights": weights,
        "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return new_report


def rejudge_run(run_dir: Path, judge: OnlineJudge,
                step_weights: dict, out_name: str = "detail_report_new-rubrics.json",
                overwrite: bool = False) -> str:
    """Re-judge a single run. Returns status: 'ok', 'skip', or 'err: <msg>'."""
    out_path = run_dir / out_name
    if out_path.exists() and not overwrite:
        return "skip"
    try:
        old_report = json.load(open(run_dir / "detail_report.json"))
        conv = load_conversation(run_dir)
        task = old_report.get("header", {}).get("task", conv.get("task", "kidney"))
        verdict = judge.judge(conv, old_report, task)
        new_report = recompute_scores(old_report, verdict.to_dict(), step_weights)
        with open(out_path, "w") as f:
            json.dump(new_report, f, indent=2, default=str)
        return "ok"
    except Exception as e:
        return f"err: {type(e).__name__}: {str(e)[:120]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True,
                    help="Group directories, e.g. runs/kuma/bench-opus4.6-kidney-lite")
    ap.add_argument("--tier", default="lite",
                    choices=["lite", "standard", "pro"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-name", default="detail_report_new-rubrics.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, stop after this many runs total (debug)")
    args = ap.parse_args()

    step_weights = STEP_WEIGHTS_LITE_STD  # same for lite/standard

    judge = OnlineJudge()
    print(f"[rejudge] Using judge: {judge.model} via {judge.api_url}",
          flush=True)

    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "err": 0}
    total_seen = 0

    for g in args.groups:
        gdir = Path(g)
        if not gdir.is_dir():
            print(f"[rejudge] skip missing: {g}", flush=True)
            continue
        print(f"\n[rejudge] ===== {gdir.name} =====", flush=True)
        runs = list(list_run_dirs(gdir))
        for i, run in enumerate(runs, 1):
            if args.limit and total_seen >= args.limit:
                break
            total_seen += 1
            status = rejudge_run(run, judge, step_weights,
                                 out_name=args.out_name,
                                 overwrite=args.overwrite)
            elapsed = time.time() - t0
            tag = "ok" if status == "ok" else ("skip" if status == "skip" else "err")
            stats[tag] += 1
            print(f"  [{i}/{len(runs)}] {run.name}  -> {status}   "
                  f"(elapsed {elapsed:.0f}s, ok={stats['ok']} skip={stats['skip']} err={stats['err']})",
                  flush=True)
        if args.limit and total_seen >= args.limit:
            break

    dt = time.time() - t0
    print(f"\n[rejudge] DONE  ok={stats['ok']} skip={stats['skip']} "
          f"err={stats['err']}  total_runs={total_seen}  time={dt:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
