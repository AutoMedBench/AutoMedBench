#!/usr/bin/env python3
"""Generate concise detail report: metrics, time, tokens, failure analysis."""

import json
from datetime import datetime, timezone

from failure_classifier import ERROR_CODES


def generate_detail_report(eval_report: dict, runtime: dict,
                           agent_name: str, model: str, task: str,
                           tool_summary: dict = None,
                           judge_verdict: dict = None,
                           tier: str = None) -> dict:
    """Merge eval results with runtime stats into a single detail report."""
    metrics = eval_report.get("metrics", {})
    agg = eval_report.get("aggregate", {})
    fmt = eval_report.get("format", {})
    pl = eval_report.get("patient_level", {})
    steps = eval_report.get("step_scores", {})
    failure = eval_report.get("failure")

    # Per-patient detail is intentionally excluded from the report
    # to keep it concise and agent-level only.

    rating = agg.get("rating", "F")

    is_multiclass = metrics.get("task_type") == "multiclass"

    if is_multiclass:
        macro_mean_dice = float(metrics.get("macro_mean_dice", 0.0))
        per_class_dice = metrics.get("per_class_dice", {})
        clinical_score = round(macro_mean_dice, 4)
    else:
        # Read dice values once to guarantee consistency across all sections.
        organ_dice = metrics.get("organ_dice", 0.0)
        lesion_dice = metrics.get("lesion_dice", 0.0)
        # Recompute clinical score from the same values used in diagnostic_metrics
        # to prevent score/dice divergence (Bug 3).
        clinical_score = round(0.50 * lesion_dice + 0.50 * organ_dice, 4) \
            if isinstance(organ_dice, (int, float)) else agg.get("clinical_score", 0.0)

    header = {
        "agent": agent_name,
        "model": model,
        "task": task,
        "organ": task,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    if tier:
        header["tier"] = tier

    if is_multiclass:
        diagnostic_metrics = {
            "task_type": "multiclass",
            "macro_mean_dice": round(macro_mean_dice, 4),
            "per_class_dice": per_class_dice,
        }
    else:
        diagnostic_metrics = {
            "organ_dice": organ_dice,
            "lesion_dice": lesion_dice,
            "patient_level": {
                "tp": pl.get("tp", 0),
                "tn": pl.get("tn", 0),
                "fp": pl.get("fp", 0),
                "fn": pl.get("fn", 0),
            },
        }

    return {
        "header": header,
        "runtime": runtime,

        # (1) Diagnostic metrics — clinical quality of the segmentation output
        "diagnostic_metrics": diagnostic_metrics,

        # (2) Agentic score — how well the agent followed the S1-S5 workflow
        "agentic_score": {
            "score": agg.get("agentic_score", 0.0),
            "step_scores": {
                "s1": steps.get("s1"),
                "s2": steps.get("s2"),
                "s3": steps.get("s3"),
                "s4": steps.get("s4"),
                "s5": steps.get("s5"),
            },
            "active_steps": agg.get("active_steps", []),
            "progress_rate": agg.get("progress_rate", 0.0),
        },

        # (3) Clinical score — organ Dice + lesion Dice + lesion AUC
        # (for multiclass: macro-mean Dice across foreground tissues)
        "clinical_score": (
            {
                "score": clinical_score,
                "macro_mean_dice": round(macro_mean_dice, 4),
                "per_class_dice": per_class_dice,
            } if is_multiclass else {
                "score": clinical_score,
                "organ_dice": organ_dice,
                "lesion_dice": lesion_dice,
                "auc": metrics.get("auc", 0.0),
            }
        ),

        # (3) Agentic tier — final letter grade
        "agentic_tier": {
            "rating": rating,
            "resolved": agg.get("resolved", False),
            "overall_score": round(
                0.5 * (agg.get("agentic_score", 0.0) or 0.0)
                + 0.5 * clinical_score, 4),
            "medal_tier": metrics.get("medal_tier", 0),
            "medal_name": metrics.get("medal_name", "N/A"),
            "description": {
                "A": "Good result",
                "B": "Okay result",
                "C": "Below baseline",
                "F": "Failed",
            }.get(rating, "Unknown"),
        },

        "format": {
            "submission_valid": fmt.get("submission_format_valid", False),
            "csv_valid": fmt.get("decision_csv_valid", None),
            "masks_valid": fmt.get("output_format_valid", False),
        },
        **_build_failure_section(failure, judge_verdict,
                                 medal_tier=metrics.get("medal_tier", 0),
                                 masks_valid=fmt.get("output_format_valid", False)),
        "tool_calls": tool_summary or {},
    }


def _build_failure_section(auto_failure: dict, judge_verdict: dict,
                           medal_tier: int = 0,
                           masks_valid: bool = False) -> dict:
    """Build two failure sections combining auto-detection and judge.

    Returns a dict with two keys to be merged into the report:
      - "error_analysis":  E-code statistics (how many of each error type)
      - "step_failures":   per-step breakdown (which step, which E-code)
    """
    # Success override: if masks are valid and result tier >= okay,
    # suppress ALL failure flags (both heuristic and judge).
    # This prevents the LLM judge from flagging E5 on successful runs
    # just because the optional CSV is missing.
    if masks_valid and medal_tier >= 1:
        return {
            "error_analysis": {
                "code_counts": {"E1": 0, "E2": 0, "E3": 0, "E4": 0, "E5": 0},
                "total_errors": 0,
            },
            "step_failures": {
                "s1": None, "s2": None, "s3": None, "s4": None, "s5": None,
                "primary_failure": None,
                "failure_explanation": "",
            },
        }

    jv = judge_verdict or {}

    # Per-step failure codes: S1-S3 from judge, S4-S5 from heuristic only.
    # S4 and S5 are deterministic (completion rate + format check) — the
    # LLM judge should not override them.
    step_codes = {
        "s1": jv.get("s1_failure"),
        "s2": jv.get("s2_failure"),
        "s3": jv.get("s3_failure"),
        "s4": None,  # deterministic — never from judge
        "s5": None,  # deterministic — never from judge
    }

    # S4/S5 failures come only from the heuristic auto-classifier
    if auto_failure:
        auto_steps = auto_failure.get("step_failures", {})
        for s in ("s4", "s5"):
            if auto_steps.get(s):
                step_codes[s] = auto_steps[s]

    # Fall back to auto-detection for S1-S3 if judge didn't provide them
    if not any(step_codes[s] for s in ("s1", "s2", "s3")) and auto_failure:
        auto_steps = auto_failure.get("step_failures", {})
        for s in ("s1", "s2", "s3"):
            if auto_steps.get(s):
                step_codes[s] = auto_steps[s]

    # Primary failure: prefer judge, fall back to auto.
    # But ignore judge failures that are purely S4/S5 (deterministic).
    primary = jv.get("detected_failure")
    explanation = jv.get("failure_explanation", "")
    if primary and primary.startswith(("S4:", "S5:")):
        primary = None  # S4/S5 failures come from heuristic, not judge
        explanation = ""
    if not primary and auto_failure:
        primary = auto_failure.get("primary_failure")
        explanation = auto_failure.get("failure_explanation", "")

    # E-code counts
    all_codes = [v for v in step_codes.values() if v]
    code_counts = {}
    for e in ("E1", "E2", "E3", "E4", "E5"):
        code_counts[e] = all_codes.count(e)

    return {
        # (1) Error Analysis — E-code statistics
        "error_analysis": {
            "code_counts": code_counts,
            "total_errors": len(all_codes),
        },
        # (2) Step-level Failure — which step failed and why
        "step_failures": {
            "s1": step_codes["s1"],
            "s2": step_codes["s2"],
            "s3": step_codes["s3"],
            "s4": step_codes["s4"],
            "s5": step_codes["s5"],
            "primary_failure": primary,
            "failure_explanation": explanation,
        },
    }


# -------------------------------------------------------------------
# Pretty printer
# -------------------------------------------------------------------

def _fmt(val, width=8):
    if val is None:
        return "N/A".rjust(width)
    if isinstance(val, float):
        return f"{val:.4f}".rjust(width)
    return str(val).rjust(width)


def print_detail_report(report: dict):
    """Print a concise, human-readable detail report to stdout."""
    h = report["header"]
    r = report["runtime"]
    dm = report["diagnostic_metrics"]
    ag = report["agentic_score"]
    tier = report["agentic_tier"]
    f = report["format"]
    ea = report.get("error_analysis", {})
    sf = report.get("step_failures", {})
    tc = report.get("tool_calls", {})

    W = 62
    line = "=" * W
    dash = "-" * W

    print(f"\n{line}")
    print(f"  MedAgentsBench -- Detail Report")
    tier_str = f"  |  Tier: {h['tier']}" if h.get('tier') else ""
    print(f"  Agent: {h['agent']}  |  Task: {h['task']}{tier_str}  |  {h['date']}")
    print(f"  Model: {h['model']}")
    print(line)

    # ── (3) AGENTIC TIER ──────────────────────────────────────────
    print(f"  AGENTIC TIER:  [{tier['rating']}] — {tier['description']}")
    resolved_tag = "PASS" if tier.get("resolved") else "FAIL"
    print(f"    Resolved: {resolved_tag}  |  Result: {tier['medal_name']} (tier {tier['medal_tier']})")
    print(dash)

    # ── (1) DIAGNOSTIC METRICS ────────────────────────────────────
    if dm.get("task_type") == "multiclass":
        print(f"  DIAGNOSTIC METRICS  (multi-tissue segmentation)")
        print(f"    Mean Dice (macro across tissues): {_fmt(dm.get('macro_mean_dice'))}")
        for tissue, d in dm.get("per_class_dice", {}).items():
            print(f"    {tissue:<10}      {_fmt(d)}")
    else:
        pl = dm["patient_level"]
        print(f"  DIAGNOSTIC METRICS")
        print(f"    Organ Dice:     {_fmt(dm['organ_dice'])}")
        print(f"    Lesion Dice:    {_fmt(dm['lesion_dice'])}")
        # Sensitivity/specificity removed from scoring — kept in report JSON for reference only
        print(f"    AUC:            {_fmt(dm.get('auc'))}")
        print(f"    F1:             {_fmt(dm.get('f1'))}")

    print(dash)

    # ── (2) AGENTIC SCORE ─────────────────────────────────────────
    ss = ag["step_scores"]
    print(f"  AGENTIC SCORE:    {ag['score']:.4f}")
    s_parts = []
    for k in ("s1", "s2", "s3", "s4", "s5"):
        v = ss.get(k)
        s_parts.append(f"{k.upper()}={v:.2f}" if v is not None else f"{k.upper()}=---")
    print(f"    Steps:          {' '.join(s_parts)}")
    active = ", ".join(ag.get("active_steps", []))
    print(f"    Active steps:   {active}")
    print(f"    Progress rate:  {ag['progress_rate']:.2f}")
    print(dash)

    # ── (3) CLINICAL SCORE ────────────────────────────────────────
    cs = report.get("clinical_score", {})
    print(f"  CLINICAL SCORE:   {_fmt(cs.get('score', 0))}")
    if "macro_mean_dice" in cs:
        print(f"    Mean Dice:      {_fmt(cs.get('macro_mean_dice', 0))}")
        for tissue, d in cs.get("per_class_dice", {}).items():
            print(f"    {tissue:<14}  {_fmt(d)}")
    else:
        print(f"    Organ Dice:     {_fmt(cs.get('organ_dice', 0))}")
        print(f"    Lesion Dice:    {_fmt(cs.get('lesion_dice', 0))}")
        print(f"    AUC:            {_fmt(cs.get('auc', 0))}")
    print(dash)

    # ── RUNTIME ───────────────────────────────────────────────────
    print(f"  RUNTIME")
    print(f"    Wall time:      {r['wall_time_s']:.1f}s")
    print(f"    API calls:      {r['api_calls']}")
    in_tok = f"{r['input_tokens']:,}"
    out_tok = f"{r['output_tokens']:,}"
    tot_tok = f"{r['total_tokens']:,}"
    print(f"    Tokens:         {in_tok} in / {out_tok} out ({tot_tok} total)")
    cost = r.get('estimated_cost_usd', 0)
    print(f"    Est. cost:      ${cost:.4f}")
    print(dash)

    # ── FORMAT CHECK ──────────────────────────────────────────────
    print(f"  FORMAT CHECK")
    csv_status = f['csv_valid']
    csv_tag = "PASS" if csv_status is True else ("FAIL" if csv_status is False else "N/A")
    print(f"    Submission: {'PASS' if f['submission_valid'] else 'FAIL'}  "
          f"CSV: {csv_tag}  "
          f"Masks: {'PASS' if f['masks_valid'] else 'FAIL'}")
    print(dash)

    # ── (4) ERROR ANALYSIS (E-codes) ────────────────────────────
    counts = ea.get("code_counts", {})
    total_errors = ea.get("total_errors", 0)

    print(f"  ERROR ANALYSIS")
    if total_errors > 0:
        for code in ("E1", "E2", "E3", "E4", "E5"):
            cnt = counts.get(code, 0)
            name = ERROR_CODES.get(code, code)
            bar = "#" * cnt
            print(f"    {code} {name:<16} {cnt}  {bar}")
    else:
        print(f"    No errors detected")
    print(dash)

    # ── (5) STEP-LEVEL FAILURE (S1-S5) ───────────────────────────
    print(f"  STEP-LEVEL FAILURE")
    has_any_failure = False
    step_names = {"s1": "Plan", "s2": "Setup", "s3": "Validate",
                  "s4": "Inference", "s5": "Submit"}
    for s in ("s1", "s2", "s3", "s4", "s5"):
        code = sf.get(s)
        name = step_names[s]
        if code:
            code_name = ERROR_CODES.get(code, code)
            print(f"    {s.upper()} {name:<10}  FAIL  {code} ({code_name})")
            has_any_failure = True
        else:
            print(f"    {s.upper()} {name:<10}  ok")
    primary = sf.get("primary_failure")
    if primary:
        print(f"    Root cause: {primary} — {sf.get('failure_explanation', '')}")
    elif not has_any_failure:
        print(f"    All steps passed")
    print(dash)

    # ── TOOL USAGE ────────────────────────────────────────────────
    if tc:
        print(f"  TOOL USAGE")
        print(f"    Total calls: {tc.get('total', 0)}  Errors: {tc.get('errors', 0)}")
        by = tc.get("by_tool", {})
        if by:
            parts = [f"{k}: {v}" for k, v in sorted(by.items())]
            print(f"    {', '.join(parts)}")

        # Phase summary
        ps = tc.get("phase_summary", {})
        if ps:
            print()
            phase_names = {"S1": "Plan", "S2": "Setup", "S3": "Validate",
                           "S4": "Inference", "S5": "Submit"}
            print(f"    {'Phase':<14} {'Calls':>5} {'Errors':>6} {'Exec(s)':>8}")
            for p in ("S1", "S2", "S3", "S4", "S5"):
                s = ps.get(p)
                if not s:
                    continue
                label = f"{p} {phase_names.get(p, '')}"
                print(f"    {label:<14} {s['calls']:>5} {s['errors']:>6} "
                      f"{s['total_exec_s']:>8.1f}")

        # Failures
        failures = tc.get("failures", [])
        if failures:
            print()
            print(f"    Failures:")
            for f_entry in failures:
                desc = f_entry.get("description", "")
                print(f"      #{f_entry['seq']:<3} [{f_entry['phase']}] {desc}")

        # Per-call log
        call_log = tc.get("call_log", [])
        if call_log:
            print()
            print(f"    {'#':>4} {'Turn':>4} {'Phase':>5} {'Lang':>6} "
                  f"{'RC':>3} {'Time':>6}  Description")
            for entry in call_log:
                lang = entry.get("language") or "—"
                rc = entry.get("exit_code")
                rc_s = str(rc) if rc is not None else "—"
                et = entry.get("exec_time_s")
                et_s = f"{et:.1f}s" if et is not None else "—"
                desc = entry.get("description", "")
                marker = " *" if rc and rc != 0 else ""
                print(f"    {entry['seq']:>4} T{entry['turn']:>3} {entry['phase']:>5} "
                      f"{lang:>6} {rc_s:>3} {et_s:>6}  {desc}{marker}")

        print(dash)

    print(f"{line}\n")


# -------------------------------------------------------------------
# CLI: read a detail_report.json and print it
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print a detail report JSON")
    parser.add_argument("report_json", help="Path to detail_report.json")
    args = parser.parse_args()

    with open(args.report_json) as f:
        report = json.load(f)
    print_detail_report(report)
