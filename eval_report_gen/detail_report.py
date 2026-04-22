#!/usr/bin/env python3
"""Generate concise detail reports for report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from failure_classifier import ERROR_CODES


def generate_detail_report(
    eval_report: dict,
    runtime: dict,
    agent_name: str,
    model: str,
    task: str,
    tool_summary: dict | None = None,
    judge_verdict: dict | None = None,
    tier: str | None = None,
) -> dict:
    metrics = eval_report.get("metrics", {})
    agg = eval_report.get("aggregate", {})
    fmt = eval_report.get("format", {})
    steps = eval_report.get("step_scores", {})
    failure = eval_report.get("failure")

    header = {
        "agent": agent_name,
        "model": model,
        "task": task,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    if tier:
        header["tier"] = tier

    clinical_score = agg.get("clinical_score", 0.0)

    return {
        "header": header,
        "runtime": runtime,
        "diagnostic_metrics": {
            "observation_f1": metrics.get("observation_f1", 0.0),
            "report_similarity": metrics.get("report_similarity", 0.0),
            "label_exact_match": metrics.get("label_exact_match", 0.0),
            "micro_precision": metrics.get("micro_precision", 0.0),
            "micro_recall": metrics.get("micro_recall", 0.0),
            "micro_f1": metrics.get("micro_f1", 0.0),
        },
        "agentic_score": {
            "score": agg.get("agentic_score", 0.0),
            "step_scores": {key: steps.get(key) for key in ("s1", "s2", "s3", "s4", "s5")},
            "active_steps": agg.get("active_steps", []),
            "progress_rate": agg.get("progress_rate", 0.0),
        },
        "clinical_score": {
            "score": clinical_score,
            "BLEU": metrics.get("BLEU", 0.0),
            "BLEU_1": metrics.get("BLEU_1", 0.0),
            "BLEU_2": metrics.get("BLEU_2", 0.0),
            "BLEU_3": metrics.get("BLEU_3", 0.0),
            "BLEU_4": metrics.get("BLEU_4", 0.0),
            "METEOR": metrics.get("METEOR", 0.0),
            "ROUGE_L": metrics.get("ROUGE_L", 0.0),
            "F1RadGraph": metrics.get("F1RadGraph", 0.0),
            "micro_average_precision": metrics.get("micro_average_precision", 0.0),
            "micro_average_recall": metrics.get("micro_average_recall", 0.0),
            "micro_average_f1": metrics.get("micro_average_f1", 0.0),
            "observation_f1": metrics.get("observation_f1", 0.0),
            "report_similarity": metrics.get("report_similarity", 0.0),
            "label_exact_match": metrics.get("label_exact_match", 0.0),
        },
        "agentic_tier": {
            "rating": agg.get("rating", "F"),
            "resolved": agg.get("resolved", False),
            "overall_score": agg.get("overall_score", 0.0),
            "medal_tier": metrics.get("medal_tier", 0),
            "medal_name": metrics.get("medal_name", "fail"),
            "description": {
                "A": "Good result",
                "B": "Okay result",
                "C": "Below baseline",
                "F": "Failed",
            }.get(agg.get("rating", "F"), "Unknown"),
        },
        "format": {
            "submission_valid": fmt.get("submission_format_valid", False),
            "reports_valid": fmt.get("output_format_valid", False),
            "completion_rate": fmt.get("completion_rate", 0.0),
        },
        **_build_failure_section(failure, judge_verdict),
        "tool_calls": tool_summary or {},
    }


def _build_failure_section(auto_failure: dict | None, judge_verdict: dict | None) -> dict:
    judge_verdict = judge_verdict or {}
    step_codes = {
        "s1": judge_verdict.get("s1_failure"),
        "s2": judge_verdict.get("s2_failure"),
        "s3": judge_verdict.get("s3_failure"),
        "s4": None,
        "s5": None,
    }
    if auto_failure:
        auto_steps = auto_failure.get("step_failures", {})
        for step_name in ("s4", "s5"):
            if auto_steps.get(step_name):
                step_codes[step_name] = auto_steps[step_name]

    all_codes = [code for code in step_codes.values() if code]
    counts = {code: all_codes.count(code) for code in ERROR_CODES}

    primary = judge_verdict.get("detected_failure")
    explanation = judge_verdict.get("failure_explanation", "")
    if not primary and auto_failure:
        primary = auto_failure.get("primary_failure")
        explanation = auto_failure.get("failure_explanation", "")

    return {
        "error_analysis": {
            "code_counts": counts,
            "total_errors": len(all_codes),
        },
        "step_failures": {
            **step_codes,
            "primary_failure": primary,
            "failure_explanation": explanation,
        },
    }


def print_detail_report(report: dict) -> None:
    header = report["header"]
    tier = report["agentic_tier"]
    diag = report["diagnostic_metrics"]
    runtime = report["runtime"]
    fmt = report["format"]
    clinical = report["clinical_score"]

    print("\n==============================================================")
    print("  MedAgentsBench -- Report Generation Detail Report")
    tier_str = f" | Tier: {header['tier']}" if header.get("tier") else ""
    print(f"  Agent: {header['agent']} | Task: {header['task']}{tier_str} | {header['date']}")
    print(f"  Model: {header['model']}")
    print("==============================================================")
    print(f"  RESULT: [{tier['rating']}] {tier['description']} ({'PASS' if tier['resolved'] else 'FAIL'})")
    print(f"  Overall: {tier['overall_score']:.4f} | Medal: {tier['medal_name']} ({tier['medal_tier']})")
    print("--------------------------------------------------------------")
    print(
        "  Clinical:"
        f" BLEU1={clinical.get('BLEU_1', 0.0):.4f}"
        f" BLEU2={clinical.get('BLEU_2', 0.0):.4f}"
        f" BLEU3={clinical.get('BLEU_3', 0.0):.4f}"
        f" BLEU4={clinical.get('BLEU_4', 0.0):.4f}"
    )
    print(
        "           "
        f" METEOR={clinical.get('METEOR', 0.0):.4f}"
        f" ROUGE_L={clinical.get('ROUGE_L', 0.0):.4f}"
        f" F1RadGraph={clinical.get('F1RadGraph', 0.0):.4f}"
    )
    print(
        "           "
        f" microP={clinical.get('micro_average_precision', 0.0):.4f}"
        f" microR={clinical.get('micro_average_recall', 0.0):.4f}"
        f" microF1={clinical.get('micro_average_f1', 0.0):.4f}"
    )
    print(
        "           "
        f" obs_f1={diag['observation_f1']:.4f}"
        f" sim={diag['report_similarity']:.4f}"
        f" exact={diag['label_exact_match']:.4f}"
    )
    print(
        "  Runtime:"
        f" {runtime.get('wall_time_s', 0):.1f}s"
        f" | api_calls={runtime.get('api_calls', 0)}"
        f" | code_exec={runtime.get('code_executions', 0)}"
    )
    print(
        "  Format:"
        f" valid={fmt['reports_valid']}"
        f" | completion={fmt['completion_rate']:.2f}"
    )
    print("==============================================================")
