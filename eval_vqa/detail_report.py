#!/usr/bin/env python3
"""Generate concise VQA detail reports."""

from __future__ import annotations

from datetime import datetime, timezone


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
    aggregate = eval_report.get("aggregate", {})
    steps = eval_report.get("step_scores", {})

    header = {
        "agent": agent_name,
        "model": model,
        "task": task,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    if tier:
        header["tier"] = tier

    report = {
        "header": header,
        "runtime": runtime,
        "task_metrics": {
            "accuracy": metrics.get("accuracy", 0.0),
            "completion_rate": metrics.get("completion_rate", 0.0),
            "parse_rate": metrics.get("parse_rate", 0.0),
            "valid_output_rate": metrics.get("valid_output_rate", 0.0),
            "counts": metrics.get("counts", {}),
            "breakdown": metrics.get("breakdown", {}),
        },
        "agentic_score": {
            "score": aggregate.get("agentic_score", 0.0),
            "step_scores": steps,
            "active_steps": aggregate.get("active_steps", []),
            "progress_rate": aggregate.get("progress_rate", 0.0),
        },
        "task_score": {
            "score": aggregate.get("task_score", 0.0),
            "medal_tier": metrics.get("medal_tier", 0),
            "medal_name": metrics.get("medal_name", "fail"),
        },
        "agentic_tier": {
            "rating": aggregate.get("rating", "F"),
            "resolved": aggregate.get("resolved", False),
            "overall_score": aggregate.get("overall_score", 0.0),
            "fail_rate": aggregate.get("fail_rate", 0.0),
        },
        "format": eval_report.get("format", {}),
        "failure": eval_report.get("failure"),
        "tool_calls": tool_summary or {},
    }
    if judge_verdict:
        report["llm_judge"] = judge_verdict
    return report


def print_detail_report(report: dict) -> None:
    header = report["header"]
    runtime = report.get("runtime", {})
    task_metrics = report["task_metrics"]
    agentic = report["agentic_score"]
    tier = report["agentic_tier"]

    print("\n" + "=" * 64)
    tier_text = f"  |  Tier: {header['tier']}" if "tier" in header else ""
    print(f"  MedAgentsBench VQA Detail Report")
    print(f"  Agent: {header['agent']}  |  Task: {header['task']}{tier_text}")
    print(f"  Model: {header['model']}  |  {header['date']}")
    print("=" * 64)
    print(
        f"  Overall: {tier['overall_score']:.4f}  [{tier['rating']}]"
        f"  {'PASS' if tier['resolved'] else 'FAIL'}"
    )
    print(
        f"  Agentic: {agentic['score']:.4f}  |  "
        f"Task: {report['task_score']['score']:.4f} ({report['task_score']['medal_name']})"
        f"  |  Fail%: {tier.get('fail_rate', 0.0)*100:.1f}%"
    )
    print(
        f"  Accuracy={task_metrics['accuracy']:.4f}  "
        f"Completion={task_metrics['completion_rate']:.4f}  "
        f"Parse={task_metrics['parse_rate']:.4f}  "
        f"Valid={task_metrics['valid_output_rate']:.4f}"
    )
    if isinstance(runtime.get("wall_time_s"), (int, float)):
        print(f"  Runtime: wall={runtime['wall_time_s']:.4f}s  questions={runtime.get('question_count', 0)}")
    phase_summary = runtime.get("phase_summary", {})
    if phase_summary:
        phase_parts = [
            f"{name}={details.get('duration_s', 0):.4f}s"
            for name, details in phase_summary.items()
        ]
        print(f"  Phases: {', '.join(phase_parts)}")
    print(f"  Step scores: {agentic['step_scores']}")
    if report.get("failure"):
        print(f"  Failure: {report['failure']['primary_failure']}  {report['failure']['failure_explanation']}")
    print("=" * 64)
