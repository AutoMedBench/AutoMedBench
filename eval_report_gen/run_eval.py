#!/usr/bin/env python3
"""Evaluation entry point for report generation."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregate import STEP_WEIGHTS, build_report
from failure_classifier import classify_failure
from format_checker import check_submission
from report_scorer import score_all


def run_eval(
    gt_dir: str,
    agent_dir: str,
    case_ids: list[str],
    task_config: dict,
    llm_judge: bool = False,
    online_judge: bool = False,
    conversation: dict | None = None,
    judge_kwargs: dict | None = None,
) -> dict:
    format_result = check_submission(
        agent_dir=agent_dir,
        case_ids=case_ids,
        task_config=task_config,
    )

    score_result = score_all(
        pred_dir=agent_dir,
        gt_dir=gt_dir,
        case_ids=case_ids,
        task_config=task_config,
    )

    report = build_report(
        format_result=format_result,
        score_result=score_result,
        clinical_weights=task_config.get("clinical_metric_weights"),
        rating_thresholds=task_config.get("rating_thresholds"),
        step_weights=STEP_WEIGHTS,
    )
    report["failure"] = classify_failure(report)
    report["_per_case"] = score_result.get("per_case", {})

    if llm_judge and conversation:
        from llm_judge import create_judge

        judge = create_judge(online=online_judge, **(judge_kwargs or {}))
        verdict = judge.judge(conversation, report, task_config["task_id"])
        report["llm_judge"] = verdict.to_dict()
        report["step_scores"]["s1"] = verdict.s1_plan_score
        report["step_scores"]["s2"] = verdict.s2_setup_score
        report["step_scores"]["s3"] = verdict.s3_validate_score

        workflow_score = sum(
            STEP_WEIGHTS[key] * (report["step_scores"][key] or 0.0)
            for key in STEP_WEIGHTS
        ) / sum(STEP_WEIGHTS.values())
        report["aggregate"]["agentic_score"] = round(workflow_score, 4)
        report["aggregate"]["active_steps"] = [
            key for key, value in report["step_scores"].items() if value is not None
        ]
        report["aggregate"]["overall_score"] = round(
            0.5 * report["aggregate"]["agentic_score"]
            + 0.5 * report["aggregate"]["clinical_score"],
            4,
        )

    return report


def print_report(name: str, task: str, report: dict) -> None:
    metrics = report["metrics"]
    agg = report["aggregate"]
    fmt = report["format"]
    steps = report["step_scores"]

    print("\n==============================================================")
    print(f"  Agent: {name} | Task: {task}")
    print("==============================================================")
    print(f"  OVERALL: {agg['overall_score']:.4f} [{agg['rating']}] ({'PASS' if agg['resolved'] else 'FAIL'})")
    print(f"  Agentic: {agg['agentic_score']:.4f} | Clinical: {agg['clinical_score']:.4f}")
    print(
        "  Steps: "
        + " | ".join(
            f"{step}={'—' if value is None else f'{value:.3f}'}"
            for step, value in steps.items()
        )
    )
    print(
        f"  Metrics: BLEU1={metrics.get('BLEU_1', 0.0):.4f} "
        f"BLEU2={metrics.get('BLEU_2', 0.0):.4f} "
        f"BLEU3={metrics.get('BLEU_3', 0.0):.4f} "
        f"BLEU4={metrics.get('BLEU_4', 0.0):.4f}"
    )
    print(
        f"           METEOR={metrics.get('METEOR', 0.0):.4f} "
        f"ROUGE_L={metrics.get('ROUGE_L', 0.0):.4f} "
        f"F1RadGraph={metrics.get('F1RadGraph', 0.0):.4f}"
    )
    print(
        f"           obs_f1={metrics['observation_f1']:.4f} "
        f"sim={metrics['report_similarity']:.4f} exact={metrics['label_exact_match']:.4f}"
    )
    print(
        f"  Format: valid={fmt['output_format_valid']} "
        f"completion={fmt['completion_rate']:.2f} medal={metrics['medal_name']}"
    )
    print("==============================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate report-generation outputs")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--cases", required=True, help="Comma-separated case IDs")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--name", default="agent")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--online-judge", action="store_true")
    parser.add_argument("--conversation", default=None)
    args = parser.parse_args()

    from config_io import load_config

    task_config = load_config(args.task_config)
    case_ids = [case.strip() for case in args.cases.split(",") if case.strip()]

    conversation = None
    if args.llm_judge and args.conversation:
        with open(args.conversation, "r", encoding="utf-8") as handle:
            conversation = json.load(handle)

    report = run_eval(
        gt_dir=args.gt_dir,
        agent_dir=args.agent_dir,
        case_ids=case_ids,
        task_config=task_config,
        llm_judge=args.llm_judge,
        online_judge=args.online_judge,
        conversation=conversation,
    )
    print_report(args.name, task_config["task_id"], report)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Saved JSON report -> {args.output_json}")


if __name__ == "__main__":
    main()
