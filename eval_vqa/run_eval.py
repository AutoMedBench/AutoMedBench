#!/usr/bin/env python3
"""Main evaluation entry point for VQA benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregate import build_report
from answer_judge import AnswerJudge
from detail_report import generate_detail_report, print_detail_report
from failure_classifier import classify_failure
from format_checker import check_submission
from inference_verifier import (
    check_postprocess_artefact,
    check_smoke_forward,
    compute_length_finish_rate,
    detect_model_call,
)
from llm_judge import create_judge
from medal_tier import assign_tier
from task_loader import discover_question_ids, load_task_config
from tier_config import get_tier_config
from tool_usage import apply_tool_usage, load_tool_calls, summarize as summarize_tool_usage
from vqa_scorer import score_all


def run_eval(
    gt_dir: str,
    agent_dir: str,
    public_dir: str,
    question_ids: list[str],
    llm_judge: bool = False,
    conversation: dict | None = None,
    tier: str = "lite",
    workspace_dir: str | None = None,
    answer_mode: str = "multiple_choice",
    conversation_path: str | None = None,
    enable_answer_judge: bool = False,
    answer_judge_model: str | None = None,
    answer_judge: AnswerJudge | None = None,
) -> dict:
    format_result = check_submission(
        agent_dir=agent_dir,
        question_ids=question_ids,
        public_dir=public_dir,
        answer_mode=answer_mode,
    )
    if (
        answer_judge is None
        and enable_answer_judge
        and answer_mode == "open_ended"
    ):
        cache_path = os.path.join(
            workspace_dir or agent_dir, "answer_judge_cache.jsonl"
        )
        answer_judge = AnswerJudge.from_env(cache_path=cache_path, model=answer_judge_model)
    score_result = score_all(
        pred_dir=agent_dir,
        gt_dir=gt_dir,
        public_dir=public_dir,
        question_ids=question_ids,
        answer_mode=answer_mode,
        answer_judge=answer_judge,
    )
    medal_result = assign_tier(score_result.get("accuracy", 0.0))

    tool_usage_summary = summarize_tool_usage(
        load_tool_calls(workspace_dir or agent_dir),
        expected_samples=len(question_ids),
    )

    smoke = check_smoke_forward(workspace_dir or agent_dir)
    smoke_forward_passed = smoke["valid"]
    postprocess = check_postprocess_artefact(workspace_dir or agent_dir)
    postprocess_valid = postprocess["valid"]
    conv_path = conversation_path or os.path.join(
        os.path.dirname(workspace_dir or agent_dir), "process", "conversation.json"
    )
    model_call_info = detect_model_call(conv_path)
    run_root = os.path.dirname(workspace_dir or agent_dir)
    length_finish_info = compute_length_finish_rate(os.path.join(run_root, "process"))

    build_kwargs = {
        "format_result": format_result,
        "score_result": score_result,
        "medal_result": medal_result,
        "step_weights": get_tier_config(tier).step_weights,
        "model_call_detected": model_call_info["detected"],
        "model_call_evidence": model_call_info["evidence"],
        "smoke_forward_passed": smoke_forward_passed,
        "postprocess_valid": postprocess_valid,
        "postprocess_info": postprocess,
    }

    def _finalize(r: dict) -> dict:
        _s4_pen = r["metrics"].get("s4_penalties")
        r["tool_usage"] = tool_usage_summary
        r["inference_verifier"] = {
            "smoke_forward": smoke,
            "model_call": model_call_info,
            "postprocess": postprocess,
            "length_finish": length_finish_info,
        }
        r["metrics"]["length_finish_rate"] = length_finish_info["length_finish_rate"]
        r["metrics"]["length_finish_count"] = length_finish_info["length_finish_count"]
        r["step_scores"] = apply_tool_usage(r["step_scores"], tool_usage_summary)
        # S2 is binary (P3) — aggregate already set it; apply_tool_usage may
        # wipe it, so restore from the report metrics.
        s2_components = r["metrics"].get("s2_components", {})
        if s2_components:
            r["step_scores"]["s2"] = round(
                sum(bool(v) for v in s2_components.values()) / 3.0, 4
            )
        if postprocess_valid is False and r["step_scores"].get("s3") is not None:
            r["step_scores"]["s3"] = round(min(float(r["step_scores"]["s3"]), 0.5), 4)
        if _s4_pen and r["step_scores"].get("s4") is not None:
            cap = 1.0
            if not model_call_info["detected"]:
                cap = min(cap, 0.3)
            if any("placeholder_rate" in p for p in _s4_pen):
                cap = min(cap, 0.2)
            if any("real_but_broken" in p for p in _s4_pen):
                cap = min(cap, 0.5)
            r["step_scores"]["s4"] = round(min(float(r["step_scores"]["s4"]), cap), 4)
        return r

    report = _finalize(build_report(**build_kwargs))

    judge_verdict = None
    if llm_judge and conversation:
        judge = create_judge()
        verdict = judge.judge(conversation, report, conversation.get("task", "unknown"))
        judge_verdict = verdict.to_dict()
        judge_steps = {
            "s1": verdict.s1_plan_score,
            "s2": verdict.s2_setup_score,
            "s3": verdict.s3_validate_score,
        }
        report = _finalize(build_report(**build_kwargs, step_scores=judge_steps))
        report["llm_judge"] = judge_verdict

    report["failure"] = classify_failure(report)

    # Surface heuristic fallback — agent_judge backend failed mid-run and the
    # judge silently degraded to F1/yes-no heuristics. Operators need to know
    # so they don't treat the judge-score column as LLM-graded.
    metrics = report.get("metrics", {})
    fb = int(metrics.get("judge_fallback_count", 0) or 0)
    samples = int(metrics.get("judge_samples", 0) or 0)
    if fb > 0:
        sys.stderr.write(
            f"[run_eval] WARNING: answer judge fell back to heuristic on "
            f"{fb}/{samples} sample(s). `accuracy_judge` is partially heuristic; "
            f"inspect judge_rationale entries tagged 'fallback:' for affected qids.\n"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="MedAgentsBench VQA evaluator")
    parser.add_argument("--gt-dir", required=True, help="Ground truth directory")
    parser.add_argument("--agent-dir", required=True, help="Agent outputs directory")
    parser.add_argument("--public-dir", required=True, help="Public staged question directory")
    parser.add_argument("--task", default="medxpertqa-mm-vqa-task")
    parser.add_argument("--tier", default="lite", choices=("lite", "standard"))
    parser.add_argument("--question-ids", default=None, help="Comma-separated question IDs")
    parser.add_argument("--split", default=None, help="Optional split filter when auto-discovering question IDs")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument(
        "--enable-answer-judge",
        action="store_true",
        help="Use LLM-as-judge for open-ended answer scoring (BUG-038).",
    )
    parser.add_argument(
        "--answer-judge-model",
        default=None,
        help="Override judge model (default: env ANSWER_JUDGE_MODEL or anthropic/claude-haiku-4.5).",
    )
    parser.add_argument("--conversation", default=None, help="Optional conversation JSON for judge scoring")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--name", default="agent")
    parser.add_argument("--model", default="unknown")
    args = parser.parse_args()

    if args.question_ids:
        question_ids = [item.strip() for item in args.question_ids.split(",") if item.strip()]
    else:
        question_ids = discover_question_ids(args.task, split=args.split)

    conversation = None
    if args.llm_judge:
        if not args.conversation:
            sys.exit("--conversation is required with --llm-judge")
        with open(args.conversation, "r", encoding="utf-8") as handle:
            conversation = json.load(handle)

    try:
        task_cfg = load_task_config(args.task)
        task_answer_mode = task_cfg.get("answer_mode", "multiple_choice")
    except Exception:
        task_answer_mode = "multiple_choice"

    report = run_eval(
        gt_dir=args.gt_dir,
        agent_dir=args.agent_dir,
        public_dir=args.public_dir,
        question_ids=question_ids,
        llm_judge=args.llm_judge,
        conversation=conversation,
        tier=args.tier,
        answer_mode=task_answer_mode,
        enable_answer_judge=(
            args.enable_answer_judge
            or os.environ.get("VQA_ANSWER_JUDGE") in ("1", "true", "True")
        ),
        answer_judge_model=args.answer_judge_model,
    )
    detail_report = generate_detail_report(
        eval_report=report,
        runtime={"question_count": len(question_ids)},
        agent_name=args.name,
        model=args.model,
        task=args.task,
        judge_verdict=report.get("llm_judge"),
        tier=args.tier,
    )
    print_detail_report(detail_report)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nJSON report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
