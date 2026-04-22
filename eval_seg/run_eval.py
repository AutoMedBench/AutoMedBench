#!/usr/bin/env python3
"""Main evaluation entry point for segmentation benchmark."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format_checker import check_submission
from dice_scorer import score_all, score_all_multiclass
from medal_tier import assign_tier
from aggregate import build_report
from failure_classifier import classify_failure


def run_eval(gt_dir: str, agent_dir: str,
             public_dir: str, patient_ids: list,
             llm_judge: bool = False, online_judge: bool = False,
             conversation: dict = None, judge_kwargs: dict = None,
             task_cfg: dict = None) -> dict:
    """Run full evaluation pipeline.

    Args:
        llm_judge: if True, run LLM-as-judge after deterministic scoring
        online_judge: if True, use Claude Opus 4.6; else local DeepSeek model
        conversation: agent conversation dict (required for LLM judge)
        judge_kwargs: extra kwargs for the judge constructor
        task_cfg: task configuration dict. When
            ``task_cfg['task_type'] == 'multiclass'`` the pipeline uses the
            multi-class scorer (per-tissue Dice + macro-mean). Otherwise
            falls back to binary organ/lesion scoring.
    """
    task_cfg = task_cfg or {}
    is_multiclass = task_cfg.get("task_type") == "multiclass"

    # Step 1: Format check
    format_result = check_submission(
        agent_dir=agent_dir,
        patient_ids=patient_ids,
        public_dir=public_dir,
        task_cfg=task_cfg,
    )

    # Step 2: Dice scoring
    if is_multiclass:
        dice_result = score_all_multiclass(
            pred_dir=agent_dir,
            gt_dir=gt_dir,
            patient_ids=patient_ids,
            tissue_labels=task_cfg.get("tissue_labels", {}),
            output_filename=task_cfg.get("output_filename", "dseg.nii.gz"),
        )
        score_for_tier = dice_result.get("macro_mean_dice", 0.0)
    else:
        dice_result = score_all(
            pred_dir=agent_dir,
            gt_dir=gt_dir,
            patient_ids=patient_ids,
        )
        score_for_tier = dice_result.get("mean_lesion_dice", 0.0)

    # Step 3: Medal tier (Dice-only)
    medal_result = assign_tier(score_for_tier)

    # Step 4: Aggregate
    report = build_report(format_result, dice_result, medal_result,
                          task_cfg=task_cfg)

    # Step 6: Failure classification (only for failed runs)
    failure = classify_failure(report)
    report["failure"] = failure

    # Attach per-patient dice detail for the detail report
    report["_dice_per_patient"] = dice_result.get("per_patient", {})

    # Step 7: LLM-as-Judge (optional)
    if llm_judge and conversation:
        from llm_judge import create_judge
        kw = judge_kwargs or {}
        judge = create_judge(online=online_judge, **kw)
        task = conversation.get("task", "unknown")
        verdict = judge.judge(conversation, report, task)
        report["llm_judge"] = verdict.to_dict()

    return report


def print_report(name: str, task: str, report: dict):
    """Pretty-print evaluation report."""
    m = report["metrics"]
    a = report["aggregate"]
    f = report["format"]
    ss = report["step_scores"]

    print(f"\n{'='*60}")
    print(f"  Agent: {name}  |  Task: {task}")
    print(f"{'='*60}")

    # Headline
    resolved_tag = "PASS" if a["resolved"] else "FAIL"
    print(f"  OVERALL SCORE:  {a['overall_score']:.4f}  [{a['rating']}]  ({resolved_tag})")
    print(f"{'─'*60}")

    # Sub-scores
    print(f"  Agentic score:  {a['agentic_score']:.4f}  (active: {', '.join(a['active_steps'])})")
    print(f"  Clinical score: {a['clinical_score']:.4f}")

    # Step detail
    step_parts = []
    for sn in ["s1", "s2", "s3", "s4", "s5"]:
        v = ss[sn]
        step_parts.append(f"{sn}={'—' if v is None else f'{v:.3f}'}")
    print(f"  Steps:   {' | '.join(step_parts)}")

    # Metrics
    print(f"{'─'*60}")
    organ_str = f"{m['organ_dice']:.4f}" if isinstance(m['organ_dice'], float) else str(m['organ_dice'])
    print(f"  Dice:    lesion={m['lesion_dice']:.4f}  organ={organ_str}")
    print(f"  Result:  {m['medal_name']} (tier {m['medal_tier']})")
    print(f"  Format:  sub={f['submission_format_valid']}  masks={f['output_format_valid']}")
    print(f"  Progress rate: {a['progress_rate']:.2f}")

    fail = report.get("failure")
    if fail:
        code = fail.get('primary_failure', '?')
        explanation = fail.get('failure_explanation', fail.get('root_cause_explanation', ''))
        print(f"  FAILURE: {code} — {explanation}")
    print(f"{'='*60}")


def print_judge_verdict(report: dict):
    """Print LLM judge verdict if present."""
    jv = report.get("llm_judge")
    if not jv:
        return
    if "error" in jv:
        print(f"\n  LLM JUDGE ERROR: {jv['error']}")
        return

    def _step_line(label, score_key, rationale_key, failure_key=None):
        score = jv.get(score_key, 0)
        rat = jv.get(rationale_key, "")
        fail = jv.get(failure_key, "") if failure_key else ""
        fail_tag = f"  {fail}" if fail else "     "
        return f"  {label:<14} {score:.2f}{fail_tag}  — {rat}"

    print(f"\n{'─'*60}")
    print(f"  LLM JUDGE VERDICT  ({jv.get('judge_backend', '?')})")
    print(f"  Model: {jv.get('judge_model', '?')}")
    print(f"{'─'*60}")
    print(_step_line("S1 Plan:",     "s1_plan_score",      "s1_rationale",      "s1_failure"))
    print(_step_line("S2 Setup:",    "s2_setup_score",     "s2_rationale",      "s2_failure"))
    print(_step_line("S3 Validate:", "s3_validate_score",  "s3_rationale",      "s3_failure"))
    print(_step_line("S4 Inference:","s4_inference_score",  "s4_rationale",      "s4_failure"))
    print(_step_line("S5 Submit:",   "s5_submit_score",     "s5_rationale",      "s5_failure"))
    print(_step_line("Tool Calling:","tool_calling_score",  "tool_calling_rationale"))
    print(_step_line("Clinical:",    "clinical_reasoning_score", "clinical_reasoning_rationale"))
    if jv.get("detected_failure"):
        print(f"  Failure:       {jv['detected_failure']} — {jv.get('failure_explanation', '')}")
    print(f"  Summary:       {jv.get('overall_rationale', '')}")
    print(f"  Latency:       {jv.get('judge_latency_s', 0):.1f}s  "
          f"Tokens: {jv.get('input_tokens', 0)}in/{jv.get('output_tokens', 0)}out")
    print(f"{'─'*60}")


def main():
    parser = argparse.ArgumentParser(description="MedAgentsBench Segmentation Evaluator")
    parser.add_argument("--gt-dir", required=True, help="Ground truth masks directory")
    parser.add_argument("--gt-csv", required=True, help="Ground truth CSV")
    parser.add_argument("--agent-dir", required=True, help="Agent outputs directory")
    parser.add_argument("--decision-csv", default=None, help="Agent decision CSV (optional, no longer required)")
    parser.add_argument("--public-dir", required=True, help="Public data directory (CT)")
    parser.add_argument("--patients", required=True, help="Comma-separated patient IDs")
    parser.add_argument("--task", default="unknown", help="Task name (kidney/liver)")
    parser.add_argument("--name", default="agent", help="Agent name for display")
    parser.add_argument("--output-json", default=None, help="Optional: save report as JSON")
    # LLM Judge options
    parser.add_argument("--llm-judge", action="store_true",
                        help="Run LLM-as-judge evaluation")
    parser.add_argument("--online-judge", action="store_true",
                        help="Use Claude Opus 4.6 (online) instead of local DeepSeek model")
    parser.add_argument("--conversation", default=None,
                        help="Path to conversation.json (required for --llm-judge)")
    parser.add_argument("--judge-model-path", default=None,
                        help="Local model path for offline judge")
    parser.add_argument("--judge-vllm-url", default=None,
                        help="URL of running vLLM server for offline judge")
    args = parser.parse_args()

    patient_ids = [p.strip() for p in args.patients.split(",")]

    # Load conversation if LLM judge requested
    conversation = None
    if args.llm_judge:
        if not args.conversation:
            sys.exit("--conversation is required when using --llm-judge")
        with open(args.conversation) as f:
            conversation = json.load(f)

    judge_kwargs = {}
    if args.judge_model_path:
        judge_kwargs["model_path"] = args.judge_model_path
    if args.judge_vllm_url:
        judge_kwargs["base_url"] = args.judge_vllm_url

    report = run_eval(
        gt_dir=args.gt_dir,
        agent_dir=args.agent_dir,
        public_dir=args.public_dir,
        patient_ids=patient_ids,
        llm_judge=args.llm_judge,
        online_judge=args.online_judge,
        conversation=conversation,
        judge_kwargs=judge_kwargs,
    )

    print_report(args.name, args.task, report)
    print_judge_verdict(report)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
