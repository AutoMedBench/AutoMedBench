#!/usr/bin/env python3
"""Test script for both offline and online LLM judge backends.

Usage:
    # Test offline judge (requires GPU + model weights)
    python test_judge.py --offline

    # Test online judge (requires ANTHROPIC_API_KEY)
    python test_judge.py --online

    # Test both
    python test_judge.py --offline --online

    # Test with a real conversation from a previous run
    python test_judge.py --offline --conversation runs/.../conversation.json \
                         --eval-report runs/.../detail_report.json
"""

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Add pip_packages to path for vllm/anthropic
PIP_PKG_DIR = os.path.join(PROJECT_DIR, "pip_packages")
if os.path.isdir(PIP_PKG_DIR):
    sys.path.insert(0, PIP_PKG_DIR)

sys.path.insert(0, SCRIPT_DIR)

from llm_judge import (
    OfflineJudge, OnlineJudge, create_judge,
    build_judge_user_prompt, _parse_judge_response, JudgeVerdict,
)


# ---- Synthetic test data ----

SYNTHETIC_CONVERSATION = {
    "agent": "test-agent",
    "model": "test-model",
    "task": "kidney",
    "messages": [
        {"role": "user", "content": "Begin the kidney lesion segmentation task. Process all patients."},
        {"role": "assistant", "content": (
            "I'll start by listing all patients and then process each one. "
            "Let me first plan my approach: I'll use the available segmentation tools "
            "to generate organ and lesion masks for each patient, then review the statistics "
            "to make clinical decisions."
        )},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "list_patients", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tc1",
         "content": '{"patient_ids": ["BDMAP_00000001", "BDMAP_00000005"], "count": 2}'},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc2", "type": "function",
             "function": {"name": "get_patient_ct", "arguments": '{"patient_id": "BDMAP_00000001"}'}}
        ]},
        {"role": "tool", "tool_call_id": "tc2",
         "content": '{"patient_id": "BDMAP_00000001", "ct_path": "/data/ct.nii.gz", "shape": [512,512,200], "spacing_mm": [0.78, 0.78, 1.5]}'},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc3", "type": "function",
             "function": {"name": "run_organ_segmentation", "arguments": '{"patient_id": "BDMAP_00000001"}'}}
        ]},
        {"role": "tool", "tool_call_id": "tc3",
         "content": '{"mask_path": "/out/organ.nii.gz", "foreground_voxels": 190438, "coverage_ratio": 0.0036}'},
        {"role": "assistant", "content": (
            "The organ segmentation shows 190,438 foreground voxels which is a reasonable "
            "kidney volume. Now let me run lesion detection."
        )},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc4", "type": "function",
             "function": {"name": "run_lesion_detection", "arguments": '{"patient_id": "BDMAP_00000001"}'}}
        ]},
        {"role": "tool", "tool_call_id": "tc4",
         "content": '{"mask_path": "/out/lesion.nii.gz", "foreground_voxels": 0, "coverage_ratio": 0.0}'},
        {"role": "assistant", "content": (
            "The lesion mask is empty (0 foreground voxels), indicating no lesion detected. "
            "I'll submit lesion_present=0 for this patient."
        )},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc5", "type": "function",
             "function": {"name": "submit_decision", "arguments": '{"patient_id": "BDMAP_00000001", "lesion_present": 0}'}}
        ]},
        {"role": "tool", "tool_call_id": "tc5",
         "content": '{"patient_id": "BDMAP_00000001", "lesion_present": 0, "decisions_so_far": 1, "remaining": 1}'},
    ],
}

SYNTHETIC_EVAL_REPORT = {
    "metrics": {
        "lesion_dice": 0.72,
        "organ_dice": 0.93,
        "medal_tier": 2,
        "medal_name": "good",
    },
    "aggregate": {
        "overall_score": 0.95,
        "rating": "A",
        "resolved": True,
        "workflow_score": 0.96,
        "clinical_score": 0.94,
    },
    "format": {
        "submission_format_valid": True,
        "decision_csv_valid": True,
        "output_format_valid": True,
    },
    "step_scores": {"s1": None, "s2": None, "s3": None, "s4": 1.0, "s5": 0.91},
    "failure": None,
}


def test_prompt_building():
    """Test that the prompt builder works correctly."""
    print("\n[Test] Prompt building...")
    prompt = build_judge_user_prompt(SYNTHETIC_CONVERSATION, SYNTHETIC_EVAL_REPORT, "kidney")
    assert "test-agent" in prompt
    assert "kidney" in prompt
    assert "0.72" in prompt  # lesion dice
    assert "good" in prompt  # result tier
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  First 200 chars: {prompt[:200]}...")
    print("  PASS")
    return True


def test_json_parsing():
    """Test JSON extraction from various response formats."""
    print("\n[Test] JSON parsing...")

    # Direct JSON
    r1 = _parse_judge_response('{"s1_plan_score": 0.8, "s1_rationale": "test"}')
    assert r1["s1_plan_score"] == 0.8, f"Direct JSON failed: {r1}"

    # Markdown fence
    r2 = _parse_judge_response('Here is my evaluation:\n```json\n{"s1_plan_score": 0.5}\n```')
    assert r2["s1_plan_score"] == 0.5, f"Markdown fence failed: {r2}"

    # With <think> block (reasoning model)
    raw = '<think>Let me analyze this...</think>\n{"s1_plan_score": 0.7, "s2_setup_score": 0.9}'
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    r3 = _parse_judge_response(cleaned)
    assert r3["s1_plan_score"] == 0.7, f"Think block failed: {r3}"

    print("  All parsing tests PASS")
    return True


def test_verdict_construction():
    """Test JudgeVerdict dataclass."""
    print("\n[Test] Verdict construction...")
    v = JudgeVerdict(
        s1_plan_score=0.8, s1_rationale="Good plan",
        judge_model="test", judge_backend="test",
    )
    d = v.to_dict()
    assert d["s1_plan_score"] == 0.8
    assert d["s1_rationale"] == "Good plan"
    assert d["judge_model"] == "test"
    print("  PASS")
    return True


def test_offline_judge(conversation, eval_report, task, model_path=None, vllm_url=None):
    """Test the offline (local) judge."""
    print("\n" + "="*60)
    print("  Testing OFFLINE Judge (DeepSeek-R1-Distill-Qwen-32B)")
    print("="*60)

    if not model_path:
        model_path = os.path.join(PROJECT_DIR, "models", "DeepSeek-R1-Distill-Qwen-32B")
    if not os.path.isdir(model_path) and not vllm_url:
        print(f"  SKIP: Model not found at {model_path} and no --vllm-url provided")
        return False

    judge = OfflineJudge(model_path=model_path, base_url=vllm_url)
    print(f"  Model: {judge.model_path}")
    if vllm_url:
        print(f"  vLLM server: {vllm_url}")

    t0 = time.time()
    verdict = judge.judge(conversation, eval_report, task)
    elapsed = time.time() - t0

    _print_verdict(verdict, elapsed)
    _validate_verdict(verdict)
    return True


def test_online_judge(conversation, eval_report, task):
    """Test the online (Claude) judge."""
    print("\n" + "="*60)
    print("  Testing ONLINE Judge (Claude Opus 4.6)")
    print("="*60)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  SKIP: ANTHROPIC_API_KEY not set")
        return False

    judge = OnlineJudge()
    print(f"  Model: {judge.model}")

    t0 = time.time()
    verdict = judge.judge(conversation, eval_report, task)
    elapsed = time.time() - t0

    _print_verdict(verdict, elapsed)
    _validate_verdict(verdict)
    return True


def _print_verdict(verdict, elapsed):
    """Pretty-print a verdict."""
    print(f"\n  Results (total {elapsed:.1f}s):")
    print(f"    S1 Plan:      {verdict.s1_plan_score:.2f}  — {verdict.s1_rationale[:80]}")
    print(f"    S2 Setup:     {verdict.s2_setup_score:.2f}  — {verdict.s2_rationale[:80]}")
    print(f"    S3 Validate:  {verdict.s3_validate_score:.2f}  — {verdict.s3_rationale[:80]}")
    print(f"    Tool Calling: {verdict.tool_calling_score:.2f}  — {verdict.tool_calling_rationale[:80]}")
    print(f"    Clinical:     {verdict.clinical_reasoning_score:.2f}  — {verdict.clinical_reasoning_rationale[:80]}")
    if verdict.detected_failure:
        print(f"    Failure:      {verdict.detected_failure} — {verdict.failure_explanation[:80]}")
    else:
        print(f"    Failure:      None")
    print(f"    Summary:      {verdict.overall_rationale[:120]}")
    print(f"    Backend:      {verdict.judge_backend}")
    print(f"    Latency:      {verdict.judge_latency_s:.1f}s")
    print(f"    Tokens:       {verdict.input_tokens} in / {verdict.output_tokens} out")


def _validate_verdict(verdict):
    """Basic validation of verdict scores."""
    errors = []
    for field in ["s1_plan_score", "s2_setup_score", "s3_validate_score",
                  "tool_calling_score", "clinical_reasoning_score"]:
        v = getattr(verdict, field)
        if not (0.0 <= v <= 1.0):
            errors.append(f"  {field}={v} out of range [0,1]")
    if not verdict.judge_backend:
        errors.append("  judge_backend is empty")
    if not verdict.judge_model:
        errors.append("  judge_model is empty")

    if errors:
        print("\n  VALIDATION ERRORS:")
        for e in errors:
            print(f"    {e}")
    else:
        print("\n  VALIDATION: PASS (all scores in [0,1], metadata populated)")


def main():
    parser = argparse.ArgumentParser(description="Test LLM judge backends")
    parser.add_argument("--offline", action="store_true", help="Test offline judge")
    parser.add_argument("--online", action="store_true", help="Test online judge")
    parser.add_argument("--conversation", default=None,
                        help="Path to real conversation.json")
    parser.add_argument("--eval-report", default=None,
                        help="Path to real eval/detail report JSON")
    parser.add_argument("--task", default="kidney", choices=["kidney", "liver"])
    parser.add_argument("--model-path", default=None,
                        help="Local model path for offline judge")
    parser.add_argument("--vllm-url", default=None,
                        help="URL of running vLLM server")
    args = parser.parse_args()

    if not args.offline and not args.online:
        print("Specify --offline, --online, or both")
        sys.exit(1)

    # Load data
    if args.conversation:
        with open(args.conversation) as f:
            conversation = json.load(f)
        task = conversation.get("task", args.task)
    else:
        conversation = SYNTHETIC_CONVERSATION
        task = args.task

    if args.eval_report:
        with open(args.eval_report) as f:
            eval_report = json.load(f)
    else:
        eval_report = SYNTHETIC_EVAL_REPORT

    # Run unit tests
    print("="*60)
    print("  Unit Tests")
    print("="*60)
    test_prompt_building()
    test_json_parsing()
    test_verdict_construction()

    # Run integration tests
    results = {}
    if args.offline:
        results["offline"] = test_offline_judge(
            conversation, eval_report, task,
            model_path=args.model_path, vllm_url=args.vllm_url,
        )
    if args.online:
        results["online"] = test_online_judge(
            conversation, eval_report, task,
        )

    # Summary
    print("\n" + "="*60)
    print("  Test Summary")
    print("="*60)
    for name, passed in results.items():
        status = "PASS" if passed else "SKIP/FAIL"
        print(f"  {name}: {status}")
    print("="*60)


if __name__ == "__main__":
    main()
