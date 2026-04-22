#!/usr/bin/env python3
"""LLM-as-Judge for MedAgentsBench segmentation benchmark.

Two backends:
  - OfflineJudge: local DeepSeek-R1-Distill-Qwen-32B via vLLM
  - OnlineJudge:  Claude Opus 4.7 via NVIDIA Inference API

Both produce identical structured JSON verdicts that plug into the
eval pipeline alongside the deterministic scorer.
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional

# =====================================================================
# Judge verdict schema
# =====================================================================

@dataclass
class JudgeVerdict:
    """Structured output from the LLM judge."""
    # S1: Plan quality — 6 binary sub-criteria averaged
    s1a_plan_md: int = 0
    s1b_plan_pipeline: int = 0
    s1c_lesion_model: int = 0
    s1d_researched_3: int = 0
    s1e_plan_plot: int = 0
    s1f_plot_pipeline: int = 0
    s1_plan_score: float = 0.0  # computed: avg of s1a-s1f
    s1_rationale: str = ""
    s1_failure: Optional[str] = None
    # S2: Setup quality — 5 binary sub-criteria averaged
    s2a_checkpoint_downloaded: int = 0
    s2b_compatibility_check: int = 0
    s2c_env_setup_success: int = 0
    s2d_env_fail_within_5: int = 0
    s2e_model_loaded: int = 0
    s2_setup_score: float = 0.0  # computed: avg of s2a-s2e
    s2_rationale: str = ""
    s2_failure: Optional[str] = None
    # S3: Validation quality (0-1)
    s3_validate_score: float = 0.0
    s3_rationale: str = ""
    s3_failure: Optional[str] = None
    # S4: Inference quality (0-1)
    s4_inference_score: float = 0.0
    s4_rationale: str = ""
    s4_failure: Optional[str] = None
    # S5: Submission quality (0-1)
    s5_submit_score: float = 0.0
    s5_rationale: str = ""
    s5_failure: Optional[str] = None
    # Tool-calling correctness (0-1)
    tool_calling_score: float = 0.0
    tool_calling_rationale: str = ""
    # Clinical reasoning quality (0-1)
    clinical_reasoning_score: float = 0.0
    clinical_reasoning_rationale: str = ""
    # Overall qualitative assessment
    overall_rationale: str = ""
    # Primary failure (e.g. "S1:E1" or None) — earliest root cause
    detected_failure: Optional[str] = None
    failure_explanation: str = ""
    # Metadata
    judge_model: str = ""
    judge_backend: str = ""
    judge_latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def step_failures(self) -> dict:
        """Return per-step failure codes (None if step passed)."""
        return {
            "s1": self.s1_failure,
            "s2": self.s2_failure,
            "s3": self.s3_failure,
            "s4": self.s4_failure,
            "s5": self.s5_failure,
        }


# =====================================================================
# Rubric prompt (shared by both backends)
# =====================================================================

# Binary organ+lesion rubric (kidney, liver, pancreas — CT).
JUDGE_SYSTEM_PROMPT = """\
You are an expert medical AI evaluator. You assess how well an AI agent \
performed a medical image segmentation task (organ + lesion segmentation \
from CT scans).

You will receive:
1. The agent's conversation trace (system prompt, messages, tool calls)
2. Quantitative evaluation results (Dice scores)
3. The task description and expected workflow (S1-S5)

## Evaluation Rubric

Score each step using ONLY the discrete values listed below (0.0, 0.5, or 1.0). \
Do NOT use intermediate values like 0.3, 0.6, 0.8, etc. You MUST pick exactly \
one of the three allowed values for each step score.

### S1 — Plan Quality (6 binary sub-criteria, each 0 or 1)
Evaluate EACH of the following independently (0 = no, 1 = yes):
  s1a_plan_md:          Did the agent create a plan.md file?
  s1b_plan_pipeline:    Does plan.md contain clear pipeline instructions? (0 if no plan.md)
  s1c_lesion_model:     Did the plan choose a LESION/TUMOR segmentation model
                        (not just a general organ segmenter)? (0 if no plan.md)
  s1d_researched_3:     Did the agent research at least 3 distinct models in the conversation?
                        (For LITE tier: default to 1 — agent is given the model.)
  s1e_plan_plot:        Did the agent create a plan visualization (plan.png)?
                        (For LITE tier: default to 1 — not required.)
  s1f_plot_pipeline:    Does the plot contain a clear pipeline diagram? (0 if no plot)
                        (For LITE tier: default to 1 — not required.)

Final s1_plan_score = average of s1a + s1b + s1c + s1d + s1e + s1f

### S2 — Setup Quality (6 binary sub-criteria, each 0 or 1)
Evaluate EACH of the following independently (0 = no, 1 = yes):
  s2a_checkpoint_downloaded: Did the agent successfully download model checkpoint/weights?
  s2b_compatibility_check:   Did the agent check input format compatibility (spacing, shape, dtype)?
  s2c_env_setup_success:     Did the environment setup succeed (venv created, packages installed)?
  s2d_env_fail_within_5:     Did the agent resolve env issues within 5 attempts? (1 if ≤5 failures or no failures, 0 if >5)
  s2e_model_loaded:          Was the model loaded into GPU memory and confirmed working?

Final s2_setup_score = average of s2a + s2b + s2c + s2d + s2e

### S3 — Validation Quality (MUST be 0.0, 0.5, or 1.0)
- 1.0: Tested on one patient first, verified output shape/values, then proceeded
- 0.5: Some validation but incomplete
- 0.0: No validation; ran batch immediately or never verified outputs

### S4 and S5 — DO NOT SCORE
S4 (Inference) and S5 (Submit) are scored deterministically by the benchmark
harness based on output file completeness and format checks. Do NOT provide
s4_inference_score, s5_submit_score, s4_failure, or s5_failure in your
response — they will be ignored. Set them to 0.0 and null respectively.

### Tool Calling Correctness (0-1)
- 1.0: Tools called in correct order, correct arguments, no wasted calls
- 0.5: Some incorrect arguments or missing tool calls
- 0.0: Did not use tools or completely wrong tool sequence

### Clinical Reasoning Quality (0-1)
- 1.0: Carefully examined mask statistics, made justified lesion decisions
- 0.5: Made decisions but with limited justification
- 0.0: Random or unjustified decisions

## Error Codes (E1-E5)
For EACH step that failed, assign exactly one error code (null if step passed). \
These describe the ROOT CAUSE of failure:
- E1: Hallucination — agent fabricated non-existent models, APIs, repos, \
packages, or functions that do not exist
- E2: Resource error — GPU OOM, execution timeout, download failure, network error
- E3: Logic error — code ran without crashing but produced incorrect results \
(wrong label mapping, orientation mismatch, bad preprocessing, wrong thresholds)
- E4: Code error — Python/bash runtime errors (syntax, import, type, exception)
- E5: Format error — output does not meet spec (wrong shape, not binary, \
missing files, malformed CSV)

A step can fail with any error code. For example, S1 could fail with E1 \
(hallucinated a model) or E4 (search code crashed). S4 could fail with E2 \
(OOM) or E3 (wrong labels).

Also identify the primary (earliest) failure step and its error code as \
"detected_failure".

## Output Format
Respond with ONLY a JSON object (no markdown, no extra text).
CRITICAL: s1 sub-scores (s1a-s1f) and s2 sub-scores (s2a-s2e) MUST be exactly 0 or 1.
s3 scores MUST be exactly 0.0, 0.5, or 1.0. Tool-calling and clinical-reasoning
scores may be any float 0-1.
{
  "s1a_plan_md": <0 | 1>,
  "s1b_plan_pipeline": <0 | 1>,
  "s1c_lesion_model": <0 | 1>,
  "s1d_researched_3": <0 | 1>,
  "s1e_plan_plot": <0 | 1>,
  "s1f_plot_pipeline": <0 | 1>,
  "s1_rationale": "<1-2 sentences>",
  "s1_failure": "<E1-E5 or null>",
  "s2a_checkpoint_downloaded": <0 | 1>,
  "s2b_compatibility_check": <0 | 1>,
  "s2c_env_setup_success": <0 | 1>,
  "s2d_env_fail_within_5": <0 | 1>,
  "s2e_model_loaded": <0 | 1>,
  "s2_rationale": "<1-2 sentences>",
  "s2_failure": "<E1-E5 or null>",
  "s3_validate_score": <0.0 | 0.5 | 1.0>,
  "s3_rationale": "<1-2 sentences>",
  "s3_failure": "<E1-E5 or null>",
  "s4_inference_score": <0.0 | 0.5 | 1.0>,
  "s4_rationale": "<1-2 sentences>",
  "s4_failure": "<E1-E5 or null>",
  "s5_submit_score": <0.0 | 0.5 | 1.0>,
  "s5_rationale": "<1-2 sentences>",
  "s5_failure": "<E1-E5 or null>",
  "tool_calling_score": <float 0-1>,
  "tool_calling_rationale": "<1-2 sentences>",
  "clinical_reasoning_score": <float 0-1>,
  "clinical_reasoning_rationale": "<1-2 sentences>",
  "overall_rationale": "<2-3 sentence summary>",
  "detected_failure": "<e.g. 'S1:E1' or null — earliest step:code pair>",
  "failure_explanation": "<explanation or empty string>"
}
"""


# Multi-class / multi-tissue rubric (e.g. FeTA fetal brain MRI).
# JSON schema is IDENTICAL to the binary rubric so downstream parsing and
# the JudgeVerdict dataclass are unchanged; only the TEXT framing of s1c
# and the metric labels change. The `s1c_lesion_model` field name is kept
# for schema compatibility but is repurposed to mean "correct label scheme
# coverage" in this rubric.
JUDGE_SYSTEM_PROMPT_MULTICLASS = """\
You are an expert medical AI evaluator. You assess how well an AI agent \
performed a MULTI-TISSUE medical image segmentation task (a single \
multi-class label map per scan, not organ+lesion binary masks). The final \
clinical metric is the MEAN Dice across foreground tissue classes.

You will receive:
1. The agent's conversation trace (system prompt, messages, tool calls)
2. Quantitative evaluation results (per-tissue Dice, macro-mean Dice)
3. The task description and expected workflow (S1-S5)

## Evaluation Rubric

Score each step using ONLY the discrete values listed below (0.0, 0.5, or 1.0). \
Do NOT use intermediate values like 0.3, 0.6, 0.8, etc. You MUST pick exactly \
one of the three allowed values for each step score.

### S1 — Plan Quality (6 binary sub-criteria, each 0 or 1)
Evaluate EACH of the following independently (0 = no, 1 = yes):
  s1a_plan_md:          Did the agent create a plan.md file?
  s1b_plan_pipeline:    Does plan.md contain clear pipeline instructions? (0 if no plan.md)
  s1c_lesion_model:     [MULTICLASS — field name is legacy] Did the plan identify a model
                        whose label scheme covers ALL target foreground tissue classes,
                        and (if a remap is needed) document the source→target label mapping?
                        Score 0 if the plan settles on a model that clearly cannot produce
                        all target tissues, or omits the label-mapping discussion. (0 if no plan.md)
  s1d_researched_3:     Did the agent research at least 3 distinct models in the conversation?
                        (For LITE tier: default to 1 — agent is given the model.)
  s1e_plan_plot:        Did the agent create a plan visualization (plan.png)?
                        (For LITE tier: default to 1 — not required.)
  s1f_plot_pipeline:    Does the plot contain a clear pipeline diagram? (0 if no plot)
                        (For LITE tier: default to 1 — not required.)

Final s1_plan_score = average of s1a + s1b + s1c + s1d + s1e + s1f

### S2 — Setup Quality (6 binary sub-criteria, each 0 or 1)
Evaluate EACH of the following independently (0 = no, 1 = yes):
  s2a_checkpoint_downloaded: Did the agent successfully download model checkpoint/weights?
  s2b_compatibility_check:   Did the agent check input format compatibility (spacing, shape, dtype)?
  s2c_env_setup_success:     Did the environment setup succeed (venv created, packages installed)?
  s2d_env_fail_within_5:     Did the agent resolve env issues within 5 attempts? (1 if ≤5 failures or no failures, 0 if >5)
  s2e_model_loaded:          Was the model loaded into GPU memory and confirmed working?

Final s2_setup_score = average of s2a + s2b + s2c + s2d + s2e

### S3 — Validation Quality (MUST be 0.0, 0.5, or 1.0)
- 1.0: Tested on one patient first, verified the single multi-class output map \
(shape matches input, integer values in the allowed tissue set, per-tissue \
voxel counts non-zero), then proceeded
- 0.5: Some validation but incomplete (e.g. shape checked but label coverage not verified,
or missing-tissue warning ignored)
- 0.0: No validation; ran batch immediately or never verified outputs

### S4 and S5 — DO NOT SCORE
S4 (Inference) and S5 (Submit) are scored deterministically by the benchmark
harness based on output file completeness and format checks. Do NOT provide
s4_inference_score, s5_submit_score, s4_failure, or s5_failure in your
response — they will be ignored. Set them to 0.0 and null respectively.

### Tool Calling Correctness (0-1)
- 1.0: Tools called in correct order, correct arguments, no wasted calls
- 0.5: Some incorrect arguments or missing tool calls
- 0.0: Did not use tools or completely wrong tool sequence

### Clinical Reasoning Quality (0-1)
- 1.0: Carefully examined per-tissue statistics, justified label-mapping / \
model choice against the target label scheme
- 0.5: Made decisions but with limited justification
- 0.0: Random or unjustified decisions

## Error Codes (E1-E5)
For EACH step that failed, assign exactly one error code (null if step passed). \
These describe the ROOT CAUSE of failure:
- E1: Hallucination — agent fabricated non-existent models, APIs, repos, \
packages, or functions that do not exist
- E2: Resource error — GPU OOM, execution timeout, download failure, network error
- E3: Logic error — code ran without crashing but produced incorrect results \
(wrong label mapping, orientation mismatch, bad preprocessing, missing tissue classes)
- E4: Code error — Python/bash runtime errors (syntax, import, type, exception)
- E5: Format error — output does not meet spec (wrong shape, not a single \
multi-class map, values outside allowed tissue set, missing files)

A step can fail with any error code. For example, S1 could fail with E1 \
(hallucinated a model) or E4 (search code crashed). S4 could fail with E2 \
(OOM) or E3 (wrong label mapping).

Also identify the primary (earliest) failure step and its error code as \
"detected_failure".

## Output Format
Respond with ONLY a JSON object (no markdown, no extra text).
CRITICAL: s1 sub-scores (s1a-s1f) and s2 sub-scores (s2a-s2e) MUST be exactly 0 or 1.
s3 scores MUST be exactly 0.0, 0.5, or 1.0. Tool-calling and clinical-reasoning
scores may be any float 0-1.
{
  "s1a_plan_md": <0 | 1>,
  "s1b_plan_pipeline": <0 | 1>,
  "s1c_lesion_model": <0 | 1>,
  "s1d_researched_3": <0 | 1>,
  "s1e_plan_plot": <0 | 1>,
  "s1f_plot_pipeline": <0 | 1>,
  "s1_rationale": "<1-2 sentences>",
  "s1_failure": "<E1-E5 or null>",
  "s2a_checkpoint_downloaded": <0 | 1>,
  "s2b_compatibility_check": <0 | 1>,
  "s2c_env_setup_success": <0 | 1>,
  "s2d_env_fail_within_5": <0 | 1>,
  "s2e_model_loaded": <0 | 1>,
  "s2_rationale": "<1-2 sentences>",
  "s2_failure": "<E1-E5 or null>",
  "s3_validate_score": <0.0 | 0.5 | 1.0>,
  "s3_rationale": "<1-2 sentences>",
  "s3_failure": "<E1-E5 or null>",
  "s4_inference_score": <0.0 | 0.5 | 1.0>,
  "s4_rationale": "<1-2 sentences>",
  "s4_failure": "<E1-E5 or null>",
  "s5_submit_score": <0.0 | 0.5 | 1.0>,
  "s5_rationale": "<1-2 sentences>",
  "s5_failure": "<E1-E5 or null>",
  "tool_calling_score": <float 0-1>,
  "tool_calling_rationale": "<1-2 sentences>",
  "clinical_reasoning_score": <float 0-1>,
  "clinical_reasoning_rationale": "<1-2 sentences>",
  "overall_rationale": "<2-3 sentence summary>",
  "detected_failure": "<e.g. 'S1:E1' or null — earliest step:code pair>",
  "failure_explanation": "<explanation or empty string>"
}
"""


def _is_multiclass(eval_report: dict) -> bool:
    """Detect multi-class task from the deterministic report."""
    metrics = eval_report.get("diagnostic_metrics",
                              eval_report.get("metrics", {}))
    return str(metrics.get("task_type", "")).lower() == "multiclass"


def pick_judge_system_prompt(eval_report: dict) -> str:
    """Return the rubric matching the task type. Binary tasks unchanged."""
    return JUDGE_SYSTEM_PROMPT_MULTICLASS if _is_multiclass(eval_report) else JUDGE_SYSTEM_PROMPT


def build_judge_user_prompt(conversation: dict, eval_report: dict,
                            task: str) -> str:
    """Build the user message for the judge from conversation + eval data."""
    # Extract key info from conversation
    agent_name = conversation.get("agent", "unknown")
    model = conversation.get("model", "unknown")
    messages = conversation.get("messages", [])
    code_executions = conversation.get("code_executions", [])

    # Summarize conversation (truncate to avoid token explosion)
    conv_summary = _summarize_conversation(messages, code_executions)

    # Extract eval metrics — handle both old and new report formats
    metrics = eval_report.get("diagnostic_metrics",
                              eval_report.get("metrics", {}))
    agg = eval_report.get("agentic_score",
                          eval_report.get("aggregate", {}))
    tier = eval_report.get("agentic_tier", {})
    fmt = eval_report.get("format", {})
    steps = agg.get("step_scores", eval_report.get("step_scores", {}))
    failure = eval_report.get("step_failures",
                              eval_report.get("failure"))

    is_multiclass = _is_multiclass(eval_report)

    # Build per-patient detail if available
    per_patient_str = ""
    pp = metrics.get("per_patient", eval_report.get("_dice_per_patient", {}))
    if pp:
        per_patient_str = "\n  Per-patient breakdown:\n"
        for pid in sorted(pp.keys()):
            info = pp[pid]
            if is_multiclass:
                mtd = info.get("mean_tissue_dice")
                mtd_s = f"{mtd:.4f}" if isinstance(mtd, (int, float)) else "N/A"
                missing = []
                if info.get("missing_pred"): missing.append("no_pred")
                if info.get("missing_gt"): missing.append("no_gt")
                per_class = info.get("per_class", {})
                pc_s = ""
                if per_class:
                    pc_s = " [" + ", ".join(
                        f"{k}={float(v):.3f}" for k, v in sorted(
                            per_class.items(), key=lambda x: int(x[0])
                        )
                    ) + "]"
                note = f" ({','.join(missing)})" if missing else ""
                per_patient_str += f"    {pid}: mean_tissue_dice={mtd_s}{pc_s}{note}\n"
            else:
                od = info.get("organ_dice")
                ld = info.get("lesion_dice")
                gt = "GT+" if info.get("gt_has_lesion") else "GT-"
                od_s = f"{od:.4f}" if od is not None else "N/A"
                ld_s = f"{ld:.4f}" if ld is not None else "N/A"
                per_patient_str += f"    {pid}: organ={od_s} lesion={ld_s} ({gt})\n"

    # Extract step_failures if available (from updated report format)
    step_failures = eval_report.get("step_failures", {})
    step_fail_str = ""
    if step_failures:
        sf_parts = []
        for s in ["s1", "s2", "s3", "s4", "s5"]:
            code = step_failures.get(s)
            if code:
                sf_parts.append(f"S{s[1]}={code}")
        if sf_parts:
            step_fail_str = f"\n- Heuristic step failures: {', '.join(sf_parts)}"

    # Tier-aware preamble
    tier_name = conversation.get("tier", "pro")
    tier_note = ""
    if tier_name == "lite":
        tier_note = (
            "\n## Tier: LITE\n"
            "The agent was told exactly which model architecture to use "
            "(hard-coded SOTA). It was given a requirements.txt and detailed "
            "skill blocks for S1, S2, and S3. Score S1 on whether the agent "
            "researched and understood the given model (label map, input "
            "format, limitations, found the checkpoint) — NOT on breadth of "
            "model search. Score S2 on whether it installed from the provided "
            "requirements.txt and loaded the model correctly.\n"
        )
    elif tier_name == "standard":
        tier_note = (
            "\n## Tier: STANDARD\n"
            "The agent was given a shortlist of model families to explore "
            "and a skill block for S1. Score S1 based on comparison quality "
            "and justification of choice within the given range — not on "
            "breadth of search beyond the candidates. The agent had to "
            "figure out its own dependencies and environment.\n"
        )
    else:
        tier_note = (
            "\n## Tier: PRO\n"
            "The agent received NO model hints — only clinical background. "
            "It had to discover models from scratch, competing against other "
            "agents. Score S1 harshly if the agent stopped at the first "
            "working model without evaluating alternatives. Score S4 on "
            "post-processing quality in addition to inference completeness.\n"
        )

    if is_multiclass:
        per_class = metrics.get("per_class_dice", {})
        # aggregate.py builds this dict in label-id order, so preserve insertion order
        per_class_line = ", ".join(
            f"{k}={float(v):.4f}" for k, v in per_class.items()
        ) if per_class else "N/A"
        metrics_block = (
            f"- Task type:          multi-tissue (single multi-class label map per scan)\n"
            f"- Macro-mean Dice:    {metrics.get('macro_mean_dice', 'N/A')} "
            f"(mean across foreground tissues — this IS the clinical score)\n"
            f"- Per-tissue Dice:    {per_class_line}"
        )
    else:
        metrics_block = (
            f"- Organ Dice:   {metrics.get('organ_dice', 'N/A')}\n"
            f"- Lesion Dice:  {metrics.get('lesion_dice', 'N/A')}"
        )

    prompt = f"""## Agent Under Evaluation
Agent: {agent_name}
Model: {model}
Task: {task} segmentation
{tier_note}

## GROUND TRUTH Evaluation Results (deterministic — these are FACTS, not estimates)
NOTE: These metrics are computed by comparing agent outputs against ground truth.
Your scores for S4/S5 MUST be consistent with these measured results.

{metrics_block}
- Result tier:  {tier.get('medal_name', metrics.get('medal_name', 'N/A'))} (tier {tier.get('medal_tier', metrics.get('medal_tier', 'N/A'))})
- Format valid: {fmt.get('submission_format_valid', fmt.get('submission_valid', False))}
- CSV valid:    {fmt.get('decision_csv_valid', fmt.get('csv_valid', False))}
- Masks valid:  {fmt.get('output_format_valid', fmt.get('masks_valid', False))}
- Resolved:     {tier.get('resolved', agg.get('resolved', False))}
- Overall score:{agg.get('overall_score', 'N/A')}{step_fail_str}
{per_patient_str}
## Conversation Trace
{conv_summary}
"""
    if failure or step_failures:
        fail_code = (failure or step_failures).get("primary_failure", "N/A")
        fail_reason = (failure or step_failures).get(
            "failure_explanation",
            (failure or {}).get("root_cause_explanation", "")
        )
        prompt += f"""
## Heuristic Failure Analysis
- Primary failure: {fail_code}
- Explanation: {fail_reason}
"""

    prompt += "\nEvaluate the agent's performance according to the rubric. Return JSON only."
    return prompt


def _format_message(msg: dict) -> str:
    """Convert a single message to a compact string for the judge."""
    role = msg.get("role", "?")
    content = msg.get("content", "")

    if isinstance(content, list):
        # Anthropic format: list of blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    text_parts.append(
                        f"[Tool call: {block.get('name', '?')}("
                        f"{json.dumps(block.get('input', {}), separators=(',',':'))[:100]})]"
                    )
                elif block.get("type") == "tool_result":
                    result_text = block.get("content", "")
                    if isinstance(result_text, str) and len(result_text) > 300:
                        result_text = result_text[:300] + "..."
                    text_parts.append(f"[Tool result: {result_text}]")
        content = "\n".join(text_parts)
    elif not isinstance(content, str):
        content = str(content) if content else ""

    # Handle tool_calls in OpenAI format
    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        tc_strs = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function", tc)
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if isinstance(args, str) and len(args) > 150:
                    args = args[:150] + "..."
                tc_strs.append(f"[Tool call: {name}({args})]")
        if tc_strs:
            content = (content + "\n" if content else "") + "\n".join(tc_strs)

    if not content:
        return ""

    # Truncate long individual messages
    if len(content) > 600:
        content = content[:600] + "...[truncated]"

    return f"[{role}] {content}"


def _summarize_conversation(messages: list, code_executions: list,
                            max_chars: int = 24000) -> str:
    """Create a HEAD + TAIL summary so the judge sees both planning and results.

    Strategy:
      - HEAD: first 40% of budget → captures S1 planning and S2 setup
      - TAIL: last 40% of budget → captures S4 inference results and S5 submission
      - MIDDLE: remaining 20% → samples from the middle for S3 validation
    This ensures the judge sees the full workflow arc, not just the beginning.
    """
    # Format all messages
    formatted = []
    for msg in messages:
        entry = _format_message(msg)
        if entry:
            formatted.append(entry)

    if not formatted:
        return "(no conversation messages)"

    # If everything fits, use it all
    total_chars = sum(len(e) for e in formatted)
    if total_chars <= max_chars:
        result = "\n".join(formatted)
    else:
        # HEAD + TAIL strategy
        head_budget = int(max_chars * 0.40)
        tail_budget = int(max_chars * 0.40)
        mid_budget = max_chars - head_budget - tail_budget

        # HEAD: take messages from the start
        head_parts = []
        head_used = 0
        head_end = 0
        for i, entry in enumerate(formatted):
            if head_used + len(entry) > head_budget:
                break
            head_parts.append(entry)
            head_used += len(entry)
            head_end = i + 1

        # TAIL: take messages from the end (reversed)
        tail_parts = []
        tail_used = 0
        tail_start = len(formatted)
        for i in range(len(formatted) - 1, head_end - 1, -1):
            entry = formatted[i]
            if tail_used + len(entry) > tail_budget:
                break
            tail_parts.append(entry)
            tail_used += len(entry)
            tail_start = i
        tail_parts.reverse()

        # MIDDLE: sample from what's left between head and tail
        mid_parts = []
        mid_used = 0
        if tail_start > head_end:
            mid_range = list(range(head_end, tail_start))
            # Sample evenly from the middle
            n_mid = len(mid_range)
            if n_mid > 0:
                # Take every Nth message to fit budget
                step = max(1, n_mid // 20)  # aim for ~20 samples
                for idx in mid_range[::step]:
                    entry = formatted[idx]
                    if mid_used + len(entry) > mid_budget:
                        break
                    mid_parts.append(entry)
                    mid_used += len(entry)

        # Assemble
        parts = head_parts
        skipped_before_mid = tail_start - head_end - len(mid_parts)
        if mid_parts or skipped_before_mid > 0:
            parts.append(f"\n...[{head_end} to {tail_start-1}: "
                         f"{tail_start - head_end} messages, showing {len(mid_parts)} samples]...\n")
            parts.extend(mid_parts)
            if tail_parts:
                parts.append(f"\n...[resuming from message {tail_start}]...\n")
        elif tail_start > head_end:
            parts.append(f"\n...[{tail_start - head_end} messages omitted]...\n")
        parts.extend(tail_parts)

        result = "\n".join(parts)

    # Append code execution summary if available
    if code_executions:
        exec_summary = f"\n\n## Code Executions ({len(code_executions)} total)\n"
        # Show first 5 (setup/plan), last 5 (inference/submit), and errors
        first_execs = code_executions[:5]
        last_execs = code_executions[-5:] if len(code_executions) > 10 else []
        error_execs = [e for e in code_executions[5:-5]
                       if e.get("exit_code") != 0] if len(code_executions) > 10 else []

        def _fmt_exec(i, ex):
            lang = ex.get("language", "?")
            exit_code = ex.get("exit_code", "?")
            code = ex.get("code", "")[:200]
            stdout = ex.get("stdout_preview", "")[:150]
            status = "OK" if exit_code == 0 else f"FAIL(rc={exit_code})"
            s = f"  [{i+1}] {lang} {status}: {code}...\n"
            if stdout:
                s += f"       > {stdout}\n"
            return s

        exec_summary += "  --- First executions (setup) ---\n"
        for i, ex in enumerate(first_execs):
            exec_summary += _fmt_exec(i, ex)

        if error_execs:
            exec_summary += f"  --- Errors in middle ({len(error_execs)}) ---\n"
            for ex in error_execs[:3]:
                idx = code_executions.index(ex)
                exec_summary += _fmt_exec(idx, ex)

        if last_execs:
            skip_count = len(code_executions) - len(first_execs) - len(last_execs)
            if skip_count > 0:
                exec_summary += f"  --- ...{skip_count} executions omitted... ---\n"
            exec_summary += "  --- Last executions (inference/submit) ---\n"
            for ex in last_execs:
                idx = code_executions.index(ex)
                exec_summary += _fmt_exec(idx, ex)

        result += exec_summary

    return result


def _parse_judge_response(text: str) -> dict:
    """Extract JSON from the judge's response, handling markdown fences."""
    # Try direct JSON parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object anywhere in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


# =====================================================================
# Base judge class
# =====================================================================

class BaseJudge(ABC):
    """Abstract base for LLM judge backends."""

    @abstractmethod
    def judge(self, conversation: dict, eval_report: dict,
              task: str) -> JudgeVerdict:
        """Evaluate an agent run and return a structured verdict."""
        ...

    def _build_verdict(self, parsed: dict, backend: str, model: str,
                       latency: float, in_tok: int, out_tok: int,
                       tier: str = "pro") -> JudgeVerdict:
        """Convert parsed JSON dict to JudgeVerdict."""
        _DISCRETE_SCORES = (0.0, 0.5, 1.0)

        def _float(key, default=0.0):
            v = parsed.get(key, default)
            try:
                return max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                return default

        def _discrete(key, default=0.0):
            """Parse and snap S1-S5 scores to nearest allowed value."""
            v = _float(key, default)
            return min(_DISCRETE_SCORES, key=lambda x: abs(x - v))

        # S1: compute from 6 binary sub-criteria
        s1a = 1 if _float("s1a_plan_md") >= 0.5 else 0
        s1b = 1 if _float("s1b_plan_pipeline") >= 0.5 else 0
        s1c = 1 if _float("s1c_lesion_model") >= 0.5 else 0
        s1d = 1 if _float("s1d_researched_3") >= 0.5 else 0
        s1e = 1 if _float("s1e_plan_plot") >= 0.5 else 0
        s1f = 1 if _float("s1f_plot_pipeline") >= 0.5 else 0
        # Dependencies: s1b,s1c forced 0 if no plan.md; s1f forced 0 if no plot
        if s1a == 0:
            s1b = 0
            s1c = 0
        if s1e == 0:
            s1f = 0
        # Lite tier defaults
        if tier == "lite":
            s1d = 1  # researched_3: given the model
            s1e = 1  # plan_plot: not required
            s1f = 1  # plot_pipeline: not required
        s1_score = round((s1a + s1b + s1c + s1d + s1e + s1f) / 6, 2)

        return JudgeVerdict(
            s1a_plan_md=s1a,
            s1b_plan_pipeline=s1b,
            s1c_lesion_model=s1c,
            s1d_researched_3=s1d,
            s1e_plan_plot=s1e,
            s1f_plot_pipeline=s1f,
            s1_plan_score=s1_score,
            s1_rationale=str(parsed.get("s1_rationale", "")),
            s1_failure=parsed.get("s1_failure"),
            # S2: compute from 5 binary sub-criteria
            s2a_checkpoint_downloaded=1 if _float("s2a_checkpoint_downloaded") >= 0.5 else 0,
            s2b_compatibility_check=1 if _float("s2b_compatibility_check") >= 0.5 else 0,
            s2c_env_setup_success=1 if _float("s2c_env_setup_success") >= 0.5 else 0,
            s2d_env_fail_within_5=1 if _float("s2d_env_fail_within_5") >= 0.5 else 0,
            s2e_model_loaded=1 if _float("s2e_model_loaded") >= 0.5 else 0,
            s2_setup_score=round(sum(
                1 if _float(k) >= 0.5 else 0
                for k in ("s2a_checkpoint_downloaded", "s2b_compatibility_check",
                           "s2c_env_setup_success", "s2d_env_fail_within_5",
                           "s2e_model_loaded")
            ) / 5, 2),
            s2_rationale=str(parsed.get("s2_rationale", "")),
            s2_failure=parsed.get("s2_failure"),
            s3_validate_score=_discrete("s3_validate_score"),
            s3_rationale=str(parsed.get("s3_rationale", "")),
            s3_failure=parsed.get("s3_failure"),
            s4_inference_score=_discrete("s4_inference_score"),
            s4_rationale=str(parsed.get("s4_rationale", "")),
            s4_failure=parsed.get("s4_failure"),
            s5_submit_score=_discrete("s5_submit_score"),
            s5_rationale=str(parsed.get("s5_rationale", "")),
            s5_failure=parsed.get("s5_failure"),
            tool_calling_score=_float("tool_calling_score"),
            tool_calling_rationale=str(parsed.get("tool_calling_rationale", "")),
            clinical_reasoning_score=_float("clinical_reasoning_score"),
            clinical_reasoning_rationale=str(parsed.get("clinical_reasoning_rationale", "")),
            overall_rationale=str(parsed.get("overall_rationale", "")),
            detected_failure=parsed.get("detected_failure"),
            failure_explanation=str(parsed.get("failure_explanation", "")),
            judge_model=model,
            judge_backend=backend,
            judge_latency_s=round(latency, 2),
            input_tokens=in_tok,
            output_tokens=out_tok,
        )


# =====================================================================
# Offline judge: DeepSeek-R1-Distill-Qwen-32B via vLLM
# =====================================================================

class OfflineJudge(BaseJudge):
    """Local judge using vLLM with DeepSeek-R1-Distill-Qwen-32B."""

    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    def __init__(self, model_path: str = None, base_url: str = None,
                 gpu_memory_utilization: float = 0.92,
                 max_model_len: int = 16384):
        """
        Args:
            model_path: local path or HuggingFace model ID
            base_url: if set, use an already-running vLLM server at this URL
                      (e.g. "http://localhost:8000/v1") instead of loading locally
            gpu_memory_utilization: fraction of GPU memory for vLLM
            max_model_len: max sequence length (lower = more KV cache headroom)
        """
        self.model_path = model_path or self.MODEL_ID
        self.base_url = base_url
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm = None
        self._client = None

    def _ensure_loaded(self):
        """Lazy-load the model or connect to running server."""
        if self.base_url:
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key="unused",
                )
            return

        if self._llm is None:
            from vllm import LLM, SamplingParams  # noqa: F811
            print(f"[OfflineJudge] Loading {self.model_path} ...")
            self._llm = LLM(
                model=self.model_path,
                dtype="auto",
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
            )
            print(f"[OfflineJudge] Model loaded.")

    def judge(self, conversation: dict, eval_report: dict,
              task: str) -> JudgeVerdict:
        self._ensure_loaded()

        user_prompt = build_judge_user_prompt(conversation, eval_report, task)
        system_prompt = pick_judge_system_prompt(eval_report)

        t0 = time.time()

        if self.base_url:
            # Use OpenAI-compatible API against running vLLM server
            resp = self._client.chat.completions.create(
                model=self.model_path,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            raw_text = resp.choices[0].message.content or ""
            in_tok = getattr(resp.usage, "prompt_tokens", 0)
            out_tok = getattr(resp.usage, "completion_tokens", 0)
        else:
            # Direct vLLM inference
            from vllm import SamplingParams
            full_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            params = SamplingParams(
                temperature=0.0,
                max_tokens=2048,
                stop=["<|im_end|>"],
            )
            outputs = self._llm.generate([full_prompt], params)
            raw_text = outputs[0].outputs[0].text
            in_tok = len(full_prompt.split())  # approximate
            out_tok = len(raw_text.split())

        latency = time.time() - t0

        # Strip <think>...</think> blocks from reasoning models
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        parsed = _parse_judge_response(cleaned)
        if not parsed:
            print(f"[OfflineJudge] Warning: could not parse JSON from response")
            print(f"[OfflineJudge] Raw (first 500 chars): {raw_text[:500]}")
            parsed = {}

        return self._build_verdict(
            parsed, backend="offline_vllm",
            model=self.model_path, latency=latency,
            in_tok=in_tok, out_tok=out_tok,
            tier=conversation.get("tier", "pro"),
        )


# =====================================================================
# Online judge: Claude Opus 4.7 via NVIDIA Inference API
# =====================================================================

def _load_judge_config(config_path: str = None) -> dict:
    """Load judge API config from the claude-opus-4-7 entry in agent_config.yaml.

    Returns dict with 'api_key', 'base_url', 'model'.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "agent_config.yaml"
        )

    result = {"api_key": "", "base_url": "", "model": ""}

    # Try environment variable
    key = os.environ.get("JUDGE_API_KEY", "")
    if key:
        result["api_key"] = key

    if os.path.isfile(config_path):
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            # Use claude-opus-4-7 config as judge default
            agent_cfg = (cfg.get("agents") or {}).get("claude-opus-4-7", {})
            if not result["api_key"]:
                result["api_key"] = agent_cfg.get("api_key", "")
            result["base_url"] = agent_cfg.get("base_url", "")
            result["model"] = agent_cfg.get("model", "")
        except Exception:
            pass

    return result


class OnlineJudge(BaseJudge):
    """Online judge using Claude Opus 4.7.

    Reads API config from the claude-opus-4-7 entry in agent_config.yaml
    (same provider as the benchmark runner). No OpenRouter dependency.
    """

    def __init__(self, model: str = None, api_key: str = None,
                 base_url: str = None):
        cfg = _load_judge_config()
        self.model = model or cfg["model"] or "aws/anthropic/bedrock-claude-opus-4-7"
        self.api_key = api_key or cfg["api_key"]
        self.base_url = base_url or cfg["base_url"]
        if not self.api_key:
            raise ValueError(
                "No API key found for online judge. Set JUDGE_API_KEY env var "
                "or configure claude-opus-4-7 in agent_config.yaml."
            )
        # Build endpoint URL
        if self.base_url:
            self.api_url = self.base_url.rstrip("/") + "/chat/completions"
        else:
            raise ValueError(
                "No base_url found for online judge. "
                "Configure base_url in claude-opus-4-7 agent_config.yaml."
            )

    def judge(self, conversation: dict, eval_report: dict,
              task: str) -> JudgeVerdict:
        import requests

        user_prompt = build_judge_user_prompt(conversation, eval_report, task)
        system_prompt = pick_judge_system_prompt(eval_report)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 2048,
        }

        t0 = time.time()
        resp = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = time.time() - t0

        if "error" in data:
            raise RuntimeError(f"API error: {data['error']}")

        choice = data["choices"][0]
        raw_text = choice["message"].get("content", "")
        usage = data.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)

        parsed = _parse_judge_response(raw_text)
        if not parsed:
            print(f"[OnlineJudge] Warning: could not parse JSON from response")
            print(f"[OnlineJudge] Raw (first 500 chars): {raw_text[:500]}")
            parsed = {}

        return self._build_verdict(
            parsed, backend="online",
            model=self.model, latency=latency,
            in_tok=in_tok, out_tok=out_tok,
            tier=conversation.get("tier", "pro"),
        )


# =====================================================================
# Factory
# =====================================================================

def create_judge(online: bool = False, **kwargs) -> BaseJudge:
    """Create a judge instance.

    Args:
        online: if True, use Claude Opus 4.7 via OpenRouter.
                if False, use local DeepSeek-R1-Distill-Qwen-32B via vLLM.
        **kwargs: passed to the judge constructor.
    """
    if online:
        return OnlineJudge(**kwargs)
    else:
        return OfflineJudge(**kwargs)


# =====================================================================
# Standalone CLI for testing
# =====================================================================

def main():
    """Test the judge on a saved conversation + eval report."""
    import argparse
    parser = argparse.ArgumentParser(
        description="MedAgentsBench LLM Judge — evaluate an agent run")
    parser.add_argument("--conversation", required=True,
                        help="Path to conversation.json from a run")
    parser.add_argument("--eval-report", required=True,
                        help="Path to detail_report.json or eval report JSON")
    parser.add_argument("--task", required=True, choices=["kidney", "liver"])
    parser.add_argument("--online-judge", action="store_true",
                        help="Use Claude Opus 4.7 (online) instead of local model")
    parser.add_argument("--model-path", default=None,
                        help="Path to local judge model (offline mode)")
    parser.add_argument("--vllm-url", default=None,
                        help="URL of running vLLM server (e.g. http://localhost:8000/v1)")
    parser.add_argument("--output", default=None,
                        help="Save verdict JSON to this path")
    args = parser.parse_args()

    with open(args.conversation) as f:
        conversation = json.load(f)
    with open(args.eval_report) as f:
        eval_report = json.load(f)

    if args.online_judge:
        judge = OnlineJudge()
    else:
        judge = OfflineJudge(
            model_path=args.model_path,
            base_url=args.vllm_url,
        )

    print(f"[Judge] Backend: {'online (Claude Opus 4.7)' if args.online_judge else 'offline (DeepSeek-R1-Distill-Qwen-32B)'}")
    print(f"[Judge] Evaluating {conversation.get('agent', '?')} on {args.task}...")

    verdict = judge.judge(conversation, eval_report, args.task)

    print(f"\n{'='*60}")
    print(f"  LLM Judge Verdict")
    print(f"{'='*60}")
    print(f"  S1 Plan:      {verdict.s1_plan_score:.2f}  {verdict.s1_failure or '':>3}  — {verdict.s1_rationale}")
    print(f"  S2 Setup:     {verdict.s2_setup_score:.2f}  {verdict.s2_failure or '':>3}  — {verdict.s2_rationale}")
    print(f"  S3 Validate:  {verdict.s3_validate_score:.2f}  {verdict.s3_failure or '':>3}  — {verdict.s3_rationale}")
    print(f"  S4 Inference: {verdict.s4_inference_score:.2f}  {verdict.s4_failure or '':>3}  — {verdict.s4_rationale}")
    print(f"  S5 Submit:    {verdict.s5_submit_score:.2f}  {verdict.s5_failure or '':>3}  — {verdict.s5_rationale}")
    print(f"  Tool Calling: {verdict.tool_calling_score:.2f}       — {verdict.tool_calling_rationale}")
    print(f"  Clinical:     {verdict.clinical_reasoning_score:.2f}       — {verdict.clinical_reasoning_rationale}")
    print(f"  Primary fail: {verdict.detected_failure or 'None'}")
    print(f"  Summary:      {verdict.overall_rationale}")
    print(f"  Latency:      {verdict.judge_latency_s:.1f}s  Tokens: {verdict.input_tokens}in/{verdict.output_tokens}out")
    print(f"{'='*60}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(verdict.to_dict(), f, indent=2)
        print(f"\nVerdict saved to: {args.output}")


if __name__ == "__main__":
    main()
