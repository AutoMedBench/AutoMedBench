#!/usr/bin/env python3
"""Judge interface for VQA workflow scoring.

The initial implementation provides a deterministic heuristic backend so
`run_eval.py` can emit valid `S1-S3` scores without requiring an external LLM
service. The transport boundary is kept small so an API-backed judge can
replace this later without changing evaluator callers.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class JudgeVerdict:
    s1a_plan_md: int = 0
    s1b_runnable_model: int = 0
    s1c_dataset_schema: int = 0
    s1d_researched_3: int = 0
    s1e_answer_normalization: int = 0
    s1f_smoke_path: int = 0
    s1_plan_score: float = 0.0
    s1_rationale: str = ""
    s1_failure: str | None = None
    s2a_dataset_ready: int = 0
    s2b_model_assets_ready: int = 0
    s2c_env_setup_success: int = 0
    s2d_env_fail_within_5: int = 0
    s2e_forward_pass: int = 0
    s2_setup_score: float = 0.0
    s2_rationale: str = ""
    s2_failure: str | None = None
    s3_validate_score: float = 0.0
    s3_rationale: str = ""
    s3_failure: str | None = None
    overall_rationale: str = ""
    detected_failure: str | None = None
    failure_explanation: str = ""
    judge_model: str = "heuristic-v1"
    judge_backend: str = "heuristic"
    judge_latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HeuristicJudge:
    def judge(self, conversation: dict[str, Any], eval_report: dict, task: str) -> JudgeVerdict:
        started = time.time()
        artifact_root = (
            conversation.get("artifacts_dir")
            or conversation.get("run_dir")
            or conversation.get("output_dir")
            or ""
        )
        messages_text = _flatten_messages(conversation.get("messages", []))
        tier = conversation.get("tier", "lite")

        plan_md_path = os.path.join(artifact_root, "plan", "plan.md") if artifact_root else ""
        plan_text = _read_text(plan_md_path)

        s1 = {
            "s1a_plan_md": int(bool(plan_text)),
            "s1b_runnable_model": int(_has_any(plan_text, messages_text, needles=["medvlthinker", "medvlsynther", "gemma-4", "qwen2.5-vl"])),
            "s1c_dataset_schema": int(_has_any(plan_text, messages_text, needles=["medxpertqa", "question.json", "answer.json", "options", "predicted_label"])),
            "s1d_researched_3": int(tier == "lite" or _count_model_mentions(plan_text + "\n" + messages_text) >= 3),
            "s1e_answer_normalization": int(_has_any(plan_text, messages_text, needles=["normalize", "predicted_label", "extract label", "A-E", "answer extraction"])),
            "s1f_smoke_path": int(_has_any(plan_text, messages_text, needles=["smoke", "1-10", "validate", "sample-limit", "before scale"])),
        }
        s1_plan_score = round(sum(s1.values()) / 6.0, 4)

        metrics = eval_report.get("metrics", {})
        counts = metrics.get("counts", {})
        s2 = {
            "s2a_dataset_ready": int(counts.get("expected", 0) > 0 or _has_any(messages_text, "", needles=["download dataset", "staged data", "public/"])),
            "s2b_model_assets_ready": int(_has_any(messages_text, "", needles=["from_pretrained", "snapshot_download", "downloaded model", "processor"])),
            "s2c_env_setup_success": int(counts.get("prediction_files", 0) > 0 or _has_any(messages_text, "", needles=["pip install", "requirements", "venv"])),
            "s2d_env_fail_within_5": int(_count_error_markers(messages_text) <= 5),
            "s2e_forward_pass": int(counts.get("prediction_files", 0) > 0 or _has_any(messages_text, "", needles=["forward pass", "generate(", "raw output"])),
        }
        s2_setup_score = round(sum(s2.values()) / 5.0, 4)

        placeholder_rate = float(metrics.get("placeholder_rate", 0.0))
        inference_mode = metrics.get("inference_mode", "real")
        if inference_mode in ("heuristic", "placeholder", "empty") or placeholder_rate > 0.5:
            s3_score = 0.0
            s3_rationale = (
                f"Outputs are fake / placeholder (mode={inference_mode}, "
                f"placeholder_rate={placeholder_rate:.2f}); validation evidence is meaningless."
            )
            s3_failure = "E5"
        elif placeholder_rate > 0.05:
            s3_score = 0.2
            s3_rationale = (
                f"Non-trivial placeholder rate ({placeholder_rate:.2f}); validation did "
                "not catch fallback strings before full inference."
            )
            s3_failure = "E3"
        elif metrics.get("completion_rate", 0.0) > 0 and metrics.get("parse_rate", 0.0) > 0:
            if _has_any(messages_text, plan_text, needles=["smoke", "validate", "raw output", "parseable", "1-10"]):
                s3_score = 1.0
                s3_rationale = "A smoke-style validation path is visible and outputs were parseable."
                s3_failure = None
            else:
                s3_score = 0.5
                s3_rationale = "The run produced outputs, but explicit validation evidence is partial."
                s3_failure = None
        else:
            s3_score = 0.0
            s3_rationale = "No meaningful validation evidence was found before or during inference."
            s3_failure = "E4" if counts.get("prediction_files", 0) == 0 else "E3"

        verdict = JudgeVerdict(
            **s1,
            s1_plan_score=s1_plan_score,
            s1_rationale=_build_s1_rationale(s1_plan_score),
            s1_failure=None if s1_plan_score >= 0.5 else "E1",
            **s2,
            s2_setup_score=s2_setup_score,
            s2_rationale=_build_s2_rationale(s2_setup_score),
            s2_failure=None if s2_setup_score >= 0.6 else "E4",
            s3_validate_score=s3_score,
            s3_rationale=s3_rationale,
            s3_failure=s3_failure,
            overall_rationale="Heuristic judge used artifact and output evidence to approximate S1-S3 workflow quality.",
            detected_failure=_earliest_failure(s1_plan_score, s2_setup_score, s3_score),
        )
        if verdict.detected_failure:
            verdict.failure_explanation = "Earliest workflow weakness detected by the heuristic judge."
        verdict.judge_latency_s = round(time.time() - started, 4)
        verdict.input_tokens = len(messages_text.split())
        verdict.output_tokens = len(json.dumps(verdict.to_dict()).split())
        return verdict


def create_judge(**_: Any) -> HeuristicJudge:
    return HeuristicJudge()


def _flatten_messages(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, str):
            parts.append(message)
        elif isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
        else:
            parts.append(str(message))
    return "\n".join(parts)


def _read_text(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _has_any(*texts: str, needles: list[str]) -> bool:
    haystack = "\n".join(texts).lower()
    return any(needle.lower() in haystack for needle in needles)


def _count_model_mentions(text: str) -> int:
    patterns = [
        "medvlthinker",
        "medvlsynther",
        "gemma-4",
        "qwen2.5-vl",
        "qwen2_5_vl",
    ]
    return sum(int(pattern in text.lower()) for pattern in patterns)


def _count_error_markers(text: str) -> int:
    return len(re.findall(r"\b(error|exception|traceback|failed|oom|timeout)\b", text, flags=re.IGNORECASE))


def _build_s1_rationale(score: float) -> str:
    if score >= 0.8:
        return "Planning evidence covered model choice, dataset schema, normalization, and smoke validation."
    if score >= 0.5:
        return "Planning evidence was partial but captured some of the expected VQA workflow."
    return "Planning evidence was weak or missing for the required VQA setup."


def _build_s2_rationale(score: float) -> str:
    if score >= 0.8:
        return "Setup evidence shows staged data, model assets, and a working forward path."
    if score >= 0.6:
        return "Setup evidence is incomplete but suggests partial environment and model preparation."
    return "Setup evidence was too weak to confirm a reliable local inference environment."


def _earliest_failure(s1_score: float, s2_score: float, s3_score: float) -> str | None:
    if s1_score < 0.5:
        return "S1:E1"
    if s2_score < 0.6:
        return "S2:E4"
    if s3_score < 0.5:
        return "S3:E3"
    return None
