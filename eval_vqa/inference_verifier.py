#!/usr/bin/env python3
"""Verify the agent actually performed local VLM inference.

Signals:
- Model-call detection: scan conversation.json / trace.jsonl for model-loading
  and .generate() invocations in successful code executions.
- Smoke-forward artefact: agent-written ``smoke_forward.json`` confirming a
  real GPU forward pass (schema: model_name, device, wall_s, raw_output_sample,
  success).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


LOAD_SIGNATURES = [
    r"\.from_pretrained\(",
    r"LlavaMistralForCausalLM",
    r"AutoModelForVision2Seq",
    r"AutoModelForCausalLM",
    r"AutoProcessor",
    r"LlavaProcessor",
    r"LlavaForConditionalGeneration",
    r"Qwen2VLForConditionalGeneration",
    r"pipeline\(",
    r"load_pretrained_model\(",
    r"snapshot_download\(",
]
GENERATE_SIGNATURES = [
    r"\.generate\(",
    r"model\.generate",
]
STDOUT_LOAD_MARKERS = [
    "Loading checkpoint shards",
    "Downloading shards",
    "model.safetensors",
    "pytorch_model.bin",
    "Loading weights",
]
# Patterns that suggest the agent coded a fake-inference shortcut.
SHORTCUT_PATTERNS = [
    r"['\"]heuristic:",
    r"['\"]fallback:",
    r"['\"]placeholder:",
    r"return\s+['\"]unknown['\"]",
]

_COMPILED_LOAD = [re.compile(p) for p in LOAD_SIGNATURES]
_COMPILED_GEN = [re.compile(p) for p in GENERATE_SIGNATURES]
_COMPILED_SHORTCUT = [re.compile(p) for p in SHORTCUT_PATTERNS]


def _matches(patterns: list[re.Pattern], text: str) -> list[str]:
    return [p.pattern for p in patterns if p.search(text)]


def detect_model_call(conversation_path: str) -> dict[str, Any]:
    """Inspect code_executions in conversation.json for VLM inference evidence.

    Returns dict with:
      detected: bool
      evidence: list[str] of matched signatures
      shortcuts: list[str] of suspicious shortcut patterns
      has_generate: bool (a .generate( call in a successful exec)
      has_loader: bool (from_pretrained/pipeline in a successful exec)
      has_stdout_marker: bool
    """
    result = {
        "detected": False,
        "evidence": [],
        "shortcuts": [],
        "has_generate": False,
        "has_loader": False,
        "has_stdout_marker": False,
    }
    if not os.path.isfile(conversation_path):
        result["evidence"].append("conversation.json missing")
        return result

    try:
        with open(conversation_path, "r", encoding="utf-8") as handle:
            conv = json.load(handle)
    except Exception as exc:
        result["evidence"].append(f"conversation parse error: {exc}")
        return result

    for ce in conv.get("code_executions") or []:
        code = ce.get("code") or ""
        stdout = ce.get("stdout_preview") or ""
        exit_code = ce.get("exit_code")
        blob = code + "\n" + stdout

        for pat in _COMPILED_SHORTCUT:
            if pat.search(code):
                result["shortcuts"].append(f"turn{ce.get('turn')}:{pat.pattern}")

        if exit_code == 0:
            gen_hits = _matches(_COMPILED_GEN, blob)
            load_hits = _matches(_COMPILED_LOAD, blob)
            if gen_hits:
                result["has_generate"] = True
                result["evidence"].extend(f"turn{ce.get('turn')}:{h}" for h in gen_hits)
            if load_hits:
                result["has_loader"] = True
                result["evidence"].extend(f"turn{ce.get('turn')}:{h}" for h in load_hits)
            for marker in STDOUT_LOAD_MARKERS:
                if marker in stdout:
                    result["has_stdout_marker"] = True
                    result["evidence"].append(f"turn{ce.get('turn')}:stdout:{marker}")

    result["detected"] = result["has_generate"] and (
        result["has_loader"] or result["has_stdout_marker"]
    )
    return result


def check_smoke_forward(workspace_dir: str | None) -> dict[str, Any]:
    """Validate <workspace>/smoke_forward.json as BUG-5 artefact.

    Valid when: file exists, JSON parses, schema keys present, success is True,
    wall_s >= 0.3, raw_output_sample non-empty and not a placeholder.

    Returns dict with valid(bool), reason(str), record(dict|None).
    """
    result = {"valid": False, "reason": "missing", "record": None, "path": None}
    if not workspace_dir:
        return result
    path = os.path.join(workspace_dir, "smoke_forward.json")
    result["path"] = path
    if not os.path.isfile(path):
        return result
    try:
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except Exception as exc:
        result["reason"] = f"parse_error: {exc}"
        return result
    result["record"] = record

    required = {"model_name", "device", "wall_s", "raw_output_sample", "success"}
    missing = required - set(record)
    if missing:
        result["reason"] = f"missing_keys:{sorted(missing)}"
        return result
    if not record.get("success"):
        result["reason"] = "success=False"
        return result
    wall_s = record.get("wall_s")
    if not isinstance(wall_s, (int, float)) or wall_s < 0.3:
        result["reason"] = f"wall_s<0.3 ({wall_s})"
        return result
    sample = str(record.get("raw_output_sample") or "").strip()
    if not sample or len(sample) < 5:
        result["reason"] = "raw_output_sample too short"
        return result
    sample_lower = sample.lower()
    if sample_lower.startswith(("heuristic:", "fallback:", "placeholder:", "mock:")):
        result["reason"] = f"raw_output_sample looks like placeholder: {sample[:40]!r}"
        return result

    result["valid"] = True
    result["reason"] = "ok"
    return result


def compute_length_finish_rate(process_dir: str | None) -> dict[str, Any]:
    """BUG-045: scan trace.jsonl for finish_reason=length turns.

    High rate signals a reasoning model burning its max_tokens budget on
    hidden thinking; report surfaces the metric without auto-failing the run.
    """
    result = {"length_finish_count": 0, "api_calls": 0, "length_finish_rate": 0.0}
    if not process_dir:
        return result
    trace_path = os.path.join(process_dir, "trace.jsonl")
    if not os.path.isfile(trace_path):
        return result
    try:
        with open(trace_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") != "api_call":
                    continue
                result["api_calls"] += 1
                if ev.get("finish_reason") == "length":
                    result["length_finish_count"] += 1
    except Exception:
        return result
    if result["api_calls"]:
        result["length_finish_rate"] = round(
            result["length_finish_count"] / result["api_calls"], 4
        )
    return result


def check_postprocess_artefact(workspace_dir: str | None) -> dict[str, Any]:
    """Validate <workspace>/answer_postprocess.py + s3_calibration.json (P1-A).

    Valid when:
      - answer_postprocess.py exists, imports without error, exposes callable
        ``postprocess(raw: str) -> str``.
      - s3_calibration.json exists, parses, contains >= 15 records with keys
        {question_id, raw_model_output, predicted_answer, gold_answer, hit}.
    """
    result = {
        "valid": False,
        "reason": "missing",
        "postprocess_ok": False,
        "calibration_ok": False,
        "calibration_n": 0,
        "calibration_hit_rate": None,
        "calibration_hit_rate_raw": None,
        "invalid_gold_n": 0,
        "invalid_gold_rate": None,
        "malformed_raw_n": 0,
        "malformed_raw_rate": None,
    }
    if not workspace_dir:
        return result

    pp_path = os.path.join(workspace_dir, "answer_postprocess.py")
    cal_path = os.path.join(workspace_dir, "s3_calibration.json")
    result["postprocess_path"] = pp_path
    result["calibration_path"] = cal_path

    if not os.path.isfile(pp_path):
        result["reason"] = "answer_postprocess.py missing"
    else:
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "_pp_module_probe", pp_path
            )
            if spec is None or spec.loader is None:
                result["reason"] = "cannot load answer_postprocess.py"
            else:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                fn = getattr(module, "postprocess", None)
                if not callable(fn):
                    result["reason"] = "postprocess() not callable"
                else:
                    try:
                        probe = fn("no, the upper dermis shows only mild edema.")
                        if not isinstance(probe, str):
                            result["reason"] = "postprocess() did not return str"
                        else:
                            result["postprocess_ok"] = True
                    except Exception as exc:
                        result["reason"] = f"postprocess() raised: {exc}"
        except Exception as exc:
            result["reason"] = f"import error: {exc}"

    if not os.path.isfile(cal_path):
        if result["postprocess_ok"]:
            result["reason"] = "s3_calibration.json missing"
    else:
        try:
            with open(cal_path, "r", encoding="utf-8") as handle:
                cal = json.load(handle)
        except Exception as exc:
            result["reason"] = f"s3_calibration parse error: {exc}"
            return result
        if not isinstance(cal, list):
            result["reason"] = "s3_calibration.json must be a list"
            return result
        required_keys = {"question_id", "raw_model_output", "predicted_answer"}
        bad = [i for i, rec in enumerate(cal) if not isinstance(rec, dict) or required_keys - set(rec)]
        if bad:
            result["reason"] = f"s3_calibration.json records missing keys at indices {bad[:3]}"
            return result
        result["calibration_n"] = len(cal)
        hits_raw = sum(1 for rec in cal if rec.get("hit"))
        result["calibration_hit_rate_raw"] = round(hits_raw / max(len(cal), 1), 4)

        invalid_gold_tokens = {"", "unknown", "n/a", "na", "none", "null", "?", "-"}

        def _is_invalid_gold(rec: dict) -> bool:
            # gold_answer is optional. Only flag records that include gold_answer
            # but set it to a placeholder token. Absent/None gold is treated as
            # "not provided" (skipped by the invalid_gold_rate guard).
            if "gold_answer" not in rec:
                return False
            g = rec.get("gold_answer")
            if g is None:
                return False
            if not isinstance(g, str):
                return True
            return g.strip().lower() in invalid_gold_tokens

        def _is_malformed_raw(rec: dict) -> bool:
            raw = rec.get("raw_model_output")
            if not isinstance(raw, str):
                return True
            s = raw.strip()
            if len(s) < 5:
                return True
            return s[0] in ",.;:!?-"

        invalid_gold_n = sum(1 for rec in cal if _is_invalid_gold(rec))
        malformed_raw_n = sum(1 for rec in cal if _is_malformed_raw(rec))
        result["invalid_gold_n"] = invalid_gold_n
        result["malformed_raw_n"] = malformed_raw_n
        result["invalid_gold_rate"] = round(invalid_gold_n / max(len(cal), 1), 4)
        result["malformed_raw_rate"] = round(malformed_raw_n / max(len(cal), 1), 4)

        valid_recs = [r for r in cal if not _is_invalid_gold(r)]
        if valid_recs:
            hits_eff = sum(1 for r in valid_recs if r.get("hit"))
            result["calibration_hit_rate"] = round(hits_eff / len(valid_recs), 4)
        else:
            result["calibration_hit_rate"] = 0.0

        if len(cal) < 10:
            result["reason"] = f"s3_calibration has only {len(cal)} records (need >= 10)"
            return result
        if result["invalid_gold_rate"] > 0.2:
            result["reason"] = (
                f"too many gold_answer=unknown/empty ({invalid_gold_n}/{len(cal)}); "
                "fake calibration hits"
            )
            return result
        if result["malformed_raw_rate"] > 0.2:
            result["reason"] = (
                f"raw_model_output looks truncated/decode-broken "
                f"({malformed_raw_n}/{len(cal)} records < 5 chars or start with punctuation)"
            )
            return result
        result["calibration_ok"] = True

    if result["postprocess_ok"] and result["calibration_ok"]:
        result["valid"] = True
        result["reason"] = "ok"
    return result
