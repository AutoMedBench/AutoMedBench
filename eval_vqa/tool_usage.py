"""Summarise medbench_vqa helper calls recorded in tool_calls.jsonl.

The VQA coding-agent exposes only `execute_code`.  Inside that tool the agent
imports three Python helpers (`inspect_image`, `public_medical_search`,
`submit_answer`) which each append a record to `$WORKSPACE_DIR/tool_calls.jsonl`.
This module reads that file and turns it into a compact signal used as
`artefact + 轻量 tool-usage 副分` on top of the deterministic scorer.
"""

from __future__ import annotations

import json
import os
from typing import Any


def load_tool_calls(workspace_dir: str) -> list[dict[str, Any]]:
    path = os.path.join(workspace_dir, "tool_calls.jsonl")
    if not os.path.isfile(path):
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize(records: list[dict[str, Any]], *, expected_samples: int) -> dict[str, Any]:
    inspect = [r for r in records if r.get("tool") == "inspect_image"]
    search = [r for r in records if r.get("tool") == "public_medical_search"]
    submits = [r for r in records if r.get("tool") == "submit_answer"]
    ok_submits = [r for r in submits if (r.get("result_summary") or {}).get("status") == "ok"]

    return {
        "inspect_image_calls": len(inspect),
        "public_medical_search_calls": len(search),
        "submit_answer_calls": len(submits),
        "submit_answer_ok": len(ok_submits),
        "expected_samples": int(expected_samples),
        "submit_coverage": round(len(ok_submits) / expected_samples, 4) if expected_samples > 0 else 0.0,
        "inspect_used": len(inspect) > 0,
        "search_used": len(search) > 0,
    }


def score_adjustments(summary: dict[str, Any]) -> dict[str, float]:
    """Map tool-usage signal into per-step auxiliary scores in [0, 1].

    These scores are *auxiliary* — the caller folds them into the main step
    score with a small weight (default 20%, see `apply_tool_usage`).
    """
    expected = max(1, int(summary.get("expected_samples", 0)))
    submit_cov = min(1.0, summary.get("submit_answer_ok", 0) / expected)
    inspect = 1.0 if summary.get("inspect_used") else 0.0
    return {
        "s3_aux": inspect,          # agent actually inspected at least one image
        "s4_aux": submit_cov,       # submit_answer matched sample count
    }


def apply_tool_usage(
    step_scores: dict[str, float | None],
    summary: dict[str, Any],
    *,
    aux_weight: float = 0.20,
) -> dict[str, float | None]:
    """Blend auxiliary tool-usage signal into S3/S4 step scores."""
    adj = score_adjustments(summary)
    out = dict(step_scores)
    for step, aux in (("s3", adj["s3_aux"]), ("s4", adj["s4_aux"])):
        base = out.get(step)
        if base is None:
            continue
        out[step] = round((1.0 - aux_weight) * float(base) + aux_weight * float(aux), 4)
    return out
