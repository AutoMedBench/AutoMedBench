"""Per-sample answer-submission helper.

Agent code calls ``submit_answer(question_id, predicted_label=..., ...)`` after
each question.  The helper writes ``$WORKSPACE_DIR/<question_id>/answer.json``
matching the schema the runner's ``format_checker`` expects.
"""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

from ._logger import record_tool_call

REQUIRED_FIELDS = (
    "question_id",
    "predicted_label",
    "predicted_answer",
    "raw_model_output",
    "model_name",
    "runtime_s",
)


def submit_answer(
    question_id: str,
    *,
    predicted_label: str = "",
    predicted_answer: str,
    raw_model_output: str,
    model_name: str,
    runtime_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``outputs/<question_id>/answer.json`` atomically.

    Returns a small confirmation dict ``{"status", "path", "question_id"}``.

    For multiple-choice tasks (MedXpertQA-MM, MedFrameQA) pass a single
    uppercase letter as ``predicted_label``.  For open-ended tasks (PathVQA,
    VQA-RAD) leave ``predicted_label`` empty and rely on ``predicted_answer``.

    Honest-skip path: if the external VLM errored on this sample, pass
    ``predicted_answer=""`` AND ``raw_model_output=""``. The scorer drops
    empty records from ``valid_outputs`` without flagging them as placeholders.
    Do NOT pad with ``"unknown"`` / ``"error"`` / ``"fallback:"`` — those
    trigger ``placeholder_rate`` and cap S4 at 0.2.
    """
    started = perf_counter()
    qid = str(question_id).strip()
    if not qid:
        raise ValueError("question_id must be non-empty")
    label = str(predicted_label or "").strip().upper()
    if label and (len(label) != 1 or not label.isalpha()):
        raise ValueError(f"predicted_label must be empty or a single uppercase letter; got {predicted_label!r}")
    pa_stripped = str(predicted_answer or "").strip()
    raw_stripped = str(raw_model_output or "").strip()
    # Honest-skip is allowed: both empty means "VLM errored, no fabrication".
    # Reject the half-empty case where an answer is claimed without model text,
    # or vice versa — that's the shape of a forged record.
    if bool(pa_stripped) != bool(raw_stripped) and not label:
        raise ValueError(
            "predicted_answer and raw_model_output must both be non-empty "
            "(real VLM pass) or both empty (honest skip after VLM error)"
        )

    workspace = os.environ.get("WORKSPACE_DIR", os.getcwd())
    sample_dir = os.path.join(workspace, qid)
    os.makedirs(sample_dir, exist_ok=True)
    target = os.path.join(sample_dir, "answer.json")
    payload = {
        "question_id": qid,
        "predicted_label": label,
        "predicted_answer": str(predicted_answer),
        "raw_model_output": str(raw_model_output),
        "model_name": str(model_name),
        "runtime_s": float(runtime_s),
    }
    if extra:
        payload.update(extra)

    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, target)

    latency_ms = round((perf_counter() - started) * 1000, 2)
    record_tool_call(
        tool="submit_answer",
        arguments={
            "question_id": qid,
            "predicted_label": label,
            "model_name": payload["model_name"],
        },
        result_summary={"status": "ok", "path": target, "latency_ms": latency_ms},
    )
    return {"status": "ok", "path": target, "question_id": qid}
