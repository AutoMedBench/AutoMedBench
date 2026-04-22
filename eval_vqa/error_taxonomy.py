"""Error taxonomy for Phase 1 medical VQA analysis.

The taxonomy is intentionally small and stable. It separates output formatting
and runtime failures from answer-level mistakes so S5 can report actionable
error breakdowns without requiring a real LLM judge.
"""

from __future__ import annotations

from typing import Any


ERROR_CATEGORIES = (
    "hallucination",
    "resource/runtime",
    "logic",
    "format",
    "visual_miss",
    "localization_error",
    "comparison_error",
    "shortcut_suspicion",
)

ERROR_CODE_TO_CATEGORY = {
    "HAL001": "hallucination",
    "RES001": "resource/runtime",
    "RES002": "resource/runtime",
    "PRV001": "resource/runtime",   # V1: provider content-policy rejection
    "LOG001": "logic",
    "LOG002": "logic",
    "FMT001": "format",
    "FMT002": "format",
    "VIS001": "visual_miss",
    "LOC001": "localization_error",
    "CMP001": "comparison_error",
    "SCT001": "shortcut_suspicion",
    "AUX001": "shortcut_suspicion",  # V2: agent declared but did not exercise an auxiliary tool
    "PLC001": "shortcut_suspicion",  # V5: VQA tool-policy violation (no tool_plan / no aux action)
}

ERROR_CODE_DESCRIPTIONS = {
    "HAL001": "Predicted answer appears unsupported or contradicts the reference.",
    "RES001": "Runtime timeout or resource-limit failure.",
    "RES002": "Tool, image loading, or external resource failure.",
    "PRV001": "Provider rejected the request (content policy / safety filter); fallback answer emitted.",
    "LOG001": "Incorrect reasoning or answer selection.",
    "LOG002": "Empty answer despite completed workflow.",
    "FMT001": "Invalid JSONL or malformed prediction structure.",
    "FMT002": "Missing required submission field.",
    "VIS001": "Likely missed visual evidence.",
    "LOC001": "Incorrect anatomical or image location.",
    "CMP001": "Incorrect comparison across images, studies, or temporal states.",
    "SCT001": "Potential shortcut or suspicious process behavior.",
    "AUX001": "Agent produced a final answer without meaningfully exercising any auxiliary tool (no-op inspect/search).",
    "PLC001": "VQA tool-use policy violation (missing tool_plan or missing auxiliary action before submit).",
}

CATEGORY_TO_DEFAULT_CODE = {
    "hallucination": "HAL001",
    "resource/runtime": "RES001",
    "logic": "LOG001",
    "format": "FMT001",
    "visual_miss": "VIS001",
    "localization_error": "LOC001",
    "comparison_error": "CMP001",
    "shortcut_suspicion": "SCT001",
}

FAILURE_TAG_TO_CODE = {
    "timeout": "RES001",
    "data_error": "RES002",
    "agent_error": "RES002",
    "format_error": "FMT001",
    "missing_field": "FMT002",
    "shortcut": "SCT001",
    "judge_warning": "SCT001",
    "wrong_answer": "LOG001",
    "empty_answer": "LOG002",
}


def category_for_code(code: str | None) -> str:
    """Return the category for an error code, falling back to `logic`."""
    if not code:
        return "logic"
    return ERROR_CODE_TO_CATEGORY.get(str(code).upper(), "logic")


def default_code_for_category(category: str | None) -> str:
    """Return the default error code for a category."""
    normalized = normalize_category(category)
    return CATEGORY_TO_DEFAULT_CODE[normalized]


def code_for_failure_tag(tag: str | None) -> str:
    """Map a judge or pipeline failure tag to a stable error code."""
    if not tag or tag == "none":
        return ""
    return FAILURE_TAG_TO_CODE.get(str(tag).strip().lower(), "LOG001")


def normalize_category(category: str | None) -> str:
    """Normalize a category string into the supported taxonomy."""
    if not category:
        return "logic"
    text = str(category).strip().lower().replace("_", " ")
    aliases = {
        "resource": "resource/runtime",
        "runtime": "resource/runtime",
        "resource runtime": "resource/runtime",
        "format error": "format",
        "localization": "localization_error",
        "comparison": "comparison_error",
        "shortcut": "shortcut_suspicion",
        "shortcut suspicion": "shortcut_suspicion",
        "visual": "visual_miss",
    }
    normalized = aliases.get(text, text.replace(" ", "_"))
    if normalized in ERROR_CATEGORIES:
        return normalized
    return "logic"


def infer_error_code(
    *,
    status: Any | None = None,
    error: Any | None = None,
    task_type: str | None = None,
    judge_tag: str | None = None,
    answer_present: bool = True,
) -> str:
    """Infer a coarse error code from available runtime and task metadata."""
    judge_code = code_for_failure_tag(judge_tag)
    if judge_code:
        return judge_code

    status_text = _safe_text(status).lower()
    error_text = _safe_text(error).lower()
    task = _safe_text(task_type).lower()

    if "timeout" in error_text:
        return "RES001"
    if any(token in error_text for token in ("json", "schema", "format", "field")):
        return "FMT001"
    if any(token in error_text for token in ("image", "path", "file not found", "tool", "resource")):
        return "RES002"
    if status_text in {"error", "failed", "failure", "invalid"}:
        return "RES002"
    if not answer_present:
        return "LOG002"
    if task == "location":
        return "LOC001"
    if task == "comparison":
        return "CMP001"
    if task == "multi_image":
        return "VIS001"
    return "LOG001"


def taxonomy_summary() -> dict[str, Any]:
    """Return the complete taxonomy as JSON-serializable metadata."""
    return {
        "categories": list(ERROR_CATEGORIES),
        "code_to_category": dict(ERROR_CODE_TO_CATEGORY),
        "code_descriptions": dict(ERROR_CODE_DESCRIPTIONS),
    }


def _safe_text(value: Any) -> str:
    """Convert arbitrary values to safe text."""
    if value is None:
        return ""
    return str(value)
