"""Conventional answer metrics for Phase 1 medical VQA leaderboard."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from answer_normalizer import is_yes_no_answer, normalize_answer
from modality_typing import MODALITY_UNKNOWN_OTHER, normalize_modality


def exact_match(pred: str | None, gold: str | None) -> float:
    """Return exact match as a 0-100 score after answer normalization."""
    pred_norm = normalize_answer(_safe_text(pred))
    gold_norm = normalize_answer(_safe_text(gold))
    if not pred_norm or not gold_norm:
        return 0.0
    return 100.0 if pred_norm == gold_norm else 0.0


def token_f1(pred: str | None, gold: str | None) -> float:
    """Return token-level F1 as a 0-100 score after answer normalization."""
    pred_tokens = normalize_answer(_safe_text(pred)).split()
    gold_tokens = normalize_answer(_safe_text(gold)).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 100.0 * (2 * precision * recall / (precision + recall))


def yes_no_accuracy(pred: str | None, gold: str | None) -> float:
    """Return yes/no accuracy as a 0-100 score.

    If the gold answer is not yes/no after normalization, the metric returns
    0.0 because the example is outside the yes/no subset.
    """
    pred_norm = normalize_answer(_safe_text(pred))
    gold_norm = normalize_answer(_safe_text(gold))
    if not is_yes_no_answer(gold_norm):
        return 0.0
    return 100.0 if pred_norm == gold_norm else 0.0


def score_predictions(predictions: Iterable[Any], gold_samples: Iterable[Any]) -> dict[str, Any]:
    """Score predictions against gold samples.

    Args:
        predictions: Iterable of dict-like or object-like predictions. Expected
            answer fields include `answer`, `final_answer`, or `prediction`.
        gold_samples: Iterable of dict-like or object-like gold samples with
            `sample_id` and `answer`.

    Returns:
        A dictionary with `overall`, `per_dataset`, and modality-primary scores.
        All reported metric values are in the 0-100 range.
    """
    gold_by_id = {_get_field(sample, "sample_id"): sample for sample in gold_samples}
    prediction_by_id: dict[str, Any] = {}
    for prediction in predictions:
        sample_id = _get_field(prediction, "sample_id")
        if sample_id in gold_by_id and sample_id not in prediction_by_id:
            prediction_by_id[sample_id] = prediction

    groups = _new_groups()

    for sample_id, gold in gold_by_id.items():
        prediction = prediction_by_id.get(sample_id)
        gold = gold_by_id[sample_id]
        pred_answer = _first_field(prediction, ("pred_answer_raw", "answer", "final_answer", "prediction"))
        gold_answer = _get_field(gold, "answer")
        gold_alternatives = _gold_answer_alternatives(gold)
        dataset = _get_field(gold, "dataset") or _get_field(prediction, "dataset") or "unknown"
        modality = _modality(gold) or _modality(prediction) or "unknown_other"
        question_task_type = _question_task_type(gold) or _question_task_type(prediction) or "unknown"

        metric_row = {
            "em": max(exact_match(pred_answer, alternative) for alternative in gold_alternatives),
            "f1": max(token_f1(pred_answer, alternative) for alternative in gold_alternatives),
            "yes_no_accuracy": yes_no_accuracy(pred_answer, gold_answer),
            "yes_no_count": 1 if is_yes_no_answer(_safe_text(gold_answer)) else 0,
        }
        _add_row(groups["overall"], metric_row)
        _add_row(groups["per_dataset"][dataset], metric_row)
        _add_row(groups["per_modality"][modality], metric_row)
        _add_row(groups["per_question_task_type"][question_task_type], metric_row)

    per_modality = {name: _finalize_group(group) for name, group in groups["per_modality"].items()}
    return {
        "overall": _finalize_group(groups["overall"]),
        "per_dataset": {name: _finalize_group(group) for name, group in groups["per_dataset"].items()},
        "per_modality": per_modality,
        "per_question_task_type": {
            name: _finalize_group(group) for name, group in groups["per_question_task_type"].items()
        },
        # per_task_type is a legacy alias for per_question_task_type (not per_modality)
        "per_task_type": {
            name: _finalize_group(group) for name, group in groups["per_question_task_type"].items()
        },
    }


def _new_groups() -> dict[str, Any]:
    """Create mutable metric aggregation groups."""
    return {
        "overall": _empty_group(),
        "per_dataset": defaultdict(_empty_group),
        "per_modality": defaultdict(_empty_group),
        "per_question_task_type": defaultdict(_empty_group),
    }


def _empty_group() -> dict[str, Any]:
    """Create one mutable metric group."""
    return {
        "count": 0,
        "em_sum": 0.0,
        "f1_sum": 0.0,
        "yes_no_sum": 0.0,
        "yes_no_count": 0,
    }


def _add_row(group: dict[str, Any], row: dict[str, float]) -> None:
    """Add one scored row to a group."""
    group["count"] += 1
    group["em_sum"] += row["em"]
    group["f1_sum"] += row["f1"]
    if row["yes_no_count"]:
        group["yes_no_sum"] += row["yes_no_accuracy"]
        group["yes_no_count"] += 1


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    """Convert metric sums into averages."""
    count = group["count"]
    yes_no_count = group["yes_no_count"]
    return {
        "count": count,
        "em": _safe_mean(group["em_sum"], count),
        "f1": _safe_mean(group["f1_sum"], count),
        "yes_no_accuracy": _safe_mean(group["yes_no_sum"], yes_no_count),
        "yes_no_count": yes_no_count,
    }


def _safe_mean(total: float, count: int) -> float:
    """Return rounded mean or 0.0 for empty groups."""
    if count <= 0:
        return 0.0
    return round(total / count, 4)


def _safe_text(value: Any) -> str:
    """Convert arbitrary values to safe text for metric computation."""
    if value is None:
        return ""
    return str(value)


def _gold_answer_alternatives(gold: Any) -> list[str]:
    """Return acceptable gold answer strings, including MC label/text aliases."""
    values: list[Any] = [_get_field(gold, "answer"), _get_field(gold, "answer_label")]

    metadata = _get_field(gold, "metadata")
    if isinstance(metadata, dict):
        values.extend([
            metadata.get("answer_text"),
            metadata.get("correct_answer_text"),
            metadata.get("gold_answer_text"),
        ])

    label = _first_present(values[:2])
    choices = _get_field(gold, "choices")
    if label is not None and isinstance(choices, list):
        values.append(_choice_text(str(label), choices))

    alternatives: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _safe_text(value).strip()
        if not text:
            continue
        normalized_key = normalize_answer(text)
        if normalized_key in seen:
            continue
        alternatives.append(text)
        seen.add(normalized_key)
    return alternatives or [""]


def _first_present(values: list[Any]) -> Any:
    for value in values:
        if value is not None and str(value).strip():
            return value
    return None


def _choice_text(label: str, choices: list[Any]) -> str | None:
    label = label.strip().upper()
    if len(label) != 1 or not label.isalpha():
        return None
    index = ord(label) - ord("A")
    if 0 <= index < len(choices):
        return _safe_text(choices[index])
    return None


def _first_field(item: Any, names: tuple[str, ...]) -> Any:
    """Return the first present field from a dict-like or object-like item."""
    for name in names:
        value = _get_field(item, name)
        if value is not None:
            return value
    return None


def _get_field(item: Any, name: str) -> Any:
    """Get a field from dict-like or object-like input."""
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _modality(item: Any) -> str | None:
    """Extract primary modality from direct fields, task_type, or metadata."""
    direct = _get_field(item, "modality")
    if direct:
        return normalize_modality(direct)
    task_type = _get_field(item, "task_type")
    if task_type:
        return normalize_modality(task_type)
    metadata = _get_field(item, "metadata")
    if isinstance(metadata, dict):
        if metadata.get("modality"):
            return normalize_modality(metadata["modality"])
        sample_metadata = metadata.get("sample_metadata")
        if isinstance(sample_metadata, dict):
            if sample_metadata.get("modality"):
                return normalize_modality(sample_metadata["modality"])
            if sample_metadata.get("task_type"):
                return normalize_modality(sample_metadata["task_type"])
    return MODALITY_UNKNOWN_OTHER


def _question_task_type(item: Any) -> str | None:
    """Extract legacy question-task metadata when present."""
    metadata = _get_field(item, "metadata")
    if isinstance(metadata, dict):
        for key in ("question_task_type", "legacy_task_type"):
            if metadata.get(key):
                return str(metadata[key])
        sample_metadata = metadata.get("sample_metadata")
        if isinstance(sample_metadata, dict):
            for key in ("question_task_type", "legacy_task_type"):
                if sample_metadata.get(key):
                    return str(sample_metadata[key])
    return None
