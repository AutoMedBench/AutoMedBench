#!/usr/bin/env python3
"""Shared answer normalization helpers for VQA inference and scoring."""

from __future__ import annotations

import re
from typing import Any

VALID_LABELS = ("A", "B", "C", "D", "E")


def normalize_options(options: Any) -> dict[str, str]:
    if isinstance(options, dict):
        return {str(key).strip().upper(): str(value).strip() for key, value in options.items()}
    if isinstance(options, list):
        labels = VALID_LABELS[: len(options)]
        return {label: str(value).strip() for label, value in zip(labels, options)}
    return {}


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in VALID_LABELS:
        return text
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)
    return None


def extract_predicted_label(raw_output: str, options: dict[str, str]) -> str | None:
    if not raw_output:
        return None

    text = str(raw_output).strip()
    upper_text = text.upper()
    patterns = [
        r"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE|PREDICTED_LABEL)\s*[:\-]?\s*\(?([A-E])\)?\b",
        r"^\(?([A-E])\)?(?:[\).\s]|$)",
        r"\b([A-E])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper_text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            label = normalize_label(match.group(1))
            if label in options:
                return label

    normalized_options = {label: _normalize_text(value) for label, value in options.items()}
    normalized_text = _normalize_text(text)
    exact_matches = [label for label, value in normalized_options.items() if value and value == normalized_text]
    if len(exact_matches) == 1:
        return exact_matches[0]

    contained_matches = [label for label, value in normalized_options.items() if value and value in normalized_text]
    if len(contained_matches) == 1:
        return contained_matches[0]

    return None


def predicted_answer_text(label: str | None, options: dict[str, str]) -> str:
    if label and label in options:
        return options[label]
    return ""


def _normalize_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value.strip().lower())
    return re.sub(r"[^a-z0-9 ]+", "", text)
