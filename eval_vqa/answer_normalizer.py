"""Answer normalization helpers for conventional medical VQA scoring."""

from __future__ import annotations

import re
import string


MEDICAL_ABBREVIATIONS = {
    # Anatomical direction — only expand as standalone tokens
    "r": "right",
    "l": "left",
    "bil": "bilateral",
    "bilat": "bilateral",
    # Cardiovascular structures (VQA-RAD common)
    "ivc": "inferior vena cava",
    "svc": "superior vena cava",
    "lv": "left ventricle",
    "rv": "right ventricle",
    "la": "left atrium",
    "ra": "right atrium",
    "lad": "left anterior descending",
    "mca": "middle cerebral artery",
    "ica": "internal carotid artery",
    "pca": "posterior cerebral artery",
    "aca": "anterior cerebral artery",
    # Common radiology abbreviations
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "gi": "gastrointestinal",
    "gu": "genitourinary",
    "cns": "central nervous system",
}

YES_NO_MAP = {
    "yes": "yes",
    "y": "yes",
    "yeah": "yes",
    "yep": "yes",
    "true": "yes",
    "present": "yes",
    "positive": "yes",
    "no": "no",
    "n": "no",
    "nope": "no",
    "false": "no",
    "absent": "no",
    "negative": "no",
}

NUMBER_WORDS = {
    "zero": "0",
    "none": "0",
    "one": "1",
    "single": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}

PUNCTUATION_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Normalize free-text answers for exact match and token F1.

    The normalization intentionally stays conservative:

    1. Lower-case.
    2. Strip punctuation.
    3. Normalize whitespace.
    4. Map common yes/no synonyms to `yes` or `no`.
    5. Normalize common number words such as `two` to `2`.

    Args:
        text: Raw model or reference answer.

    Returns:
        A normalized answer string.
    """
    normalized = str(text or "").lower()
    normalized = normalized.translate(PUNCTUATION_TRANSLATION)
    normalized = _normalize_whitespace(normalized)

    if normalized in YES_NO_MAP:
        return YES_NO_MAP[normalized]

    tokens = normalized.split()
    tokens = [NUMBER_WORDS.get(t, t) for t in tokens]
    tokens = [MEDICAL_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(tokens)


def is_yes_no_answer(text: str) -> bool:
    """Return true when an answer is a yes/no answer or a supported synonym."""
    normalized = normalize_answer(text)
    return normalized in {"yes", "no"}


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim edges."""
    return re.sub(r"\s+", " ", text).strip()

