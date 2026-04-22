"""Modality-based taxonomy for the medical VQA benchmark.

The primary benchmark grouping is intentionally modality-based. Legacy
question-task labels such as abnormality, location, comparison, and multi_image
are preserved in metadata for diagnostics, but they are no longer the primary
slice used by manifests or score breakdowns.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


MODALITY_PATHOLOGY = "pathology"
MODALITY_RADIOLOGY_XRAY = "radiology_xray"
MODALITY_RADIOLOGY_CT = "radiology_ct"
MODALITY_RADIOLOGY_MRI = "radiology_mri"
MODALITY_ULTRASOUND = "ultrasound"
MODALITY_CLINICAL_MULTIMODAL = "clinical_multimodal"
MODALITY_UNKNOWN_OTHER = "unknown_other"

SUPPORTED_MODALITIES = {
    MODALITY_PATHOLOGY,
    MODALITY_RADIOLOGY_XRAY,
    MODALITY_RADIOLOGY_CT,
    MODALITY_RADIOLOGY_MRI,
    MODALITY_ULTRASOUND,
    MODALITY_CLINICAL_MULTIMODAL,
    MODALITY_UNKNOWN_OTHER,
}

_EXPLICIT_ALIASES = {
    "pathology": MODALITY_PATHOLOGY,
    "histology": MODALITY_PATHOLOGY,
    "histopathology": MODALITY_PATHOLOGY,
    "microscopy": MODALITY_PATHOLOGY,
    "microscopic": MODALITY_PATHOLOGY,
    "pathology_figure": MODALITY_PATHOLOGY,
    "pathology figure": MODALITY_PATHOLOGY,
    "gross": MODALITY_PATHOLOGY,
    "gross pathology": MODALITY_PATHOLOGY,
    "xray": MODALITY_RADIOLOGY_XRAY,
    "x ray": MODALITY_RADIOLOGY_XRAY,
    "x-ray": MODALITY_RADIOLOGY_XRAY,
    "radiograph": MODALITY_RADIOLOGY_XRAY,
    "chest xray": MODALITY_RADIOLOGY_XRAY,
    "chest x ray": MODALITY_RADIOLOGY_XRAY,
    "chest x-ray": MODALITY_RADIOLOGY_XRAY,
    "cxr": MODALITY_RADIOLOGY_XRAY,
    "ct": MODALITY_RADIOLOGY_CT,
    "computed tomography": MODALITY_RADIOLOGY_CT,
    "cta": MODALITY_RADIOLOGY_CT,
    "mri": MODALITY_RADIOLOGY_MRI,
    "mr": MODALITY_RADIOLOGY_MRI,
    "magnetic resonance": MODALITY_RADIOLOGY_MRI,
    "ultrasound": MODALITY_ULTRASOUND,
    "us": MODALITY_ULTRASOUND,
    "sonography": MODALITY_ULTRASOUND,
    "sonogram": MODALITY_ULTRASOUND,
    "clinical_multimodal": MODALITY_CLINICAL_MULTIMODAL,
    "clinical multimodal": MODALITY_CLINICAL_MULTIMODAL,
    "clinical_reasoning": MODALITY_CLINICAL_MULTIMODAL,
    "clinical reasoning": MODALITY_CLINICAL_MULTIMODAL,
    "multimodal": MODALITY_CLINICAL_MULTIMODAL,
    "multi modal": MODALITY_CLINICAL_MULTIMODAL,
    "multi-image": MODALITY_CLINICAL_MULTIMODAL,
    "multi image": MODALITY_CLINICAL_MULTIMODAL,
}

_FIELD_PRIORITY = (
    "modality",
    "image_modality",
    "modality_type",
    "imaging_modality",
    "study_type",
    "scan_type",
    "image_type",
    "domain",
    "system",
    "category",
)


def normalize_modality(value: Any) -> str:
    """Normalize a free-form modality value into the supported taxonomy."""
    if value in (None, ""):
        return MODALITY_UNKNOWN_OTHER
    text = _normalize_text(value)
    if not text:
        return MODALITY_UNKNOWN_OTHER
    if text in SUPPORTED_MODALITIES:
        return text
    if text in _EXPLICIT_ALIASES:
        return _EXPLICIT_ALIASES[text]
    return _classify_text(text)


def infer_modality(
    *,
    dataset: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    image_path: str | Path | None = None,
    image_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    question: str | None = None,
    image_count: int = 0,
) -> str:
    """Infer the primary benchmark modality from public sample context."""
    metadata = metadata or {}
    normalized_metadata = {str(key).strip().lower(): value for key, value in metadata.items()}

    for key in _FIELD_PRIORITY:
        if key in normalized_metadata:
            modality = normalize_modality(normalized_metadata.get(key))
            if modality != MODALITY_UNKNOWN_OTHER:
                return modality

    text_parts: list[str] = []
    if dataset:
        text_parts.append(str(dataset))
    if question:
        text_parts.append(str(question))
    if image_path:
        text_parts.append(str(image_path))
    for path in image_paths or ():
        text_parts.append(str(path))
    for key in ("organ", "keyword", "video_id", "source_dataset", "source_record_keys"):
        value = normalized_metadata.get(key)
        if value is not None:
            text_parts.append(" ".join(str(item) for item in value) if isinstance(value, list) else str(value))

    text = _normalize_text(" ".join(text_parts))
    classified = _classify_text(text)
    if classified != MODALITY_UNKNOWN_OTHER:
        return classified

    dataset_text = _normalize_text(dataset or normalized_metadata.get("source_dataset") or "")
    if "pathvqa" in dataset_text or "path vqa" in dataset_text:
        return MODALITY_PATHOLOGY
    if "medframe" in dataset_text or _truthy(normalized_metadata.get("is_multi_image")) or image_count > 1:
        return MODALITY_CLINICAL_MULTIMODAL
    return MODALITY_UNKNOWN_OTHER


def _classify_text(text: str) -> str:
    if _contains_any(text, ("histology", "histopathology", "microscopy", "pathology", "gross pathology")):
        return MODALITY_PATHOLOGY
    if _contains_any(text, ("computed tomography", " ct ", "ct-", "_ct", " cta ", "cta-", "tomography")):
        return MODALITY_RADIOLOGY_CT
    if _contains_any(text, ("magnetic resonance", " mri ", "mri-", "_mri", " mr ", "mr-")):
        return MODALITY_RADIOLOGY_MRI
    if _contains_any(text, ("ultrasound", " sonography", "sonogram", " us ", "us-", "_us")):
        return MODALITY_ULTRASOUND
    if _contains_any(text, ("x ray", "x-ray", "xray", "radiograph", " cxr ", "cxr-", "_cxr")):
        return MODALITY_RADIOLOGY_XRAY
    if _contains_any(text, ("clinical reasoning", "clinical multimodal", "multimodal", "multi image", "multi study")):
        return MODALITY_CLINICAL_MULTIMODAL
    return MODALITY_UNKNOWN_OTHER


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(needle in padded for needle in needles)


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower().replace("/", " ").replace("\\", " ")
    text = re.sub(r"[_:]+", " ", text)
    text = re.sub(r"[^a-z0-9+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "multi_image", "multi-study"}
    return bool(value)
