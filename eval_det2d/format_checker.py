#!/usr/bin/env python3
"""Validate 2D detection submissions against the benchmark spec."""

import json
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image


REQUIRED_BOX_KEYS = {"class", "x1", "y1", "x2", "y2"}


def _load_json(path: str) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        with open(path) as f:
            return json.load(f), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)


def _get_image_size(image_path: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, None, str(exc)


def _check_single_box(box: Dict, width: Optional[int], height: Optional[int]) -> List[str]:
    errors = []
    missing = REQUIRED_BOX_KEYS - set(box.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
        return errors

    for key in ("x1", "y1", "x2", "y2"):
        if not isinstance(box[key], (int, float)):
            errors.append(f"{key} must be numeric")

    if errors:
        return errors

    if box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
        errors.append("box has non-positive width/height")

    if width is not None:
        if not (0 <= box["x1"] < width and 0 < box["x2"] <= width):
            errors.append(f"x coords out of bounds for width={width}")
    if height is not None:
        if not (0 <= box["y1"] < height and 0 < box["y2"] <= height):
            errors.append(f"y coords out of bounds for height={height}")

    if "score" in box and not (isinstance(box["score"], (int, float)) and 0.0 <= box["score"] <= 1.0):
        errors.append("score must be a float in [0, 1]")

    if not isinstance(box["class"], str) or not box["class"].strip():
        errors.append("class must be a non-empty string")

    return errors


def check_prediction_file(prediction_path: str, image_path: Optional[str] = None) -> Dict:
    result = {"exists": False, "valid": False, "errors": []}
    if not os.path.isfile(prediction_path):
        result["errors"].append(f"prediction.json not found: {prediction_path}")
        return result
    result["exists"] = True

    payload, err = _load_json(prediction_path)
    if err:
        result["errors"].append(f"JSON parse error: {err}")
        return result

    if not isinstance(payload, dict):
        result["errors"].append("prediction.json must contain a JSON object")
        return result

    boxes = payload.get("boxes")
    if not isinstance(boxes, list):
        result["errors"].append("prediction.json must contain a list field 'boxes'")
        return result

    width = height = None
    if image_path and os.path.isfile(image_path):
        width, height, img_err = _get_image_size(image_path)
        if img_err:
            result["errors"].append(f"failed to read image size: {img_err}")

    for idx, box in enumerate(boxes):
        if not isinstance(box, dict):
            result["errors"].append(f"boxes[{idx}] must be an object")
            continue
        box_errors = _check_single_box(box, width, height)
        result["errors"].extend([f"boxes[{idx}]: {msg}" for msg in box_errors])

    if not result["errors"]:
        result["valid"] = True
    return result


def check_submission(agent_dir: str, patient_ids: List, public_dir: Optional[str] = None) -> Dict:
    """Validate prediction.json for each patient."""
    report = {
        "submission_format_valid": False,
        "output_format_valid": False,
        "per_patient": {},
        "errors": [],
    }

    all_valid = True
    for pid in patient_ids:
        image_path = os.path.join(public_dir, pid, "image.png") if public_dir else None
        pred_path = os.path.join(agent_dir, pid, "prediction.json")
        pred_result = check_prediction_file(pred_path, image_path)
        report["per_patient"][pid] = {"prediction": pred_result}
        if not pred_result["valid"]:
            all_valid = False
            report["errors"].extend([f"{pid}: {msg}" for msg in pred_result["errors"]])

    report["output_format_valid"] = all_valid
    report["submission_format_valid"] = all_valid
    return report
