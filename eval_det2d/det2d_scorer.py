#!/usr/bin/env python3
"""Metrics for 2D bounding-box detection tasks."""

import json
import os
from typing import Dict, List, Tuple


def _load_boxes(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        payload = json.load(f)
    return payload.get("boxes", [])


def _iou(box_a: dict, box_b: dict) -> float:
    xa1, ya1, xa2, ya2 = box_a["x1"], box_a["y1"], box_a["x2"], box_a["y2"]
    xb1, yb1, xb2, yb2 = box_b["x1"], box_b["y1"], box_b["x2"], box_b["y2"]

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _class_name(box: dict) -> str:
    return str(box.get("class", "abnormality")).strip().lower()


def _classes_match(pred_class: str, gt_class: str) -> bool:
    return pred_class == gt_class or pred_class == "abnormality" or gt_class == "abnormality"


def match_boxes(pred_boxes: List[Dict], gt_boxes: List[Dict], iou_threshold: float = 0.4) -> Dict:
    """Greedy class-aware matching ordered by prediction score."""
    pred_sorted = sorted(pred_boxes, key=lambda x: x.get("score", 1.0), reverse=True)
    used_gt = set()
    matches = []
    tp = fp = 0

    for pred_idx, pred in enumerate(pred_sorted):
        best_gt_idx = None
        best_iou = 0.0
        pred_class = _class_name(pred)
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue
            gt_class = _class_name(gt)
            if not _classes_match(pred_class, gt_class):
                continue
            iou = _iou(pred, gt)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is None:
            fp += 1
            matches.append({"pred_index": pred_idx, "gt_index": None, "iou": 0.0, "matched": False})
        else:
            tp += 1
            used_gt.add(best_gt_idx)
            matches.append({"pred_index": pred_idx, "gt_index": best_gt_idx, "iou": best_iou, "matched": True})

    fn = len(gt_boxes) - len(used_gt)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def _collect_classes(predictions_by_pid: Dict[str, List[Dict]], gt_by_pid: Dict[str, List[Dict]]) -> List[str]:
    classes = set()
    for boxes in list(predictions_by_pid.values()) + list(gt_by_pid.values()):
        for box in boxes:
            classes.add(_class_name(box))
    return sorted(classes or {"abnormality"})


def _compute_ap_101(tp_flags: List[int], fp_flags: List[int], total_gt: int) -> Tuple[float, List[float], List[float]]:
    if total_gt == 0:
        return 0.0, [], []

    cum_tp = []
    cum_fp = []
    t = f = 0
    for tp, fp in zip(tp_flags, fp_flags):
        t += tp
        f += fp
        cum_tp.append(t)
        cum_fp.append(f)

    precisions = []
    recalls = []
    for t, f in zip(cum_tp, cum_fp):
        precisions.append(t / (t + f) if (t + f) else 0.0)
        recalls.append(t / total_gt)

    ap = 0.0
    for recall_threshold in range(101):
        r = recall_threshold / 100.0
        precision_at_r = 0.0
        for precision, recall in zip(precisions, recalls):
            if recall >= r and precision > precision_at_r:
                precision_at_r = precision
        ap += precision_at_r
    ap /= 101.0
    return ap, precisions, recalls


def compute_map(predictions_by_pid: Dict[str, List[Dict]], gt_by_pid: Dict[str, List[Dict]], iou_threshold: float) -> Dict:
    """Compute per-class AP and mean AP using 101-point interpolation."""
    classes = _collect_classes(predictions_by_pid, gt_by_pid)
    ap_per_class = {}

    for class_name in classes:
        total_gt = 0
        detections = []

        for pid, gt_boxes_all in gt_by_pid.items():
            gt_boxes = [b for b in gt_boxes_all if _classes_match(class_name, _class_name(b))]
            pred_boxes = [b for b in predictions_by_pid.get(pid, []) if _classes_match(class_name, _class_name(b))]
            total_gt += len(gt_boxes)

            pred_sorted = sorted(pred_boxes, key=lambda x: x.get("score", 1.0), reverse=True)
            used_gt = set()

            for pred in pred_sorted:
                best_gt_idx = None
                best_iou = 0.0
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_idx in used_gt:
                        continue
                    iou = _iou(pred, gt)
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx is None:
                    detections.append({"score": float(pred.get("score", 1.0)), "tp": 0, "fp": 1})
                else:
                    used_gt.add(best_gt_idx)
                    detections.append({"score": float(pred.get("score", 1.0)), "tp": 1, "fp": 0})

        detections.sort(key=lambda x: x["score"], reverse=True)
        tp_flags = [d["tp"] for d in detections]
        fp_flags = [d["fp"] for d in detections]
        ap, precisions, recalls = _compute_ap_101(tp_flags, fp_flags, total_gt)
        ap_per_class[class_name] = {
            "ap": ap,
            "n_gt": total_gt,
            "n_predictions": len(detections),
            "precision_curve": precisions,
            "recall_curve": recalls,
        }

    mean_ap = (
        sum(v["ap"] for v in ap_per_class.values()) / len(ap_per_class)
        if ap_per_class else 0.0
    )
    return {"mAP": mean_ap, "per_class": ap_per_class}


def score_all(pred_dir: str, gt_dir: str, patient_ids: List, iou_threshold: float = 0.4) -> Dict:
    """Score all patients with mAP@IoU and per-patient precision/recall/F1."""
    per_patient = {}
    total_tp = total_fp = total_fn = 0
    predictions_by_pid = {}
    gt_by_pid = {}

    for pid in patient_ids:
        pred_path = os.path.join(pred_dir, pid, "prediction.json")
        gt_path = os.path.join(gt_dir, pid, "boxes.json")
        pred_boxes = _load_boxes(pred_path) if os.path.isfile(pred_path) else []
        gt_boxes = _load_boxes(gt_path)

        predictions_by_pid[pid] = pred_boxes
        gt_by_pid[pid] = gt_boxes

        if not os.path.isfile(pred_path):
            per_patient[pid] = {
                "tp": 0,
                "fp": 0,
                "fn": len(gt_boxes),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "gt_box_count": len(gt_boxes),
                "pred_box_count": 0,
                "completed": False,
            }
            total_fn += len(gt_boxes)
            continue

        matched = match_boxes(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
        per_patient[pid] = {
            **matched,
            "gt_box_count": len(gt_boxes),
            "pred_box_count": len(pred_boxes),
            "completed": True,
        }
        total_tp += matched["tp"]
        total_fp += matched["fp"]
        total_fn += matched["fn"]

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    micro_f1 = 0.0 if (micro_precision + micro_recall) == 0 else (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )
    completion_rate = sum(1 for r in per_patient.values() if r["completed"]) / max(len(patient_ids), 1)
    map_result = compute_map(predictions_by_pid, gt_by_pid, iou_threshold=iou_threshold)

    return {
        "per_patient": per_patient,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "mAP": map_result["mAP"],
        "per_class_ap": map_result["per_class"],
        "completion_rate": completion_rate,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "iou_threshold": iou_threshold,
        "n_patients": len(patient_ids),
    }
