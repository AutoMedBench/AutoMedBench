#!/usr/bin/env python3
"""Stage a small VinDr-CXR detection benchmark from an existing local download.

This script does NOT fetch the VinDr-CXR/VinBigData files for you. Instead it:
1. reads a local download,
2. selects a small subset (default: 10 samples),
3. converts source images to PNG,
4. writes MedAgentsBench-style public/private folders.
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr -= arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr /= max_val
    return (arr * 255).clip(0, 255).astype(np.uint8)


def dicom_to_png(src: Path, dst: Path) -> Tuple[int, int]:
    ds = pydicom.dcmread(str(src))
    pixels = ds.pixel_array
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels
    img = Image.fromarray(normalize_to_uint8(pixels)).convert("L")
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    return img.size


def standard_image_to_png(src: Path, dst: Path) -> Tuple[int, int]:
    with Image.open(src) as img:
        img = img.convert("L")
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst)
        return img.size


def convert_image_to_png(src: Path, dst: Path) -> Tuple[int, int]:
    if src.suffix.lower() in {".dicom", ".dcm"}:
        return dicom_to_png(src, dst)
    return standard_image_to_png(src, dst)


def resolve_annotation_csv(source_dir: Path, split: str) -> Path:
    candidates = [
        source_dir / f"annotations_{split}.csv",
        source_dir / f"{split}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(str(path.name) for path in candidates)
    raise FileNotFoundError(f"missing annotation CSV for split '{split}'; tried: {expected}")


def resolve_image_path(split_dir: Path, image_id: str) -> Path:
    candidates = [
        split_dir / f"{image_id}.dicom",
        split_dir / f"{image_id}.dcm",
        split_dir / f"{image_id}.png",
        split_dir / f"{image_id}.jpg",
        split_dir / f"{image_id}.jpeg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"missing image for {image_id} in {split_dir}; tried: {expected}")


def try_resolve_image_path(split_dir: Path, image_id: str) -> Optional[Path]:
    try:
        return resolve_image_path(split_dir, image_id)
    except FileNotFoundError:
        return None


def load_annotations(csv_path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            class_name = (row.get("class_name") or "").strip()
            if not class_name or class_name.lower() == "no finding":
                grouped.setdefault(image_id, [])
                continue
            box = {
                "class": "abnormality",
                "source_class": class_name,
                "x1": float(row["x_min"]),
                "y1": float(row["y_min"]),
                "x2": float(row["x_max"]),
                "y2": float(row["y_max"]),
            }
            grouped.setdefault(image_id, []).append(box)
    return grouped


def select_cases(
    annotations: Dict[str, List[dict]],
    total: int,
    seed: int,
    positive_count: Optional[int] = None,
) -> List[str]:
    rng = random.Random(seed)
    positives = [k for k, v in annotations.items() if v]
    negatives = [k for k, v in annotations.items() if not v]

    if positive_count is None:
        positive_count = max(total - 2, 0)
    negative_count = max(total - positive_count, 0)

    if len(positives) < positive_count or len(negatives) < negative_count:
        raise ValueError("Not enough positive/negative samples for the requested subset")

    selected = rng.sample(positives, positive_count) + rng.sample(negatives, negative_count)
    rng.shuffle(selected)
    return selected


def stage_vindr(source_dir: Path, output_name: str, split: str, num_samples: int, seed: int):
    split_dir = source_dir / split
    ann_path = resolve_annotation_csv(source_dir, split)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing image directory: {split_dir}")

    annotations = load_annotations(ann_path)
    available_annotations: Dict[str, List[dict]] = {}
    missing_image_count = 0
    for image_id, boxes in annotations.items():
        image_path = try_resolve_image_path(split_dir, image_id)
        if image_path is None:
            missing_image_count += 1
            continue
        available_annotations[image_id] = boxes

    if not available_annotations:
        raise FileNotFoundError(f"no annotated samples have matching image files under {split_dir}")

    if missing_image_count:
        print(
            f"Warning: skipped {missing_image_count} annotated image_ids with no matching image file in {split_dir}"
        )
    selected = select_cases(available_annotations, total=num_samples, seed=seed)

    out_root = DATA_DIR / output_name
    public_dir = out_root / "public"
    private_dir = out_root / "private"
    gt_rows = []

    for idx, image_id in enumerate(selected, start=1):
        patient_id = f"VINDR_{idx:06d}"
        image_path = resolve_image_path(split_dir, image_id)

        width, height = convert_image_to_png(image_path, public_dir / patient_id / "image.png")
        boxes = available_annotations.get(image_id, [])
        gt_payload = {
            "image_id": image_id,
            "width": width,
            "height": height,
            "boxes": boxes,
        }
        target_path = private_dir / patient_id / "boxes.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w") as f:
            json.dump(gt_payload, f, indent=2)

        gt_rows.append({
            "patient_id": patient_id,
            "image_id": image_id,
            "num_boxes": len(boxes),
            "has_finding": int(bool(boxes)),
        })

    private_dir.mkdir(parents=True, exist_ok=True)
    with open(private_dir / "ground_truth.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "image_id", "num_boxes", "has_finding"])
        writer.writeheader()
        writer.writerows(gt_rows)

    print(f"Staged {len(selected)} VinDr-CXR cases to {out_root}")


def main():
    parser = argparse.ArgumentParser(description="Stage a small VinDr-CXR 2D detection dataset")
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Local dataset root containing train|test/ and either annotations_<split>.csv or <split>.csv",
    )
    parser.add_argument("--output-name", default="VinDrCXR_Detection10", help="Folder name under data/")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split to sample from")
    parser.add_argument("--num-samples", type=int, default=10, help="How many samples to stage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset selection")
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing output folder before writing")
    args = parser.parse_args()

    out_root = DATA_DIR / args.output_name
    if out_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_root)

    stage_vindr(
        source_dir=Path(args.source_dir),
        output_name=args.output_name,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
