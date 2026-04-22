#!/usr/bin/env python3
"""Stage a study-level MIMIC-CXR report-generation pilot.

The raw dataset is a flat directory of paired files with names like:
  p10_p10046166_s51738740_<uuid>.jpg
  p10_p10046166_s51738740_<uuid>.txt

The canonical raw study ID is the first three underscore-delimited fields:
  p10_p10046166_s51738740

For each raw study ID in the configured pilot split, this script creates a
local staged benchmark tree under:

  data/MIMIC_CXR_Report/pilot_10/
    public/CXR0001/
      images/01.jpg
      images/02.jpg
      manifest.json
    private/CXR0001/
      report.txt
      manifest.json
    manifest.csv

Public cases use neutral benchmark case IDs so later agent runs do not depend
on raw MIMIC study identifiers. The staged reference text is findings-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from report_scorer import (
    OBSERVATION_SCHEMA,
    extract_findings_text,
    extract_observations,
    positive_labels,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "mimic-cxr-report-task" / "config.yaml"


def load_simple_yaml(path: Path) -> dict[str, object]:
    """Parse a small flat YAML-like config without external dependencies."""
    data: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = ""
            continue
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            value = value[1:-1]
        if value.isdigit():
            data[key] = int(value)
        else:
            data[key] = value
    return data


def parse_study_id(filename: str) -> str:
    parts = filename.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected raw filename format: {filename}")
    return "_".join(parts[:3])


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_case_id(index: int, prefix: str) -> str:
    return f"{prefix}{index:04d}"


def stage_pilot(config_path: Path, force: bool = False) -> Path:
    config = load_simple_yaml(config_path)

    source_root = Path(str(config["source_root"]))
    split_path = SCRIPT_DIR / "splits" / str(config["pilot_split"])
    data_dir_name = str(config.get("data_dir_name", "MIMIC_CXR_Report"))
    split_name = str(config.get("staged_split_name", "pilot_10"))
    case_prefix = str(config.get("case_id_prefix", "CXR"))
    public_images_subdir = str(config.get("public_images_subdir", "images"))

    out_root = REPO_ROOT / "data" / data_dir_name / split_name
    public_root = out_root / "public"
    private_root = out_root / "private"

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Pilot split not found: {split_path}")

    if out_root.exists():
        if not force:
            raise FileExistsError(
                f"Staged output already exists: {out_root}\n"
                "Re-run with --force to replace it."
            )
        shutil.rmtree(out_root)

    public_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)

    study_ids = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows: list[dict[str, object]] = []

    for index, study_id in enumerate(study_ids, start=1):
        image_files = sorted(source_root.glob(f"{study_id}_*.jpg"))
        report_files = sorted(source_root.glob(f"{study_id}_*.txt"))

        if not image_files:
            raise RuntimeError(f"No JPEG files found for study {study_id}")
        if not report_files:
            raise RuntimeError(f"No report files found for study {study_id}")
        if len(image_files) != len(report_files):
            raise RuntimeError(
                f"Study {study_id} has {len(image_files)} JPEGs but "
                f"{len(report_files)} report files"
            )
        for path in [*image_files, *report_files]:
            if parse_study_id(path.name) != study_id:
                raise RuntimeError(f"Mis-grouped file for {study_id}: {path.name}")

        report_texts = [path.read_text(encoding="utf-8") for path in report_files]
        unique_report_texts = list(dict.fromkeys(report_texts))
        if len(unique_report_texts) != 1:
            raise RuntimeError(
                f"Study {study_id} has {len(unique_report_texts)} unique reports; "
                "expected exactly one canonical report per study"
            )
        canonical_report = unique_report_texts[0]
        findings_report = extract_findings_text(canonical_report, strict=True)

        case_id = build_case_id(index, case_prefix)
        public_case_dir = public_root / case_id
        private_case_dir = private_root / case_id
        staged_images_dir = public_case_dir / public_images_subdir
        staged_images_dir.mkdir(parents=True, exist_ok=True)
        private_case_dir.mkdir(parents=True, exist_ok=True)

        width = max(2, len(str(len(image_files))))
        staged_image_names = []
        for image_index, image_path in enumerate(image_files, start=1):
            staged_name = f"{image_index:0{width}d}{image_path.suffix.lower()}"
            copy_file(image_path, staged_images_dir / staged_name)
            staged_image_names.append(staged_name)

        report_out = private_case_dir / str(config.get("output_filename", "report.txt"))
        report_out.write_text(findings_report + "\n", encoding="utf-8")

        report_hash = sha256_text(findings_report)
        labels = extract_observations(findings_report)
        labels_payload = {
            "schema_version": int(config.get("observation_schema_version", 1)),
            "reference_text_mode": str(config.get("reference_text_mode", "findings_only")),
            "labels": labels,
            "positive_labels": positive_labels(labels),
            "label_descriptions": {
                label: OBSERVATION_SCHEMA[label]["description"]
                for label in OBSERVATION_SCHEMA
            },
        }
        (private_case_dir / "labels.json").write_text(
            json.dumps(labels_payload, indent=2) + "\n",
            encoding="utf-8",
        )

        public_manifest = {
            "case_id": case_id,
            "task_id": str(config["task_id"]),
            "split_name": split_name,
            "image_count": len(staged_image_names),
            "image_files": staged_image_names,
        }
        (public_case_dir / "manifest.json").write_text(
            json.dumps(public_manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        private_manifest = {
            "case_id": case_id,
            "task_id": str(config["task_id"]),
            "split_name": split_name,
            "raw_study_id": study_id,
            "image_count": len(image_files),
            "source_image_files": [path.name for path in image_files],
            "source_report_files": [path.name for path in report_files],
            "reference_report_file": report_out.name,
            "reference_labels_file": "labels.json",
            "reference_text_mode": str(config.get("reference_text_mode", "findings_only")),
            "report_sha256": report_hash,
            "report_char_count": len(findings_report),
        }
        (private_case_dir / "manifest.json").write_text(
            json.dumps(private_manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        rows.append({
            "case_id": case_id,
            "raw_study_id": study_id,
            "image_count": len(image_files),
            "public_image_files": ";".join(staged_image_names),
            "source_image_files": ";".join(path.name for path in image_files),
            "source_report_files": ";".join(path.name for path in report_files),
            "reference_report_file": report_out.name,
            "reference_labels_file": "labels.json",
            "reference_text_mode": str(config.get("reference_text_mode", "findings_only")),
            "report_sha256": report_hash,
            "report_char_count": len(findings_report),
            "public_case_dir": str(public_case_dir.relative_to(REPO_ROOT)),
            "private_case_dir": str(private_case_dir.relative_to(REPO_ROOT)),
        })

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Staged {len(rows)} study-level cases -> {out_root}")
    for row in rows:
        print(
            f"  {row['case_id']}: {row['raw_study_id']} "
            f"({row['image_count']} views)"
        )
    print(f"Manifest -> {manifest_path}")
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage the 10-study pilot for eval_report_gen."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to mimic-cxr-report-task config.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the staged output root if it already exists.",
    )
    args = parser.parse_args()

    stage_pilot(Path(args.config), force=args.force)


if __name__ == "__main__":
    main()
