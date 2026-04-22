#!/usr/bin/env python3
"""Stage MedXpertQA MM into MedAgentsBench public/private layout."""

from __future__ import annotations

import argparse
import io
import csv
import json
import os
import random
import shutil
import zipfile
from typing import Any


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _validate_output_root(output_root: str) -> None:
    normalized = os.path.abspath(output_root)
    if normalized in {"/", os.path.abspath(os.sep)}:
        raise ValueError("Refusing to stage into filesystem root.")


def _load_dataset_bundle(dataset_source: str, dataset_subset: str, cache_dir: str | None = None):
    if os.path.isdir(dataset_source):
        return _load_local_snapshot_bundle(os.path.abspath(dataset_source), dataset_subset)

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    load_kwargs: dict[str, Any] = {}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    dataset = load_dataset(dataset_source, dataset_subset, **load_kwargs)
    images_zip_path = hf_hub_download(
        repo_id=dataset_source,
        repo_type="dataset",
        filename="images.zip",
        cache_dir=cache_dir,
    )
    return {
        "splits": {
            "dev": list(dataset["dev"]),
            "test": list(dataset["test"]),
        },
        "image_loader": _build_zip_image_loader(images_zip_path),
        "images_zip_path": images_zip_path,
        "source_type": "huggingface_dataset",
    }


def _resolve_subset_dir(dataset_root: str, dataset_subset: str) -> str:
    candidates = [
        os.path.join(dataset_root, dataset_subset),
        dataset_root if os.path.basename(dataset_root) == dataset_subset else "",
    ]
    for candidate in candidates:
        if candidate and all(os.path.isfile(os.path.join(candidate, f"{split}.jsonl")) for split in ("dev", "test")):
            return candidate
    raise FileNotFoundError(
        f"Could not find {dataset_subset!r} split files under {dataset_root!r}. "
        f"Expected {dataset_subset}/dev.jsonl and {dataset_subset}/test.jsonl."
    )


def _resolve_images_zip(dataset_root: str, subset_dir: str) -> str:
    candidates = [
        os.path.join(dataset_root, "images.zip"),
        os.path.join(subset_dir, "images.zip"),
        os.path.join(os.path.dirname(subset_dir), "images.zip"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find images.zip for local snapshot under {dataset_root!r} or {subset_dir!r}."
    )


def _load_jsonl_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(record)
    return records


def _build_zip_image_loader(images_zip_path: str):
    from PIL import Image

    with zipfile.ZipFile(images_zip_path, "r") as archive:
        image_entries = {
            os.path.basename(info.filename): info.filename
            for info in archive.infolist()
            if not info.is_dir()
        }

    def load_image(image_name: str):
        entry_name = image_entries.get(os.path.basename(image_name))
        if not entry_name:
            raise FileNotFoundError(f"Image {image_name!r} was not found in {images_zip_path!r}")
        with zipfile.ZipFile(images_zip_path, "r") as archive:
            with archive.open(entry_name, "r") as handle:
                image = Image.open(io.BytesIO(handle.read()))
                image.load()
                return image

    return load_image


def _sort_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(samples, key=lambda sample: str(sample.get("id", "")))


def _load_local_snapshot_bundle(dataset_root: str, dataset_subset: str) -> dict[str, Any]:
    subset_dir = _resolve_subset_dir(dataset_root, dataset_subset)
    images_zip_path = _resolve_images_zip(dataset_root, subset_dir)
    return {
        "splits": {
            "dev": _sort_samples(_load_jsonl_records(os.path.join(subset_dir, "dev.jsonl"))),
            "test": _sort_samples(_load_jsonl_records(os.path.join(subset_dir, "test.jsonl"))),
        },
        "image_loader": _build_zip_image_loader(images_zip_path),
        "images_zip_path": images_zip_path,
        "source_type": "local_snapshot",
        "snapshot_root": dataset_root,
        "subset_dir": subset_dir,
    }


def _materialize_sample(
    sample: dict[str, Any],
    public_root: str,
    private_root: str,
    image_loader=None,
) -> str:
    question_id = str(sample["id"])
    public_dir = os.path.join(public_root, question_id)
    private_dir = os.path.join(private_root, question_id)
    image_dir = os.path.join(public_dir, "images")
    _ensure_dir(image_dir)
    _ensure_dir(private_dir)

    image_paths: list[str] = []
    for index, image_ref in enumerate(sample.get("images", [])):
        image = image_ref
        if not hasattr(image_ref, "save"):
            if image_loader is None:
                raise TypeError(f"Image loader is required for non-image references on question {question_id!r}")
            image = image_loader(str(image_ref))
        ext = _guess_ext(getattr(image, "format", None))
        filename = f"{question_id}-{chr(ord('a') + index)}.{ext}"
        dest_path = os.path.join(image_dir, filename)
        image.save(dest_path)
        image_paths.append(os.path.join("images", filename))

    options = _normalize_options(sample.get("options", []))
    answer_label = str(sample["label"]).strip().upper()

    question_payload = {
        "question_id": question_id,
        "question": sample["question"],
        "options": options,
        "images": image_paths,
        "medical_task": sample.get("medical_task", ""),
        "body_system": sample.get("body_system", ""),
        "question_type": sample.get("question_type", ""),
        "split": sample.get("_split", ""),
        "dataset": "MedXpertQA-MM",
    }
    answer_payload = {
        "question_id": question_id,
        "answer_label": answer_label,
        "answer_text": options.get(answer_label, ""),
    }

    with open(os.path.join(public_dir, "question.json"), "w", encoding="utf-8") as handle:
        json.dump(question_payload, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(private_dir, "answer.json"), "w", encoding="utf-8") as handle:
        json.dump(answer_payload, handle, indent=2, ensure_ascii=False)
    return question_id


def _normalize_options(options: Any) -> dict[str, str]:
    if isinstance(options, dict):
        return {str(key).strip().upper(): str(value).strip() for key, value in options.items()}
    if isinstance(options, list):
        labels = ["A", "B", "C", "D", "E"]
        if len(options) > len(labels):
            raise ValueError(f"Expected at most {len(labels)} options, got {len(options)}")
        return {labels[index]: str(value).strip() for index, value in enumerate(options)}
    raise TypeError(f"Unsupported options type: {type(options)!r}")


def _guess_ext(image_format: str | None) -> str:
    if not image_format:
        return "jpeg"
    image_format = image_format.lower()
    if image_format == "jpeg":
        return "jpeg"
    return image_format


def _write_ground_truth_csv(private_root: str, rows: list[dict[str, str]]) -> None:
    path = os.path.join(private_root, "ground_truth.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question_id", "answer_label", "medical_task", "body_system", "question_type", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_subset(output_root: str, filename: str, question_ids: list[str]) -> None:
    with open(os.path.join(output_root, filename), "w", encoding="utf-8") as handle:
        for question_id in question_ids:
            handle.write(question_id + "\n")


def _build_subsets(test_ids: list[str], smoke_size: int, calibration_size: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    ordered_ids = sorted(test_ids)
    smoke_ids = sorted(rng.sample(ordered_ids, min(smoke_size, len(ordered_ids))))
    remaining_ids = [question_id for question_id in ordered_ids if question_id not in set(smoke_ids)]
    calibration_pool = remaining_ids if remaining_ids else ordered_ids
    calibration_ids = sorted(rng.sample(calibration_pool, min(calibration_size, len(calibration_pool))))
    return {
        "seed": seed,
        "smoke_ids": smoke_ids,
        "calibration_ids": calibration_ids,
        "smoke_count": len(smoke_ids),
        "calibration_count": len(calibration_ids),
        "overlap_count": len(set(smoke_ids) & set(calibration_ids)),
    }


def _validate_sample_ids(split_name: str, samples: list[dict[str, Any]], seen_ids: set[str]) -> None:
    for sample in samples:
        question_id = str(sample.get("id", "")).strip()
        if not question_id:
            raise ValueError(f"Encountered empty question id in split {split_name!r}")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question id detected while staging: {question_id}")
        seen_ids.add(question_id)


def stage_dataset(
    output_root: str,
    smoke_size: int,
    calibration_size: int,
    seed: int,
    dataset_source: str,
    dataset_subset: str,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    dataset = _load_dataset_bundle(dataset_source=dataset_source, dataset_subset=dataset_subset, cache_dir=cache_dir)
    _validate_output_root(output_root)
    if os.path.isdir(output_root):
        shutil.rmtree(output_root)

    public_root = os.path.join(output_root, "public")
    private_root = os.path.join(output_root, "private")
    _ensure_dir(public_root)
    _ensure_dir(private_root)

    counts: dict[str, int] = {}
    all_test_ids: list[str] = []
    gt_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    split_map = dataset["splits"]
    image_loader = dataset.get("image_loader")
    for split_name in ("dev", "test"):
        split_dataset = _sort_samples(list(split_map[split_name]))
        _validate_sample_ids(split_name, split_dataset, seen_ids)
        counts[split_name] = len(split_dataset)
        for sample in split_dataset:
            sample = dict(sample)
            sample["_split"] = split_name
            question_id = _materialize_sample(sample, public_root, private_root, image_loader=image_loader)
            if split_name == "test":
                all_test_ids.append(question_id)
            gt_rows.append(
                {
                    "question_id": question_id,
                    "answer_label": str(sample["label"]).strip().upper(),
                    "medical_task": str(sample.get("medical_task", "")),
                    "body_system": str(sample.get("body_system", "")),
                    "question_type": str(sample.get("question_type", "")),
                    "split": split_name,
                }
            )

    _write_ground_truth_csv(private_root, gt_rows)

    subsets = _build_subsets(
        test_ids=all_test_ids,
        smoke_size=smoke_size,
        calibration_size=calibration_size,
        seed=seed,
    )
    _write_subset(output_root, "smoke_ids.txt", subsets["smoke_ids"])
    _write_subset(output_root, "calibration_ids.txt", subsets["calibration_ids"])
    with open(os.path.join(output_root, "subsets.json"), "w", encoding="utf-8") as handle:
        json.dump(subsets, handle, indent=2)

    manifest = {
        "dataset": dataset_source,
        "subset": dataset_subset,
        "output_root": os.path.abspath(output_root),
        "source_type": dataset.get("source_type", "huggingface_dataset"),
        "snapshot_root": dataset.get("snapshot_root", ""),
        "subset_dir": dataset.get("subset_dir", ""),
        "images_zip_path": dataset.get("images_zip_path", ""),
        "counts": counts,
        "materialized_question_count": len(gt_rows),
        "expected_test_ids": len(all_test_ids),
        "subsets": subsets,
        "public_root": os.path.abspath(public_root),
        "private_root": os.path.abspath(private_root),
    }
    with open(os.path.join(output_root, "staging_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "dataset": dataset_source,
        "subset": dataset_subset,
        "output_root": os.path.abspath(output_root),
        "counts": counts,
        "test_ids": len(all_test_ids),
        "smoke_size": subsets["smoke_count"],
        "calibration_size": subsets["calibration_count"],
        "subset_overlap": subsets["overlap_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage MedXpertQA MM into MedAgentsBench layout")
    parser.add_argument("--output-root", default=os.path.join("data", "MedXpertQA_MM"))
    parser.add_argument("--dataset-source", default="TsinghuaC3I/MedXpertQA")
    parser.add_argument("--dataset-subset", default="MM")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--smoke-size", type=int, default=10)
    parser.add_argument("--calibration-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260416)
    args = parser.parse_args()

    summary = stage_dataset(
        output_root=args.output_root,
        smoke_size=args.smoke_size,
        calibration_size=args.calibration_size,
        seed=args.seed,
        dataset_source=args.dataset_source,
        dataset_subset=args.dataset_subset,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
