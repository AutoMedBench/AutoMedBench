#!/usr/bin/env python3
"""Stage HF VQA datasets into eval_vqa_v2's per-qid public/private layout.

Usage::

    python eval_vqa_v2/stage_hf_datasets.py \
        --pathvqa-count 500 --medframeqa-count 500  # vqa-rad defaults to all

Writes::

    data/PathVQA/{public,private}/<qid>/...
    data/VQA_RAD/{public,private}/<qid>/...
    data/MedFrameQA/{public,private}/<qid>/...

Images are materialised next to ``question.json`` so an agent can read them
with relative paths.  ``question.json`` carries ``image_paths`` (list) so
single-image and multi-frame datasets use the same schema.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable

from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_DIR, "data")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _save_image(image, path: str) -> None:
    image = image.convert("RGB") if image.mode not in ("RGB", "L") else image
    image.save(path, format="JPEG", quality=92)


def _subsample(rows: list[int], count: int | None, seed: int) -> list[int]:
    if count is None or count >= len(rows):
        return rows
    rng = random.Random(seed)
    return sorted(rng.sample(rows, count))


def _stage_simple(
    name: str,
    repo_id: str,
    data_dir_name: str,
    answer_mode: str,
    count: int | None,
    seed: int,
    qid_prefix: str,
    question_type_fn: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    """Handle PathVQA / VQA-RAD style (image, question, answer)."""
    ds = load_dataset(repo_id, split="test")
    indexes = _subsample(list(range(len(ds))), count, seed)

    task_root = os.path.join(DATA_ROOT, data_dir_name)
    pub_root = os.path.join(task_root, "public")
    prv_root = os.path.join(task_root, "private")
    _ensure_dir(pub_root); _ensure_dir(prv_root)

    for pos, idx in enumerate(indexes):
        row = ds[idx]
        qid = f"{qid_prefix}_{pos:05d}"
        qdir = os.path.join(pub_root, qid); _ensure_dir(qdir)
        adir = os.path.join(prv_root, qid); _ensure_dir(adir)

        img_name = "image.jpg"
        _save_image(row["image"], os.path.join(qdir, img_name))

        qtype = question_type_fn(row) if question_type_fn else "open"
        question = {
            "question_id": qid,
            "dataset": name,
            "question": row["question"],
            "image_paths": [img_name],
            "split": "test",
            "medical_task": name.lower(),
            "body_system": "unknown",
            "question_type": qtype,
        }
        answer = {
            "question_id": qid,
            "answer_text": str(row["answer"]).strip(),
            "answer_label": None,
        }
        _write_json(os.path.join(qdir, "question.json"), question)
        _write_json(os.path.join(adir, "answer.json"), answer)

    return {"dataset": name, "staged": len(indexes), "output_root": task_root}


def _vqa_rad_qtype(row: dict[str, Any]) -> str:
    ans = str(row.get("answer") or "").strip().lower()
    return "closed_yes_no" if ans in {"yes", "no"} else "open"


SLAKE_REPO = "BoKelvin/SLAKE"
SLAKE_TEST_JSON_URL = f"https://huggingface.co/datasets/{SLAKE_REPO}/resolve/main/test.json"
SLAKE_IMGS_ZIP_URL = f"https://huggingface.co/datasets/{SLAKE_REPO}/resolve/main/imgs.zip"


def _http_get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _stage_slake_en(count: int | None, seed: int, cache_dir: str) -> dict[str, Any]:
    """Stage the English subset of the SLAKE 1.0 test split.

    HF's ``datasets`` loader for BoKelvin/SLAKE hits an intermittent brotli
    decoding bug on large files, so we fetch the annotations + images
    directly from the HF repo.
    """
    _ensure_dir(cache_dir)
    test_json_path = os.path.join(cache_dir, "slake_test.json")
    imgs_zip_path = os.path.join(cache_dir, "slake_imgs.zip")
    imgs_extract_dir = os.path.join(cache_dir, "slake_imgs")

    if not os.path.isfile(test_json_path):
        data = _http_get(SLAKE_TEST_JSON_URL, timeout=120)
        with open(test_json_path, "wb") as handle:
            handle.write(data)
    with open(test_json_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    en_records = [row for row in records if row.get("q_lang") == "en"]

    if not os.path.isdir(imgs_extract_dir) or not os.listdir(imgs_extract_dir):
        if not os.path.isfile(imgs_zip_path):
            # imgs.zip is ~200MB — stream to disk.
            print(f"downloading SLAKE imgs.zip ({SLAKE_IMGS_ZIP_URL}) ...")
            req = urllib.request.Request(SLAKE_IMGS_ZIP_URL, headers={"Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=1200) as response, open(imgs_zip_path, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        _ensure_dir(imgs_extract_dir)
        with zipfile.ZipFile(imgs_zip_path, "r") as zf:
            zf.extractall(imgs_extract_dir)

    # SLAKE zips extract a top-level `imgs/<xmlabN>/source.jpg` structure.
    imgs_root = imgs_extract_dir
    if os.path.isdir(os.path.join(imgs_root, "imgs")):
        imgs_root = os.path.join(imgs_root, "imgs")

    indexes = _subsample(list(range(len(en_records))), count, seed)

    task_root = os.path.join(DATA_ROOT, "SLAKE_EN")
    pub_root = os.path.join(task_root, "public")
    prv_root = os.path.join(task_root, "private")
    _ensure_dir(pub_root); _ensure_dir(prv_root)

    from PIL import Image  # lazy import — Pillow already available via datasets.

    staged = 0
    skipped_missing_image = 0
    for pos, idx in enumerate(indexes):
        row = en_records[idx]
        qid = f"slake_{pos:05d}"
        img_rel = row.get("img_name") or ""
        src_path = os.path.join(imgs_root, img_rel)
        if not os.path.isfile(src_path):
            skipped_missing_image += 1
            continue
        qdir = os.path.join(pub_root, qid); _ensure_dir(qdir)
        adir = os.path.join(prv_root, qid); _ensure_dir(adir)

        try:
            with Image.open(src_path) as img:
                _save_image(img, os.path.join(qdir, "image.jpg"))
        except Exception as exc:  # noqa: BLE001 - malformed image shouldn't nuke run
            skipped_missing_image += 1
            continue

        answer_type = str(row.get("answer_type") or "").upper()
        qtype = "closed_yes_no" if answer_type == "CLOSED" else "open"
        question = {
            "question_id": qid,
            "dataset": "SLAKE-EN",
            "question": row.get("question"),
            "image_paths": ["image.jpg"],
            "split": "test",
            "medical_task": "slake",
            "body_system": str(row.get("location") or "unknown"),
            "question_type": qtype,
            "modality": row.get("modality"),
            "source_qid": row.get("qid"),
            "source_img_name": img_rel,
        }
        answer = {
            "question_id": qid,
            "answer_text": str(row.get("answer") or "").strip(),
            "answer_label": None,
        }
        _write_json(os.path.join(qdir, "question.json"), question)
        _write_json(os.path.join(adir, "answer.json"), answer)
        staged += 1

    return {
        "dataset": "SLAKE-EN",
        "staged": staged,
        "skipped_missing_image": skipped_missing_image,
        "en_total": len(en_records),
        "output_root": task_root,
    }


def _stage_medframeqa(count: int | None, seed: int) -> dict[str, Any]:
    ds = load_dataset("SuhaoYu1020/MedFrameQA", split="test")
    indexes = _subsample(list(range(len(ds))), count, seed)

    task_root = os.path.join(DATA_ROOT, "MedFrameQA")
    pub_root = os.path.join(task_root, "public")
    prv_root = os.path.join(task_root, "private")
    _ensure_dir(pub_root); _ensure_dir(prv_root)

    staged = 0
    for pos, idx in enumerate(indexes):
        row = ds[idx]
        qid = str(row.get("question_id") or f"medframe_{pos:05d}")
        qdir = os.path.join(pub_root, qid); _ensure_dir(qdir)
        adir = os.path.join(prv_root, qid); _ensure_dir(adir)

        image_paths: list[str] = []
        for i in range(1, 6):
            img = row.get(f"image_{i}")
            if img is None:
                continue
            name = f"frame_{i}.jpg"
            _save_image(img, os.path.join(qdir, name))
            image_paths.append(name)
        if not image_paths:
            continue

        options = row.get("options") or []
        options_map = {chr(ord("A") + i): str(opt) for i, opt in enumerate(options)}
        question = {
            "question_id": qid,
            "dataset": "MedFrameQA",
            "question": row.get("question"),
            "options": options_map,
            "image_paths": image_paths,
            "split": "test",
            "medical_task": "medframeqa",
            "body_system": str(row.get("system") or "unknown"),
            "question_type": "multi_frame_mcq",
            "modality": row.get("modality"),
            "organ": row.get("organ"),
        }
        answer = {
            "question_id": qid,
            "answer_label": str(row.get("correct_answer") or "").strip().upper() or None,
            "answer_text": options_map.get(str(row.get("correct_answer") or "").strip().upper(), ""),
        }
        _write_json(os.path.join(qdir, "question.json"), question)
        _write_json(os.path.join(adir, "answer.json"), answer)
        staged += 1

    return {"dataset": "MedFrameQA", "staged": staged, "output_root": task_root}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage HF VQA datasets into eval_vqa_v2 layout.")
    parser.add_argument("--pathvqa-count", type=int, default=500, help="-1 for all.")
    parser.add_argument("--vqa-rad-count", type=int, default=-1, help="Default: all (~451).")
    parser.add_argument("--medframeqa-count", type=int, default=500, help="-1 for all.")
    parser.add_argument("--slake-count", type=int, default=-1, help="-1 for all EN test (~1061).")
    parser.add_argument(
        "--slake-cache-dir",
        default=os.path.join(DATA_ROOT, ".cache", "slake"),
        help="Where to cache SLAKE annotations + imgs.zip.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", default="", help="Comma-separated: pathvqa,vqa_rad,medframeqa,slake")
    args = parser.parse_args()

    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    results: list[dict[str, Any]] = []

    def _count(value: int) -> int | None:
        return None if value < 0 else value

    if "pathvqa" not in skip:
        results.append(_stage_simple(
            name="PathVQA", repo_id="flaviagiammarino/path-vqa",
            data_dir_name="PathVQA", answer_mode="open_ended",
            count=_count(args.pathvqa_count), seed=args.seed,
            qid_prefix="pathvqa",
        ))
    if "vqa_rad" not in skip:
        results.append(_stage_simple(
            name="VQA-RAD", repo_id="flaviagiammarino/vqa-rad",
            data_dir_name="VQA_RAD", answer_mode="open_ended",
            count=_count(args.vqa_rad_count), seed=args.seed,
            qid_prefix="vqarad", question_type_fn=_vqa_rad_qtype,
        ))
    if "medframeqa" not in skip:
        results.append(_stage_medframeqa(count=_count(args.medframeqa_count), seed=args.seed))
    if "slake" not in skip:
        results.append(_stage_slake_en(
            count=_count(args.slake_count),
            seed=args.seed,
            cache_dir=args.slake_cache_dir,
        ))

    for r in results:
        print(f"staged {r['dataset']:10s} n={r['staged']:5d} → {r['output_root']}")


if __name__ == "__main__":
    main()
