"""Structured image-inspection helper for VQA agents.

Simplified from ``eval_vqa/tools/image_inspection.py`` (VQA_V1).  Returns
metadata (dimensions / format / size) about one or more local image files and
records the call into ``tool_calls.jsonl`` for trajectory scoring.

PIL is a soft dependency — when unavailable, dimension fields are omitted but
the inspection record still carries the agent's declared intent.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from time import perf_counter
from typing import Any

from ._logger import record_tool_call


def inspect_image(
    image_paths: str | Path | list[str | Path] | None = None,
    *,
    regions: list[str] | None = None,
    zoom: bool = True,
    crop: bool = False,
    crop_box: tuple[int, int, int, int] | None = None,
    compare_images: bool | None = None,
    return_thumbnail: bool = False,
    thumbnail_size: tuple[int, int] = (256, 256),
) -> dict[str, Any]:
    """Inspect one or more local images and return metadata + intent record.

    The call is logged to ``$WORKSPACE_DIR/tool_calls.jsonl`` so S3/S4 scoring
    can confirm that visual inspection actually happened.
    """
    started = perf_counter()

    if image_paths is None:
        paths: list[Path] = []
    elif isinstance(image_paths, (str, Path)):
        paths = [Path(image_paths).expanduser()]
    else:
        paths = [Path(p).expanduser() for p in image_paths if str(p).strip()]

    region_list = [str(r).strip() for r in (regions or []) if str(r).strip()]
    multi_image = len(paths) > 1 if compare_images is None else bool(compare_images)

    actions: list[str] = ["whole_image_review"]
    if zoom:
        actions.append("zoom_in")
    if crop or crop_box:
        actions.append("crop_region")
    if region_list:
        actions.append("region_inspection")
    if multi_image:
        actions.append("multi_image_comparison")

    per_image: list[dict[str, Any]] = []
    errors: list[str] = []
    pil_available = _pil_available()

    for path in paths:
        meta = _inspect_single(
            path,
            pil_available=pil_available,
            return_thumbnail=return_thumbnail,
            thumbnail_size=thumbnail_size,
            crop_box=crop_box,
        )
        if meta.get("error"):
            errors.append(meta["error"])
        per_image.append(meta)

    latency_ms = round((perf_counter() - started) * 1000, 2)
    has_real_dims = pil_available and any(img.get("width") for img in per_image)

    result = {
        "status": "ok" if not errors else "partial_error",
        "tool_name": "inspect_image",
        "image_count": len(paths),
        "checked_paths": [str(p) for p in paths],
        "regions": region_list,
        "actions": actions,
        "whole_image_review": True,
        "zoom_action_used": zoom,
        "crop_action_used": bool(crop or crop_box),
        "region_inspection_used": bool(region_list),
        "multi_image_comparison_used": multi_image,
        "per_image": per_image,
        "pil_available": pil_available,
        "placeholder": not has_real_dims,
        "latency_ms": latency_ms,
        "errors": errors,
    }

    record_tool_call(
        tool="inspect_image",
        arguments={
            "image_count": len(paths),
            "regions": region_list,
            "zoom": zoom,
            "crop": bool(crop or crop_box),
        },
        result_summary={
            "status": result["status"],
            "image_count": result["image_count"],
            "placeholder": result["placeholder"],
        },
    )
    return result


def _pil_available() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def _inspect_single(
    path: Path,
    *,
    pil_available: bool,
    return_thumbnail: bool,
    thumbnail_size: tuple[int, int],
    crop_box: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "error": f"File not found: {path}"}
    if not path.is_file():
        return {"path": str(path), "exists": False, "error": f"Not a file: {path}"}

    stat = path.stat()
    mime = mimetypes.guess_type(str(path))[0] or "image/unknown"
    meta: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "name": path.name,
        "suffix": path.suffix.lower(),
        "mime_type": mime,
        "size_bytes": stat.st_size,
        "size_kb": round(stat.st_size / 1024, 1),
    }

    if not pil_available:
        meta["pil_available"] = False
        return meta

    try:
        from PIL import Image  # type: ignore

        img = Image.open(path)
        width, height = img.size
        meta.update(
            {
                "pil_available": True,
                "width": width,
                "height": height,
                "mode": img.mode,
                "format": img.format or path.suffix.upper().lstrip("."),
            }
        )
        if crop_box is not None:
            try:
                cropped = img.crop(crop_box)
                cw, ch = cropped.size
                meta["crop_box"] = list(crop_box)
                meta["crop_size"] = [cw, ch]
            except Exception as exc:  # noqa: BLE001
                meta["crop_error"] = str(exc)
        if return_thumbnail:
            thumb = img.copy()
            thumb.thumbnail(thumbnail_size)
            import io

            buf = io.BytesIO()
            fmt = (img.format or "PNG").upper()
            if fmt not in {"JPEG", "PNG", "WEBP"}:
                fmt = "PNG"
            thumb.save(buf, format=fmt)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            meta["thumbnail_b64"] = encoded
            meta["thumbnail_mime"] = f"image/{fmt.lower()}"
    except Exception as exc:  # noqa: BLE001
        meta["pil_error"] = str(exc)

    return meta
