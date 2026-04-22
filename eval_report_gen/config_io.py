#!/usr/bin/env python3
"""Small config loader for eval_report_gen.

This module avoids a hard dependency on PyYAML. It supports:

- JSON documents (JSON is valid YAML)
- a small YAML subset used by eval_report_gen task files
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


def _parse_scalar(text: str):
    text = text.strip()
    if text == "":
        return ""

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return ast.literal_eval(text)

    if text[0] in {"'", '"'} and text[-1] == text[0]:
        return text[1:-1]

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        pass

    return text


def _clean_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip("\n")
        if line.strip():
            lines.append(line.rstrip())
    return lines


def _parse_block(lines: list[str], indent: int = 0, start: int = 0):
    data = None
    index = start

    while index < len(lines):
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Invalid indentation near: {line}")

        stripped = line.strip()
        if stripped.startswith("- "):
            if data is None:
                data = []
            elif not isinstance(data, list):
                raise ValueError(f"Mixed list/dict structure near: {line}")

            item_text = stripped[2:].strip()
            if not item_text:
                item, index = _parse_block(lines, indent + 2, index + 1)
                data.append(item)
                continue

            if item_text.endswith(":"):
                key = item_text[:-1].strip()
                nested, index = _parse_block(lines, indent + 2, index + 1)
                data.append({key: nested})
                continue

            data.append(_parse_scalar(item_text))
            index += 1
            continue

        if data is None:
            data = {}
        elif not isinstance(data, dict):
            raise ValueError(f"Mixed list/dict structure near: {line}")

        if ":" not in stripped:
            raise ValueError(f"Expected key/value pair near: {line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value:
            data[key] = _parse_scalar(value)
            index += 1
            continue

        nested, index = _parse_block(lines, indent + 2, index + 1)
        data[key] = nested

    return data if data is not None else {}, index


def load_config(path: str | Path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    lines = _clean_lines(text)
    parsed, _ = _parse_block(lines, 0, 0)
    return parsed
