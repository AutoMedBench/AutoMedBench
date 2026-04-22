#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


API_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]+")
EXPECTED_KEY_COUNT = 4


def load_api_keys(path: str | Path) -> list[str]:
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) != EXPECTED_KEY_COUNT:
        raise ValueError(
            f"{file_path} must contain exactly {EXPECTED_KEY_COUNT} API keys, one key per line; found {len(lines)} line(s)."
        )

    invalid_lines = []
    keys = []
    for idx, line in enumerate(lines, start=1):
        key = line.strip()
        if not API_KEY_PATTERN.fullmatch(key):
            invalid_lines.append(idx)
            continue
        keys.append(key)

    if invalid_lines:
        joined = ", ".join(str(idx) for idx in invalid_lines)
        raise ValueError(
            f"{file_path} must contain only API keys, one per line; invalid content on line(s): {joined}."
        )

    return keys
