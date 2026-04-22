"""Shared tool-call logger for medbench_vqa helpers.

Every helper call appends one JSON line to ``$WORKSPACE_DIR/tool_calls.jsonl``
so the runner's artefact scorer can attribute tool usage to S1-S5 phases.

The log line schema mirrors the one written by the benchmark runner for
``execute_code`` calls — this way the downstream scorer sees a unified
tool-call stream.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


def _log_path() -> str:
    workspace = os.environ.get("WORKSPACE_DIR", os.getcwd())
    return os.path.join(workspace, "tool_calls.jsonl")


def record_tool_call(tool: str, arguments: dict[str, Any], result_summary: dict[str, Any]) -> None:
    entry = {
        "ts": round(time.time(), 4),
        "tool": tool,
        "source": "medbench_vqa_helper",
        "arguments": arguments,
        "result_summary": result_summary,
    }
    try:
        with open(_log_path(), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        # Best-effort logging; never fail the agent's helper call because of it.
        pass
