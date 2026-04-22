"""Recover OpenAI-style tool_calls from text-wrapped XML emitted by some models.

Background:
- BUG-043 (minimax-m2.5): emits ``<minimax:tool_call><invoke name="X">
  <parameter name="Y">Z</parameter></invoke></minimax:tool_call>`` as assistant
  content instead of native ``tool_calls``.
- BUG-040 (qwen3.5-397b): under long context, degrades to
  ``<tool_call><function=X><parameter=Y>Z</parameter></function></tool_call>``.

The NVDA OpenAI-compatible adapter passes these through as plain text, so the
benchmark runner's native ``tool_calls`` list is empty and the agent is
classified as ``inference_mode=empty`` even though it is actually *trying* to
call tools.

This module provides ``recover_tool_calls(content)`` which scans the string for
both dialects and returns a list of ``{"id", "name", "arguments"}`` records
compatible with the runner's native tool_call shape. Safe to call on any
string — returns ``[]`` if nothing matches.
"""

from __future__ import annotations

import re
import uuid
from typing import Any


# Dialect A — minimax: <invoke name="X"> <parameter name="Y">Z</parameter> </invoke>
_INVOKE_RE = re.compile(
    r"<invoke\s+name\s*=\s*\"(?P<name>[^\"]+)\"\s*>(?P<body>.*?)</invoke>",
    re.DOTALL | re.IGNORECASE,
)
_INVOKE_OPEN_RE = re.compile(
    r"<invoke\s+name\s*=\s*\"(?P<name>[^\"]+)\"\s*>",
    re.IGNORECASE,
)
_PARAM_NAMED_RE = re.compile(
    r"<parameter\s+name\s*=\s*\"(?P<key>[^\"]+)\"\s*>(?P<val>.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_NAMED_OPEN_RE = re.compile(
    r"<parameter\s+name\s*=\s*\"(?P<key>[^\"]+)\"\s*>",
    re.IGNORECASE,
)

# Dialect B — qwen: <function=X> <parameter=Y>Z</parameter> </function>
_FUNCTION_RE = re.compile(
    r"<function\s*=\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*>(?P<body>.*?)</function>",
    re.DOTALL | re.IGNORECASE,
)
_FUNCTION_OPEN_RE = re.compile(
    r"<function\s*=\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*>",
    re.IGNORECASE,
)
_PARAM_EQ_RE = re.compile(
    r"<parameter\s*=\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*>(?P<val>.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_EQ_OPEN_RE = re.compile(
    r"<parameter\s*=\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*>",
    re.IGNORECASE,
)

# Wrapper sentinels that unambiguously identify tool-call intent (not prose).
_WRAPPER_RE = re.compile(
    r"<(?:minimax:tool_call|tool_call|function_calls|antml:function_calls)\b",
    re.IGNORECASE,
)


def _has_wrapper(content: str) -> bool:
    return bool(_WRAPPER_RE.search(content))


def _is_anchored(content: str, start: int) -> bool:
    """True if the tag at ``start`` is not embedded in a prose sentence.

    Accept when the tag is preceded (since the last newline) only by
    whitespace or other angle-bracket markup — i.e. it starts its own line
    or directly follows another tag. Reject when a word-character sequence
    precedes it on the same line (e.g. "Please call <invoke ...>").
    """
    line_start = content.rfind("\n", 0, start) + 1
    prefix = content[line_start:start]
    # Strip angle-bracket markup that legitimately precedes the tag.
    stripped = re.sub(r"<[^>]*>", "", prefix)
    return not re.search(r"\w", stripped)


def _parse_params_tolerant(body: str, closed_re, open_re) -> dict[str, str]:
    """Parse parameters with closing-tag fallback for truncated content."""
    result = {}
    # First pass: well-formed closed tags
    for m in closed_re.finditer(body):
        result[m.group("key")] = m.group("val").strip()
    # Second pass: orphan opens — take everything after the last open until EOS
    # (only if not already captured with a closer)
    opens = list(open_re.finditer(body))
    if opens:
        last = opens[-1]
        key = last.group("key")
        if key not in result:
            tail = body[last.end():]
            # Stop at a new <parameter ... > open if one appears after (shouldn't,
            # since we took last open, but be safe)
            cut = re.search(r"<parameter[\s=]", tail, re.IGNORECASE)
            if cut:
                tail = tail[:cut.start()]
            # Also trim any orphan closing ancestor tags
            tail = re.sub(r"</(parameter|invoke|function|minimax:tool_call|tool_call)>.*$",
                          "", tail, flags=re.DOTALL | re.IGNORECASE)
            if tail.strip():
                result[key] = tail.strip()
    return result


def _parse_params_named(body: str) -> dict[str, str]:
    return _parse_params_tolerant(body, _PARAM_NAMED_RE, _PARAM_NAMED_OPEN_RE)


def _parse_params_eq(body: str) -> dict[str, str]:
    return _parse_params_tolerant(body, _PARAM_EQ_RE, _PARAM_EQ_OPEN_RE)


def recover_tool_calls(content: str) -> list[dict[str, Any]]:
    """Return a list of synthesized tool_calls parsed from ``content``.

    Each entry has shape ``{"id": str, "name": str, "arguments": dict}``,
    matching the runner's internal tool-call representation. Returns an empty
    list if no recognized XML-wrapped tool call is present.
    """
    if not isinstance(content, str) or not content:
        return []

    # Gate: only recover when the content looks like tool-call XML, not prose
    # that happens to mention an <invoke> tag.
    wrapper_present = _has_wrapper(content)

    def _accept(start: int) -> bool:
        return wrapper_present or _is_anchored(content, start)

    recovered: list[dict[str, Any]] = []

    # Dialect A: <invoke name="X">...</invoke> (may be truncated)
    seen_spans: list[tuple[int, int]] = []
    for m in _INVOKE_RE.finditer(content):
        if not _accept(m.start()):
            continue
        name = m.group("name").strip()
        args = _parse_params_named(m.group("body"))
        if name:
            seen_spans.append((m.start(), m.end()))
            recovered.append({
                "id": f"recovered_{uuid.uuid4().hex[:8]}",
                "name": name,
                "arguments": args,
            })
    # Orphan <invoke> opens not closed (truncated by max_tokens)
    for m in _INVOKE_OPEN_RE.finditer(content):
        # skip ones already captured by closed-form pass
        if any(s <= m.start() < e for s, e in seen_spans):
            continue
        if not _accept(m.start()):
            continue
        name = m.group("name").strip()
        body_tail = content[m.end():]
        args = _parse_params_named(body_tail)
        if name and args:
            recovered.append({
                "id": f"recovered_{uuid.uuid4().hex[:8]}",
                "name": name,
                "arguments": args,
            })

    # Dialect B: <function=X>...</function> (may be truncated)
    seen_spans_b: list[tuple[int, int]] = []
    for m in _FUNCTION_RE.finditer(content):
        if not _accept(m.start()):
            continue
        name = m.group("name").strip()
        args = _parse_params_eq(m.group("body"))
        if name:
            seen_spans_b.append((m.start(), m.end()))
            recovered.append({
                "id": f"recovered_{uuid.uuid4().hex[:8]}",
                "name": name,
                "arguments": args,
            })
    for m in _FUNCTION_OPEN_RE.finditer(content):
        if any(s <= m.start() < e for s, e in seen_spans_b):
            continue
        if not _accept(m.start()):
            continue
        name = m.group("name").strip()
        body_tail = content[m.end():]
        args = _parse_params_eq(body_tail)
        if name and args:
            recovered.append({
                "id": f"recovered_{uuid.uuid4().hex[:8]}",
                "name": name,
                "arguments": args,
            })

    return recovered
