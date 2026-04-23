#!/usr/bin/env python3
"""Coding-agent benchmark runner for eval_vqa_v2.

Single-conversation VQA coding agent aligned with ``eval_seg/benchmark_runner.py``:

- The LLM only sees one framework tool: ``execute_code``.
- The agent writes Python that loops over all question ids, loads one local
  VLM once, produces ``<qid>/answer.json`` for every question, and finally
  stops issuing tool calls to end the run.
- ``inspect_image`` / ``public_medical_search`` / ``submit_answer`` are
  provided as Python helpers importable from the agent sandbox via
  ``from medbench_vqa import ...``.

Derived from vqa_hard/eval_vqa/agent_benchmark_runner.py, with the
``submit_results`` framework tool removed and the medbench_vqa helper
package wired onto ``PYTHONPATH`` inside the sandbox.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import requests

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detail_report import generate_detail_report, print_detail_report
from format_checker import check_submission
from inference_verifier import check_smoke_forward
from run_eval import run_eval
from task_loader import (
    get_task_data_root,
    load_skill,
    load_task_config,
    load_yaml_file,
    resolve_agent_config_path,
)
from tier_config import get_task_model_info, get_tier_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")

BLOCKED_ESCAPE_REGEXES = [
    r"find\s+/\s",
    r"\bls\s+(-[a-zA-Z]+\s+)?/\s",
    r"os\.walk\s*\(\s*['\"]\/['\"]",
    r"os\.listdir\s*\(\s*['\"]\/['\"]",
    r"os\.scandir\s*\(\s*['\"]\/['\"]",
    r"\.\./\.\.",
    r"\/proc\/",
    r"\/sys\/",
    r"\/etc\/",
    r"\/var\/run\/",
    r"ground_truth",
]
ESCAPE_RES = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in BLOCKED_ESCAPE_REGEXES]


def _iter_answer_files(output_dir: str):
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        p = os.path.join(output_dir, name, "answer.json")
        if os.path.isfile(p):
            yield p

SANDBOX_PREAMBLE_PY = f"""\
import os as _os, sys as _sys
_ws = _os.environ.get("WORKSPACE_DIR", "/workspace")
_shared_hf = _os.environ.get("SHARED_HF_CACHE")
if _shared_hf:
    _os.makedirs(_shared_hf, exist_ok=True)
    _os.environ["HF_HOME"] = _shared_hf
    _os.environ["HF_HUB_CACHE"] = _os.path.join(_shared_hf, "hub")
    _os.environ["TRANSFORMERS_CACHE"] = _shared_hf
else:
    _os.environ["HF_HOME"] = _os.path.join(_ws, ".cache", "huggingface")
_os.environ["TORCH_HOME"] = _os.path.join(_ws, ".cache", "torch")
_os.environ["XDG_CACHE_HOME"] = _os.path.join(_ws, ".cache")
_os.environ["PIP_CACHE_DIR"] = _os.path.join(_ws, ".cache", "pip")
_os.environ["TMPDIR"] = _os.path.join(_ws, ".cache", "tmp")
_os.makedirs(_os.path.join(_ws, ".cache", "tmp"), exist_ok=True)
# Make medbench_vqa helpers importable inside agent-authored code.
_sys.path.insert(0, {SCRIPT_DIR!r})
def _sandbox_hook(_event, _args):
    _forbidden = ("/data/private", "/results/", "/eval/")
    _write_forbidden = ("/root/", "/tmp/")
    if _event in ("open", "os.listdir", "os.scandir", "os.chdir", "os.mkdir", "os.rename", "os.remove") and _args:
        _path = str(_args[0])
        _resolved = _os.path.realpath(_path)
        for _prefix in _forbidden:
            if _resolved.startswith(_prefix) or _path.startswith(_prefix):
                _sys.stderr.write(f"SANDBOX VIOLATION: access to {{_resolved}} is FORBIDDEN.\\n")
                _sys.stderr.flush()
                _os._exit(99)
        if _event in ("os.mkdir", "os.rename", "os.remove") or (_event == "open" and len(_args) > 1 and any(flag in str(_args[1]) for flag in ("w", "a", "x"))):
            for _prefix in _write_forbidden:
                if _resolved.startswith(_prefix) or _path.startswith(_prefix):
                    _sys.stderr.write(f"SANDBOX VIOLATION: write to {{_resolved}} is FORBIDDEN.\\n")
                    _sys.stderr.flush()
                    _os._exit(99)
_sys.addaudithook(_sandbox_hook)
"""

SANDBOX_PREAMBLE_BASH = f"""\
_WS="${{WORKSPACE_DIR:-/workspace}}"
if [ -n "${{SHARED_HF_CACHE:-}}" ]; then
  mkdir -p "${{SHARED_HF_CACHE}}"
  export HF_HOME="${{SHARED_HF_CACHE}}"
  export HF_HUB_CACHE="${{SHARED_HF_CACHE}}/hub"
  export TRANSFORMERS_CACHE="${{SHARED_HF_CACHE}}"
else
  export HF_HOME="${{_WS}}/.cache/huggingface"
fi
export TORCH_HOME="${{_WS}}/.cache/torch"
export XDG_CACHE_HOME="${{_WS}}/.cache"
export PIP_CACHE_DIR="${{_WS}}/.cache/pip"
export TMPDIR="${{_WS}}/.cache/tmp"
export PYTHONPATH="{SCRIPT_DIR}:${{PYTHONPATH:-}}"
mkdir -p "${{_WS}}/.cache/tmp"
"""


def _check_isolation(code: str) -> str:
    for regex in ESCAPE_RES:
        match = regex.search(code)
        if match:
            return f"BLOCKED: sandbox escape detected — matched '{match.group()}'"
    return ""


def execute_code(language: str, code: str, cwd: str, timeout: int | None = None) -> dict[str, Any]:
    violation = _check_isolation(code)
    if violation:
        return {"exit_code": -1, "stdout": "", "stderr": violation}

    suffix = ".py" if language == "python" else ".sh"
    full_code = (SANDBOX_PREAMBLE_PY if language == "python" else SANDBOX_PREAMBLE_BASH) + "\n" + code
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, dir=cwd, delete=False) as handle:
        handle.write(full_code)
        script_path = handle.name
    command = ["python3", script_path] if language == "python" else ["bash", script_path]
    run_env = os.environ.copy()
    run_env["WORKSPACE_DIR"] = cwd
    # Ensure the medbench_vqa helper package is importable by agent code.
    existing_pypath = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = f"{SCRIPT_DIR}{os.pathsep}{existing_pypath}" if existing_pypath else SCRIPT_DIR
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=run_env,
            timeout=timeout,
        )
        stderr = result.stderr[-4000:] if len(result.stderr) > 4000 else result.stderr
        stdout = result.stdout[-6000:] if len(result.stdout) > 6000 else result.stdout
        if result.returncode == 99 and "SANDBOX VIOLATION" in stderr:
            return {"exit_code": -1, "stdout": stdout, "stderr": f"BLOCKED: {stderr.strip()}"}
        return {"exit_code": result.returncode, "stdout": stdout, "stderr": stderr}
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"TIMEOUT: execution exceeded {timeout}s"}
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _try_recover_json_arguments(raw: str) -> dict[str, Any] | None:
    """Best-effort recovery of malformed tool_call.arguments JSON.

    kimik2.5 sometimes emits multi-line python source inside the `code` field
    with real newlines / tabs that aren't JSON-escaped. Try common fixes:
      1. escape unescaped control characters inside string literals
      2. strip trailing commas
    Returns the parsed dict on success, else None.
    """
    if not raw:
        return None
    attempts = [raw]
    # Escape bare newlines/tabs/carriage returns that appear inside strings.
    escaped = []
    in_string = False
    prev = ""
    for ch in raw:
        if ch == '"' and prev != "\\":
            in_string = not in_string
            escaped.append(ch)
        elif in_string and ch == "\n":
            escaped.append("\\n")
        elif in_string and ch == "\r":
            escaped.append("\\r")
        elif in_string and ch == "\t":
            escaped.append("\\t")
        else:
            escaped.append(ch)
        prev = ch
    attempts.append("".join(escaped))
    # Strip trailing commas before } or ]
    attempts.append(re.sub(r",(\s*[}\]])", r"\1", attempts[-1]))
    for candidate in attempts:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def call_api(
    api_key: str,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    temperature: float = 0.0,
    reasoning: bool = True,
    base_url: str | None = None,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    endpoint = (base_url.rstrip("/") + "/chat/completions" if base_url else "https://openrouter.ai/api/v1/chat/completions")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
    }
    # Reasoning-mode Claude (opus-4-7, sonnet-4-6) rejects `temperature`;
    # only send it for non-reasoning requests.
    if not reasoning:
        payload["temperature"] = temperature
    if reasoning and not base_url:
        payload["reasoning"] = {"enabled": True}
    transient = {429, 500, 502, 503, 504}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = json.dumps(payload)
    last_exc: Exception | None = None
    response = None
    data: dict[str, Any] | None = None
    for attempt in range(1, 5):
        try:
            response = requests.post(endpoint, headers=headers, data=body, timeout=600)
            if response.status_code in transient and attempt < 4:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else 0.0
                except (TypeError, ValueError):
                    wait = 0.0
                if wait <= 0:
                    wait = min(60.0, 2 ** attempt + random.uniform(0, 1))
                try:
                    body_preview = (response.text or "")[:200]
                except Exception:
                    body_preview = ""
                sys.stderr.write(
                    f"[call_api] HTTP {response.status_code} attempt {attempt}/4 "
                    f"body_preview={body_preview!r} sleeping {wait:.1f}s\n"
                )
                sys.stderr.flush()
                time.sleep(wait)
                continue
            if response.status_code >= 400:
                try:
                    body_preview = response.text[:800]
                except Exception:
                    body_preview = ""
                sys.stderr.write(
                    f"[call_api] HTTP {response.status_code} body={body_preview}\n"
                )
                sys.stderr.flush()
            response.raise_for_status()
            # Some providers (NVDA / OpenRouter) occasionally return HTTP 200
            # with a non-JSON body (HTML error page, truncated SSE chunk).
            # Treat that as transient: log a body preview and retry instead of
            # crashing the whole worker (BUG: kimik2.5-r02 lost in formal sweep).
            try:
                data = response.json()
            except (json.JSONDecodeError, requests.exceptions.JSONDecodeError) as exc:
                last_exc = exc
                if attempt == 4:
                    raise
                preview = (response.text or "")[:400]
                wait = min(60.0, 2 ** attempt + random.uniform(0, 1))
                sys.stderr.write(
                    f"[call_api] non-JSON body (HTTP {response.status_code}) "
                    f"attempt {attempt}/4 sleeping {wait:.1f}s preview={preview!r}\n"
                )
                sys.stderr.flush()
                time.sleep(wait)
                continue
            break
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt == 4:
                raise
            wait = min(60.0, 2 ** attempt + random.uniform(0, 1))
            sys.stderr.write(
                f"[call_api] {type(exc).__name__} attempt {attempt}/4 sleeping {wait:.1f}s\n"
            )
            sys.stderr.flush()
            time.sleep(wait)
    else:
        if last_exc:
            raise last_exc
    assert response is not None
    # BUG-048: defend against non-JSON / malformed-JSON / empty-choices
    # responses. Upstream CDNs and proxies occasionally return HTML error
    # pages with 200, or SSE bodies with application/json content-type, or
    # {"error":...} dicts with missing/empty choices. Prior code crashed
    # the worker with JSONDecodeError / KeyError / IndexError and silently
    # zeroed the whole run.
    try:
        data = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        preview = (response.text or "")[:800]
        raise RuntimeError(
            f"API returned non-JSON (status={response.status_code}, "
            f"len={len(response.text or '')}): {exc}; body_preview={preview!r}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"API returned non-dict JSON (status={response.status_code}): "
            f"type={type(data).__name__} preview={str(data)[:200]!r}"
        )
    if "error" in data and not data.get("choices"):
        raise RuntimeError(
            f"API error (status={response.status_code}): {data['error']}"
        )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(
            f"API returned no choices (status={response.status_code}, "
            f"keys={list(data)}): preview={str(data)[:500]!r}"
        )
    choice = choices[0]
    message = choice.get("message") or {}
    tool_calls = []
    malformed_tool_calls = 0
    for tool_call in message.get("tool_calls") or []:
        arguments = tool_call["function"]["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError) as exc:
                # kimik2.5 has been observed to emit malformed JSON in the
                # tool_call.function.arguments field (unescaped newlines in
                # Python code literals). Previously crashed the whole run with
                # JSONDecodeError; instead, try to recover by stripping common
                # kimi-side artefacts, else skip this tool_call and let the
                # agent retry next turn.
                recovered = _try_recover_json_arguments(arguments)
                if recovered is not None:
                    arguments = recovered
                else:
                    sys.stderr.write(
                        f"[call_api] malformed tool_call arguments "
                        f"({type(exc).__name__}: {exc}); dropping this call, "
                        f"preview={arguments[:200]!r}\n"
                    )
                    sys.stderr.flush()
                    malformed_tool_calls += 1
                    continue
        tool_calls.append({"id": tool_call["id"], "name": tool_call["function"]["name"], "arguments": arguments})
    # BUG-043 / BUG-040: some models (minimax, qwen under long context) emit
    # tool calls as XML inside assistant content instead of native tool_calls.
    # Recover those so the run isn't silently classified as empty.
    content_text = message.get("content") or ""
    recovered_flag = False
    if not tool_calls and content_text:
        from tool_call_recovery import recover_tool_calls
        recovered = recover_tool_calls(content_text)
        if recovered:
            tool_calls = recovered
            recovered_flag = True
    usage = data.get("usage", {})
    return {
        "content": content_text,
        "tool_calls": tool_calls,
        "tool_calls_recovered": recovered_flag,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "finish_reason": choice.get("finish_reason", "stop"),
        "reasoning_details": message.get("reasoning_details") or message.get("reasoning_content"),
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Execute Python or bash code in the isolated VQA benchmark workspace. "
                "This is the only tool you have. You must use it to plan, install, "
                "download the VLM, run inference, and write every answer.json. "
                "Helpers `from medbench_vqa import inspect_image, public_medical_search, submit_answer` "
                "are importable inside your Python. When all answers are written, stop issuing "
                "tool_calls to end the run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python", "bash"]},
                    "code": {"type": "string"},
                },
                "required": ["language", "code"],
            },
        },
    },
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_dotenv(repo_root: str) -> None:
    dotenv_path = os.path.join(repo_root, ".env")
    if not os.path.isfile(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _load_prompt(relpath: str, fallback: str) -> str:
    path = os.path.join(PROMPTS_DIR, relpath)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return fallback.strip()


_LLAVA_MED_SKILL = _load_prompt("common/llava_med_skill.md", "")


def _load_llava_med_skill_if_applicable(tier: str, lite_spec: dict[str, Any]) -> str:
    """Return the shared LLaVA-Med skill markdown for lite+llava_med tasks, else ''."""
    if tier != "lite":
        return ""
    loader = str(lite_spec.get("loader_family") or "").lower()
    model_name = str(lite_spec.get("model_name") or "").lower()
    if loader == "llava_med" or "llava-med" in model_name:
        return _LLAVA_MED_SKILL
    return ""


def _substitute_workspace(text: str, output_dir: str) -> str:
    """BUG-046: substitute ${WORKSPACE_DIR} with the concrete outputs path.

    Skill markdown files use ``${WORKSPACE_DIR}/...`` as the canonical
    artefact path. Some agents (qwen, seen on pathvqa) read that literal as
    a hint rather than as an env var and hardcode a wrong value (e.g. the
    run_dir root instead of run_dir/outputs). Substituting at render time
    removes the ambiguity — the prompt now shows the absolute path agents
    must write to.
    """
    if not text:
        return text
    return text.replace("${WORKSPACE_DIR}", output_dir).replace(
        "{WORKSPACE_DIR}", output_dir
    )


PREAMBLE = _load_prompt(
    "README.md",
    "Use the VQA workflow.",
)
S1_LITE = _load_prompt("s1_plan/lite.md", "")
S1_STANDARD = _load_prompt("s1_plan/standard.md", "")
S2_LITE = _load_prompt("s2_setup/lite.md", "")
S2_STANDARD = _load_prompt("s2_setup/standard.md", "")
S3_ALL = _load_prompt("s3_validate/lite_standard.md", "")
S4_ALL = _load_prompt("s4_inference/all.md", "")
S5_ALL = _load_prompt("s5_submit/all.md", "")
CRITICAL_MODEL_USE = _load_prompt(
    "common/critical_model_use.md",
    "You must load an external VLM and use its .generate output for every answer. "
    "Do not answer from your own knowledge or heuristics.",
)
_COMMON_PREAMBLE = _load_prompt("common/preamble.md", "")
_COMMON_OPEN_ENDED = _load_prompt("common/open_ended_contract.md", "")
_COMMON_ENV_LITE = _load_prompt("common/env_lite.md", "")
_COMMON_ENV_STANDARD = _load_prompt("common/env_standard.md", "")
_COMMON_IMPORTANT_LITE = _load_prompt("common/important_lite.md", "")
_COMMON_IMPORTANT_STANDARD = _load_prompt("common/important_standard.md", "")
_COMMON_KICKOFF = _load_prompt("common/kickoff.md", "")


def _kickoff_for_tier(tier_name: str) -> str:
    """Extract the per-tier kickoff message from common/kickoff.md.

    Falls back to the historical inline strings when the file is missing or
    does not contain the requested tier section.
    """
    fallback = {
        "lite": "Begin. Use the fixed medical VLM (LLaVA-Med) and follow S1 through S5 using only `execute_code`.",
        "standard": "Begin. Compare the candidate VLMs, choose one, and follow S1 through S5 using only `execute_code`.",
    }
    if not _COMMON_KICKOFF:
        return fallback.get(tier_name, fallback["lite"])
    header = f"## {tier_name}"
    chunks = _COMMON_KICKOFF.split("## ")
    for chunk in chunks:
        if chunk.startswith(f"{tier_name}\n") or chunk.startswith(f"{tier_name} "):
            body = chunk.split("\n", 1)[1].strip() if "\n" in chunk else ""
            if body:
                return body
    return fallback.get(tier_name, fallback["lite"])


def build_tier_system_prompt(
    task_id: str,
    tier: str,
    question_ids: list[str],
    subset: str,
    sample_limit: int | None,
    data_dir: str = "/data/public",
    output_dir: str = "/workspace",
) -> str:
    task_config = load_task_config(task_id)
    model_info = get_task_model_info(task_id)
    requirements_path = os.path.join(output_dir, "requirements.txt")
    sample_count = len(question_ids)
    is_open_ended = task_config.get("answer_mode", "multiple_choice") == "open_ended"
    sample_limit_line = (
        f"sample limit override: `{sample_limit}`" if sample_limit is not None
        else "sample limit override: none"
    )
    preamble_rendered = _COMMON_PREAMBLE.format(
        critical_model_use=CRITICAL_MODEL_USE,
        data_dir=data_dir,
        output_dir=output_dir,
        task_description=task_config["task_description"],
        subset=subset,
        sample_count=sample_count,
        sample_limit_line=sample_limit_line,
    )
    lines = [preamble_rendered, ""]
    if is_open_ended:
        lines.extend([_COMMON_OPEN_ENDED, ""])
    env_template = _COMMON_ENV_LITE if tier == "lite" else _COMMON_ENV_STANDARD
    lines.extend([
        env_template.format(output_dir=output_dir),
        "",
        "Workflow:",
        "S1 PLAN",
        S1_LITE if tier == "lite" else S1_STANDARD,
        load_skill(task_id, "lite_s1.md" if tier == "lite" else "standard_s1.md").strip(),
        "",
        "S2 SETUP",
        _substitute_workspace(
            S2_LITE if tier == "lite" else (S2_STANDARD or S2_LITE),
            output_dir,
        ),
        _substitute_workspace(
            load_skill(task_id, "lite_s2.md" if tier == "lite" else "").strip(),
            output_dir,
        ),
        # Lite tier + LLaVA-Med tasks: append the shared canonical skill with
        # working conv_templates / tokenizer_image_token / output_ids-slice
        # code. Without this, most agents call .generate() on a bare question
        # and the decode returns "", which trips the smoke_forward verifier
        # and burns S3/S4 budget on broken runs. Aligned with eval_seg's
        # per-task skill pattern (kidney_lite_s2.md ships executable code).
        _substitute_workspace(
            _load_llava_med_skill_if_applicable(
                tier, model_info.get("lite_model_spec") or {}
            ),
            output_dir,
        ),
        "",
        "S3 VALIDATE",
        _substitute_workspace(S3_ALL, output_dir),
        _substitute_workspace(
            load_skill(task_id, "lite_s3.md" if tier == "lite" else "standard_s3.md").strip(),
            output_dir,
        ),
        "",
        "S4 INFERENCE",
        _substitute_workspace(S4_ALL, output_dir),
        "",
        "S5 SUBMIT",
        _substitute_workspace(S5_ALL, output_dir),
        "",
        "Tier/model guidance:",
    ])
    if tier == "lite":
        lines.append(f"- fixed model: `{model_info['lite_model']}`")
        lines.append(f"- provided task requirements file will be copied to `{requirements_path}`")
    else:
        lines.append("- candidate models:")
        for candidate in model_info["standard_candidate_specs"]:
            lines.append(
                f"  - `{candidate['model_name']}` | access={candidate.get('accessibility')} | notes={candidate.get('notes')}"
            )
        if model_info.get("selection_guidance"):
            lines.append("- selection guidance:")
            for item in model_info["selection_guidance"]:
                lines.append(f"  - {item}")
    lines.append("")
    important_block = _COMMON_IMPORTANT_LITE if tier == "lite" else _COMMON_IMPORTANT_STANDARD
    if not important_block:
        important_block = (
            "When every required `answer.json` is written and you have verified the workspace, "
            "stop issuing `tool_calls`. The runner interprets an empty tool_calls response as "
            "'agent is done'."
        )
    lines.append(important_block)
    return "\n".join(line for line in lines if line != "").strip()


def resolve_question_ids(
    task_id: str,
    split: str | None,
    sample_limit: int | None,
    question_ids_arg: str | None,
    subset: str,
) -> list[str]:
    """Resolve the list of question ids to run for this session."""
    from task_loader import discover_question_ids, load_subset_ids

    config = load_task_config(task_id)
    if question_ids_arg:
        explicit = [qid.strip() for qid in question_ids_arg.split(",") if qid.strip()]
        if sample_limit is not None:
            explicit = explicit[:sample_limit]
        return explicit
    if subset in {"smoke", "calibration"}:
        subset_ids = load_subset_ids(task_id, subset)
        if subset_ids:
            return subset_ids[:sample_limit] if sample_limit else subset_ids
    ids = discover_question_ids(task_id, split=split)
    if sample_limit is not None:
        ids = ids[:sample_limit]
    _ = config  # reserved for future per-task filtering
    return ids


def resolve_agent_api_key(agent_name: str, agent_spec: dict[str, Any]) -> str:
    if agent_spec.get("api_key"):
        return str(agent_spec["api_key"])
    provider = str(agent_spec.get("provider", "")).lower()
    candidates: list[str]
    if provider == "google":
        candidates = ["GOOGLE_API_KEY", "GEMINI_API_KEY"]
    elif agent_name.startswith("deepseek-"):
        candidates = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY"]
    elif agent_name == "glm-5.1":
        candidates = ["BIGMODEL_API_KEY", "GLM_API_KEY", "ZAI_API_KEY", "OPENAI_API_KEY"]
    else:
        candidates = ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "NVDA_API_KEY"]
    for env_name in candidates:
        if os.environ.get(env_name):
            return str(os.environ[env_name])
    raise RuntimeError(f"No API key available for agent {agent_name!r}. Checked {candidates}.")


def check_vqa_submission(
    output_dir: str,
    public_dir: str,
    question_ids: list[str],
    answer_mode: str = "multiple_choice",
) -> dict[str, Any]:
    return check_submission(
        agent_dir=output_dir,
        question_ids=question_ids,
        public_dir=public_dir,
        answer_mode=answer_mode,
    )


def _code_description(code: str) -> str:
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:80]
    return ""


def _classify_phase(code: str, question_ids: list[str], prev_phase: str) -> str:
    lowered = code.lower()
    desc = _code_description(code).lower()
    if "plan.md" in lowered or "candidate" in lowered or "model_info" in lowered or "comparison" in lowered:
        return "S1"
    if "pip install" in lowered or "venv" in lowered or "from_pretrained" in lowered or "snapshot_download" in lowered:
        return "S2"
    if any(token in lowered for token in ("smoke", "sample-limit", "1-10", "validate", "raw output")):
        return "S3"
    if any(qid.lower() in lowered for qid in question_ids[:5]) or "submit_answer" in lowered or "answer.json" in lowered or "predicted_label" in lowered:
        return "S4" if prev_phase in ("S3", "S4") else "S3"
    if "check_submission" in lowered or "completeness" in lowered or "schema" in lowered:
        return "S5"
    if desc.startswith(("s1", "s2", "s3", "s4", "s5")):
        return desc[:2].upper()
    return prev_phase or "S1"


def build_tool_summary(code_executions: list[dict[str, Any]], question_ids: list[str]) -> dict[str, Any]:
    call_log: list[dict[str, Any]] = []
    phase = ""
    for index, execution in enumerate(code_executions):
        phase = _classify_phase(execution.get("code", ""), question_ids, phase)
        call_log.append(
            {
                "seq": index + 1,
                "turn": execution.get("turn"),
                "phase": phase,
                "language": execution.get("language"),
                "exit_code": execution.get("exit_code"),
                "exec_time_s": execution.get("exec_time_s"),
                "description": _code_description(execution.get("code", "")),
            }
        )
    phase_summary: dict[str, dict[str, Any]] = {}
    for entry in call_log:
        phase_entry = phase_summary.setdefault(entry["phase"], {"calls": 0, "errors": 0, "duration_s": 0.0})
        phase_entry["calls"] += 1
        if entry["exit_code"] not in (None, 0):
            phase_entry["errors"] += 1
        if isinstance(entry["exec_time_s"], (int, float)):
            phase_entry["duration_s"] += float(entry["exec_time_s"])
    for value in phase_summary.values():
        value["duration_s"] = round(value["duration_s"], 4)
    return {
        "total": len(call_log),
        "by_tool": {
            "execute_code": len(code_executions),
        },
        "errors": sum(1 for item in code_executions if item.get("exit_code") not in (None, 0)),
        "call_log": call_log,
        "phase_summary": phase_summary,
    }


class AgentBenchmarkRunner:
    def __init__(
        self,
        agent_name: str,
        task: str,
        tier: str,
        subset: str,
        sample_limit: int | None,
        question_ids_arg: str | None,
        split: str,
        output_dir: str | None,
    ) -> None:
        self.tier = get_tier_config(tier)
        self.subset = subset
        self.sample_limit = sample_limit
        self.split = split
        self.question_ids_arg = question_ids_arg

        config_path = resolve_agent_config_path()
        self.config = load_yaml_file(config_path)
        if agent_name not in self.config.get("agents", {}):
            raise ValueError(f"Unknown agent {agent_name!r}")
        self.agent_name = agent_name
        self.agent_cfg = self.config["agents"][agent_name]
        self.model = str(self.agent_cfg["model"])
        self.api_key = resolve_agent_api_key(agent_name, self.agent_cfg)
        self.base_url = self.agent_cfg.get("base_url")
        self.reasoning = bool(self.agent_cfg.get("reasoning", True))
        # BUG-045: reasoning models (minimax, claude, kimi) can burn the
        # default 4096 max_tokens budget on hidden "thinking" tokens mid-run.
        # Allow an explicit override in agent_config.yaml; otherwise bump
        # reasoning-flagged agents to 8192.
        default_max = 8192 if self.reasoning else 4096
        self.max_tokens = int(self.agent_cfg.get("max_tokens", default_max))

        task_config = load_task_config(task)
        self.task_id = task_config["_task_id"]
        self.task_config = task_config
        self.data_root = get_task_data_root(self.task_id)
        self.public_dir = os.path.join(self.data_root, "public")
        self.private_dir = os.path.join(self.data_root, "private")
        self.question_ids = resolve_question_ids(self.task_id, split, sample_limit, question_ids_arg, subset)
        if not self.question_ids:
            raise ValueError("No question IDs resolved for the requested run.")

        self.run_dir = os.path.abspath(output_dir) if output_dir else os.path.join(
            SCRIPT_DIR,
            "runs",
            "agent-benchmark",
            tier,
            agent_name,
            self.task_id,
            time.strftime("%y%m%d-%H%M%S"),
        )
        self.process_dir = os.path.join(self.run_dir, "process")
        self._real_output_dir = os.path.join(self.run_dir, "outputs")
        os.makedirs(self.process_dir, exist_ok=True)
        os.makedirs(self._real_output_dir, exist_ok=True)

        self.output_dir = self._real_output_dir
        self.data_dir = os.path.join(self.run_dir, "public_data")
        if os.path.islink(self.data_dir):
            os.unlink(self.data_dir)
        elif os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        # Stage only the resolved question_ids so `ls data_dir` matches the
        # `sample_count` declared in the prompt. Previously we symlinked the
        # full public/ tree, so agents always discovered all 500/451/1061
        # questions and the `Process ALL discovered questions` instruction
        # forced them to ignore --sample-limit.
        if os.path.isdir(self.public_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            for qid in self.question_ids:
                src = os.path.join(self.public_dir, qid)
                if os.path.isdir(src):
                    os.symlink(src, os.path.join(self.data_dir, qid))
        os.makedirs(os.path.join(self.output_dir, "plan"), exist_ok=True)

        if self.tier.name == "lite":
            requirements_src = os.path.join(task_config["_task_dir"], "requirements.txt")
            if os.path.isfile(requirements_src):
                shutil.copy2(requirements_src, os.path.join(self.output_dir, "requirements.txt"))
        self.run_started_at = utc_now_iso()

        self.system = build_tier_system_prompt(
            task_id=self.task_id,
            tier=self.tier.name,
            question_ids=self.question_ids,
            subset=self.subset,
            sample_limit=self.sample_limit,
            data_dir=self.data_dir,
            output_dir=self.output_dir,
        )

    def run(self) -> dict[str, Any]:
        kickoff = _kickoff_for_tier(self.tier.name)
        messages: list[dict[str, Any]] = [{"role": "user", "content": kickoff}]

        started = time.time()
        total_in = 0
        total_out = 0
        api_calls = 0
        code_executions: list[dict[str, Any]] = []
        trace_events: list[dict[str, Any]] = []

        trace_path = os.path.join(self.process_dir, "trace.jsonl")
        tool_log_path = os.path.join(self.process_dir, "tool_calls.jsonl")
        fail_fast_reason: str | None = None
        fail_fast_warned = False
        smoke_invalid_warned = False
        format_breakdown_count = 0
        breakdown_dir = os.path.join(self.process_dir, "format_breakdown_dumps")
        s3_budget_warned = False
        length_finish_count = 0
        with open(trace_path, "w", encoding="utf-8") as trace_file, open(tool_log_path, "w", encoding="utf-8") as tool_file:
            while True:
                elapsed = time.time() - started
                if elapsed > self.task_config.get("time_limit_s", 3600):
                    break
                # P2 fail-fast: if agent has started producing answers but no
                # smoke_forward.json exists, they skipped S2 smoke. Warn once,
                # then abort to save compute.
                if not fail_fast_reason:
                    try:
                        answers_so_far = sum(
                            1 for _ in _iter_answer_files(self._real_output_dir)
                        )
                    except Exception:
                        answers_so_far = 0
                    smoke_path = os.path.join(self._real_output_dir, "smoke_forward.json")
                    # Hard-block: if smoke_forward.json exists but fails the
                    # verifier (success!=True, empty raw, wall_s<0.3,
                    # placeholder prefix), abort before S3/S4 burn API budget.
                    # Prior behavior only capped S2=0 and let the agent keep
                    # going — wasted $$ on gpt-5.4/slake (raw_output_sample=""
                    # but agent ran full S4 anyway).
                    if os.path.isfile(smoke_path):
                        smoke_check = check_smoke_forward(self._real_output_dir)
                        if not smoke_check.get("valid"):
                            reason = str(smoke_check.get("reason", "invalid"))
                            if not smoke_invalid_warned:
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        "[SMOKE INVALID] Your smoke_forward.json "
                                        f"fails verification: {reason}. "
                                        "Fix the S2 smoke pass (non-empty real "
                                        "decode, wall_s>=0.3, success=true) "
                                        "before writing any more answer.json "
                                        "files — the runner will abort this "
                                        "session on the next check if the "
                                        "artefact is still invalid and you have "
                                        "produced any answers."
                                    ),
                                })
                                smoke_invalid_warned = True
                            elif answers_so_far >= 1:
                                fail_fast_reason = (
                                    f"smoke_invalid: {reason} "
                                    f"(answers_so_far={answers_so_far})"
                                )
                                break
                    if answers_so_far > 0 and not os.path.isfile(smoke_path):
                        if not fail_fast_warned and answers_so_far >= 5:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "[FAIL-FAST WARNING] You have written "
                                        f"{answers_so_far} answer.json files but "
                                        f"{smoke_path} is still missing. S2 smoke "
                                        "forward artefact is mandatory before S4 "
                                        "inference. Stop and produce "
                                        "smoke_forward.json with a real GPU "
                                        "forward pass now, or the runner will "
                                        "abort this session."
                                    ),
                                }
                            )
                            fail_fast_warned = True
                        elif answers_so_far >= 30:
                            fail_fast_reason = (
                                f"fail_fast: {answers_so_far} answers written "
                                "without smoke_forward.json"
                            )
                            break
                # BUG-044: at turn 15, if agent still hasn't committed S3
                # artefacts (answer_postprocess.py OR s3_calibration.json),
                # nudge it to stop debugging and move to S4. Either missing
                # file indicates the agent is stuck in the calibration loop.
                if not s3_budget_warned and len(code_executions) >= 15:
                    pp_path = os.path.join(self._real_output_dir, "answer_postprocess.py")
                    cal_path = os.path.join(self._real_output_dir, "s3_calibration.json")
                    pp_missing = not os.path.isfile(pp_path)
                    cal_missing = not os.path.isfile(cal_path)
                    if pp_missing or cal_missing:
                        missing = []
                        if pp_missing:
                            missing.append("answer_postprocess.py")
                        if cal_missing:
                            missing.append("s3_calibration.json")
                        messages.append({
                            "role": "user",
                            "content": (
                                "[S3 BUDGET WARNING] You have used 15 "
                                "execute_code turns and the following S3 "
                                f"artefact(s) are still missing: {', '.join(missing)}. "
                                "Stop debugging and (a) write a minimal "
                                "answer_postprocess.py, (b) write "
                                "s3_calibration.json from the smoke samples "
                                "you already have, (c) proceed to the S4 "
                                "inference loop over all question_ids. "
                                "Partial results score higher than a stalled "
                                "run with zero outputs."
                            ),
                        })
                        s3_budget_warned = True
                response = call_api(
                    self.api_key,
                    self.model,
                    self.system,
                    messages,
                    TOOLS,
                    reasoning=self.reasoning,
                    base_url=self.base_url,
                    max_tokens=self.max_tokens,
                )
                api_calls += 1
                total_in += int(response["input_tokens"])
                total_out += int(response["output_tokens"])

                if response["finish_reason"] == "length":
                    length_finish_count += 1
                trace_event = {
                    "ts": round(time.time() - started, 4),
                    "type": "api_call",
                    "turn": len(code_executions) + 1,
                    "input_tokens": response["input_tokens"],
                    "output_tokens": response["output_tokens"],
                    "finish_reason": response["finish_reason"],
                    "tool_calls": [tool["name"] for tool in response["tool_calls"]],
                    "content_preview": (response["content"] or "")[:300],
                    "tool_calls_recovered": response.get("tool_calls_recovered", False),
                }
                trace_events.append(trace_event)
                trace_file.write(json.dumps(trace_event) + "\n")
                trace_file.flush()

                assistant_message: dict[str, Any] = {"role": "assistant", "content": response["content"] or None}
                if response["tool_calls"]:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["arguments"]),
                            },
                        }
                        for tool_call in response["tool_calls"]
                    ]
                if response.get("reasoning_details"):
                    assistant_message["reasoning_details"] = response["reasoning_details"]
                messages.append(assistant_message)

                if not response["tool_calls"]:
                    # BUG-040: distinguish clean completion (finish_reason=stop)
                    # from model-side tool-call schema breakdown (finish_reason=
                    # length with no parseable tool_calls — typically qwen
                    # emitting nested <think>/<tool_call>/<function=> XML).
                    # BUG-045: count *consecutive* length-no-tool turns only;
                    # reasoning models (minimax) sometimes burn max_tokens on
                    # thinking for a single turn then recover the next turn.
                    if response["finish_reason"] == "length":
                        format_breakdown_count += 1
                        os.makedirs(breakdown_dir, exist_ok=True)
                        dump_turn = len(code_executions) + 1
                        dump_path = os.path.join(
                            breakdown_dir, f"turn_{dump_turn:03d}.txt"
                        )
                        with open(dump_path, "w", encoding="utf-8") as dump_handle:
                            dump_handle.write(response["content"] or "")
                        if format_breakdown_count >= 2:
                            fail_fast_reason = (
                                f"format_breakdown: agent {self.agent_name} "
                                f"returned finish_reason=length with no parseable "
                                f"tool_calls on {format_breakdown_count} consecutive "
                                f"turns (last at turn {dump_turn}); raw dumps in "
                                f"{breakdown_dir}"
                            )
                            break
                        # Single breakdown — try once more by injecting a
                        # corrective user message so a recoverable model can
                        # get back on track.
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[FORMAT WARNING] Your previous response hit "
                                    "max_tokens with no parseable tool_calls. Use "
                                    "the OpenAI tool_calls API only — do NOT emit "
                                    "<tool_call>, <function=>, or <think> XML "
                                    "tags in your output. Issue a single "
                                    "execute_code call now."
                                ),
                            }
                        )
                        continue
                    # Agent signaled completion by not issuing any further tool call.
                    break

                # BUG-045: a turn with parseable tool_calls resets the
                # consecutive-length-breakdown counter. Only persistent
                # breakdowns (qwen's degraded XML) should fail-fast.
                format_breakdown_count = 0

                for tool_call in response["tool_calls"]:
                    if tool_call["name"] == "execute_code":
                        lang = tool_call["arguments"].get("language", "python")
                        code = tool_call["arguments"].get("code", "")
                        exec_started = time.time()
                        result = execute_code(
                            language=lang,
                            code=code,
                            cwd=self.output_dir,
                            timeout=max(60, int(self.task_config.get("time_limit_s", 3600) - (time.time() - started))),
                        )
                        exec_elapsed = round(time.time() - exec_started, 4)
                        execution_entry = {
                            "turn": len(code_executions) + 1,
                            "language": lang,
                            "code": code,
                            "exit_code": result["exit_code"],
                            "exec_time_s": exec_elapsed,
                            "stdout_preview": result["stdout"][:200],
                        }
                        code_executions.append(execution_entry)
                        tool_file.write(
                            json.dumps(
                                {
                                    "ts": round(time.time() - started, 4),
                                    "tool": "execute_code",
                                    "turn": execution_entry["turn"],
                                    "arguments": tool_call["arguments"],
                                    "result": result,
                                    "exec_time_s": exec_elapsed,
                                }
                            )
                            + "\n"
                        )
                        tool_file.flush()
                        tool_result = json.dumps(result)
                    else:
                        tool_result = json.dumps({"error": f"Unknown tool: {tool_call['name']}"})

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": tool_result,
                        }
                    )

        conversation_payload = {
            "agent": self.agent_name,
            "model": self.model,
            "task": self.task_id,
            "tier": self.tier.name,
            "messages": messages,
            "artifacts_dir": self._real_output_dir,
            "run_dir": self.run_dir,
            "code_executions": code_executions,
            "trace_events": len(trace_events),
            "fail_fast_reason": fail_fast_reason,
            "api_calls": api_calls,
            "length_finish_count": length_finish_count,
            "length_finish_rate": round(length_finish_count / max(api_calls, 1), 4),
        }
        save_json(os.path.join(self.process_dir, "conversation.json"), conversation_payload)

        eval_report = run_eval(
            gt_dir=self.private_dir,
            agent_dir=self._real_output_dir,
            public_dir=self.public_dir,
            question_ids=self.question_ids,
            llm_judge=True,
            conversation=conversation_payload,
            tier=self.tier.name,
            workspace_dir=self._real_output_dir,
            answer_mode=self.task_config.get("answer_mode", "multiple_choice"),
            conversation_path=os.path.join(self.process_dir, "conversation.json"),
            enable_answer_judge=(
                os.environ.get("VQA_ANSWER_JUDGE") in ("1", "true", "True")
            ),
            answer_judge_model=os.environ.get("ANSWER_JUDGE_MODEL") or None,
        )

        tool_summary = build_tool_summary(code_executions, self.question_ids)
        pricing = self.config.get("pricing", {}).get(self.agent_name, {})
        estimated_cost = (
            total_in * float(pricing.get("input_per_1m", 0.0))
            + total_out * float(pricing.get("output_per_1m", 0.0))
        ) / 1_000_000.0
        runtime = {
            "wall_time_s": round(time.time() - started, 4),
            "api_calls": api_calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "estimated_cost_usd": round(estimated_cost, 4),
            "code_executions": len(code_executions),
            "question_count": len(self.question_ids),
            "phase_summary": tool_summary["phase_summary"],
        }

        detail_report = generate_detail_report(
            eval_report=eval_report,
            runtime=runtime,
            agent_name=self.agent_name,
            model=self.model,
            task=self.task_id,
            tool_summary=tool_summary,
            judge_verdict=eval_report.get("llm_judge"),
            tier=self.tier.name,
        )
        save_json(os.path.join(self.run_dir, "report.json"), {
            "generated_at": utc_now_iso(),
            "task": self.task_id,
            "tier": self.tier.name,
            "agent": self.agent_name,
            "model_name": self.model,
            "question_ids": self.question_ids,
            "eval_report": eval_report,
            "detail_report": detail_report,
        })

        summary = {
            "task": self.task_id,
            "tier": self.tier.name,
            "subset": self.subset,
            "agent": self.agent_name,
            "agent_provider": self.agent_cfg.get("provider"),
            "agent_model": self.model,
            "model_name": self.model,
            "question_count": len(self.question_ids),
            "question_ids": self.question_ids,
            "status": "completed",
            "started_at": self.run_started_at,
            "ended_at": utc_now_iso(),
            "fail_fast_reason": fail_fast_reason,
            "completed_outputs": eval_report.get("metrics", {}).get("counts", {}).get("prediction_files", 0),
            "evaluation": {
                "accuracy": eval_report.get("metrics", {}).get("accuracy"),
                "completion_rate": eval_report.get("metrics", {}).get("completion_rate"),
                "parse_rate": eval_report.get("metrics", {}).get("parse_rate"),
                "placeholder_rate": eval_report.get("metrics", {}).get("placeholder_rate"),
                "inference_mode": eval_report.get("metrics", {}).get("inference_mode"),
                "model_call_detected": eval_report.get("metrics", {}).get("model_call_detected"),
                "smoke_forward_passed": eval_report.get("metrics", {}).get("smoke_forward_passed"),
                "step_scores": eval_report.get("step_scores"),
                "rating": eval_report.get("aggregate", {}).get("rating"),
                "resolved": eval_report.get("aggregate", {}).get("resolved"),
                "report_path": os.path.join(self.run_dir, "report.json"),
            },
            "runtime": runtime,
            "tool_summary": tool_summary,
        }
        save_json(os.path.join(self._real_output_dir, "run_summary.json"), summary)
        print_detail_report(detail_report)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one eval_vqa_v2 coding-agent benchmark.")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--task", default="medxpertqa-mm-task")
    parser.add_argument("--tier", default="lite", choices=("lite", "standard"))
    parser.add_argument("--subset", default="all", choices=("all", "smoke", "calibration"))
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--question-ids", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--shared-hf-cache", default=None,
                        help="Shared HF cache dir; overrides per-run HF_HOME to avoid redownloads.")
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR)
    if args.shared_hf_cache:
        shared = os.path.abspath(args.shared_hf_cache)
        os.makedirs(shared, exist_ok=True)
        os.environ["SHARED_HF_CACHE"] = shared
    runner = AgentBenchmarkRunner(
        agent_name=args.agent,
        task=args.task,
        tier=args.tier,
        subset=args.subset,
        sample_limit=args.sample_limit,
        question_ids_arg=args.question_ids,
        split=args.split,
        output_dir=args.output_dir,
    )
    try:
        runner.run()
    except Exception as exc:
        failed_summary = {
            "task": runner.task_id,
            "tier": runner.tier.name,
            "subset": runner.subset,
            "agent": runner.agent_name,
            "agent_provider": runner.agent_cfg.get("provider"),
            "agent_model": runner.model,
            "model_name": runner.model,
            "question_count": len(runner.question_ids),
            "question_ids": runner.question_ids,
            "status": "failed",
            "started_at": runner.run_started_at,
            "ended_at": utc_now_iso(),
            "completed_outputs": 0,
            "error_reason": exc.__class__.__name__,
            "error_message": str(exc),
            "runtime": {
                "wall_time_s": 0.0,
                "question_count": len(runner.question_ids),
                "phase_summary": {},
            },
        }
        save_json(os.path.join(runner._real_output_dir, "run_summary.json"), failed_summary)
        raise


if __name__ == "__main__":
    main()
