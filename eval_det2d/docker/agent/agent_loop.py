#!/usr/bin/env python3
"""Agent loop for the Docker container.

Reads configuration from environment variables and /workspace/tier_prompt.txt.
Calls the LLM API, dispatches tool calls (execute_code, submit_results),
and writes process logs to /workspace/process/.

Environment variables:
  AGENT_NAME, MODEL, API_KEY, TASK, TIER, PATIENT_IDS (comma-separated)
  PROVIDER (default: openrouter)
"""

import json
import os
import sys
import time

import requests

from agent_code_executor import execute_code

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
AGENT_NAME = os.environ.get("AGENT_NAME", "unknown")
MODEL = os.environ.get("MODEL", "")
API_KEY = os.environ.get("API_KEY", "")
TASK = os.environ.get("TASK", "")
TIER = os.environ.get("TIER", "lite")
PATIENT_IDS = os.environ.get("PATIENT_IDS", "").split(",")
PROVIDER = os.environ.get("PROVIDER", "openrouter")
EXTRA_BLOCKED = os.environ.get("EXTRA_BLOCKED", "").split(",") if os.environ.get("EXTRA_BLOCKED") else []

WORKSPACE = "/workspace"
DATA_PUBLIC = "/data/public"

# Tool schemas (OpenAI format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Execute Python or bash code in your isolated GPU environment. "
                "Pre-installed: PyTorch, MONAI, nnU-Net, nibabel, etc. "
                "You can pip install additional packages. "
                "Returns stdout and stderr. No timeout on execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "bash"],
                    },
                    "code": {
                        "type": "string",
                        "description": "The code to execute",
                    },
                },
                "required": ["language", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_results",
            "description": (
                "Call this when all outputs are saved and verified. "
                "Signals that the agent has completed and is done."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Kickoff messages per tier
KICKOFF = {
    "lite": "Begin. The model architecture has been chosen for you. "
            "Research it, then follow S1 through S5.",
    "standard": "Begin. Choose from the candidate model families, "
                "then follow S1 through S5.",
    "pro": "Begin. Follow S1 through S5.",
}


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def call_api(api_key, model, system, messages, tools,
             temperature=0.0, reasoning=True):
    """Call OpenRouter with tool use + reasoning."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
    }
    if reasoning:
        payload["reasoning"] = {"enabled": True}

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")

    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning_details = msg.get("reasoning_details")

    tc = []
    if msg.get("tool_calls"):
        for t in msg["tool_calls"]:
            args = t["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            tc.append({
                "id": t["id"],
                "name": t["function"]["name"],
                "arguments": args,
            })

    usage = data.get("usage", {})
    return {
        "content": content,
        "tool_calls": tc,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "finish_reason": choice.get("finish_reason", "stop"),
        "reasoning_details": reasoning_details,
    }


# ---------------------------------------------------------------------------
# Submission checker (lightweight — just checks files exist)
# ---------------------------------------------------------------------------
def check_submission(output_dir, patient_ids):
    """Quick check: do expected output files exist?"""
    agents_out = os.path.join(output_dir, "agents_outputs")
    missing = []
    for pid in patient_ids:
        p = os.path.join(agents_out, pid, "prediction.json")
        if not os.path.isfile(p):
            missing.append(f"{pid}/prediction.json")
    return {
        "complete": len(missing) == 0,
        "missing_predictions": missing,
        "patients_checked": len(patient_ids),
    }


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------
def run_agent():
    """Run the full agent loop."""
    # Read system prompt from file (pre-rendered by orchestrator)
    prompt_path = os.path.join(WORKSPACE, "tier_prompt.txt")
    if not os.path.isfile(prompt_path):
        sys.exit(f"System prompt not found at {prompt_path}")
    with open(prompt_path) as f:
        system_prompt = f.read()

    # Create process log directory
    process_dir = os.path.join(WORKSPACE, "process")
    os.makedirs(process_dir, exist_ok=True)
    os.makedirs(os.path.join(WORKSPACE, "agents_outputs"), exist_ok=True)
    os.makedirs(os.path.join(WORKSPACE, "plan"), exist_ok=True)

    kickoff = KICKOFF.get(TIER, KICKOFF["pro"])
    messages = [{"role": "user", "content": kickoff}]

    t_start = time.time()
    total_in = 0
    total_out = 0
    api_calls = 0
    submitted = False
    violation_warnings = 0     # 1 warning before kill
    code_executions = []

    trace_path = os.path.join(process_dir, "trace.jsonl")
    trace_f = open(trace_path, "w")
    tool_log_path = os.path.join(process_dir, "tool_calls.jsonl")
    tool_log_f = open(tool_log_path, "w")

    def _trace(event_type, data):
        entry = {"ts": round(time.time() - t_start, 2),
                 "type": event_type, **data}
        trace_f.write(json.dumps(entry, default=str) + "\n")
        trace_f.flush()

    def _log_tool(turn, name, arguments, result, exec_time_s=None):
        entry = {
            "ts": round(time.time() - t_start, 2),
            "turn": turn, "tool": name,
            "arguments": arguments, "result": result,
        }
        if exec_time_s is not None:
            entry["exec_time_s"] = exec_time_s
        tool_log_f.write(json.dumps(entry, default=str) + "\n")
        tool_log_f.flush()

    print(f"\n{'='*60}")
    print(f"  MedAgentsBench Agent Container — {TIER.upper()} tier")
    print(f"  Agent: {AGENT_NAME}  Model: {MODEL}")
    print(f"  Task: {TASK}  Patients: {len(PATIENT_IDS)}")
    print(f"{'='*60}\n")

    reasoning = os.environ.get("REASONING", "true").lower() == "true"

    for turn in range(999999):
        elapsed = time.time() - t_start

        # API call with retry
        try:
            resp = call_api(API_KEY, MODEL, system_prompt, messages,
                            TOOLS, reasoning=reasoning)
        except Exception as e:
            print(f"\n[Agent] API ERROR: {e}")
            time.sleep(5)
            try:
                resp = call_api(API_KEY, MODEL, system_prompt, messages,
                                TOOLS, reasoning=reasoning)
            except Exception as e2:
                print(f"[Agent] RETRY FAILED: {e2} — stopping.")
                break

        api_calls += 1
        total_in += resp["input_tokens"]
        total_out += resp["output_tokens"]

        _trace("api_call", {
            "turn": turn + 1,
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
            "finish_reason": resp["finish_reason"],
            "tool_calls": [tc["name"] for tc in resp["tool_calls"]],
            "content_preview": (resp["content"] or "")[:300],
        })

        if resp["content"]:
            preview = resp["content"][:200].replace("\n", " ")
            print(f"  [Turn {turn+1} | {elapsed:.0f}s] {preview}...")

        if not resp["tool_calls"]:
            print(f"  [Turn {turn+1}] No tool calls — agent stopped.")
            break

        # Build assistant message
        asst_msg = {"role": "assistant", "content": resp["content"] or None}
        if resp["reasoning_details"]:
            asst_msg["reasoning_details"] = resp["reasoning_details"]
        if resp["tool_calls"]:
            asst_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc["arguments"])}}
                for tc in resp["tool_calls"]
            ]
        messages.append(asst_msg)

        # Execute tool calls
        for tc in resp["tool_calls"]:
            name = tc["name"]
            args = tc["arguments"]

            if name == "execute_code":
                lang = args.get("language", "python")
                code = args.get("code", "")
                code_preview = code[:80].replace("\n", "\\n")
                print(f"  [Code:{lang}] {code_preview}...")

                t_exec = time.time()
                result = execute_code(
                    lang, code, cwd=WORKSPACE,
                    extra_blocked=EXTRA_BLOCKED,
                )
                exec_time = time.time() - t_exec

                code_executions.append({
                    "turn": turn + 1,
                    "language": lang,
                    "exit_code": result["exit_code"],
                    "exec_time_s": round(exec_time, 1),
                })

                _trace("code_exec", {
                    "turn": turn + 1,
                    "language": lang,
                    "exit_code": result["exit_code"],
                    "exec_time_s": round(exec_time, 1),
                    "code_lines": code.count("\n") + 1,
                    "stdout_len": len(result["stdout"]),
                    "stderr_len": len(result["stderr"]),
                })

                # ISOLATION VIOLATION — 1 warning, then kill
                if result["exit_code"] == -1 and "BLOCKED" in result.get("stderr", ""):
                    violation_warnings += 1

                    if violation_warnings < 2:
                        # ── First violation: WARNING ──
                        warning_msg = (
                            f"WARNING ({violation_warnings}/1): SANDBOX VIOLATION at turn {turn+1}.\n"
                            f"Violation: {result['stderr']}\n\n"
                            f"Your code was BLOCKED and NOT executed. You may continue,\n"
                            f"but the NEXT violation will IMMEDIATELY TERMINATE your run:\n"
                            f"  - All remaining S-step scores → 0\n"
                            f"  - Clinical score → 0\n"
                            f"  - Rating → F (Disqualified)\n"
                            f"Fix your approach and stay within /data/public/ and /workspace/."
                        )
                        print(f"\n  *** VIOLATION WARNING ({violation_warnings}/1) at turn {turn+1} ***")
                        print(f"  {result['stderr']}")
                        print(f"  Agent warned — next violation = kill.\n")
                        _trace("violation_warning", {
                            "turn": turn + 1,
                            "warning_number": violation_warnings,
                            "violation": result["stderr"],
                        })
                        result["stderr"] = warning_msg
                        tool_result = json.dumps(result)
                        _log_tool(turn + 1, name, args, result,
                                  exec_time_s=round(exec_time, 1))
                        # Continue — don't kill
                    else:
                        # ── Second violation: KILL ──
                        breach_msg = (
                            f"SANDBOX VIOLATION DETECTED at turn {turn+1} (2nd offense).\n"
                            f"Violation: {result['stderr']}\n\n"
                            f"PENALTY: Agent execution terminated immediately.\n"
                            f"  - All remaining S-step scores → 0\n"
                            f"  - Clinical score → 0\n"
                            f"  - Rating → F (Disqualified)\n"
                            f"  - Resolved → False\n"
                            f"Any partial outputs produced before this violation are "
                            f"discarded for scoring purposes."
                        )
                        print(f"\n  *** ISOLATION BREACH at turn {turn+1} (2nd violation) ***")
                        print(f"  {result['stderr']}")
                        print(f"  KILLED — all remaining scores → 0.\n")
                        _trace("isolation_breach", {
                            "turn": turn + 1,
                            "violation": result["stderr"],
                            "penalty": "zero_all_remaining_scores",
                            "prior_warnings": violation_warnings - 1,
                        })
                        result["stderr"] = breach_msg
                        _log_tool(turn + 1, name, args, result,
                                  exec_time_s=round(exec_time, 1))
                        trace_f.close()
                        tool_log_f.close()
                        sys.exit(99)

                status = "OK" if result["exit_code"] == 0 else f"FAIL(rc={result['exit_code']})"
                print(f"           {status} ({exec_time:.1f}s)")
                if result["stdout"].strip():
                    for line in result["stdout"].strip().split("\n")[-5:]:
                        print(f"           > {line[:100]}")
                if result["exit_code"] != 0 and result["stderr"]:
                    for line in result["stderr"].strip().split("\n")[-3:]:
                        print(f"           ! {line[:100]}")

                tool_result = json.dumps(result)
                _log_tool(turn + 1, name, args, result,
                          exec_time_s=round(exec_time, 1))

            elif name == "submit_results":
                check = check_submission(WORKSPACE, PATIENT_IDS)
                print(f"  [Submit] complete={check['complete']} "
                      f"missing={len(check['missing_predictions'])}")
                tool_result = json.dumps(check)
                submitted = True
                _trace("submit", check)
                _log_tool(turn + 1, name, args, check)
            else:
                tool_result = json.dumps({"error": f"Unknown tool: {name}"})

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_result,
            })

        if submitted:
            print(f"\n  [Turn {turn+1}] Agent called submit_results — done.")
            break

    wall_time = time.time() - t_start

    _trace("run_end", {
        "wall_time_s": round(wall_time, 2),
        "api_calls": api_calls,
        "total_in": total_in,
        "total_out": total_out,
        "submitted": submitted,
    })
    trace_f.close()
    tool_log_f.close()

    # Save conversation
    conv_path = os.path.join(process_dir, "conversation.json")
    with open(conv_path, "w") as f:
        json.dump({
            "agent": AGENT_NAME,
            "model": MODEL,
            "task": TASK,
            "tier": TIER,
            "system_prompt": system_prompt,
            "messages": messages,
            "code_executions": code_executions,
        }, f, indent=2, default=str)

    print(f"\n[Agent] Wall time: {wall_time:.1f}s  API calls: {api_calls}  "
          f"Tokens: {total_in}in/{total_out}out")
    print(f"[Agent] Submitted: {submitted}")

    return 0 if submitted else 1


if __name__ == "__main__":
    sys.exit(run_agent())
