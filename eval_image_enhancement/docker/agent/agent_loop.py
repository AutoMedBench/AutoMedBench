#!/usr/bin/env python3
"""Agent loop for the image-enhancement Docker container.

Reads config from environment variables and /workspace/tier_prompt.txt.
Calls the LLM API, dispatches tool calls (execute_code, submit_results),
and writes process logs to /workspace/process/.

Env:
  AGENT_NAME, MODEL, API_KEY, BASE_URL, TASK, TIER, REPEAT_IDX,
  PATIENT_IDS (comma-separated)
"""

import json
import os
import sys
import time

from openai import OpenAI

from agent_code_executor import execute_code


AGENT_NAME  = os.environ.get("AGENT_NAME", "unknown")
MODEL       = os.environ.get("MODEL", "")
API_KEY     = os.environ.get("API_KEY", "")
BASE_URL    = os.environ.get("BASE_URL") or "https://api.openai.com/v1"
TASK        = os.environ.get("TASK", "")
TIER        = os.environ.get("TIER", "lite")
REPEAT_IDX  = os.environ.get("REPEAT_IDX", "0")
PATIENT_IDS = [p for p in os.environ.get("PATIENT_IDS", "").split(",") if p]
MAX_TURNS   = int(os.environ.get("MAX_TURNS", "150"))
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", "3600"))
# Max wall time for a single execute_code call. Prevents one runaway exec
# (hung pip install, huge HF download, infinite loop) from eating the whole
# agent budget without yielding a turn boundary.
EXEC_TIMEOUT = int(os.environ.get("EXEC_TIMEOUT", "600"))

WORKSPACE   = "/workspace"
DATA_PUBLIC = "/data/public"


TOOLS = [
    {"type": "function", "function": {
        "name": "execute_code",
        "description": ("Run Python or bash code in the isolated GPU container. "
                        "Pre-installed: PyTorch, skimage, bm3d, lpips, deepinv, "
                        "transformers. You may `pip install` additional packages. "
                        "Returns stdout and stderr."),
        "parameters": {
            "type": "object",
            "properties": {
                "language": {"type": "string", "enum": ["python", "bash"]},
                "code":     {"type": "string"},
            },
            "required": ["language", "code"],
        }}},
    {"type": "function", "function": {
        "name": "submit_results",
        "description": ("Submit final enhanced outputs. The container's runner "
                        "pre-checks >=50% valid enhanced.npy files before "
                        "accepting; otherwise it returns REJECTED and you may "
                        "fix and retry."),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string",
                            "description": "One-line summary of method used."},
            },
            "required": ["summary"],
        }}},
]


def pre_submit_check(agent_outputs_dir: str, expected_pids: list) -> dict:
    """Count valid enhanced.npy files (2D float, no NaN/Inf)."""
    import numpy as np
    seen = set()
    for pid in expected_pids:
        path = os.path.join(agent_outputs_dir, pid, "enhanced.npy")
        if not os.path.isfile(path):
            continue
        try:
            arr = np.load(path, mmap_mode="r")
            if arr.ndim == 2 and not (np.isnan(arr).any() or np.isinf(arr).any()):
                seen.add(pid)
        except Exception:
            pass
    return {
        "n_valid": len(seen),
        "n_patients_expected": len(expected_pids),
        "expected_dir": agent_outputs_dir,
        "expected_patients": expected_pids,
    }


def call_api(client, model, system, messages, tools):
    """Call the OpenAI-compatible chat completions endpoint."""
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + messages,
        tools=tools, tool_choice="auto",
        max_tokens=4096,
    )


def run_agent() -> int:
    # Read tier prompt (pre-rendered by orchestrator)
    prompt_path = os.path.join(WORKSPACE, "tier_prompt.txt")
    if not os.path.isfile(prompt_path):
        print(f"FATAL: {prompt_path} not found"); return 1
    with open(prompt_path) as f:
        system_prompt = f.read()
    with open(os.path.join(WORKSPACE, "kickoff.txt")) as f:
        kickoff = f.read()

    process_dir = os.path.join(WORKSPACE, "process")
    agent_outputs_dir = os.path.join(WORKSPACE, "agents_outputs")
    for d in (process_dir, agent_outputs_dir):
        os.makedirs(d, exist_ok=True)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "user", "content": kickoff}]

    t_start = time.time()
    turn = 0
    api_times, exec_times = [], []
    total_in = total_out = 0
    violations = 0
    submitted = False
    submit_summary = ""
    tool_log = []

    # Stream tool events so debug info survives a container kill.
    tool_log_f = open(os.path.join(process_dir, "tool_log.jsonl"), "w", buffering=1)
    trace_f    = open(os.path.join(process_dir, "trace.jsonl"),    "w", buffering=1)

    def _emit_tool(entry: dict) -> None:
        tool_log.append(entry)
        tool_log_f.write(json.dumps(entry, default=str) + "\n")

    def _emit_trace(entry: dict) -> None:
        trace_f.write(json.dumps(entry, default=str) + "\n")

    print(f"\n{'='*60}")
    print(f"  IE agent — {TIER.upper()} tier, task={TASK}, repeat={REPEAT_IDX}")
    print(f"  model={MODEL}  base_url={BASE_URL}")
    print(f"  {len(PATIENT_IDS)} patients, max_turns={MAX_TURNS}, max_seconds={MAX_SECONDS}")
    print(f"{'='*60}\n")

    while turn < MAX_TURNS:
        if time.time() - t_start > MAX_SECONDS:
            print(f"[agent] TIME LIMIT {MAX_SECONDS}s at turn {turn}")
            break
        turn += 1

        t_api = time.time()
        try:
            resp = call_api(client, MODEL, system_prompt, messages, TOOLS)
        except Exception as exc:
            print(f"[turn {turn}] API error {type(exc).__name__}: {str(exc)[:200]}")
            time.sleep(3)
            try:
                resp = call_api(client, MODEL, system_prompt, messages, TOOLS)
            except Exception as exc2:
                print(f"[turn {turn}] RETRY FAILED: {str(exc2)[:200]}")
                break
        api_dt = time.time() - t_api
        api_times.append(api_dt)

        msg = resp.choices[0].message
        if resp.usage:
            total_in  += resp.usage.prompt_tokens
            total_out += resp.usage.completion_tokens
        ntc = len(msg.tool_calls or [])
        tok_out = resp.usage.completion_tokens if resp.usage else "?"
        print(f"[turn {turn:02d}] api={api_dt:.1f}s tools={ntc} tok_out={tok_out}")
        _emit_trace({"ts": round(time.time() - t_start, 2), "turn": turn,
                     "api_s": round(api_dt, 2),
                     "prompt_tokens": total_in, "completion_tokens": total_out,
                     "tool_calls": [tc.function.name for tc in (msg.tool_calls or [])]})

        ae = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            ae["tool_calls"] = [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name,
                             "arguments": tc.function.arguments},
            } for tc in msg.tool_calls]
        messages.append(ae)

        if not msg.tool_calls:
            messages.append({"role": "user",
                             "content": "Continue. Use a tool call."})
            continue

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {"_raw": tc.function.arguments}

            if name == "execute_code":
                lang = args.get("language", "python")
                code = args.get("code", "")
                t_exec = time.time()
                result = execute_code(lang, code, cwd=WORKSPACE,
                                      timeout=EXEC_TIMEOUT)
                exec_dt = time.time() - t_exec
                exec_times.append(exec_dt)

                # Sandbox violation handling: 1 warning then kill
                if result["exit_code"] == -1 and "BLOCKED" in result.get("stderr", ""):
                    violations += 1
                    if violations < 2:
                        warn = (f"WARNING ({violations}/1): SANDBOX VIOLATION at turn {turn}.\n"
                                f"Violation: {result['stderr']}\n\n"
                                f"Your code was BLOCKED and NOT executed. The NEXT "
                                f"violation will IMMEDIATELY terminate the run — "
                                f"rating F, all step scores zeroed. Stay within "
                                f"/data/public/ and /workspace/.")
                        print(f"  *** VIOLATION WARNING ({violations}/1): {result['stderr'][:200]}")
                        _emit_tool({"turn": turn, "name": name,
                                    "language": lang, "code": code,
                                    "violation": True, "warning": True})
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": warn})
                        continue
                    else:
                        breach = (f"SANDBOX VIOLATION (2nd offense) at turn {turn}.\n"
                                  f"Violation: {result['stderr']}\n\n"
                                  f"Agent execution terminated. Rating = F, all "
                                  f"step scores zeroed, clinical score = 0.")
                        print(f"  *** ISOLATION BREACH: {result['stderr'][:200]}")
                        _emit_tool({"turn": turn, "name": name,
                                    "language": lang, "code": code,
                                    "violation": True, "kill": True})
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": breach})
                        _flush_logs(process_dir, messages, tool_log, t_start,
                                    turn, api_times, exec_times,
                                    total_in, total_out,
                                    submitted=False, aborted=True,
                                    violations=violations, submit_summary="")
                        return 99

                _emit_tool({"turn": turn, "name": name,
                            "language": lang, "code": code,
                            "result": result,
                            "elapsed_s": round(exec_dt, 2)})
                print(f"  [exec/{lang}] exit={result['exit_code']} took={exec_dt:.1f}s")
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps({
                        "exit_code": result["exit_code"],
                        "stdout":    result["stdout"],
                        "stderr":    result["stderr"],
                        "elapsed_s": round(exec_dt, 2),
                    }),
                })

            elif name == "submit_results":
                pre = pre_submit_check(agent_outputs_dir, PATIENT_IDS)
                if pre["n_valid"] < max(1, int(0.5 * pre["n_patients_expected"])):
                    rej = (f"REJECTED: {pre['n_valid']}/{pre['n_patients_expected']} "
                           f"valid enhanced.npy files. Save each patient output "
                           f"to {pre['expected_dir']}/<pid>/enhanced.npy, then "
                           f"call submit_results again.")
                    _emit_tool({"turn": turn, "name": name, "args": args,
                                "rejected": True, "pre": pre})
                    print(f"  [submit] REJECTED {pre['n_valid']}/{pre['n_patients_expected']}")
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": rej})
                    continue

                submit_summary = args.get("summary", "")
                submitted = True
                _emit_tool({"turn": turn, "name": name, "args": args,
                            "accepted": True, "pre": pre})
                print(f"  [submit] ACCEPTED {pre['n_valid']}/{pre['n_patients_expected']}")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": f"ACCEPTED {pre['n_valid']}/{pre['n_patients_expected']}"})

            else:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps({"error": f"Unknown tool: {name}"})})

        # Checkpoint agent_summary each turn — survives container kill
        _write_agent_summary(process_dir, t_start, turn, api_times, exec_times,
                             total_in, total_out, violations, submitted,
                             aborted=False, submit_summary=submit_summary)

        if submitted:
            break

    tool_log_f.close()
    trace_f.close()
    _flush_logs(process_dir, messages, tool_log, t_start, turn,
                api_times, exec_times, total_in, total_out,
                submitted=submitted, aborted=False,
                violations=violations, submit_summary=submit_summary)

    return 0 if submitted else 1


def _write_agent_summary(process_dir, t_start, turn, api_times, exec_times,
                         total_in, total_out, violations, submitted,
                         aborted, submit_summary):
    summary = {
        "agent":             AGENT_NAME,
        "model":             MODEL,
        "task":              TASK,
        "tier":              TIER,
        "repeat_idx":        REPEAT_IDX,
        "patient_ids":       PATIENT_IDS,
        "turns":             turn,
        "elapsed_s":         round(time.time() - t_start, 2),
        "api_time_s":        round(sum(api_times), 2),
        "exec_time_s":       round(sum(exec_times), 2),
        "n_api_calls":       len(api_times),
        "n_exec_calls":      len(exec_times),
        "prompt_tokens":     total_in,
        "completion_tokens": total_out,
        "violations":        violations,
        "submitted":         submitted,
        "aborted":           aborted,
        "submit_summary":    submit_summary,
    }
    with open(os.path.join(process_dir, "agent_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)


def _flush_logs(process_dir, messages, tool_log, t_start, turn,
                api_times, exec_times, total_in, total_out,
                submitted, aborted, violations, submit_summary):
    _write_agent_summary(process_dir, t_start, turn, api_times, exec_times,
                         total_in, total_out, violations, submitted,
                         aborted, submit_summary)
    # Full OpenAI-format message transcript (system/user/assistant/tool).
    # Named `conversation.json` to match eval_seg's naming; we also keep
    # `messages.json` as an alias so legacy tooling keeps working.
    conv_path = os.path.join(process_dir, "conversation.json")
    with open(conv_path, "w") as f:
        json.dump(messages, f, indent=2, default=str)
    try:
        alias = os.path.join(process_dir, "messages.json")
        if not os.path.exists(alias):
            os.symlink("conversation.json", alias)
    except OSError:
        # Symlinks may fail on read-only mounts; fall back to a copy
        with open(os.path.join(process_dir, "messages.json"), "w") as f:
            json.dump(messages, f, indent=2, default=str)
    print(f"\n[agent] wall={time.time() - t_start:.1f}s turns={turn} "
          f"api_calls={len(api_times)} tokens={total_in}in/{total_out}out "
          f"violations={violations} submitted={submitted}")


if __name__ == "__main__":
    sys.exit(run_agent())
